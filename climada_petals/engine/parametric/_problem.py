"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

The data of one design problem, its scaled arrays, its solution and its record.

Shared by the routes of :mod:`~climada_petals.engine.parametric.solvers`:
the linear programs there, and the mean squared error's ``SLSQP`` route,
and by the grid route. :mod:`~climada_petals.engine.parametric.solvers`
exports :class:`DesignProblem`, :class:`Solution` and :class:`SolverInfo`,
and :mod:`~climada_petals.engine.parametric.calibrate` exports
:class:`CoverProblem` and :class:`DesignSample`, the records of a
calibration result.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .measure import OutcomeMeasure
from .objectives import Objective
from .parameters import AmountConstraints
from .sample import OutcomeSample, SplitRecord

__all__ = [
    "MIN_SCALE_LOSS",
    "CoverProblem",
    "DesignProblem",
    "DesignSample",
    "Scaled",
    "Solution",
    "SolverInfo",
    "objective_norm",
]

MIN_SCALE_LOSS: Final[float] = 1e-6
"""The problem is scaled by its largest loss in one row only above this value."""


@dataclass(frozen=True)
class SolverInfo:
    """How a design problem was solved.

    Parameters
    ----------
    route : {"lp", "qp"}
        One linear program, or the quadratic program of the mean squared
        error.
    solver : str
        ``"scipy-highs"`` or ``"scipy-slsqp"``.
    version : str
        The version of scipy.
    method : str or None
        The ``linprog`` method, ``"highs"`` or ``"highs-ipm"``; ``"slsqp"``
        for the quadratic program.
    status : str
        The solver's status; always a success in a returned solution.
    message : str
        The solver's message.
    iterations : int or None
        Iterations, where the solver reports them; on the quadratic program,
        summed over the rounds.
    seconds : float
        Wall time of the assembly and the solve.
    rounds : int or None, default None
        The quadratic program's ``SLSQP`` solves: 1, or one per cutting
        plane under a cost-of-capital load (0 if no amount is free). None
        for a linear program.
    """

    route: str
    solver: str
    version: str
    method: str | None
    status: str
    message: str
    iterations: int | None
    seconds: float
    rounds: int | None = None


@dataclass(frozen=True, eq=False)
class DesignProblem:
    """The data of one design problem, in loss units.

    Parameters
    ----------
    problem_id : str
        A unit id (individual scope) or ``"joint"``.
    basis : scipy.sparse.csr_matrix
        ``B``, shape ``(n_rows, n_amounts)``: the units' bases side by side,
        so the problem's payout of row ``r`` is ``B[r] @ amounts``.
    loss : numpy.ndarray
        The problem's loss per row, summed over its units, >= 0.
    constraints : sequence of AmountConstraints
        One per unit, in the order of the amounts.
    measure : OutcomeMeasure
        The outcomes of the rows and their weights.
    bound : float or None
        The largest premium, or None without a budget.
    """

    problem_id: str
    basis: sp.csr_matrix
    loss: NDArray[np.float64]
    constraints: Sequence[AmountConstraints]
    measure: OutcomeMeasure
    bound: float | None = field(default=None)

    @property
    def scale(self) -> float:
        """The loss scale, the largest loss in one row, or 1 if it is tiny."""
        largest = float(self.loss.max(initial=0.0))
        return largest if largest > MIN_SCALE_LOSS else 1.0

    @cached_property
    def lower(self) -> NDArray[np.float64]:
        """Lower bounds of the amounts, side by side."""
        return np.concatenate([part.lower for part in self.constraints])

    @cached_property
    def upper(self) -> NDArray[np.float64]:
        """Upper bounds of the amounts, side by side; ``inf`` where none."""
        return np.concatenate([part.upper for part in self.constraints])

    @cached_property
    def rows(self) -> tuple[sp.csr_matrix, NDArray[np.float64]]:
        """The units' rows ``a_ub @ amounts <= b_ub``, block-diagonal and sparse."""
        a_ub = sp.block_diag(
            [sp.csr_matrix(part.a_ub) for part in self.constraints], format="csr"
        )
        b_ub = np.concatenate([part.b_ub for part in self.constraints])
        return sp.csr_matrix(a_ub, shape=(b_ub.size, self.lower.size)), b_ub

    def scaled(self) -> Scaled:
        """Return the problem's data divided by its loss scale."""
        scale = self.scale
        rows, rhs = self.rows
        return Scaled(
            loss=self.loss / scale,
            basis=sp.csr_matrix(self.basis),
            lower=self.lower / scale,
            upper=self.upper / scale,
            rows=rows,
            rhs=rhs / scale,
            event_weight=np.asarray(self.measure.event_weight, dtype=np.float64),
            group=self.measure.grouping,
            weights=np.asarray(self.measure.weights, dtype=np.float64),
        )


@dataclass(frozen=True, eq=False)
class Solution:
    """The solution of one design problem, in loss units.

    Parameters
    ----------
    amounts : numpy.ndarray
        The amounts as the solver returned them, shape ``(n_amounts,)``,
        before they are snapped to their bounds.
    objective_value : float
        The solver's optimal objective value.
    info : SolverInfo
        Route, solver, method, status and timing.
    unique : bool or None, default None
        Whether the identified amounts are unique: the mean squared error's
        Hessian test, or on the LP route the ranges' two-tolerance test.
        None where the route doesn't determine it.
    ranges : numpy.ndarray or None, default None
        On the LP route with ``ranges=True``: each amount's smallest and
        largest value over the near-optimal set, in loss units, shape
        ``(n_amounts, 2)``, NaN for an amount that is not identified
        (:attr:`Scaled.identified`). None otherwise.
    """

    amounts: NDArray[np.float64]
    objective_value: float
    info: SolverInfo
    unique: bool | None = None
    ranges: NDArray[np.float64] | None = None


@dataclass(frozen=True, eq=False)
class Scaled:
    """The scaled arrays every formulation needs."""

    loss: NDArray[np.float64]  # (n,) problem loss / scale
    basis: sp.csr_matrix  # (n, amounts)
    lower: NDArray[np.float64]  # (amounts,) / scale
    upper: NDArray[np.float64]  # (amounts,) / scale
    rows: sp.csr_matrix  # (r, amounts)
    rhs: NDArray[np.float64]  # (r,) / scale
    event_weight: NDArray[np.float64]  # (n,)
    group: sp.csr_matrix  # (O, n)
    weights: NDArray[np.float64]  # (O,)

    @property
    def expected_loss(self) -> float:
        """``E[L]`` in scaled units."""
        return float(self.event_weight @ self.loss)

    @property
    def premium_row(self) -> NDArray[np.float64]:
        """``E[P] = premium_row @ theta``, shape ``(n_amounts,)``."""
        return np.asarray(self.basis.T @ self.event_weight, dtype=np.float64)

    @property
    def identified(self) -> NDArray[np.bool_]:
        """The identified amounts, those not fixed and reached by a row with a weight."""
        weighted = (self.event_weight > 0.0).astype(np.float64)
        reached = np.asarray(abs(self.basis).T @ weighted).reshape(-1) > 0.0
        return (self.lower < self.upper) & reached

    @property
    def budget_unit(self) -> float:
        """The divisor of the budget row, ``E[L]``, or 1 if it is 0."""
        expected = self.expected_loss
        return expected if expected > 0.0 else 1.0


def objective_norm(objective: Objective, arr: Scaled, measure: OutcomeMeasure) -> float:
    """Return the objective of the zero cover in scaled units, or 1 if it is 0.

    The solver minimises the objective divided by this value, which is then
    at most 1 at the optimum, whatever the weights: with 10,000 simulation
    years an outcome has weight ``1e-4``, and an unnormalised objective
    would be close to the solvers' tolerances.
    """
    value = objective.evaluate(arr.loss, np.zeros_like(arr.loss), measure)
    return value if value > 0.0 else 1.0


@dataclass(frozen=True)
class CoverProblem:
    """The solution of one design problem.

    Parameters
    ----------
    problem_id : str
        The unit id as text (individual scope), or ``"joint"``.
    unit_id : tuple of str or int
        The units of the problem.
    objective_value : float
        The objective of the resolved cover on the design sample, in loss
        units (squared for the mean squared error), recomputed with numpy.
    premium : float
        The premium of the problem's payout under the design's pricing, on
        the design sample.
    expected_loss : float
        ``E[L]`` of the problem's loss on the design sample.
    budget_bound : float or None
        The largest premium the budget allows, or None without a budget.
    loss_weight : float
        The weight of the outcomes with a loss: ``P(L > 0)`` on an annual
        route, the frequency of the events with a loss on the event basis.
    degenerate_tail : {"expected_uncovered", "single_outcome"} or None
        The weighted CVaR's degenerate tail (module docstring), else None.
    solver : SolverInfo
        Route, solver, version, method, status and time.
    unique : bool or None, default None
        Whether the problem's identified amounts are unique: on the mean
        squared error's route the Hessian test, on the LP route the ranges'
        test (``ranges=True``), on a grid false with ties, else the route's
        report at the chosen candidate (on the numpy grid, true). None where
        the route doesn't determine it.
    ties : int or None, default None
        On a grid, the candidates tied with the chosen one; None without a
        grid.
    n_candidates : int or None, default None
        On a grid, its candidates; None without a grid.
    ranges : tuple or None, default None
        On the LP route with ``ranges=True``: ``(unit_id, amount, low, high)``
        for each identified amount, its smallest and largest value over the
        near-optimal set, in payout units. None otherwise.
    """

    problem_id: str
    unit_id: tuple[Hashable, ...]
    objective_value: float
    premium: float
    expected_loss: float
    budget_bound: float | None
    loss_weight: float
    degenerate_tail: str | None
    solver: SolverInfo
    unique: bool | None = None
    ties: int | None = None
    n_candidates: int | None = None
    ranges: tuple[tuple[Hashable, str, float, float], ...] | None = None

    @property
    def budget_use(self) -> float | None:
        """``premium / budget_bound``; None without a budget, NaN for a bound of 0."""
        if self.budget_bound is None:
            return None
        return self.premium / self.budget_bound if self.budget_bound > 0 else np.nan


@dataclass(frozen=True, eq=False)
class DesignSample:
    """What a design records of its design sample.

    Enough to state the setup of a design, and for
    :func:`~climada_petals.engine.parametric.sample.check_out_of_sample` to
    compare an evaluation sample with it.

    Parameters
    ----------
    basis : {"event", "annual"}
        What an outcome is.
    route : {"frequency", "catalogue", "yearsets"}
        How the outcomes were built.
    n_years : int or None
        The years the outcomes stand for; None on the event basis.
    n_events : int
        The sample's events.
    n_draws : int
        Draws per event.
    n_outcomes : int
        The outcomes of the measure.
    scenario : str
        The sample's label.
    event_id : numpy.ndarray of int
        The sample's events.
    block : numpy.ndarray
        Each event's block, the unit a split keeps together.
    years : numpy.ndarray of int or None
        The simulation years of a sample on catalogue years; else None.
    split : SplitRecord or None
        How the sample was cut from another; None for a whole sample.
    source : str or None, default None
        The sample's source token
        (:attr:`~climada_petals.engine.parametric.sample.OutcomeSample.source`).
        A record without one is compared with no evaluation sample by its
        events and blocks.
    parents : mapping of int to str, default empty
        The parent storm of each of the sample's events that has one
        (:attr:`~climada_petals.engine.parametric.sample.OutcomeSample.parents`),
        compared with every evaluation sample's.
    """

    basis: str
    route: str
    n_years: int | None
    n_events: int
    n_draws: int
    n_outcomes: int
    scenario: str
    event_id: NDArray[np.int64]
    block: NDArray[Any]
    years: NDArray[np.int64] | None
    split: SplitRecord | None
    source: str | None = None
    parents: Mapping[int, str] = field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: OutcomeSample) -> DesignSample:
        """Return the record of a sample.

        Parameters
        ----------
        sample : OutcomeSample
            The design sample.

        Returns
        -------
        DesignSample
            Its setup, events, blocks, years, split, source token and the
            parent storms of its events.
        """
        return cls(
            basis=sample.kind,
            route=sample.route,
            n_years=sample.n_years,
            n_events=sample.n_events,
            n_draws=sample.n_draws,
            n_outcomes=sample.measure.n_outcomes,
            scenario=sample.scenario,
            event_id=sample.event_id,
            block=sample.block,
            years=sample.years,
            split=sample.split,
            source=sample.source,
            parents={
                event: sample.parents[event]
                for event in sample.event_id.tolist()
                if event in sample.parents
            },
        )
