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

The data of one design problem, its scaled arrays and its solution.

Shared by the routes of :mod:`~climada_petals.engine.parametric.solvers`:
the linear programs there, and the mean squared error's ``SLSQP`` route.
:mod:`~climada_petals.engine.parametric.solvers` exports
:class:`DesignProblem`, :class:`Solution` and :class:`SolverInfo`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Final

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .measure import OutcomeMeasure
from .objectives import Objective
from .parameters import AmountConstraints

__all__ = [
    "MIN_SCALE_LOSS",
    "DesignProblem",
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

    Attributes
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

    Attributes
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
        """Loss scale: the largest loss in one row, or 1 if it is tiny."""
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

    Attributes
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
        Hessian test. None where the route doesn't determine it.
    """

    amounts: NDArray[np.float64]
    objective_value: float
    info: SolverInfo
    unique: bool | None = None


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
    def budget_unit(self) -> float:
        """Divisor of the budget row: ``E[L]``, or 1 if it is 0."""
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
