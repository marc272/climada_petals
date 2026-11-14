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

Calibrate a parametric cover: choose the free parameters of a template.

:func:`calibrate` takes a template, a
:class:`~climada_petals.engine.parametric.cover.ParametricCover` with
:class:`~climada_petals.engine.parametric.parameters.Free` parameters, and a
design sample, an :class:`~climada_petals.engine.parametric.sample.OutcomeSample`
on the template's index definition and units. It returns a
:class:`CalibrationResult`, which holds the resolved cover and how it was
found. (``climada.util.calibrate`` calibrates impact functions and is
unrelated.)

The scope
---------
The **scope** says which units share a design problem. It is required.

``"individual"``
    One design problem per unit: its own objective on its own loss and
    payout, its own budget and its own premium.
``"joint"``
    One design problem for one risk owner. Each unit keeps its own index
    and its own payout function, and pays by its own terms, but the
    objective is on the owner's **total** per event,
    ``L_e = sum_j L_{e,j}`` and ``P_e = sum_j P_{e,j}``, under one budget on
    the total premium. Within an event, an over-payment at one unit offsets
    a shortfall at another.

With one ``Budget(ratio=kappa)`` and under the pure premium, both scopes
spend the same total. The individual designs taken together meet the joint
problem's constraints (under a cost-of-capital load too, because the loaded
premium of a total is at most the sum of the loaded premiums), so on the
design sample the joint optimum is at most their objective on the total.
That is a property of the optimisation; nothing here says which scope gives
the better cover on other samples.

**A joint design's per-unit amounts are often not unique.** The joint
objective depends on the units' payouts only through their total per event,
so other splits across the units are often optimal too. The cover holds the
one split the solver returned. The evaluation
(:func:`~climada_petals.engine.parametric.evaluate.evaluate`) gives a joint
design a total row, and per-unit rows only under a stated allocation.

Routes
------
This version has the exact route for a template whose payouts are linear
in their amounts
(:class:`~climada_petals.engine.parametric.payout.LinearInAmounts`: the four
built-in families) with every shape parameter, such as a breakpoint, fixed:

* ``ExpectedUncovered`` and ``WeightedCVaR``: one linear program per design
  problem, with scipy's HiGHS
  (:mod:`~climada_petals.engine.parametric.solvers`);
* ``MSE``: one quadratic program, with the optional dependency cvxpy.

A free shape parameter, a payout that is not linear in its amounts, and
``CalibrationOptions(ranges=True)`` take routes this version doesn't have,
and raise. In a joint design, thresholds are always fixed.

The constraints are the template's own
(:mod:`~climada_petals.engine.parametric.constraints`): each amount's bounds
and its family's rows, such as monotone levels; the cover's
:class:`~climada_petals.engine.parametric.cover.Limit`, with
``Limit.largest_loss()`` resolved into each unit's largest loss on the
design sample; and the premium budget,
:class:`~climada_petals.engine.parametric.constraints.Budget`. ``E[U]`` and
the weighted CVaR need a budget or bounded amounts: otherwise any payout at
least as large as the loss would be optimal.

No silent failures
------------------
After a solve, numpy recomputes the objective, the premium, the budget, the
limit and every amount constraint from the returned amounts, within
``CalibrationOptions.feasibility_tolerance`` (relative). A violation, or a
solver status other than optimal, raises
:class:`~climada_petals.engine.parametric.errors.CalibrationError` or one of
its subclasses. No problem is skipped and nothing falls back to other
amounts. The amounts are snapped to their bounds and their monotone order
before they go into the cover, since a solver meets them only to its
tolerance.

Non-uniqueness and degenerate tails
-----------------------------------
* **Unidentified amounts.** An amount whose basis is zero on every row of
  the design sample with a weight, such as a step no event reaches, changes
  nothing on the sample. :attr:`CalibrationResult.unidentified` lists them,
  and the fill rule sets them: ``fill="lowest"`` (the default) the smallest
  values the constraints allow, which under monotone levels is the previous
  level, or ``"highest"`` the largest. The evaluation reports how much of a
  sample lands on them (``unidentified_share``).
* ``unique`` and ``ties`` are None in this version: the ranges of the LP
  route, the uniqueness test of the mean squared error and the grid route
  come later.
* **Degenerate tails** of the weighted CVaR are recorded per design
  problem, not raised: ``"expected_uncovered"`` when the tail holds every
  outcome with a loss (``1 - beta`` at least their weight), so the design
  minimises ``E[U]``; ``"single_outcome"`` when ``1 - beta`` is at most the
  smallest weight of an outcome with a loss, so the CVaR is the worst
  outcome alone.

The objective, the premium and the budget all use the sample's outcome
measure (:mod:`~climada_petals.engine.parametric.measure`): on an annual
route the events of a year are summed and the CVaR is a CVaR over years; on
the event basis the weights are the frequencies, whatever they sum to.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from ._checks import check_real
from .constraints import (
    FILLS,
    Budget,
    UnitAmounts,
    check_bounded,
    check_unpaid_at_zero,
    final_amounts,
    resolve_limit,
    resolved_cover,
    unit_amounts,
)
from .cover import ParametricCover
from .errors import CalibrationError, ParametricInputError
from .index import ParametricIndex
from .objectives import MSE, ExpectedUncovered, Objective, WeightedCVaR, tail_degeneracy
from .payout import LinearInAmounts
from .pricing import CostOfCapitalLoad, Pricing, PurePremium
from .sample import OutcomeSample, SplitRecord
from .solvers import (
    LP_METHODS,
    DesignProblem,
    Solution,
    SolverInfo,
    check_amounts,
    lp_method,
    solve_design,
    verify_solution,
)

__all__ = [
    "DEGENERATE_TAILS",
    "JOINT",
    "SCOPES",
    "CalibrationOptions",
    "CalibrationResult",
    "CoverProblem",
    "DesignSample",
    "calibrate",
]

SCOPES: Final[tuple[str, ...]] = ("individual", "joint")
"""The calibration scopes (module docstring)."""

JOINT: Final[str] = "joint"
"""The joint scope, and the ``problem_id`` of its single design problem."""

DEGENERATE_TAILS: Final[tuple[str, ...]] = ("expected_uncovered", "single_outcome")
"""The values of :attr:`CoverProblem.degenerate_tail`, besides None."""


@dataclass(frozen=True, kw_only=True)
class CalibrationOptions:
    """How :func:`calibrate` solves and reports a design.

    Parameters
    ----------
    fill : {"lowest", "highest"}, default "lowest"
        The rule that sets unidentified amounts (module docstring).
    ranges : bool, default False
        Whether to range each amount over the optimal set. Not in this
        version: True raises.
    lp_method : {"auto", "highs", "highs-ipm"}, default "auto"
        The ``linprog`` method of the linear programs: ``"auto"`` takes
        ``"highs"`` for a problem with one unit and ``"highs-ipm"`` for
        several. The method used is recorded per design problem.
    feasibility_tolerance : float, default 1e-7
        The relative tolerance of the checks after the solve (module
        docstring), > 0.
    time_limit : float, optional
        Seconds each linear program may take; reaching it raises
        :class:`~climada_petals.engine.parametric.errors.CalibrationLimitError`.
        Default: no limit.

    Raises
    ------
    ParametricInputError
        If an option is invalid.
    """

    fill: str = "lowest"
    ranges: bool = False
    lp_method: str = "auto"
    feasibility_tolerance: float = 1e-7
    time_limit: float | None = None

    def __post_init__(self) -> None:
        """Validate the options."""
        if self.fill not in FILLS:
            raise ParametricInputError(
                f"fill must be one of {list(FILLS)}, got {self.fill!r}"
            )
        if not isinstance(self.ranges, bool):
            raise ParametricInputError(f"ranges must be a bool, got {self.ranges!r}")
        if self.lp_method not in LP_METHODS:
            raise ParametricInputError(
                f"lp_method must be one of {list(LP_METHODS)}, got {self.lp_method!r}"
            )
        object.__setattr__(
            self,
            "feasibility_tolerance",
            check_real(
                "feasibility_tolerance",
                self.feasibility_tolerance,
                low=0.0,
                include_low=False,
            ),
        )
        if self.time_limit is not None:
            object.__setattr__(
                self,
                "time_limit",
                check_real("time_limit", self.time_limit, low=0.0, include_low=False),
            )


@dataclass(frozen=True)
class CoverProblem:
    """The solution of one design problem.

    Attributes
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
    :func:`~climada_petals.engine.parametric.sample.check_no_leakage` to
    compare an evaluation sample with it.

    Attributes
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

    @classmethod
    def from_sample(cls, sample: OutcomeSample) -> DesignSample:
        """Return the record of a sample."""
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
        )


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    """A design: the resolved cover and how it was found.

    Attributes
    ----------
    cover : ParametricCover
        The resolved cover: the template with every free parameter set and
        ``Limit.largest_loss()`` written as numbers. It pays by its own
        terms on any sample (:meth:`ParametricCover.payout_matrix`).
    template_hash : str
        ``definition_hash`` of the template.
    objective : MSE, ExpectedUncovered or WeightedCVaR
        What the design minimises; :meth:`summary` holds its document.
    pricing : PurePremium or CostOfCapitalLoad
        The pricing of the premium and the budget.
    budget : Budget or None
        The premium budget, or None.
    scope : {"individual", "joint"}
        Which units shared a design problem.
    problems : tuple of CoverProblem
        One per design problem: one per unit, or one for the joint scope.
    calibrated : mapping of unit id to tuple of str
        The parameters the calibration set, per unit: the template's free
        parameters.
    unidentified : mapping of unit id to tuple of str
        The calibrated amounts that no outcome of the design sample reaches,
        per unit (module docstring).
    fill : {"lowest", "highest"}
        The rule that set the unidentified amounts.
    design_sample : DesignSample
        The basis, route, size, events and blocks of the design sample.
    unique : bool or None, default None
        Whether every design problem's identified amounts are unique. None:
        not determined in this version.
    ties : int or None, default None
        Grid candidates tied with the best. None: no grid.
    """

    cover: ParametricCover
    template_hash: str
    objective: Objective
    pricing: Pricing
    budget: Budget | None
    scope: str
    problems: tuple[CoverProblem, ...]
    calibrated: Mapping[Hashable, tuple[str, ...]]
    unidentified: Mapping[Hashable, tuple[str, ...]]
    fill: str
    design_sample: DesignSample
    unique: bool | None = None
    ties: int | None = None

    def __post_init__(self) -> None:
        """Check the scope and store the mappings read-only."""
        if self.scope not in SCOPES:
            raise ParametricInputError(
                f"scope must be one of {list(SCOPES)}, got {self.scope!r}"
            )
        object.__setattr__(self, "problems", tuple(self.problems))
        for name in ("calibrated", "unidentified"):
            mapping = {
                unit: tuple(names) for unit, names in getattr(self, name).items()
            }
            object.__setattr__(self, name, MappingProxyType(mapping))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy through the constructor, with the mappings as dicts."""
        fields = {
            item.name: getattr(self, item.name) for item in dataclasses.fields(self)
        }
        fields["calibrated"] = dict(self.calibrated)
        fields["unidentified"] = dict(self.unidentified)
        return (partial(CalibrationResult, **fields), ())

    @property
    def objective_value(self) -> float:
        """The sum of the design problems' objective values.

        In the joint scope this is the one problem's value; in the
        individual scope, what the separate designs minimise together.
        """
        return float(sum(problem.objective_value for problem in self.problems))

    @property
    def premium(self) -> float:
        """The sum of the design problems' premiums, on the design sample.

        In the joint scope this is the one problem's premium. In the
        individual scope it is the price of buying each unit's cover
        separately; under a cost-of-capital load the premium of the combined
        payout can be lower, because the load is sub-additive.
        """
        return float(sum(problem.premium for problem in self.problems))

    @property
    def expected_loss(self) -> float:
        """``E[L]`` of the owner's total loss on the design sample."""
        return float(sum(problem.expected_loss for problem in self.problems))

    @property
    def basis(self) -> str:
        """The design sample's basis, ``"event"`` or ``"annual"``."""
        return self.design_sample.basis

    @property
    def route(self) -> str:
        """The design sample's route."""
        return self.design_sample.route

    @property
    def degenerate_tail(self) -> dict[str, str | None]:
        """Each design problem's degenerate tail, by ``problem_id``."""
        return {
            problem.problem_id: problem.degenerate_tail for problem in self.problems
        }

    def filled(self, index: ParametricIndex) -> NDArray[np.bool_]:
        """Return where the cover's payout comes from an unidentified amount.

        Parameters
        ----------
        index : ParametricIndex
            An index on the cover's definition, with the cover's units in
            order, e.g. an evaluation sample's ``index``.

        Returns
        -------
        numpy.ndarray of bool
            Shape ``(index.n_events, n_units)``: whether the basis of an
            amount the fill rule set is non-zero at the event and unit.

        Raises
        ------
        ParametricInputError
            If the index is not on the cover's definition and units.
        """
        if tuple(index.unit_id) != tuple(self.cover.unit_id) or (
            index.definition != self.cover.index
        ):
            raise ParametricInputError(
                "the index is not on the cover's definition and units"
            )
        out = np.zeros((index.n_events, self.cover.n_units), dtype=bool)
        for position, unit in enumerate(self.cover.unit_id):
            names = self.unidentified.get(unit, ())
            payout = self.cover.payout[unit]
            if not names or not isinstance(payout, LinearInAmounts):
                continue
            columns = [payout.amount_names().index(name) for name in names]
            basis = sp.csc_matrix(payout.basis(index.values[:, position]))[:, columns]
            out[:, position] = np.asarray(abs(basis).sum(axis=1)).reshape(-1) > 0
        return out

    def summary(self) -> pd.DataFrame:
        """Return one row per design problem: objective, premium, budget, solver.

        Returns
        -------
        pandas.DataFrame
            Columns ``problem_id``, ``scope``, ``objective`` (the document as
            text), ``objective_value``, ``premium``, ``expected_loss``,
            ``budget_bound``, ``budget_use``, ``loss_weight``,
            ``degenerate_tail``, ``n_unidentified``, ``route``, ``solver``,
            ``version``, ``method``, ``status``, ``iterations`` and
            ``seconds``.
        """
        rows = []
        for problem in self.problems:
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "scope": self.scope,
                    "objective": _document_text(self.objective.to_dict()),
                    "objective_value": problem.objective_value,
                    "premium": problem.premium,
                    "expected_loss": problem.expected_loss,
                    "budget_bound": problem.budget_bound,
                    "budget_use": problem.budget_use,
                    "loss_weight": problem.loss_weight,
                    "degenerate_tail": problem.degenerate_tail,
                    "n_unidentified": sum(
                        len(self.unidentified.get(unit, ())) for unit in problem.unit_id
                    ),
                    "route": problem.solver.route,
                    "solver": problem.solver.solver,
                    "version": problem.solver.version,
                    "method": problem.solver.method,
                    "status": problem.solver.status,
                    "iterations": problem.solver.iterations,
                    "seconds": problem.solver.seconds,
                }
            )
        return pd.DataFrame(rows)

    def parameters_frame(self) -> pd.DataFrame:
        """Return every parameter of the cover's payouts as a table.

        Returns
        -------
        pandas.DataFrame
            One row per unit and parameter, in the order of the units and of
            each payout's parameters: ``unit_id``, ``parameter`` (its name,
            e.g. ``"levels[2]"``), ``value``, ``role`` (``"amount"``, in
            payout units, or ``"shape"``, in index units), ``calibrated``
            (whether the calibration set it) and ``unidentified``.
        """
        rows = []
        for unit in self.cover.unit_id:
            payout = self.cover.payout[unit]
            amounts = (
                set(payout.amount_names())
                if isinstance(payout, LinearInAmounts)
                else set()
            )
            for name, value in payout.parameters().items():
                rows.append(
                    {
                        "unit_id": unit,
                        "parameter": name,
                        "value": float(value),  # a resolved cover holds numbers
                        "role": "amount" if name in amounts else "shape",
                        "calibrated": name in self.calibrated.get(unit, ()),
                        "unidentified": name in self.unidentified.get(unit, ()),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "unit_id",
                "parameter",
                "value",
                "role",
                "calibrated",
                "unidentified",
            ],
        )


def _document_text(document: Mapping[str, Any]) -> str:
    """Return ``kind(key=value, ...)`` for an objective's document."""
    rest = ", ".join(
        f"{key}={value}" for key, value in document.items() if key != "kind"
    )
    return f"{document['kind']}({rest})"


class _Setup(NamedTuple):
    """The checked arguments of one calibration."""

    template: ParametricCover
    sample: OutcomeSample
    objective: Objective
    scope: str
    pricing: Pricing
    budget: Budget | None
    options: CalibrationOptions


def _check_types(arguments: _Setup) -> None:
    """Raise unless every argument of :func:`calibrate` has its type.

    Raises
    ------
    ParametricInputError
        If an argument is not one the module accepts.
    """
    checks: list[tuple[str, object, Any, str]] = [
        ("template", arguments.template, ParametricCover, "a ParametricCover"),
        ("sample", arguments.sample, OutcomeSample, "an OutcomeSample"),
        (
            "objective",
            arguments.objective,
            MSE | ExpectedUncovered | WeightedCVaR,
            "MSE(), ExpectedUncovered() or WeightedCVaR(alpha, beta)",
        ),
        (
            "pricing",
            arguments.pricing,
            PurePremium | CostOfCapitalLoad,
            "PurePremium() or CostOfCapitalLoad(rate, q)",
        ),
        ("options", arguments.options, CalibrationOptions, "CalibrationOptions"),
    ]
    for name, value, kind, text in checks:
        if not isinstance(value, kind):
            raise ParametricInputError(f"{name} must be {text}, got {value!r}")
    if arguments.budget is not None and not isinstance(arguments.budget, Budget):
        raise ParametricInputError(
            "budget must be Budget(ratio=...) or Budget(amount=...), or None; got "
            f"{arguments.budget!r}"
        )
    if arguments.scope not in SCOPES:
        raise ParametricInputError(
            f"scope must be one of {list(SCOPES)}, got {arguments.scope!r}"
        )


def _check_arguments(arguments: _Setup) -> None:
    """Raise unless the arguments of :func:`calibrate` fit together.

    Raises
    ------
    ParametricInputError
        If an argument is invalid, the sample is not on the template's
        index definition and units, or the options ask for a route this
        version doesn't have.
    """
    _check_types(arguments)
    template, sample = arguments.template, arguments.sample
    if tuple(template.unit_id) != sample.unit_id:
        raise ParametricInputError(
            "the template's units are not the sample's: "
            f"{list(template.unit_id)[:5]} vs {list(sample.unit_id)[:5]}"
        )
    if sample.index.definition != template.index:
        raise ParametricInputError(
            f"the sample's index is on {sample.index.definition!r}, and the template "
            f"pays on {template.index!r}"
        )
    if arguments.budget is not None:
        arguments.budget.check(arguments.scope, sample.unit_id)
    options = arguments.options
    if options.ranges:
        raise ParametricInputError(
            "ranges=True ranges each amount over the optimal set, which this "
            "version doesn't do yet"
        )
    if isinstance(arguments.objective, MSE) and options.time_limit is not None:
        raise ParametricInputError(
            "time_limit applies to the linear programs; the mean squared error's "
            "route in this version takes none"
        )


def _problems(scope: str, units: Sequence[UnitAmounts]) -> list[tuple[str, list[int]]]:
    """Return ``(problem_id, unit positions)`` for each design problem."""
    if scope == JOINT:
        return [(JOINT, list(range(len(units))))]
    return [(str(unit.unit_id), [position]) for position, unit in enumerate(units)]


def _bound(setup: _Setup, unit_id: tuple[Hashable, ...], loss: Any) -> float | None:
    """Return a problem's budget bound, or None."""
    if setup.budget is None:
        return None
    return setup.budget.bound(unit_id, setup.sample.measure.expectation(loss))


def _log(problem: CoverProblem, objective: Objective) -> None:
    """Log a solved design problem, and warn when its tail degenerates."""
    LOGGER.info(
        "design problem %r: %s = %.6g, premium %.6g (%s, %s, %.2f s)",
        problem.problem_id,
        objective.name,
        problem.objective_value,
        problem.premium,
        problem.solver.solver,
        problem.solver.method,
        problem.solver.seconds,
    )
    if problem.degenerate_tail == "expected_uncovered":
        LOGGER.warning(
            "design problem %r: 1 - beta = %.4g is at least the weight of the "
            "outcomes with a loss, %.4g, so the weighted-CVaR design minimises "
            "E[U] only",
            problem.problem_id,
            1.0 - getattr(objective, "beta", 0.0),
            problem.loss_weight,
        )
    if problem.degenerate_tail == "single_outcome":
        LOGGER.warning(
            "design problem %r: 1 - beta is at most the smallest weight of an "
            "outcome with a loss, so the CVaR is the single worst outcome",
            problem.problem_id,
        )


class _Solved(NamedTuple):
    """A solved design problem: its data, its units' positions and the solution."""

    problem: DesignProblem
    positions: list[int]
    solution: Solution


def _solve(
    setup: _Setup, units: Sequence[UnitAmounts]
) -> tuple[list[_Solved], dict[Hashable, NDArray[np.float64]]]:
    """Solve every design problem; return them and each unit's final amounts."""
    sample = setup.sample
    loss = sp.csc_matrix(sample.loss)
    solved = []
    amounts: dict[Hashable, NDArray[np.float64]] = {}
    for problem_id, positions in _problems(setup.scope, units):
        members = [units[position] for position in positions]
        problem_loss = np.asarray(loss[:, positions].sum(axis=1)).reshape(-1)
        problem = DesignProblem(
            problem_id=problem_id,
            basis=sp.hstack([unit.basis for unit in members], format="csr"),
            loss=problem_loss,
            constraints=[unit.constraints for unit in members],
            measure=sample.measure,
            bound=_bound(setup, tuple(unit.unit_id for unit in members), problem_loss),
        )
        solution = solve_design(
            problem,
            setup.objective,
            setup.pricing,
            method=lp_method(setup.options.lp_method, len(members)),
            time_limit=setup.options.time_limit,
        )
        check_amounts(problem, solution.amounts, setup.options.feasibility_tolerance)
        amounts.update(final_amounts(members, solution.amounts, setup.options.fill))
        solved.append(_Solved(problem, positions, solution))
    return solved, amounts


def _check_limit(
    problem: DesignProblem,
    paid: NDArray[np.float64],
    caps: list[float | None],
    tolerance: float,
) -> None:
    """Raise if a unit's payout is above its limit, beyond the tolerance."""
    for column, cap in enumerate(caps):
        largest = float(paid[:, column].max(initial=0.0))
        if cap is not None and largest > cap + tolerance * max(cap, problem.scale):
            raise CalibrationError(
                f"a payout of {largest:.12g} is above its limit {cap:.12g}",
                problem_id=problem.problem_id,
            )


def _record(
    setup: _Setup,
    solved: _Solved,
    paid: NDArray[np.float64],
    cover: ParametricCover,
) -> CoverProblem:
    """Check one solved problem on the resolved cover; return its record.

    ``paid`` is the cover's payout per row and unit of the sample.
    """
    problem, positions, solution = solved
    unit_id = tuple(cover.unit_id[p] for p in positions)
    caps = [None if cover.limit is None else cover.limit.amount_of(u) for u in unit_id]
    tolerance = setup.options.feasibility_tolerance
    _check_limit(problem, paid[:, positions], caps, tolerance)
    value, premium = verify_solution(
        problem,
        paid[:, positions],
        objective=setup.objective,
        pricing=setup.pricing,
        solution=solution,
        tolerance=tolerance,
    )
    weight, degenerate = tail_degeneracy(setup.objective, problem.loss, problem.measure)
    record = CoverProblem(
        problem_id=problem.problem_id,
        unit_id=unit_id,
        objective_value=value,
        premium=premium,
        expected_loss=problem.measure.expectation(problem.loss),
        budget_bound=problem.bound,
        loss_weight=weight,
        degenerate_tail=degenerate,
        solver=solution.info,
    )
    _log(record, setup.objective)
    return record


def _design(setup: _Setup) -> CalibrationResult:
    """Solve every design problem of a checked calibration."""
    sample = setup.sample
    limit = resolve_limit(setup.template.limit, sample.unit_id, sample.loss)
    units = unit_amounts(setup.template, sample, scope=setup.scope, limit=limit)
    if setup.budget is None and not isinstance(setup.objective, MSE):
        check_bounded(units, setup.objective.name)
    check_unpaid_at_zero(sample, units)
    solved, amounts = _solve(setup, units)
    cover = resolved_cover(setup.template, units, amounts, limit)
    paid = cover.payout_matrix(sample.index)[sample.event_row]
    return CalibrationResult(
        cover=cover,
        template_hash=setup.template.definition_hash,
        objective=setup.objective,
        pricing=setup.pricing,
        budget=setup.budget,
        scope=setup.scope,
        problems=tuple(_record(setup, item, paid, cover) for item in solved),
        calibrated={
            unit.unit_id: setup.template.payout[unit.unit_id].free_parameters()
            for unit in units
        },
        unidentified={unit.unit_id: unit.unidentified_names for unit in units},
        fill=setup.options.fill,
        design_sample=DesignSample.from_sample(sample),
    )


def calibrate(
    template: ParametricCover,
    sample: OutcomeSample,
    objective: Objective,
    *,
    scope: str,
    pricing: Pricing = PurePremium(),  # frozen, so safe to share
    budget: Budget | None = None,
    options: CalibrationOptions = CalibrationOptions(),  # frozen, so safe to share
) -> CalibrationResult:
    """Choose the free parameters of a template on a design sample (module docstring).

    Parameters
    ----------
    template : ParametricCover
        The cover to design: its free amounts are calibrated. Every payout is
        linear in its amounts, with its shape parameters fixed, in this
        version. Its limit, if any, bounds every amount;
        ``Limit.largest_loss()`` is resolved on ``sample``.
    sample : OutcomeSample
        The design sample, on the template's index definition and units,
        e.g. the training side of
        :func:`~climada_petals.engine.parametric.sample.holdout`.
    objective : MSE, ExpectedUncovered or WeightedCVaR
        What the design minimises. ``MSE()`` needs cvxpy in this version.
    scope : {"individual", "joint"}
        One design problem per unit, or one on the owner's total. Required.
    pricing : PurePremium or CostOfCapitalLoad, default PurePremium()
        The premium the budget bounds and the result records.
    budget : Budget, optional
        The premium budget of each design problem. Default: none, which
        ``E[U]`` and the weighted CVaR accept only with bounded amounts.
    options : CalibrationOptions, default CalibrationOptions()
        The fill rule, the LP method, the tolerance and the time limit.

    Returns
    -------
    CalibrationResult
        The resolved cover, one :class:`CoverProblem` per design problem,
        the unidentified amounts and the design sample's record.

    Raises
    ------
    ParametricInputError
        If an argument is invalid or the arguments don't fit together, a
        route this version doesn't have is needed (module docstring), a
        linear objective has neither a budget nor bounded amounts, or the
        sample left events out and a payout may pay at an index of 0
        (:func:`~climada_petals.engine.parametric.sample.check_left_out_unpaid`).
    CoverDefinitionError
        If a fixed amount is above its unit's limit.
    CalibrationError
        If a design problem is not solved to optimality, or its solution
        fails a check: :class:`InfeasibleCalibrationError`,
        :class:`UnboundedCalibrationError`, :class:`CalibrationLimitError`
        (``time_limit``) or the base class. No problem is skipped.
    MissingDependencyError
        If the objective is the mean squared error and cvxpy is missing.

    Examples
    --------
    A tail objective calibrated jointly across the units:

    >>> template = ParametricCover(  # doctest: +SKIP
    ...     index.definition,
    ...     StepPayout.uniform(attachment=33.0, exhaustion=80.0, n_steps=5),
    ...     unit_id=units.unit_id,
    ...     payout_unit="USD",
    ...     limit=Limit.largest_loss(),
    ... )
    >>> result = calibrate(  # doctest: +SKIP
    ...     template,
    ...     train,
    ...     WeightedCVaR(alpha=0.9, beta=0.99),
    ...     scope="joint",
    ...     budget=Budget(ratio=1.0),
    ... )
    >>> result.cover.write_json("cover.json")  # doctest: +SKIP
    """
    setup = _Setup(template, sample, objective, scope, pricing, budget, options)
    _check_arguments(setup)
    return _design(setup)
