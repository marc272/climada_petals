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
The **exact routes** take a template whose payouts are linear in their
amounts (:class:`~climada_petals.engine.parametric.payout.LinearInAmounts`:
the four built-in families) with every shape parameter, such as a
breakpoint, fixed:

* ``ExpectedUncovered`` and ``WeightedCVaR``: one linear program per design
  problem, with scipy's HiGHS
  (:mod:`~climada_petals.engine.parametric.solvers`);
* ``MSE``: one quadratic program per design problem, with scipy's
  ``SLSQP``, and cutting planes under a cost-of-capital load
  (:mod:`~climada_petals.engine.parametric.solvers`). It has stated limits:
  ``CalibrationOptions.max_slsqp_amounts`` and ``max_cut_rounds``.

The **grid route** takes the rest, in the individual scope only
(:mod:`~climada_petals.engine.parametric._grid`): free shape parameters,
such as a free attachment, on a grid with the exact route at each
candidate; and a payout that is not linear in its amounts, or a
:class:`~climada_petals.engine.parametric.objectives.UserObjective`, on a
grid over every free parameter, evaluated in numpy (``route="grid"``).
Each free parameter it searches has a stated grid, and
``CalibrationOptions.max_grid_points`` caps the candidates of one design
problem. In a joint design, thresholds are always fixed, and the grid
route raises there.

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
* **Uniqueness.** ``unique`` says whether every design problem's
  identified amounts are unique. On the mean squared error's route: the
  objective's Hessian restricted to them is positive definite (smallest
  eigenvalue above ``1e-10`` times the largest). On the LP route, with
  ``CalibrationOptions(ranges=True)``: every identified amount is pinned by
  the ranges' two-tolerance test, four more linear programs per amount
  (:mod:`~climada_petals.engine.parametric.solvers`), and
  :attr:`CalibrationResult.ranges` holds each amount's range over the
  near-optimal set; without it, None. On a grid: false when another
  candidate ties with the chosen one (``ties``), else the route's report at
  that candidate. A design that is not unique is one member of a set of
  optimal designs.
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

The budget frontier
-------------------
:func:`calibrate_frontier` calibrates one template at each of several
budgets and returns the designs as one table: what another unit of premium
buys, on the design sample. It is a measurement. Each row is the
calibration at its budget, solved on its own, and the table names no best
budget.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from ._checks import check_int, check_real
from ._design import (
    Setup,
    design_units,
    record,
    resolved_design,
    solve_problems,
    unidentified,
)
from ._problem import CoverProblem, DesignSample
from .constraints import FILLS, Budget
from .cover import ParametricCover
from .errors import ParametricError, ParametricInputError
from .index import ParametricIndex
from .objectives import MSE, ExpectedUncovered, Objective, UserObjective, WeightedCVaR
from .payout import LinearInAmounts
from .pricing import CostOfCapitalLoad, Pricing, PurePremium
from .sample import OutcomeSample
from .solvers import LP_METHODS, QPLimits, Settings

__all__ = [
    "DEGENERATE_TAILS",
    "JOINT",
    "SCOPES",
    "CalibrationOptions",
    "CalibrationResult",
    "CoverProblem",
    "DesignSample",
    "calibrate",
    "calibrate_frontier",
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
        Whether the LP route ranges each identified amount over the
        near-optimal set, which decides ``unique`` there (module docstring):
        four more linear programs per amount. The mean squared error's
        route and the numpy grid have no ranges.
    lp_method : {"auto", "highs", "highs-ipm"}, default "auto"
        The ``linprog`` method of the linear programs: ``"auto"`` takes
        ``"highs"`` for a problem with one unit and ``"highs-ipm"`` for
        several. The method used is recorded per design problem.
    feasibility_tolerance : float, default 1e-7
        The relative tolerance of the checks after the solve (module
        docstring), and within which the cutting planes of the mean squared
        error bring the loaded premium to the budget, > 0.
    time_limit : float, optional
        Seconds each linear program, or each mean squared error's design
        problem with all its rounds, may take; reaching it raises
        :class:`~climada_petals.engine.parametric.errors.CalibrationLimitError`.
        Default: no limit.
    max_slsqp_amounts : int, default 1_000
        The most amounts the mean squared error's ``SLSQP`` route solves for
        in one design problem, >= 1. Above it, ``CalibrationLimitError``.
    max_cut_rounds : int, default 100
        The most ``SLSQP`` solves of the mean squared error under a
        cost-of-capital load, each adding a cutting plane, >= 1. Reaching it
        with the premium above the budget raises ``CalibrationLimitError``.
    max_grid_points : int, default 10_000
        The most candidates of one design problem on the grid route, >= 1.
        Above it, ``CalibrationLimitError``, before anything is solved.

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
    max_slsqp_amounts: int = 1_000
    max_cut_rounds: int = 100
    max_grid_points: int = 10_000

    def __post_init__(self) -> None:
        """Validate the options."""
        for name in ("max_slsqp_amounts", "max_cut_rounds", "max_grid_points"):
            object.__setattr__(self, name, check_int(name, getattr(self, name), low=1))
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

    @property
    def settings(self) -> Settings:
        """What the design problems read of the options."""
        return Settings(
            self.fill,
            self.ranges,
            self.lp_method,
            self.time_limit,
            QPLimits(
                self.max_slsqp_amounts, self.max_cut_rounds, self.feasibility_tolerance
            ),
            self.max_grid_points,
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
    objective : MSE, ExpectedUncovered, WeightedCVaR or UserObjective
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
        per unit (module docstring). On the numpy grid the tie rule set
        them, not the fill rule.
    fill : {"lowest", "highest"}
        The rule that set the unidentified amounts.
    design_sample : DesignSample
        The basis, route, size, events and blocks of the design sample.
    unique : bool or None, default None
        Whether every design problem's identified amounts are unique; None
        if a route doesn't determine it (the LP route without
        ``ranges=True``).
    ties : int or None, default None
        The grid candidates tied with the chosen ones, summed over the
        design problems on a grid. None: no grid.
    profile : pandas.DataFrame or None, default None
        On a grid, the objective at every candidate: one row per candidate
        of each design problem on a grid, with ``problem_id``,
        ``candidate`` (its position in lexicographic order), one column per
        parameter on the grid, ``objective_value`` and ``premium`` (NaN
        where the exact route found no feasible amounts), ``status``
        (``"feasible"``, ``"infeasible"``, ``"above_budget"`` or
        ``"above_limit"``), ``tied`` (within the tolerance of the best) and
        ``chosen``. None: no grid.
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
    profile: pd.DataFrame | None = None

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
    def ranges(self) -> dict[Hashable, dict[str, tuple[float, float]]] | None:
        """Each identified amount's range over the near-optimal set, by unit.

        ``{unit_id: {amount: (low, high)}}`` in payout units, from the LP
        route with ``CalibrationOptions(ranges=True)``; None if no design
        problem was ranged.
        """
        found = [problem.ranges for problem in self.problems]
        if all(item is None for item in found):
            return None
        out: dict[Hashable, dict[str, tuple[float, float]]] = {}
        for unit, name, low, high in (row for item in found for row in item or ()):
            out.setdefault(unit, {})[name] = (low, high)
        return out

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
            ``degenerate_tail``, ``n_unidentified``, ``unique``, ``ties`` and
            ``n_candidates`` (on a grid, else None), ``route``, ``solver``,
            ``version``, ``method``, ``status``, ``iterations``, ``rounds``
            and ``seconds`` (on a grid, the whole grid's).
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
                    "unique": problem.unique,
                    "ties": problem.ties,
                    "n_candidates": problem.n_candidates,
                    "route": problem.solver.route,
                    "solver": problem.solver.solver,
                    "version": problem.solver.version,
                    "method": problem.solver.method,
                    "status": problem.solver.status,
                    "iterations": problem.solver.iterations,
                    "rounds": problem.solver.rounds,
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


def _check_types(arguments: Setup) -> None:
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
            MSE | ExpectedUncovered | WeightedCVaR | UserObjective,
            "MSE(), ExpectedUncovered(), WeightedCVaR(alpha, beta) or a "
            "UserObjective",
        ),
        (
            "pricing",
            arguments.pricing,
            PurePremium | CostOfCapitalLoad,
            "PurePremium() or CostOfCapitalLoad(rate, q)",
        ),
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


def _check_arguments(arguments: Setup) -> None:
    """Raise unless the arguments of :func:`calibrate` fit together.

    Raises
    ------
    ParametricInputError
        If an argument is invalid, or the sample is not on the template's
        index definition and units.
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


def _design(setup: Setup) -> CalibrationResult:
    """Solve every design problem of a checked calibration (:mod:`._design`)."""
    units = design_units(setup)
    solved = solve_problems(setup, units)
    cover = resolved_design(setup, units, solved)
    sample = setup.sample
    paid = cover.payout_matrix(sample.index)[sample.event_row]
    problems = tuple(record(setup, item, paid, cover) for item in solved)
    flags = [problem.unique for problem in problems]
    ties = [problem.ties for problem in problems if problem.ties is not None]
    profiles = [item.grid.profile for item in solved if item.grid is not None]
    return CalibrationResult(
        cover=cover,
        template_hash=setup.template.definition_hash,
        objective=setup.objective,
        pricing=setup.pricing,
        budget=setup.budget,
        scope=setup.scope,
        problems=problems,
        calibrated={
            unit: setup.template.payout[unit].free_parameters()
            for unit in sample.unit_id
        },
        unidentified=unidentified(units, solved, sample.unit_id),
        fill=setup.settings.fill,
        design_sample=DesignSample.from_sample(sample),
        unique=None if None in flags else all(flags),
        ties=sum(ties) if ties else None,
        profile=pd.concat(profiles, ignore_index=True) if profiles else None,
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
        The cover to design: its free parameters are calibrated. Free
        amounts of a payout linear in its amounts are solved for exactly;
        free shape parameters, the parameters of another payout and those
        of a ``UserObjective`` design are searched on their grids, in the
        individual scope (module docstring). Its limit, if any, bounds every
        payout; ``Limit.largest_loss()`` is resolved on ``sample``.
    sample : OutcomeSample
        The design sample, on the template's index definition and units,
        e.g. the training side of
        :func:`~climada_petals.engine.parametric.sample.holdout`.
    objective : MSE, ExpectedUncovered, WeightedCVaR or UserObjective
        What the design minimises.
    scope : {"individual", "joint"}
        One design problem per unit, or one on the owner's total. Required.
    pricing : PurePremium or CostOfCapitalLoad, default PurePremium()
        The premium the budget bounds and the result records.
    budget : Budget, optional
        The premium budget of each design problem. Default: none, which
        ``E[U]`` and the weighted CVaR accept only with bounded amounts.
    options : CalibrationOptions, default CalibrationOptions()
        The fill rule, the ranges, the LP method, the tolerance, the time
        limit and the size limits.

    Returns
    -------
    CalibrationResult
        The resolved cover, one :class:`CoverProblem` per design problem,
        the unidentified amounts, the ranges and the grid's profile where
        there are, and the design sample's record.

    Raises
    ------
    ParametricInputError
        If an argument is invalid or the arguments don't fit together, the
        grid route is needed in the joint scope or for a free parameter
        without a grid, a linear objective has neither a budget nor bounded
        amounts, or the sample left events out and a payout may pay at an
        index of 0
        (:func:`~climada_petals.engine.parametric.sample.check_left_out_unpaid`).
    CoverDefinitionError
        If a fixed amount is above its unit's limit.
    CalibrationError
        If a design problem is not solved to optimality, or its solution
        fails a check: :class:`InfeasibleCalibrationError` (on a grid: no
        feasible candidate), :class:`UnboundedCalibrationError`,
        :class:`CalibrationLimitError` (``time_limit``,
        ``max_slsqp_amounts``, ``max_cut_rounds``, ``max_grid_points`` or a
        solver's iteration limit) or the base class. No problem is skipped.

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
    if not isinstance(options, CalibrationOptions):
        raise ParametricInputError(
            f"options must be CalibrationOptions, got {options!r}"
        )
    setup = Setup(template, sample, objective, scope, pricing, budget, options.settings)
    _check_arguments(setup)
    return _design(setup)


def _frontier_budgets(budgets: object) -> list[Budget]:
    """Return the frontier's budgets as a list, checked.

    Raises
    ------
    ParametricInputError
        If ``budgets`` is not a non-empty sequence of distinct budgets.
    """
    if isinstance(budgets, Budget | str) or not isinstance(budgets, Sequence):
        raise ParametricInputError(
            f"budgets must be a sequence of Budget, got {budgets!r}"
        )
    found = list(budgets)
    if not found or not all(isinstance(budget, Budget) for budget in found):
        raise ParametricInputError(
            f"budgets must be a non-empty sequence of Budget, got {found!r}"
        )
    repeated = [b for k, b in enumerate(found) if b in found[:k]]
    if repeated:
        raise ParametricInputError(
            f"budgets must be distinct, got {repeated[0]!r} twice: two rows would "
            "hold the same design"
        )
    return found


def _frontier_rows(result: CalibrationResult, budget: Budget) -> pd.DataFrame:
    """Return a design's rows of the frontier: budget, summary and parameters."""
    rows = []
    for problem in result.problems:
        amount = (
            budget.amount[problem.unit_id[0]]
            if isinstance(budget.amount, Mapping)
            else budget.amount
        )
        row: dict[str, Any] = {
            "budget_ratio": np.nan if budget.ratio is None else budget.ratio,
            "budget_amount": np.nan if amount is None else amount,
        }
        for unit in problem.unit_id:
            parameters = result.cover.payout[unit].parameters()
            row.update(
                (f"{unit}:{name}", float(value)) for name, value in parameters.items()
            )
        rows.append(row)
    found = pd.DataFrame(rows)
    return pd.concat([found.iloc[:, :2], result.summary(), found.iloc[:, 2:]], axis=1)


def calibrate_frontier(
    template: ParametricCover,
    sample: OutcomeSample,
    objective: Objective,
    *,
    budgets: Sequence[Budget],
    scope: str,
    pricing: Pricing = PurePremium(),  # frozen, so safe to share
    options: CalibrationOptions = CalibrationOptions(),  # frozen, so safe to share
) -> pd.DataFrame:
    """Calibrate a template at each of several budgets; return the designs as one table.

    The budget frontier: what another unit of premium buys, on the design
    sample. Each budget's design is
    ``calibrate(template, sample, objective, scope=scope, pricing=pricing,
    budget=budget, options=options)``, solved on its own, so a row doesn't
    depend on the other budgets or on their order. It is a measurement: the
    table has no best point, no rank and no ordering by a metric, and a
    reader takes differences between rows. Rows follow the order of
    ``budgets``. The objective values and premiums are on the design
    sample: evaluate each design on another sample
    (:func:`~climada_petals.engine.parametric.evaluate.evaluate`) for
    out-of-sample numbers.

    Each design's non-uniqueness is in its rows (``unique``, ``ties``,
    ``n_unidentified``): where a design is not unique, its amounts are one
    member of a set of optimal designs, and they may jump between
    neighbouring budgets. ``CalibrationOptions(ranges=True)`` ranges every
    budget's amounts on the LP route. Each budget of the mean squared error
    is one ``SLSQP`` solve, with cutting planes under a load, within the
    stated limits.

    Parameters
    ----------
    template : ParametricCover
        The cover to design, as for :func:`calibrate`.
    sample : OutcomeSample
        The design sample.
    objective : MSE, ExpectedUncovered, WeightedCVaR or UserObjective
        What every design minimises.
    budgets : sequence of Budget
        The budgets, distinct, at least one. Keyword-only.
    scope : {"individual", "joint"}
        As for :func:`calibrate`. Required.
    pricing : PurePremium or CostOfCapitalLoad, default PurePremium()
        The premium the budgets bound.
    options : CalibrationOptions, default CalibrationOptions()
        As for :func:`calibrate`, for every budget.

    Returns
    -------
    pandas.DataFrame
        One row per design problem of each budget's design (one per unit,
        or one for the joint scope), budget by budget: ``budget_ratio`` and
        ``budget_amount`` (the budget as stated, the other NaN; an amount by
        unit gives the row's unit's amount), the columns of
        :meth:`CalibrationResult.summary`, and every parameter of the
        problem's units in the resolved cover, as ``"<unit_id>:<name>"``
        (e.g. ``"parish_1:levels[0]"``; NaN on the rows of other units).

    Raises
    ------
    ParametricInputError
        If ``budgets`` is not a non-empty sequence of distinct budgets, or
        as :func:`calibrate`.
    CalibrationError
        As :func:`calibrate`, at a budget the template can't meet, for
        example: no budget's row is dropped. The error's notes name the
        budget.

    Examples
    --------
    >>> frontier = calibrate_frontier(  # doctest: +SKIP
    ...     template,
    ...     train,
    ...     WeightedCVaR(alpha=0.9, beta=0.99),
    ...     budgets=[Budget(ratio=r) for r in (0.5, 1.0, 1.5)],
    ...     scope="individual",
    ... )
    >>> frontier[["budget_ratio", "problem_id", "objective_value", "premium"]]  # doctest: +SKIP
    """
    frames = []
    for budget in _frontier_budgets(budgets):
        try:
            result = calibrate(
                template,
                sample,
                objective,
                scope=scope,
                pricing=pricing,
                budget=budget,
                options=options,
            )
        except ParametricError as error:
            error.add_note(f"raised by calibrate_frontier at {budget!r}")
            raise
        frames.append(_frontier_rows(result, budget))
    return pd.concat(frames, ignore_index=True)
