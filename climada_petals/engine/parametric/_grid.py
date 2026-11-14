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

The grid route of a calibration, and the exact solve of one design problem.

The exact routes (:mod:`~climada_petals.engine.parametric.solvers`) solve
for the amounts of a payout that is linear in them, with every shape
parameter fixed. What they can't take goes on a **grid**, in the
individual scope only (in a joint design thresholds are fixed, and a
unit's grid would multiply with every other unit's):

* **Free shape parameters** of a
  :class:`~climada_petals.engine.parametric.payout.LinearInAmounts` payout
  under a built-in objective, such as a free attachment or a free
  breakpoint: the grid runs over the shape parameters, and at each
  candidate the exact route solves for the amounts (an LP, or the mean
  squared error's QP). The problem is recorded with that route, ``"lp"``
  or ``"qp"``, and with its number of candidates.
* **Any other payout, or a**
  :class:`~climada_petals.engine.parametric.objectives.UserObjective`: the
  grid runs over every free parameter, the amounts included, so each needs
  a grid, and each candidate's payout is evaluated in numpy. The route is
  ``"grid"``. A candidate whose premium is above the budget (beyond the
  feasibility tolerance), or which pays above the unit's limit, is not
  chosen: on a payout linear in its amounts, an amount above the limit;
  on any other payout, a payout on the design sample above it. Nothing
  bounds the time of this route.

**The candidates** are the Cartesian product of the grids of the
parameters, in the order of the payout's parameters, restricted to the
payouts the family accepts: thresholds in order of severity and, under
``monotone=True``, levels that don't fall. A combination the payout refuses
(:class:`~climada_petals.engine.parametric.errors.CoverDefinitionError`)
is no candidate; whole branches of the product are pruned that way, so a
large product with few ordered candidates is cheap. Above
``CalibrationOptions.max_grid_points`` candidates in one design problem the
calibration raises
:class:`~climada_petals.engine.parametric.errors.CalibrationLimitError`
before it solves anything.

**The choice.** The candidates are taken in lexicographic order: by the
first parameter's grid value, then the second's, and so on, each grid
increasing. The best objective wins. Every candidate within the tolerance
of the best, ``feasibility_tolerance * max(r, |best|)``, ties with it, with
``r`` the objective of the zero cover for a built-in objective and 1 for a
``UserObjective``, which has no known scale and is compared in its own
units. The first tied candidate in lexicographic order is chosen, and
``ties`` counts the others, so a unique best gives 0. A design with ties
is not unique; one without takes the exact route's report at the chosen
candidate (on the numpy grid: unique). The objective, the premium and the
status of every candidate are kept (the ``profile``).

**No silent failures.** At each candidate of the shape grid, the solver's
amounts are checked and the objective recomputed with numpy, as on the
exact route; a candidate whose amounts the solver finds infeasible is
recorded as ``"infeasible"``, and any other failure raises. A design
problem with no feasible candidate raises
:class:`~climada_petals.engine.parametric.errors.InfeasibleCalibrationError`.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from ._problem import DesignProblem, Solution, SolverInfo
from .constraints import (
    UnitAmounts,
    check_bounded,
    final_amounts,
    payout_at_zero,
    snap_amounts,
    unit_amount,
)
from .errors import (
    CalibrationLimitError,
    CoverDefinitionError,
    InfeasibleCalibrationError,
    ParametricInputError,
)
from .measure import OutcomeMeasure
from .objectives import MSE, Objective, UserObjective
from .parameters import AmountConstraints, Free
from .payout import LinearInAmounts, PayoutFunction
from .pricing import Pricing
from .solvers import Settings, solve_problem, verify_solution

__all__ = [
    "GRID_ROUTE",
    "STATUSES",
    "GridSolved",
    "UnitData",
    "UnitGrid",
    "grid_paid_at_zero",
    "plan_grids",
    "solve_grid",
]

GRID_ROUTE: Final[str] = "grid"
"""The route of a design problem evaluated on a grid in numpy."""

STATUSES: Final[tuple[str, ...]] = (
    "feasible",
    "infeasible",
    "above_budget",
    "above_limit",
)
"""The values of the profile's ``status`` column (module docstring)."""

_JOINT: Final[str] = "joint"


class UnitGrid(NamedTuple):
    """The grid of one unit's design problem.

    Parameters
    ----------
    position : int
        The unit's column in the sample.
    unit_id : str or int
        The unit.
    payout : PayoutFunction
        The template's payout of the unit.
    names : tuple of str
        The parameters on the grid, in the payout's parameter order.
    exact : bool
        True on the shape grid (the amounts are solved exactly at each
        candidate), False on the numpy grid.
    candidates : tuple of tuple of float
        The values of ``names`` at each candidate, in lexicographic order.
    """

    position: int
    unit_id: Hashable
    payout: PayoutFunction
    names: tuple[str, ...]
    exact: bool
    candidates: tuple[tuple[float, ...], ...]

    def resolved(self, candidate: Sequence[float]) -> PayoutFunction:
        """Return the payout with the grid parameters at a candidate's values."""
        return self.payout.with_values(dict(zip(self.names, candidate)))


def _grid_names(
    payout: PayoutFunction, objective: Objective
) -> tuple[tuple[str, ...], bool] | None:
    """Return a payout's grid parameters and whether the grid is exact; None: no grid."""
    if isinstance(objective, UserObjective) or not isinstance(payout, LinearInAmounts):
        return payout.free_parameters(), False
    amounts = set(payout.amount_names())
    shapes = tuple(name for name in payout.free_parameters() if name not in amounts)
    return (shapes, True) if shapes else None


def _accepts(
    payout: PayoutFunction, names: Sequence[str], values: Sequence[float]
) -> bool:
    """Whether the payout accepts these values of its first grid parameters."""
    try:
        payout.with_values(dict(zip(names, values)))
    except CoverDefinitionError:
        return False
    return True


def _candidates(
    payout: PayoutFunction, names: tuple[str, ...], max_points: int, unit: Hashable
) -> tuple[tuple[float, ...], ...]:
    """Return the candidates of a grid in lexicographic order (module docstring).

    Raises
    ------
    CalibrationLimitError
        Above ``max_points`` candidates.
    ParametricInputError
        If the payout accepts no point of the grid.
    """
    parameters = payout.parameters()
    grids = [parameters[name].grid or () for name in names]
    found: list[tuple[float, ...]] = []
    stack: list[tuple[float, ...]] = [()]
    while stack:
        prefix = stack.pop()
        if len(prefix) == len(names):
            found.append(prefix)
            if len(found) > max_points:
                raise CalibrationLimitError(
                    f"the grid of {list(names)} has more than max_grid_points="
                    f"{max_points} candidates (CalibrationOptions). Coarsen a grid, "
                    "fix a parameter, or raise the option",
                    problem_id=str(unit),
                    status="max_grid_points",
                )
            continue
        children = [prefix + (value,) for value in grids[len(prefix)]]
        stack.extend(
            reversed([child for child in children if _accepts(payout, names, child)])
        )
    if not found:
        raise ParametricInputError(
            f"the {payout.kind} payout of unit {unit!r} accepts no point of the "
            f"grids of {list(names)}: no combination keeps its thresholds in order "
            "(and its levels monotone)"
        )
    return tuple(found)


def _check_grid(payout: PayoutFunction, names: tuple[str, ...], unit: Hashable) -> None:
    """Raise if a free parameter of the numpy grid has no grid.

    Raises
    ------
    ParametricInputError
        If a free parameter has no grid.
    """
    parameters = payout.parameters()
    missing = [
        name
        for name in names
        if isinstance(parameter := parameters[name], Free) and parameter.grid is None
    ]
    if missing:
        raise ParametricInputError(
            f"the grid route evaluates every free parameter on a grid, and the "
            f"{payout.kind} payout of unit {unit!r} has none for {missing}: give "
            "each a grid, e.g. Free(0.0, 1e6, grid=11)"
        )


def plan_grids(
    template: Any, objective: Objective, *, scope: str, max_points: int
) -> dict[int, UnitGrid]:
    """Return the grid of each unit that needs one, by the unit's position.

    Parameters
    ----------
    template : ParametricCover
        The template.
    objective : Objective
        The design's objective.
    scope : {"individual", "joint"}
        The calibration scope.
    max_points : int
        ``CalibrationOptions.max_grid_points``.

    Returns
    -------
    dict of int to UnitGrid
        Empty when every unit takes the exact route.

    Raises
    ------
    ParametricInputError
        In the joint scope, if a unit needs a grid; if a free parameter of
        the numpy grid has no grid; or if a payout accepts no candidate.
    CalibrationLimitError
        Above ``max_points`` candidates in one design problem.
    """
    grids = {}
    for position, unit in enumerate(template.unit_id):
        payout = template.payout[unit]
        found = _grid_names(payout, objective)
        if found is None:
            continue
        names, exact = found
        if scope == _JOINT:
            if exact:
                raise ParametricInputError(
                    "thresholds are fixed in a joint design; unit "
                    f"{unit!r} has the free shape parameters {list(names)}. Fix "
                    "them, and compare thresholds by running the design several "
                    "times"
                )
            raise ParametricInputError(
                "the grid route takes the individual scope only, and unit "
                f"{unit!r} needs it: {_why_numpy(payout, objective)}"
            )
        if not exact:
            _check_grid(payout, names, unit)
        candidates = _candidates(payout, names, max_points, unit)
        grids[position] = UnitGrid(position, unit, payout, names, exact, candidates)
    return grids


def _why_numpy(payout: PayoutFunction, objective: Objective) -> str:
    """Return why a unit takes the numpy grid."""
    if isinstance(objective, UserObjective):
        return f"the objective {objective.name!r} is a UserObjective"
    return f"its {payout.kind} payout is not linear in its amounts"


def grid_paid_at_zero(grid: UnitGrid, cap: float | None) -> float:
    """Return the most any candidate of a grid may pay at an index of 0.

    A free amount counts at its upper bound, or at 1 if it has none (any
    positive value says that it may pay).
    """
    paid = 0.0
    for candidate in grid.candidates:
        payout = grid.resolved(candidate)
        if grid.exact:
            amounts = cast(LinearInAmounts, payout)
            paid = max(paid, payout_at_zero(amounts, amounts.amount_constraints(cap)))
        else:
            paid = max(paid, float(np.max(payout.evaluate(np.zeros(1)))))
    return paid


class UnitData(NamedTuple):
    """One unit's design problem on the design sample.

    Parameters
    ----------
    values : numpy.ndarray
        The unit's index per row of the sample.
    loss : numpy.ndarray
        The unit's loss per row.
    measure : OutcomeMeasure
        The sample's measure.
    cap : float or None
        The unit's limit, resolved.
    bound : float or None
        The unit's budget bound.
    objective : Objective
        What the design minimises.
    pricing : Pricing
        The premium the budget bounds.
    """

    values: NDArray[np.float64]
    loss: NDArray[np.float64]
    measure: OutcomeMeasure
    cap: float | None
    bound: float | None
    objective: Objective
    pricing: Pricing

    @property
    def weighted(self) -> NDArray[np.float64]:
        """1 for a row with a weight, else 0."""
        return (self.measure.event_weight > 0).astype(np.float64)

    def problem(self, unit: UnitAmounts | None, problem_id: str) -> DesignProblem:
        """Return the design problem of a candidate; without ``unit``, no amounts."""
        if unit is None:
            basis = sp.csr_matrix((self.loss.size, 0))
            constraints = AmountConstraints(np.zeros(0), np.zeros(0))
        else:
            basis, constraints = unit.basis, unit.constraints
        return DesignProblem(
            problem_id, basis, self.loss, [constraints], self.measure, self.bound
        )

    def reference(self) -> float:
        """The scale of the tie tolerance: the zero cover's objective, or 1."""
        if isinstance(self.objective, UserObjective):
            return 1.0
        zero = self.objective.evaluate(
            self.loss, np.zeros_like(self.loss), self.measure
        )
        return zero if zero > 0.0 else 1.0


class GridSolved(NamedTuple):
    """A design problem solved on a grid.

    Parameters
    ----------
    problem : DesignProblem
        The chosen candidate's design problem (no amounts on the numpy grid).
    solution : Solution
        The chosen candidate's solution, whose ``info.seconds`` is the whole
        grid's and whose ``unique`` follows the tie rule.
    values : dict of str to float
        The unit's free parameters, all set.
    unidentified : tuple of str
        The free amounts no row with a weight reaches at the chosen
        candidate.
    ties : int
        The candidates tied with the chosen one.
    n_candidates : int
        The grid's candidates.
    profile : pandas.DataFrame
        One row per candidate (module docstring).
    """

    problem: DesignProblem
    solution: Solution
    values: dict[str, float]
    unidentified: tuple[str, ...]
    ties: int
    n_candidates: int
    profile: pd.DataFrame


@dataclass(eq=False)
class _Profile:
    """The objective, premium and status of every candidate, and the choice."""

    value: NDArray[np.float64]
    premium: NDArray[np.float64]
    status: list[str]
    chosen: int = -1
    tied: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @classmethod
    def empty(cls, count: int) -> _Profile:
        """Return a profile of ``count`` candidates, none evaluated yet."""
        return cls(np.full(count, np.nan), np.full(count, np.nan), [""] * count)

    @property
    def ties(self) -> int:
        """The candidates tied with the chosen one."""
        return int(self.tied.sum()) - 1

    def set(self, position: int, found: tuple[str, float, float]) -> None:
        """Record one candidate's status, objective and premium."""
        self.status[position], self.value[position], self.premium[position] = found

    def choose(self, tolerance: float, reference: float, problem_id: str) -> int:
        """Choose the candidate (module docstring); return its position.

        Raises
        ------
        InfeasibleCalibrationError
            If no candidate is feasible.
        """
        feasible = np.array([status == STATUSES[0] for status in self.status])
        if not feasible.any():
            counts = {name: self.status.count(name) for name in STATUSES[1:]}
            raise InfeasibleCalibrationError(
                f"no candidate of the grid is feasible: {counts}",
                problem_id=problem_id,
                status="infeasible",
            )
        best = float(self.value[feasible].min())
        self.tied = feasible & (
            self.value <= best + tolerance * max(reference, abs(best))
        )
        self.chosen = int(np.flatnonzero(self.tied)[0])
        return self.chosen

    def frame(self, grid: UnitGrid) -> pd.DataFrame:
        """Return the profile as a table."""
        count = len(grid.candidates)
        columns: dict[str, Any] = {
            "problem_id": [str(grid.unit_id)] * count,
            "candidate": np.arange(count),
        }
        values = np.array(grid.candidates, dtype=np.float64).reshape(count, -1)
        for position, name in enumerate(grid.names):
            columns[name] = values[:, position]
        columns.update(
            objective_value=self.value,
            premium=self.premium,
            status=self.status,
            tied=self.tied,
            chosen=np.arange(count) == self.chosen,
        )
        return pd.DataFrame(columns)


def _unidentified(
    payout: PayoutFunction, template: PayoutFunction, unit_inputs: UnitData
) -> tuple[str, ...]:
    """Return the template's free amounts that the resolved payout's basis never reaches."""
    if not isinstance(payout, LinearInAmounts):
        return ()
    basis = sp.csr_matrix(payout.basis(unit_inputs.values))
    reached = np.asarray(abs(basis).T @ unit_inputs.weighted).reshape(-1) > 0
    parameters = template.parameters()
    return tuple(
        name
        for name, hit in zip(payout.amount_names(), reached)
        if isinstance(parameters[name], Free) and not hit
    )


def _numpy_status(
    payout: PayoutFunction,
    paid: NDArray[np.float64],
    unit_inputs: UnitData,
    tolerance: float,
) -> tuple[str, float]:
    """Return a numpy candidate's status and premium."""
    premium = unit_inputs.pricing.premium(paid, unit_inputs.measure)
    if unit_inputs.cap is not None:
        above = (
            not payout.amount_constraints(unit_inputs.cap).bounds_feasible
            if isinstance(payout, LinearInAmounts)
            else float(paid.max(initial=0.0)) > unit_inputs.cap
        )
        if above:
            return STATUSES[3], premium
    if unit_inputs.bound is not None:
        scale = unit_inputs.problem(None, "").scale
        reference = max(
            abs(unit_inputs.bound),
            unit_inputs.measure.expectation(unit_inputs.loss),
            scale,
        )
        if premium > unit_inputs.bound + tolerance * reference:
            return STATUSES[2], premium
    return STATUSES[0], premium


def _evaluate(
    payout: PayoutFunction, values: NDArray[np.float64], unit: Hashable
) -> NDArray[np.float64]:
    """Return a resolved payout on the rows, checked as a cover checks it.

    Raises
    ------
    CoverDefinitionError
        If the payouts are not finite, are negative or have another shape.
    """
    paid = np.asarray(payout.evaluate(values), dtype=np.float64)
    if paid.shape != values.shape or not np.all(np.isfinite(paid) & (paid >= 0)):
        raise CoverDefinitionError(
            f"the {payout.kind} payout of unit {unit!r} must return finite payouts "
            f">= 0 of shape {values.shape}"
        )
    return paid


def _solve_numpy(
    grid: UnitGrid, unit_inputs: UnitData, settings: Settings
) -> GridSolved:
    """Evaluate every candidate of a numpy grid; return the chosen one."""
    start = time.perf_counter()
    profile = _Profile.empty(len(grid.candidates))
    for position, candidate in enumerate(grid.candidates):
        payout = grid.resolved(candidate)
        paid = _evaluate(payout, unit_inputs.values, grid.unit_id)
        status, premium = _numpy_status(payout, paid, unit_inputs, settings.tolerance)
        value = unit_inputs.objective.evaluate(
            unit_inputs.loss, paid, unit_inputs.measure
        )
        profile.set(position, (status, value, premium))
    chosen = profile.choose(
        settings.tolerance, unit_inputs.reference(), str(grid.unit_id)
    )
    info = SolverInfo(
        route=GRID_ROUTE,
        solver="numpy",
        version=np.__version__,
        method=None,
        status="0",
        message=f"{len(grid.candidates)} candidates evaluated",
        iterations=None,
        seconds=time.perf_counter() - start,
    )
    return GridSolved(
        unit_inputs.problem(None, str(grid.unit_id)),
        Solution(
            np.zeros(0), float(profile.value[chosen]), info, unique=profile.ties == 0
        ),
        dict(zip(grid.names, grid.candidates[chosen])),
        _unidentified(grid.resolved(grid.candidates[chosen]), grid.payout, unit_inputs),
        profile.ties,
        len(grid.candidates),
        profile.frame(grid),
    )


def _shape_candidate(
    grid: UnitGrid, candidate: tuple[float, ...], unit_inputs: UnitData
) -> tuple[UnitAmounts, DesignProblem]:
    """Return a shape candidate's amounts and design problem."""
    payout = cast(LinearInAmounts, grid.resolved(candidate))
    unit = unit_amount(
        grid.unit_id, payout, unit_inputs.values, unit_inputs.weighted, unit_inputs.cap
    )
    return unit, unit_inputs.problem(unit, str(grid.unit_id))


def _shape_value(
    grid: UnitGrid,
    candidate: tuple[float, ...],
    unit_inputs: UnitData,
    settings: Settings,
) -> tuple[str, float, float]:
    """Solve a shape candidate's amounts exactly; return its status, objective and premium.

    A candidate whose amounts the solver finds infeasible is ``"infeasible"``;
    any other failure raises.
    """
    unit, problem = _shape_candidate(grid, candidate, unit_inputs)
    try:
        solution = solve_problem(
            problem, unit_inputs.objective, unit_inputs.pricing, settings, 1
        )
    except InfeasibleCalibrationError:
        return STATUSES[1], np.nan, np.nan
    paid = unit.basis @ snap_amounts(solution.amounts, unit.constraints)
    value, premium = verify_solution(
        problem,
        np.asarray(paid, dtype=np.float64).reshape(-1, 1),
        objective=unit_inputs.objective,
        pricing=unit_inputs.pricing,
        solution=solution,
        tolerance=settings.tolerance,
    )
    return STATUSES[0], value, premium


def _solve_shapes(
    grid: UnitGrid, unit_inputs: UnitData, settings: Settings
) -> GridSolved:
    """Solve the amounts exactly at every shape candidate; return the chosen one.

    The chosen candidate is solved again, with the ranges if the settings
    ask for them, and its amounts are snapped and filled.
    """
    start = time.perf_counter()
    profile = _Profile.empty(len(grid.candidates))
    plain = dataclasses.replace(settings, ranges=False)
    for position, candidate in enumerate(grid.candidates):
        profile.set(position, _shape_value(grid, candidate, unit_inputs, plain))
    chosen = profile.choose(
        settings.tolerance, unit_inputs.reference(), str(grid.unit_id)
    )
    unit, problem = _shape_candidate(grid, grid.candidates[chosen], unit_inputs)
    solution = solve_problem(
        problem, unit_inputs.objective, unit_inputs.pricing, settings, 1
    )
    amounts = final_amounts([unit], solution.amounts, settings.fill)[grid.unit_id]
    values = dict(zip(grid.names, grid.candidates[chosen]))
    values.update(
        (name, float(value))
        for name, value, free in zip(unit.payout.amount_names(), amounts, unit.free)
        if free
    )
    solution = dataclasses.replace(
        solution,
        info=dataclasses.replace(solution.info, seconds=time.perf_counter() - start),
        unique=False if profile.ties else solution.unique,
    )
    return GridSolved(
        problem,
        solution,
        values,
        unit.unidentified_names,
        profile.ties,
        len(grid.candidates),
        profile.frame(grid),
    )


def solve_grid(grid: UnitGrid, unit_inputs: UnitData, settings: Settings) -> GridSolved:
    """Solve one unit's design problem on its grid (module docstring).

    Raises
    ------
    ParametricInputError
        If the amounts of a shape grid need a budget or bounds, or a
        ``UserObjective`` returns no finite number.
    InfeasibleCalibrationError
        If no candidate is feasible.
    CalibrationError
        If a solve at a candidate fails otherwise, or its solution fails a
        check.
    CoverDefinitionError
        If a payout returns invalid payouts.
    """
    if grid.exact:
        if unit_inputs.bound is None and not isinstance(unit_inputs.objective, MSE):
            first, _ = _shape_candidate(grid, grid.candidates[0], unit_inputs)
            check_bounded([first], unit_inputs.objective.name)
        solved = _solve_shapes(grid, unit_inputs, settings)
    else:
        solved = _solve_numpy(grid, unit_inputs, settings)
    feasible = solved.profile["status"].eq(STATUSES[0]).sum()
    LOGGER.info(
        "design problem %r: %d grid candidates of %s, %d feasible, %d tied with the "
        "chosen one",
        str(grid.unit_id),
        solved.n_candidates,
        list(grid.names),
        feasible,
        solved.ties,
    )
    return solved
