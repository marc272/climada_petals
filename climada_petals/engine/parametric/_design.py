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

The design problems of a calibration: set up, solved and recorded.

:func:`~climada_petals.engine.parametric.calibrate.calibrate` checks its
arguments and assembles the
:class:`~climada_petals.engine.parametric.calibrate.CalibrationResult`;
this module does what lies between. A design problem is one unit
(individual scope) or every unit (joint scope).

* :func:`design_units` meets the template's units with the design sample
  and runs every check that comes before a solve: the limit resolved, the
  grids planned (:mod:`~climada_petals.engine.parametric._grid`), the
  amounts bounded where the objective needs it, and no payout at an index
  of 0 on a sample that left events out.
* :func:`solve_problems` solves each problem on its route: the exact route
  (:mod:`~climada_petals.engine.parametric.solvers`), with the amounts
  snapped and the unidentified ones filled, or the grid route.
* :func:`resolved_design` writes every unit's parameters into the cover,
  and :func:`record` checks each problem on that cover and returns its
  :class:`~climada_petals.engine.parametric._problem.CoverProblem`.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Final, NamedTuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from ._grid import (
    GridSolved,
    UnitData,
    UnitGrid,
    grid_paid_at_zero,
    plan_grids,
    solve_grid,
)
from ._problem import CoverProblem
from .constraints import (
    Budget,
    UnitAmounts,
    check_bounded,
    final_amounts,
    payout_at_zero,
    resolve_limit,
    resolved_cover,
    unit_amounts,
)
from .cover import Limit, ParametricCover
from .objectives import MSE, Objective, tail_degeneracy
from .pricing import Pricing
from .sample import OutcomeSample, check_left_out_unpaid
from .solvers import (
    DesignProblem,
    Settings,
    Solution,
    check_limit,
    solve_problem,
    verify_solution,
)

__all__ = [
    "DesignUnits",
    "Setup",
    "Solved",
    "design_units",
    "record",
    "resolved_design",
    "solve_problems",
    "unidentified",
]

_JOINT: Final[str] = "joint"


class Setup(NamedTuple):
    """The checked arguments of one calibration, with the options as settings."""

    template: ParametricCover
    sample: OutcomeSample
    objective: Objective
    scope: str
    pricing: Pricing
    budget: Budget | None
    settings: Settings


class DesignUnits(NamedTuple):
    """The template's units on the design sample, by position in the sample.

    Parameters
    ----------
    exact : dict of int to UnitAmounts
        The units of the exact route.
    grids : dict of int to UnitGrid
        The units of the grid route.
    limit : Limit or None
        The template's limit, resolved on the design sample.
    """

    exact: dict[int, UnitAmounts]
    grids: dict[int, UnitGrid]
    limit: Limit | None


class Solved(NamedTuple):
    """A solved design problem.

    Parameters
    ----------
    problem : DesignProblem
        Its data.
    positions : list of int
        Its units' positions in the sample.
    solution : Solution
        The solution.
    amounts : dict of unit id to numpy.ndarray
        On the exact route, each unit's final amounts; empty on a grid.
    grid : GridSolved or None
        On a grid, its record.
    """

    problem: DesignProblem
    positions: list[int]
    solution: Solution
    amounts: dict[Hashable, NDArray[np.float64]]
    grid: GridSolved | None = None


def _cap(limit: Limit | None, unit: Hashable) -> float | None:
    """Return a unit's resolved limit, or None."""
    return None if limit is None else limit.amount_of(unit)


def design_units(setup: Setup) -> DesignUnits:
    """Return the template's units on the design sample; check them before solving.

    Raises
    ------
    ParametricInputError, CoverDefinitionError, CalibrationLimitError
        As :func:`~climada_petals.engine.parametric.calibrate.calibrate`,
        before anything is solved.
    """
    template, sample = setup.template, setup.sample
    limit = resolve_limit(template.limit, sample.unit_id, sample.loss)
    grids = plan_grids(
        template,
        setup.objective,
        scope=setup.scope,
        max_points=setup.settings.max_grid_points,
    )
    positions = [p for p in range(len(sample.unit_id)) if p not in grids]
    exact = unit_amounts(
        template, sample, scope=setup.scope, limit=limit, positions=positions
    )
    if setup.budget is None and not isinstance(setup.objective, MSE):
        check_bounded(exact, setup.objective.name)
    at_zero = [payout_at_zero(unit.payout, unit.constraints) for unit in exact]
    if sample.n_left_out:
        at_zero += [
            grid_paid_at_zero(grid, _cap(limit, grid.unit_id))
            for grid in grids.values()
        ]
    check_left_out_unpaid(sample, at_zero)
    return DesignUnits(dict(zip(positions, exact)), grids, limit)


def _solve_one(
    setup: Setup, units: DesignUnits, positions: list[int], loss: NDArray[np.float64]
) -> Solved:
    """Solve one design problem on its route."""
    sample = setup.sample
    unit_id = tuple(sample.unit_id[position] for position in positions)
    bound = (
        None
        if setup.budget is None
        else setup.budget.bound(unit_id, sample.measure.expectation(loss))
    )
    grid = units.grids.get(positions[0])
    if grid is not None:
        data = UnitData(
            sample.index_values[:, grid.position],
            loss,
            sample.measure,
            _cap(units.limit, grid.unit_id),
            bound,
            setup.objective,
            setup.pricing,
        )
        found = solve_grid(grid, data, setup.settings)
        return Solved(found.problem, positions, found.solution, {}, found)
    members = [units.exact[position] for position in positions]
    problem = DesignProblem(
        problem_id=_JOINT if setup.scope == _JOINT else str(unit_id[0]),
        basis=sp.hstack([unit.basis for unit in members], format="csr"),
        loss=loss,
        constraints=[unit.constraints for unit in members],
        measure=sample.measure,
        bound=bound,
    )
    solution = solve_problem(
        problem, setup.objective, setup.pricing, setup.settings, len(members)
    )
    amounts = final_amounts(members, solution.amounts, setup.settings.fill)
    return Solved(problem, positions, solution, amounts)


def solve_problems(setup: Setup, units: DesignUnits) -> list[Solved]:
    """Solve every design problem: one per unit, or one for the joint scope.

    Raises
    ------
    CalibrationError
        If a problem is not solved to optimality, or its solution fails a
        check (a subclass where the solver says why).
    """
    n_units = len(setup.sample.unit_id)
    groups = (
        [list(range(n_units))]
        if setup.scope == _JOINT
        else [[position] for position in range(n_units)]
    )
    loss = sp.csc_matrix(setup.sample.loss)
    return [
        _solve_one(
            setup,
            units,
            positions,
            np.asarray(loss[:, positions].sum(axis=1)).reshape(-1),
        )
        for positions in groups
    ]


def resolved_design(
    setup: Setup, units: DesignUnits, solved: Sequence[Solved]
) -> ParametricCover:
    """Return the template with every free parameter set and its limit resolved."""
    amounts: dict[Hashable, NDArray[np.float64]] = {}
    others: dict[Hashable, dict[str, float]] = {}
    for item in solved:
        amounts.update(item.amounts)
        if item.grid is not None:
            others[setup.sample.unit_id[item.positions[0]]] = item.grid.values
    return resolved_cover(
        setup.template, list(units.exact.values()), amounts, units.limit, others
    )


def unidentified(
    units: DesignUnits, solved: Sequence[Solved], unit_id: Sequence[Hashable]
) -> dict[Hashable, tuple[str, ...]]:
    """Return the unidentified amounts of every unit, in the sample's order."""
    found = {unit.unit_id: unit.unidentified_names for unit in units.exact.values()}
    for item in solved:
        if item.grid is not None:
            found[unit_id[item.positions[0]]] = item.grid.unidentified
    return {unit: found[unit] for unit in unit_id}


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


def _ranges(
    solution: Solution, unit_id: tuple[Hashable, ...], cover: ParametricCover
) -> tuple[tuple[Hashable, str, float, float], ...] | None:
    """Return the ranges of a problem's amounts with their names, or None."""
    if solution.ranges is None:
        return None
    payouts: list[Any] = [cover.payout[unit] for unit in unit_id]
    names = [
        (unit, name)
        for unit, payout in zip(unit_id, payouts)
        for name in payout.amount_names()
    ]
    return tuple(
        (unit, name, float(low), float(high))
        for (unit, name), (low, high) in zip(names, solution.ranges)
        if not np.isnan(low)
    )


def record(
    setup: Setup, solved: Solved, paid: NDArray[np.float64], cover: ParametricCover
) -> CoverProblem:
    """Check one solved problem on the resolved cover; return its record.

    ``paid`` is the cover's payout per row and unit of the sample.

    Raises
    ------
    CalibrationError
        If the cover pays above a limit, its objective is not the solver's,
        or its premium is above the budget, beyond the tolerance.
    """
    problem, grid = solved.problem, solved.grid
    unit_id = tuple(cover.unit_id[p] for p in solved.positions)
    paid = paid[:, solved.positions]
    check_limit(
        problem,
        paid,
        [_cap(cover.limit, unit) for unit in unit_id],
        setup.settings.tolerance,
    )
    value, premium = verify_solution(
        problem,
        paid,
        objective=setup.objective,
        pricing=setup.pricing,
        solution=solved.solution,
        tolerance=setup.settings.tolerance,
    )
    weight, degenerate = tail_degeneracy(setup.objective, problem.loss, problem.measure)
    found = CoverProblem(
        problem_id=problem.problem_id,
        unit_id=unit_id,
        objective_value=value,
        premium=premium,
        expected_loss=problem.measure.expectation(problem.loss),
        budget_bound=problem.bound,
        loss_weight=weight,
        degenerate_tail=degenerate,
        solver=solved.solution.info,
        unique=solved.solution.unique,
        ties=None if grid is None else grid.ties,
        n_candidates=None if grid is None else grid.n_candidates,
        ranges=_ranges(solved.solution, unit_id, cover),
    )
    _log(found, setup.objective)
    return found
