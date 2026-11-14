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

The design problem as a linear or quadratic program, and its solution.

One design problem has ``m`` units (one in the individual scope, every unit
in the joint scope), each with a payout that is linear in its amounts
(:class:`~climada_petals.engine.parametric.payout.LinearInAmounts`), ``n``
rows (events, or events and draws) and ``O`` outcomes with weights ``w_o``
(:mod:`~climada_petals.engine.parametric.measure`). The payout of row ``r``
is ``P_r = B[r] @ theta``, with ``B`` the units' bases side by side, and
``L_r`` is the problem's loss (sums over its units). Everything is scaled:
losses, amounts and bounds are divided by the problem's largest loss in one
row, so the data are at most 1.

Variables and constraints of the linear program:

``theta``
    The amounts, within the bounds of each unit's
    :class:`~climada_petals.engine.parametric.parameters.AmountConstraints`
    (``lower >= 0``, a free amount's ``upper``, the cover's limit) and with
    their rows, such as ``theta[j, k] <= theta[j, k + 1]`` for monotone
    levels.
``u`` (one per row with ``L_r > 0``)
    The shortfall, ``u_r >= L_r - P_r``, ``u_r >= 0``. Every objective uses
    ``u`` only through non-decreasing terms, so at the optimum
    ``u_r = (L_r - P_r)^+``, and ``U_o = sum_{r in o} u_r``. A row without
    loss has no shortfall for any payout, so it needs no variable.
``zeta``, ``v``: weighted CVaR only
    The Rockafellar-Uryasev auxiliaries: ``zeta >= 0``, ``v_o >= U_o - zeta``,
    ``v_o >= 0``, one ``v_o`` per outcome with a loss. The objective is
    ``alpha zeta + alpha / (1 - beta) sum_o w_o v_o + (1 - alpha) E[U]``;
    minimising over ``zeta`` gives ``alpha CVaR_beta(U) + (1 - alpha) E[U]``.
    Because ``U >= 0``, the bound ``zeta >= 0`` gives the CVaR of
    :mod:`~climada_petals.engine.parametric.riskmeasures` for weights that
    sum to 1, to less (the missing mass counts as zero) or to more (the
    event basis above one event per year), and outcomes without loss need
    no variable.
``eta``, ``x``: cost-of-capital load with a budget only
    ``eta >= 0``, ``x_o >= P_o - eta``, ``x_o >= 0`` for the outcomes that
    can pay, and the budget
    ``(1 - r) E[P] + r (eta + sum_o w_o x_o / (1 - q)) <= bound``. Its left
    side is at least the loaded premium for every ``eta``, with equality at
    the best one, so the constraint admits exactly the payouts whose loaded
    premium meets the budget.

With the pure premium, the budget is ``E[P] <= bound``. The objective is
divided by its value for the zero cover and the budget row by ``E[L]``, so
that both are of order one for the solver whatever the weights.

The linear programs are solved with :func:`scipy.optimize.linprog`. Its
method follows :data:`LP_METHODS`: ``"auto"`` takes ``"highs"`` for a
problem with one unit and ``"highs-ipm"`` (the interior-point method, with
crossover to a vertex) for several, which solves large joint problems much
faster; both give the same objective value. A ``time_limit`` passes through
to HiGHS. A solve that doesn't end optimal raises: infeasible
(:class:`~climada_petals.engine.parametric.errors.InfeasibleCalibrationError`),
unbounded (:class:`~climada_petals.engine.parametric.errors.UnboundedCalibrationError`),
a time or iteration limit
(:class:`~climada_petals.engine.parametric.errors.CalibrationLimitError`),
or anything else (:class:`~climada_petals.engine.parametric.errors.CalibrationError`).

The mean squared error is a QP, ``sum_o w_o (L_o - P_o)^2`` under the same
constraints on the amounts, with no auxiliary variables. scipy's public
interface has no QP solver, so it is solved with
:func:`scipy.optimize.minimize` and ``method="SLSQP"`` on the amounts, with
the analytic gradient. Under a cost-of-capital load, the premium is held by
cutting planes: each round adds the linear piece of the loaded premium at
the round's solution, until the premium meets the budget within
``feasibility_tolerance``. The route has stated limits (:class:`QPLimits`):
``max_slsqp_amounts`` amounts, ``max_cut_rounds`` rounds and the
``time_limit``; reaching one raises
:class:`~climada_petals.engine.parametric.errors.CalibrationLimitError`,
which names it. The route also reports whether the design's identified
amounts are unique, from the smallest eigenvalue of the objective's Hessian
restricted to them (``Solution.unique``).

**Ranges on the LP route** (``ranges=True``). One optimal vertex is one
member of what may be a set of optimal designs. With ``ranges=True`` each
identified amount (not fixed, and reached by a row with a weight) is
ranged over the near-optimal set
``{feasible x : c x <= c* + eps max(1, |c*|)}``, with ``c`` the LP's cost
(the objective divided by that of the zero cover) and ``c*`` its optimum,
re-solved first at HiGHS's tolerances of ``1e-10``
(:data:`RANGE_HIGHS_OPTIONS`). Each amount is ranged twice, at a wide and a
narrow tolerance (:data:`RANGE_TOLERANCES`, ``1e-9`` and ``1e-11``): its
smallest and its largest value, so four LPs per amount, each as large as
the design problem's own LP. Near the optimum
the width of a range is ``w0 + eps / s``, with ``w0`` the width of the
optimal set along the amount and ``s`` the objective's slope there. At a
unique vertex ``w0 = 0``, so the width falls with ``eps``; on a set of
optimal designs it keeps ``w0``. An amount is **pinned** when its narrow
range is at most half its wide one, or narrower than ``1e-7`` of the scale
(:data:`PINNED_SHARE`, :data:`PINNED_WIDTH`), and the design is unique when
every identified amount is pinned. ``Solution.ranges`` holds the wide
ranges, in loss units. An optimal set narrower than about ``1e-9 / s`` is
reported as unique.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import scipy
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import linprog

from climada_petals.util.config import LOGGER

from ._problem import (
    MIN_SCALE_LOSS,
    DesignProblem,
    Scaled,
    Solution,
    SolverInfo,
    objective_norm,
)
from ._qp import SLSQP_OPTIONS, UNIQUE_RTOL, QPLimits, solve_mse
from .constraints import violation
from .errors import (
    CalibrationError,
    CalibrationLimitError,
    InfeasibleCalibrationError,
    UnboundedCalibrationError,
)
from .objectives import MSE, Objective, WeightedCVaR
from .pricing import CostOfCapitalLoad, Pricing

__all__ = [
    "HIGHS_OPTIONS",
    "LP_METHODS",
    "MIN_SCALE_LOSS",
    "PINNED_SHARE",
    "PINNED_WIDTH",
    "RANGE_HIGHS_OPTIONS",
    "RANGE_TOLERANCES",
    "SLSQP_OPTIONS",
    "UNIQUE_RTOL",
    "DesignProblem",
    "QPLimits",
    "Settings",
    "Solution",
    "SolverInfo",
    "check_amounts",
    "check_limit",
    "lp_method",
    "solve_design",
    "solve_problem",
    "verify_solution",
]

LP_METHODS: Final[tuple[str, ...]] = ("auto", "highs", "highs-ipm")
"""The values of ``CalibrationOptions.lp_method`` (:func:`lp_method`)."""

HIGHS_OPTIONS: Final[dict[str, Any]] = {
    "presolve": True,
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
}
"""Options passed to :func:`scipy.optimize.linprog` with either HiGHS method."""

RANGE_TOLERANCES: Final[tuple[float, float]] = (1e-9, 1e-11)
"""The wide and the narrow tolerance of the near-optimal set (module docstring)."""

RANGE_HIGHS_OPTIONS: Final[dict[str, Any]] = {
    "presolve": True,
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}
"""Options of the LPs that range the amounts, and of the optimum they re-solve."""

PINNED_SHARE: Final[float] = 0.5
"""An amount is pinned when its narrow range is at most this share of its wide one."""

PINNED_WIDTH: Final[float] = 1e-7
"""An amount is pinned when its narrow range is narrower than this, in scaled units."""

_LIMIT_STATUS: Final[int] = 1
_INFEASIBLE_STATUS: Final[int] = 2
_UNBOUNDED_STATUS: Final[int] = 3


def lp_method(option: str, n_units: int) -> str:
    """Return the ``linprog`` method of a design problem.

    Parameters
    ----------
    option : {"auto", "highs", "highs-ipm"}
        ``CalibrationOptions.lp_method``.
    n_units : int
        The problem's units.

    Returns
    -------
    str
        ``option`` if it names a method; under ``"auto"``, ``"highs"`` for
        one unit and ``"highs-ipm"`` for several.
    """
    if option != "auto":
        return option
    return "highs" if n_units == 1 else "highs-ipm"


@dataclass
class _LinearProgram:
    """A linear program ``min c x, A x <= b, lower <= x <= upper``, block by block."""

    sizes: list[int] = field(default_factory=list)
    lower: list[NDArray[np.float64]] = field(default_factory=list)
    upper: list[NDArray[np.float64]] = field(default_factory=list)
    cost: list[NDArray[np.float64]] = field(default_factory=list)
    rows: list[dict[int, Any]] = field(default_factory=list)
    rhs: list[NDArray[np.float64]] = field(default_factory=list)

    def add(
        self,
        size: int,
        *,
        lower: float | NDArray[np.float64] = 0.0,
        upper: float | NDArray[np.float64] = np.inf,
    ) -> int:
        """Add a block of ``size`` variables; return its id."""
        self.sizes.append(size)
        self.lower.append(np.broadcast_to(np.asarray(lower, float), (size,)).copy())
        self.upper.append(np.broadcast_to(np.asarray(upper, float), (size,)).copy())
        self.cost.append(np.zeros(size))
        return len(self.sizes) - 1

    def constrain(self, blocks: dict[int, Any], bound: NDArray[np.float64]) -> None:
        """Add the rows ``sum_b blocks[b] @ x_b <= bound`` (sparse blocks)."""
        self.rows.append(blocks)
        self.rhs.append(np.asarray(bound, dtype=np.float64))

    def offset(self, var: int) -> slice:
        """Return the positions of a variable block in the solution vector."""
        start = sum(self.sizes[:var])
        return slice(start, start + self.sizes[var])

    def matrix(self) -> sp.csr_matrix:
        """Return the rows of every constraint as one sparse matrix."""
        return sp.vstack(
            [
                sp.hstack(
                    [
                        (
                            sp.csr_matrix(block[var])
                            if var in block
                            else sp.csr_matrix((len(bound), size))
                        )
                        for var, size in enumerate(self.sizes)
                    ]
                )
                for block, bound in zip(self.rows, self.rhs, strict=True)
            ],
            format="csr",
        )

    def tail(
        self, blocks: dict[int, Any], weights: NDArray[np.float64]
    ) -> tuple[int, int]:
        """Add Rockafellar-Uryasev rows ``x_o >= value_o - level``, ``x_o, level >= 0``.

        ``value_o = sum_b blocks[b][o] @ x_b``, one row per outcome given.

        Returns
        -------
        level, excess : int
            The ids of the level (``zeta`` or ``eta``) and of the excess
            variables, one per outcome, whose weights are ``weights``.
        """
        n_vars = weights.size
        level = self.add(1)
        excess = self.add(n_vars)
        if n_vars:
            rows: dict[int, Any] = dict(blocks)
            rows[level] = sp.csr_matrix(-np.ones((n_vars, 1)))
            rows[excess] = -sp.identity(n_vars, format="csr")
            self.constrain(rows, np.zeros(n_vars))
        return level, excess


def _add_budget(
    program: _LinearProgram, arr: Scaled, theta: int, pricing: Pricing, bound: float
) -> None:
    """Add ``premium / E[L] <= bound / E[L]`` (module docstring); ``bound`` scaled."""
    unit = arr.budget_unit
    rhs = np.array([bound / unit])
    premium_row = sp.csr_matrix(arr.premium_row[None, :] / unit)
    if not (isinstance(pricing, CostOfCapitalLoad) and pricing.rate > 0.0):
        program.constrain({theta: premium_row}, rhs)
        return
    rate, level = pricing.rate, pricing.q
    pays = sp.csr_matrix(arr.group @ arr.basis)
    active = pays.getnnz(axis=1) > 0
    weights = arr.weights[active]
    eta, tail = program.tail({theta: pays[active]}, weights)
    program.constrain(
        {
            theta: (1.0 - rate) * premium_row,
            eta: sp.csr_matrix([[rate / unit]]),
            tail: sp.csr_matrix(rate / (1.0 - level) * weights[None, :] / unit),
        },
        rhs,
    )


def _add_shortfall(
    program: _LinearProgram, arr: Scaled, theta: int, lossy: NDArray[np.bool_]
) -> int:
    """Add the shortfall ``u`` of the rows with loss and its rows; return its id."""
    n_lossy = int(lossy.sum())
    short = program.add(n_lossy)
    if n_lossy:
        program.constrain(
            {theta: -arr.basis[lossy], short: -sp.identity(n_lossy, format="csr")},
            -arr.loss[lossy],
        )
    return short


def _set_cost(
    program: _LinearProgram,
    arr: Scaled,
    objective: Objective,
    *,
    short: int,
    lossy: NDArray[np.bool_],
    norm: float,
) -> None:
    """Set the objective's cost, adding the CVaR auxiliaries if it has a tail."""
    if isinstance(objective, WeightedCVaR) and objective.alpha > 0.0:
        alpha, beta = objective.alpha, objective.beta
        group = sp.csr_matrix(arr.group[:, lossy])
        active = group.getnnz(axis=1) > 0
        weights = arr.weights[active]
        zeta, excess = program.tail({short: group[active]}, weights)
        program.cost[zeta][:] = alpha / norm
        program.cost[excess][:] = alpha / (1.0 - beta) * weights / norm
        program.cost[short][:] = (1.0 - alpha) * arr.event_weight[lossy] / norm
    else:
        program.cost[short][:] = arr.event_weight[lossy] / norm


def _linprog_error(result: Any, problem_id: str, method: str) -> CalibrationError:
    """Return the typed error of a ``linprog`` result that is not optimal."""
    status, message = int(result.status), str(result.message)
    kinds: dict[int, tuple[type[CalibrationError], str]] = {
        _LIMIT_STATUS: (CalibrationLimitError, "HiGHS reached a limit"),
        _INFEASIBLE_STATUS: (InfeasibleCalibrationError, "the problem is infeasible"),
        _UNBOUNDED_STATUS: (UnboundedCalibrationError, "the problem is unbounded"),
    }
    kind, what = kinds.get(
        status, (CalibrationError, "HiGHS found no optimal solution")
    )
    return kind(
        f"{what} ({method}): {message}",
        problem_id=problem_id,
        status=str(status),
        solver_message=message,
    )


def _assemble(
    problem: DesignProblem, objective: Objective, pricing: Pricing
) -> tuple[_LinearProgram, int, float]:
    """Return the LP of the module docstring, the id of ``theta`` and the norm."""
    arr = problem.scaled()
    norm = objective_norm(objective, arr, problem.measure)
    program = _LinearProgram()
    theta = program.add(arr.lower.size, lower=arr.lower, upper=arr.upper)
    lossy = arr.loss > 0.0
    short = _add_shortfall(program, arr, theta, lossy)
    if arr.rhs.size:
        program.constrain({theta: arr.rows}, arr.rhs)
    _set_cost(program, arr, objective, short=short, lossy=lossy, norm=norm)
    if problem.bound is not None:
        _add_budget(program, arr, theta, pricing, problem.bound / problem.scale)
    return program, theta, norm


@dataclass(frozen=True, eq=False)
class _Arrays:
    """An assembled LP as ``linprog`` takes it, with its solve's settings."""

    cost: NDArray[np.float64]
    matrix: sp.csr_matrix
    rhs: NDArray[np.float64]
    bounds: NDArray[np.float64]
    method: str
    time_limit: float | None
    problem_id: str

    @classmethod
    def from_program(
        cls,
        program: _LinearProgram,
        method: str,
        time_limit: float | None,
        problem_id: str,
    ) -> _Arrays:
        """Return a program's arrays."""
        cost = np.concatenate(program.cost)
        return cls(
            cost,
            program.matrix() if program.rows else sp.csr_matrix((0, cost.size)),
            np.concatenate(program.rhs) if program.rows else np.zeros(0),
            np.column_stack(
                [np.concatenate(program.lower), np.concatenate(program.upper)]
            ),
            method,
            time_limit,
            problem_id,
        )

    def solve(
        self,
        cost: NDArray[np.float64],
        options: dict[str, Any],
        cut: float | None = None,
    ) -> Any:
        """Return ``linprog``'s result for ``cost``; ``cut`` adds ``self.cost @ x <= cut``."""
        matrix, rhs = self.matrix, self.rhs
        if cut is not None:
            matrix = sp.vstack([matrix, sp.csr_matrix(self.cost)], format="csr")
            rhs = np.append(rhs, cut)
        settings = dict(options)
        if self.time_limit is not None:
            settings["time_limit"] = self.time_limit
        return linprog(
            cost,
            A_ub=matrix if rhs.size else None,
            b_ub=rhs if rhs.size else None,
            bounds=self.bounds,
            method=self.method,
            options=settings,
        )

    def ranging(self, cost: NDArray[np.float64], cut: float | None = None) -> float:
        """Return the optimum of ``cost`` at the ranges' tolerances; raise if not optimal."""
        result = self.solve(cost, RANGE_HIGHS_OPTIONS, cut)
        if result.status != 0 or result.x is None:
            error = _linprog_error(result, self.problem_id, self.method)
            error.add_note("raised while ranging the amounts (ranges=True)")
            raise error
        return float(result.fun)

    def extent(self, column: int, cut: float) -> tuple[float, float]:
        """Return the smallest and the largest ``x[column]`` with ``cost @ x <= cut``."""
        unit = np.zeros(self.cost.size)
        unit[column] = 1.0
        return self.ranging(unit, cut), -self.ranging(-unit, cut)


def _ranges(
    arrays: _Arrays, theta: slice, problem: DesignProblem
) -> tuple[NDArray[np.float64], bool]:
    """Range the identified amounts over the near-optimal set (module docstring).

    Returns
    -------
    ranges : numpy.ndarray
        The wide range of each amount in loss units, shape ``(n_amounts, 2)``;
        NaN where an amount is not identified.
    unique : bool
        Whether every identified amount is pinned; true if none is.

    Raises
    ------
    CalibrationError
        If an LP doesn't end optimal (its typed subclass, as for the solve).
    """
    optimum = arrays.ranging(arrays.cost)
    size = max(1.0, abs(optimum))
    wide = optimum + RANGE_TOLERANCES[0] * size
    narrow = optimum + RANGE_TOLERANCES[1] * size
    ranges = np.full((theta.stop - theta.start, 2), np.nan)
    unique = True
    for amount in np.flatnonzero(problem.scaled().identified):
        ranges[amount] = arrays.extent(theta.start + amount, wide)
        low, high = arrays.extent(theta.start + amount, narrow)
        width = float(ranges[amount, 1] - ranges[amount, 0])
        unique = unique and (
            high - low <= PINNED_SHARE * width or high - low < PINNED_WIDTH
        )
    return ranges * problem.scale, unique


def _lp_info(result: Any, method: str, start: float) -> SolverInfo:
    """Return the record of a solved linear program."""
    nit = getattr(result, "nit", None)
    return SolverInfo(
        route="lp",
        solver="scipy-highs",
        version=scipy.__version__,
        method=method,
        status=str(result.status),
        message=str(result.message),
        iterations=None if nit is None else int(nit),
        seconds=time.perf_counter() - start,
    )


def _solve_lp(
    problem: DesignProblem,
    objective: Objective,
    pricing: Pricing,
    *,
    method: str,
    time_limit: float | None,
    ranges: bool = False,
) -> Solution:
    """Assemble the LP of the module docstring and solve it with HiGHS.

    With ``ranges``, also range the amounts (module docstring).
    """
    start = time.perf_counter()
    program, theta, norm = _assemble(problem, objective, pricing)
    arrays = _Arrays.from_program(program, method, time_limit, problem.problem_id)
    result = arrays.solve(arrays.cost, HIGHS_OPTIONS)
    if result.status != 0 or result.x is None:
        raise _linprog_error(result, problem.problem_id, method)
    ranged, unique = (
        _ranges(arrays, program.offset(theta), problem) if ranges else (None, None)
    )
    amounts = np.asarray(result.x, dtype=np.float64)[program.offset(theta)]
    return Solution(
        amounts * problem.scale,
        float(result.fun) * norm * problem.scale,
        _lp_info(result, method, start),
        unique=unique,
        ranges=ranged,
    )


def check_amounts(
    problem: DesignProblem, amounts: NDArray[np.float64], tolerance: float
) -> None:
    """Raise if a solver's amounts break a bound or a row beyond the tolerance.

    Parameters
    ----------
    problem : DesignProblem
        The problem solved.
    amounts : numpy.ndarray
        The solver's amounts, before they are snapped to their bounds.
    tolerance : float
        Relative to the problem's loss scale.

    Raises
    ------
    CalibrationError
        If a bound or a row is broken by more than ``tolerance`` times the
        largest loss in one row.
    """
    start = 0
    for constraints in problem.constraints:
        part = amounts[start : start + constraints.lower.size]
        start += constraints.lower.size
        gap = violation(part, constraints)
        if gap > tolerance * problem.scale:
            raise CalibrationError(
                f"the solver's amounts break their bounds or rows by {gap:.3g}, "
                f"above the tolerance {tolerance:g} x {problem.scale:.6g}",
                problem_id=problem.problem_id,
            )


def check_limit(
    problem: DesignProblem,
    paid: NDArray[np.float64],
    caps: Sequence[float | None],
    tolerance: float,
) -> None:
    """Raise if a unit's payout is above its limit, beyond the tolerance.

    Parameters
    ----------
    problem : DesignProblem
        The problem solved.
    paid : numpy.ndarray
        The resolved cover's payout per row and unit of the problem,
        ``(n_rows, n_units)``.
    caps : sequence of float or None
        Each unit's limit, or None.
    tolerance : float
        Relative to the larger of the limit and the largest loss in one row.

    Raises
    ------
    CalibrationError
        If a payout is above its unit's limit beyond the tolerance.
    """
    for column, cap in enumerate(caps):
        largest = float(paid[:, column].max(initial=0.0))
        if cap is not None and largest > cap + tolerance * max(cap, problem.scale):
            raise CalibrationError(
                f"a payout of {largest:.12g} is above its limit {cap:.12g}",
                problem_id=problem.problem_id,
            )


def verify_solution(
    problem: DesignProblem,
    paid: NDArray[np.float64],
    *,
    objective: Objective,
    pricing: Pricing,
    solution: Solution,
    tolerance: float,
) -> tuple[float, float]:
    """Recompute a problem's objective and premium with numpy; raise on a violation.

    Parameters
    ----------
    problem : DesignProblem
        The problem solved.
    paid : numpy.ndarray
        The resolved cover's payout per row and unit of the problem,
        ``(n_rows, n_units)``.
    objective : Objective
        What was minimised.
    pricing : Pricing
        The premium the budget bounds.
    solution : Solution
        The solver's solution.
    tolerance : float
        Relative: the objective against the solver's, relative to the
        larger of the two and the zero cover's objective; the premium
        against the budget, relative to the larger of the bound, ``E[L]``
        and the largest loss in one row.

    Returns
    -------
    objective_value, premium : float
        Of the resolved cover, on the design sample.

    Raises
    ------
    CalibrationError
        If the objective is not the solver's, or the premium is above the
        budget, beyond the tolerance.
    """
    measure = problem.measure
    total = paid.sum(axis=1)
    value = objective.evaluate(problem.loss, total, measure)
    zero = objective.evaluate(problem.loss, np.zeros_like(total), measure)
    solver_value = solution.objective_value
    if abs(value - solver_value) > tolerance * max(abs(value), abs(solver_value), zero):
        raise CalibrationError(
            f"the objective of the returned amounts is {value:.12g}, and the "
            f"solver's {solver_value:.12g}",
            problem_id=problem.problem_id,
        )
    premium = pricing.premium(total, measure)
    if problem.bound is not None:
        reference = max(
            abs(problem.bound), measure.expectation(problem.loss), problem.scale
        )
        if premium > problem.bound + tolerance * reference:
            raise CalibrationError(
                f"the premium {premium:.12g} is above the budget {problem.bound:.12g}",
                problem_id=problem.problem_id,
            )
    return value, premium


def solve_design(
    problem: DesignProblem,
    objective: Objective,
    pricing: Pricing,
    *,
    method: str,
    time_limit: float | None = None,
    limits: QPLimits = QPLimits(),  # a tuple, so safe to share
    ranges: bool = False,
) -> Solution:
    """Solve one design problem (module docstring).

    Parameters
    ----------
    problem : DesignProblem
        The problem's basis, losses, constraints, measure and budget.
    objective : Objective
        What is minimised.
    pricing : Pricing
        The premium the budget constrains.
    method : {"highs", "highs-ipm"}
        The ``linprog`` method of a linear program (:func:`lp_method`).
    time_limit : float, optional
        Seconds HiGHS may take on a linear program, or the mean squared
        error's route on the whole problem.
    limits : QPLimits, default QPLimits()
        The stated limits of the mean squared error's route.
    ranges : bool, default False
        On the LP route, whether to range the amounts over the near-optimal
        set (module docstring); four more LPs per identified amount. The
        mean squared error's route has no ranges, and ignores it.

    Returns
    -------
    Solution
        The amounts and the objective value in loss units, the solver info,
        whether the design is unique (the mean squared error's Hessian test,
        or the ranges), and the ranges.

    Raises
    ------
    InfeasibleCalibrationError, UnboundedCalibrationError
        If the solver finds the problem infeasible or unbounded.
    CalibrationLimitError
        If ``time_limit``, the solver's iteration limit, or a limit of
        ``limits`` is reached.
    CalibrationError
        If the solver finds no optimal solution for another reason.
    """
    if isinstance(objective, MSE):
        solution = solve_mse(problem, pricing, time_limit=time_limit, limits=limits)
    else:
        solution = _solve_lp(
            problem,
            objective,
            pricing,
            method=method,
            time_limit=time_limit,
            ranges=ranges,
        )
    LOGGER.debug(
        "design problem %r: %s (%s) %s in %.3f s",
        problem.problem_id,
        solution.info.solver,
        solution.info.method,
        solution.info.status,
        solution.info.seconds,
    )
    return solution


@dataclass(frozen=True)
class Settings:
    """What the design problems read of ``CalibrationOptions``.

    Attributes
    ----------
    fill : {"lowest", "highest"}
        The fill rule of unidentified amounts.
    ranges : bool
        Whether the LP route ranges the amounts.
    lp_method : {"auto", "highs", "highs-ipm"}
        The LP method option.
    time_limit : float or None
        Seconds per LP, or per mean squared error's design problem.
    limits : QPLimits
        The mean squared error's limits and the feasibility tolerance.
    max_grid_points : int
        The most grid candidates of one design problem.
    """

    fill: str
    ranges: bool
    lp_method: str
    time_limit: float | None
    limits: QPLimits
    max_grid_points: int

    @property
    def tolerance(self) -> float:
        """The feasibility tolerance, relative."""
        return self.limits.feasibility_tolerance


def solve_problem(
    problem: DesignProblem,
    objective: Objective,
    pricing: Pricing,
    settings: Settings,
    n_units: int,
) -> Solution:
    """Solve one design problem on the exact route; check the solver's amounts.

    Raises
    ------
    CalibrationError
        As :func:`~climada_petals.engine.parametric.solvers.solve_design`
        and :func:`~climada_petals.engine.parametric.solvers.check_amounts`.
    """
    solution = solve_design(
        problem,
        objective,
        pricing,
        method=lp_method(settings.lp_method, n_units),
        time_limit=settings.time_limit,
        limits=settings.limits,
        ranges=settings.ranges,
    )
    check_amounts(problem, solution.amounts, settings.tolerance)
    return solution
