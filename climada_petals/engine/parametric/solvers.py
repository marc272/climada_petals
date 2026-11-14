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
constraints. scipy's public interface has no QP solver, so this version
solves it with cvxpy and its bundled Clarabel solver, an optional
dependency imported only then.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Final

import numpy as np
import scipy
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import linprog

from climada_petals.util.config import LOGGER

from .errors import (
    CalibrationError,
    CalibrationLimitError,
    InfeasibleCalibrationError,
    MissingDependencyError,
    UnboundedCalibrationError,
)
from .measure import OutcomeMeasure
from .constraints import violation
from .objectives import MSE, Objective, WeightedCVaR
from .parameters import AmountConstraints
from .pricing import CostOfCapitalLoad, Pricing

__all__ = [
    "CLARABEL_OPTIONS",
    "HIGHS_OPTIONS",
    "LP_METHODS",
    "MIN_SCALE_LOSS",
    "DesignProblem",
    "Solution",
    "SolverInfo",
    "check_amounts",
    "lp_method",
    "solve_design",
    "verify_solution",
]

MIN_SCALE_LOSS: Final[float] = 1e-6
"""The problem is scaled by its largest loss in one row only above this value."""

LP_METHODS: Final[tuple[str, ...]] = ("auto", "highs", "highs-ipm")
"""The values of ``CalibrationOptions.lp_method`` (:func:`lp_method`)."""

HIGHS_OPTIONS: Final[dict[str, Any]] = {
    "presolve": True,
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
}
"""Options passed to :func:`scipy.optimize.linprog` with either HiGHS method."""

_QP_SOLVER: Final[str] = "CLARABEL"
"""cvxpy solver of the mean squared error; installed with cvxpy itself."""

CLARABEL_OPTIONS: Final[dict[str, float]] = {
    "tol_gap_abs": 1e-10,
    "tol_gap_rel": 1e-10,
    "tol_feas": 1e-10,
}
"""Clarabel tolerances of the mean squared error (its defaults are ``1e-8``).

The objective is flat at its optimum, so the defaults leave optimal values
up to about ``1e-6`` relative above the optimum on a 10,000-year hazard set;
these tolerances bring them to about ``1e-8``."""

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


@dataclass(frozen=True)
class SolverInfo:
    """How a design problem was solved.

    Attributes
    ----------
    route : {"lp", "qp"}
        One linear program, or one quadratic program.
    solver : str
        ``"scipy-highs"`` or ``"cvxpy-CLARABEL"``.
    version : str
        The version of scipy or cvxpy.
    method : str or None
        The ``linprog`` method, ``"highs"`` or ``"highs-ipm"``; None for the
        quadratic program.
    status : str
        The solver's status; always a success in a returned solution.
    message : str
        The solver's message.
    iterations : int or None
        Iterations, where the solver reports them.
    seconds : float
        Wall time of the assembly and the solve.
    """

    route: str
    solver: str
    version: str
    method: str | None
    status: str
    message: str
    iterations: int | None
    seconds: float


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
    """

    amounts: NDArray[np.float64]
    objective_value: float
    info: SolverInfo


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


@dataclass(frozen=True, eq=False)
class _Scaled:
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


def _scaled(problem: DesignProblem) -> _Scaled:
    """Scale the problem data."""
    scale = problem.scale
    rows, rhs = problem.rows
    return _Scaled(
        loss=problem.loss / scale,
        basis=sp.csr_matrix(problem.basis),
        lower=problem.lower / scale,
        upper=problem.upper / scale,
        rows=rows,
        rhs=rhs / scale,
        event_weight=np.asarray(problem.measure.event_weight, dtype=np.float64),
        group=problem.measure.grouping,
        weights=np.asarray(problem.measure.weights, dtype=np.float64),
    )


def _objective_norm(
    objective: Objective, arr: _Scaled, measure: OutcomeMeasure
) -> float:
    """Return the objective of the zero cover in scaled units, or 1 if it is 0.

    The solver minimises the objective divided by this value, which is then
    at most 1 at the optimum, whatever the weights: with 10,000 simulation
    years an outcome has weight ``1e-4``, and an unnormalised objective
    would be close to the solvers' tolerances.
    """
    value = objective.evaluate(arr.loss, np.zeros_like(arr.loss), measure)
    return value if value > 0.0 else 1.0


def _add_budget(
    program: _LinearProgram, arr: _Scaled, theta: int, pricing: Pricing, bound: float
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
    program: _LinearProgram, arr: _Scaled, theta: int, lossy: NDArray[np.bool_]
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
    arr: _Scaled,
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
    arr = _scaled(problem)
    norm = _objective_norm(objective, arr, problem.measure)
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


def _solve_lp(
    problem: DesignProblem,
    objective: Objective,
    pricing: Pricing,
    *,
    method: str,
    time_limit: float | None,
) -> Solution:
    """Assemble the LP of the module docstring and solve it with HiGHS."""
    start = time.perf_counter()
    program, theta, norm = _assemble(problem, objective, pricing)
    options = dict(HIGHS_OPTIONS)
    if time_limit is not None:
        options["time_limit"] = time_limit
    result = linprog(
        np.concatenate(program.cost),
        A_ub=program.matrix() if program.rows else None,
        b_ub=np.concatenate(program.rhs) if program.rows else None,
        bounds=np.column_stack(
            [np.concatenate(program.lower), np.concatenate(program.upper)]
        ),
        method=method,
        options=options,
    )
    if result.status != 0 or result.x is None:
        raise _linprog_error(result, problem.problem_id, method)
    nit = getattr(result, "nit", None)
    info = SolverInfo(
        route="lp",
        solver="scipy-highs",
        version=scipy.__version__,
        method=method,
        status=str(result.status),
        message=str(result.message),
        iterations=None if nit is None else int(nit),
        seconds=time.perf_counter() - start,
    )
    amounts = np.asarray(result.x, dtype=np.float64)[program.offset(theta)]
    return Solution(
        amounts * problem.scale, float(result.fun) * norm * problem.scale, info
    )


def _import_cvxpy() -> Any:
    """Import cvxpy, or explain how to get by without it.

    Raises
    ------
    MissingDependencyError
        If cvxpy is not installed.
    """
    try:
        return importlib.import_module("cvxpy")
    except ImportError as exc:
        raise MissingDependencyError(
            "the mean squared error objective is a quadratic program, which this "
            "version solves with the optional dependency cvxpy, not installed. "
            "Install cvxpy, or use ExpectedUncovered or WeightedCVaR, which are "
            "linear programs solved by scipy's HiGHS"
        ) from exc


def _qp_budget(
    cvx: Any,
    arr: _Scaled,
    theta: Any,
    outcome_payout: Any,
    pricing: Pricing,
    *,
    bound: float,
) -> list[Any]:
    """Return the QP's budget constraints, ``premium / E[L] <= bound / E[L]`` (cvxpy)."""
    expected_payout = arr.premium_row @ theta
    extra: list[Any] = []
    if isinstance(pricing, CostOfCapitalLoad) and pricing.rate > 0.0:
        eta = cvx.Variable()
        extra.append(eta >= 0)
        tail = eta + arr.weights @ cvx.pos(outcome_payout - eta) / (1.0 - pricing.q)
        premium = (1.0 - pricing.rate) * expected_payout + pricing.rate * tail
    else:
        premium = expected_payout
    unit = arr.budget_unit
    return [premium / unit <= bound / unit, *extra]


def _run_qp(cvx: Any, problem: Any, problem_id: str, start: float) -> SolverInfo:
    """Solve a cvxpy problem with Clarabel; return the solver info.

    Raises
    ------
    CalibrationError
        If cvxpy raises a solver error.
    """
    try:
        problem.solve(solver=_QP_SOLVER, **CLARABEL_OPTIONS)
    except cvx.SolverError as exc:
        raise CalibrationError(
            f"cvxpy's {_QP_SOLVER} failed: {exc}", problem_id=problem_id
        ) from exc
    stats = problem.solver_stats
    iterations = None if stats is None else stats.num_iters
    return SolverInfo(
        route="qp",
        solver=f"cvxpy-{_QP_SOLVER}",
        version=str(getattr(cvx, "__version__", "")),
        method=None,
        status=str(problem.status),
        message=str(problem.status),
        iterations=None if iterations is None else int(iterations),
        seconds=time.perf_counter() - start,
    )


def _solve_qp(problem: DesignProblem, pricing: Pricing) -> Solution:
    """Solve the mean squared error design with cvxpy and Clarabel."""
    cvx = _import_cvxpy()
    start = time.perf_counter()
    arr = _scaled(problem)
    norm = _objective_norm(MSE(), arr, problem.measure)
    theta = cvx.Variable(arr.lower.size)
    outcome_payout = arr.group @ (arr.basis @ theta)
    residual = arr.group @ arr.loss - outcome_payout
    bounded = np.isfinite(arr.upper)
    constraints = [theta >= arr.lower]
    if bounded.any():
        constraints.append(theta[bounded] <= arr.upper[bounded])
    if arr.rhs.size:
        constraints.append(arr.rows @ theta <= arr.rhs)
    if problem.bound is not None:
        constraints += _qp_budget(
            cvx,
            arr,
            theta,
            outcome_payout,
            pricing,
            bound=problem.bound / problem.scale,
        )
    program = cvx.Problem(
        cvx.Minimize((arr.weights / norm) @ cvx.square(residual)), constraints
    )
    info = _run_qp(cvx, program, problem.problem_id, start)
    if program.status != "optimal" or theta.value is None:
        raise CalibrationError(
            f"cvxpy's {_QP_SOLVER} found no optimal solution: {program.status}",
            problem_id=problem.problem_id,
            status=str(program.status),
            solver_message=str(program.status),
        )
    amounts = np.asarray(theta.value, dtype=np.float64)
    return Solution(
        amounts * problem.scale,
        float(program.value) * norm * problem.scale**2,
        info,
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
        Seconds HiGHS may take; a linear program only.

    Returns
    -------
    Solution
        The amounts and the objective value in loss units, and the solver
        info.

    Raises
    ------
    InfeasibleCalibrationError, UnboundedCalibrationError
        If HiGHS finds the problem infeasible or unbounded.
    CalibrationLimitError
        If HiGHS reaches ``time_limit`` or its iteration limit.
    CalibrationError
        If the solver finds no optimal solution for another reason.
    MissingDependencyError
        If the objective is the mean squared error and cvxpy is missing.
    """
    if isinstance(objective, MSE):
        solution = _solve_qp(problem, pricing)
    else:
        solution = _solve_lp(
            problem, objective, pricing, method=method, time_limit=time_limit
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
