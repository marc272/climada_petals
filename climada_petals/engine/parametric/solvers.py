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
in the portfolio scope), ``K`` steps per unit, ``n`` events and ``O``
outcomes. ``L_e`` and ``P_e`` are the problem's loss and payout of event
``e`` (sums over its units), in scaled units: every loss is divided by the
problem's largest event loss, so the data are at most 1.

Variables and constraints of the linear program:

``theta`` (``m K``)
    The payout levels, ``0 <= theta[j, k] <= cap[j]`` with ``cap[j]`` the
    unit's largest loss in the calibration events, and
    ``theta[j, k] <= theta[j, k + 1]``. The payout of event ``e`` is
    ``P_e = B[e] @ theta``, with ``B`` the one-hot step membership of every
    unit side by side (:meth:`~climada_petals.engine.parametric.payout.StepPayout.basis`).
``u`` (one per event with ``L_e > 0``)
    The shortfall, ``u_e >= L_e - P_e``, ``u_e >= 0``. Every objective uses
    ``u`` only through non-decreasing terms, so at the optimum
    ``u_e = (L_e - P_e)^+``, and ``U_o = sum_{e in o} u_e``. An event without
    loss has no shortfall for any payout, so it needs no variable.
``zeta``, ``v``: weighted CVaR only
    The Rockafellar-Uryasev auxiliaries: ``v_o >= U_o - zeta``, ``v_o >= 0``,
    one ``v_o`` per outcome with loss and one for all outcomes without loss
    together (they have ``U_o = 0`` for every payout, so one variable with
    their summed probability is exact). The objective is
    ``alpha zeta + alpha / (1 - beta) sum_o p_o v_o + (1 - alpha) E[U]``;
    minimising over ``zeta`` gives ``alpha CVaR_beta(U) + (1 - alpha) E[U]``.
``eta``, ``w``: cost-of-capital load with a budget only
    ``w_o >= P_o - eta``, ``w_o >= 0`` (pooled in the same way over the
    outcomes that pay nothing for every ``theta``), and the budget
    ``(1 - r) E[P] + r (eta + sum_o p_o w_o / (1 - q)) <= budget E[L]``.
    Its left side is at least the loaded premium for every ``eta``, with
    equality at the best one, so the constraint admits exactly the payouts
    whose loaded premium meets the budget.

With the pure premium, the budget is ``E[P] <= budget E[L]``. The objective
is divided by its value for the zero cover and the budget row by ``E[L]``,
so that both are of order one for the solver whatever the probabilities;
the reported objective value is in loss units.

The linear objectives are solved with :func:`scipy.optimize.linprog`
(``method="highs"``), so they need no dependency beyond scipy. The mean
squared error is a QP, ``sum_o p_o (L_o - P_o)^2`` under the same
constraints; scipy has no QP solver, so it is solved with cvxpy and its
bundled Clarabel solver, an optional dependency imported only then.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import linprog

from climada_petals.util.config import LOGGER

from .errors import CalibrationError, MissingDependencyError
from .measure import OutcomeMeasure
from .objectives import MeanSquaredError, Objective, WeightedCVaR
from .payout import StepPayout
from .pricing import CostOfCapitalLoad, Pricing

__all__ = [
    "CLARABEL_OPTIONS",
    "HIGHS_OPTIONS",
    "MIN_SCALE_LOSS",
    "DesignData",
    "Solution",
    "SolverInfo",
    "solve_design",
]

MIN_SCALE_LOSS: Final[float] = 1e-6
"""The problem is scaled by its largest event loss only above this value."""

HIGHS_OPTIONS: Final[dict[str, Any]] = {
    "presolve": True,
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
}
"""Options passed to :func:`scipy.optimize.linprog` with ``method="highs"``."""

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


@dataclass(frozen=True)
class SolverInfo:
    """How a design problem was solved.

    Attributes
    ----------
    solver : str
        ``"scipy-highs"`` or ``"cvxpy-CLARABEL"``.
    status : str
        The solver's status; always a success in a returned solution.
    message : str
        The solver's message.
    iterations : int or None
        Iterations, where the solver reports them.
    seconds : float
        Wall time of the assembly and the solve.
    """

    solver: str
    status: str
    message: str
    iterations: int | None
    seconds: float


@dataclass(frozen=True, eq=False)
class DesignData:
    """The data of one design problem, in loss units.

    Attributes
    ----------
    problem_id : str
        A unit id (individual scope) or ``"portfolio"``.
    index : numpy.ndarray
        Index values, shape ``(n_events, n_units)``.
    loss : numpy.ndarray
        Losses, shape ``(n_events, n_units)``, >= 0.
    measure : OutcomeMeasure
        Measure over the events.
    """

    problem_id: str
    index: NDArray[np.float64]
    loss: NDArray[np.float64]
    measure: OutcomeMeasure

    @property
    def total_loss(self) -> NDArray[np.float64]:
        """Per-event loss summed over the units, shape ``(n_events,)``."""
        return self.loss.sum(axis=1)

    @property
    def caps(self) -> NDArray[np.float64]:
        """Largest loss of each unit over the events, shape ``(n_units,)``."""
        return self.loss.max(axis=0, initial=0.0)

    @property
    def scale(self) -> float:
        """Loss scale: the largest event loss, or 1 if it is tiny."""
        largest = float(self.total_loss.max(initial=0.0))
        return largest if largest > MIN_SCALE_LOSS else 1.0


@dataclass(frozen=True, eq=False)
class Solution:
    """The solution of one design problem, in loss units.

    Attributes
    ----------
    levels : numpy.ndarray
        ``theta``, shape ``(n_units, n_steps)``.
    objective_value : float
        The optimal objective value.
    info : SolverInfo
        Solver, status and timing.
    """

    levels: NDArray[np.float64]
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
        lower: float = 0.0,
        upper: float | NDArray[np.float64] = np.inf,
    ) -> int:
        """Add a block of ``size`` variables; return its id."""
        self.sizes.append(size)
        self.lower.append(np.full(size, lower))
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

    def solve(self, problem_id: str) -> tuple[NDArray[np.float64], float, SolverInfo]:
        """Solve with HiGHS; return the solution, its value and the solver info.

        Raises
        ------
        CalibrationError
            If HiGHS does not report an optimal solution.
        """
        start = time.perf_counter()
        matrix = sp.vstack(
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
        result = linprog(
            np.concatenate(self.cost),
            A_ub=matrix,
            b_ub=np.concatenate(self.rhs),
            bounds=np.column_stack(
                [np.concatenate(self.lower), np.concatenate(self.upper)]
            ),
            method="highs",
            options=HIGHS_OPTIONS,
        )
        nit = getattr(result, "nit", None)
        info = SolverInfo(
            solver="scipy-highs",
            status=str(result.status),
            message=str(result.message),
            iterations=None if nit is None else int(nit),
            seconds=time.perf_counter() - start,
        )
        if result.status != 0 or result.x is None:
            raise CalibrationError(
                f"HiGHS found no optimal solution: {result.message}",
                problem_id=problem_id,
                status=str(result.status),
            )
        return np.asarray(result.x, dtype=np.float64), float(result.fun), info

    def tail(
        self,
        blocks: dict[int, Any],
        active: NDArray[np.bool_],
        prob: NDArray[np.float64],
    ) -> tuple[int, int, NDArray[np.float64]]:
        """Add Rockafellar-Uryasev rows ``x_o >= value_o - level``, ``x_o >= 0``.

        ``value_o = sum_b blocks[b][o] @ x_b`` for the ``active`` outcomes;
        the other outcomes have value 0 for every solution and share one
        variable, weighted with their summed probability.

        Returns
        -------
        level, excess : int
            The ids of the free level (``zeta`` or ``eta``) and of the
            excess variables.
        weights : numpy.ndarray
            The probability of each excess variable.
        """
        pooled = bool((~active).any())
        weights = prob[active]
        if pooled:
            weights = np.append(weights, prob[~active].sum())
        n_vars = weights.size
        level = self.add(1, lower=-np.inf)
        excess = self.add(n_vars)
        rows: dict[int, Any] = {
            var: sp.vstack(
                [block[active], sp.csr_matrix((int(pooled), _n_cols(block)))]
            )
            for var, block in blocks.items()
        }
        rows[level] = sp.csr_matrix(-np.ones((n_vars, 1)))
        rows[excess] = -sp.identity(n_vars, format="csr")
        self.constrain(rows, np.zeros(n_vars))
        return level, excess, weights


def _n_rows(matrix: Any) -> int:
    """Return the number of rows of a sparse matrix."""
    return int(matrix.shape[0])


def _n_cols(matrix: Any) -> int:
    """Return the number of columns of a sparse matrix."""
    return int(matrix.shape[1])


def _monotone_matrix(n_units: int, n_steps: int) -> sp.csr_matrix:
    """Rows ``theta[j, k] - theta[j, k + 1]`` for every unit and step."""
    first = np.array(
        [j * n_steps + k for j in range(n_units) for k in range(n_steps - 1)],
        dtype=np.int64,
    )
    n_rows = first.size
    return sp.csr_matrix(
        (
            np.tile([1.0, -1.0], n_rows),
            (
                np.repeat(np.arange(n_rows), 2),
                np.column_stack([first, first + 1]).ravel(),
            ),
        ),
        shape=(n_rows, n_units * n_steps),
    )


@dataclass(frozen=True, eq=False)
class _Scaled:
    """The scaled arrays every formulation needs."""

    loss: NDArray[np.float64]  # (n,) problem loss / scale
    caps: NDArray[np.float64]  # (m K,) upper bound of each level / scale
    basis: sp.csr_matrix  # (n, m K) one-hot step membership
    event_prob: NDArray[np.float64]  # (n,)
    group: sp.csr_matrix  # (O, n)
    prob: NDArray[np.float64]  # (O,)
    monotone: sp.csr_matrix | None  # (m (K - 1), m K)

    @property
    def expected_loss(self) -> float:
        """``E[L]`` in scaled units."""
        return float(self.event_prob @ self.loss)

    @property
    def premium_row(self) -> NDArray[np.float64]:
        """``E[P] = premium_row @ theta``, shape ``(m K,)``."""
        return np.asarray(self.basis.T @ self.event_prob, dtype=np.float64)

    @property
    def budget_unit(self) -> float:
        """Divisor of the budget row: ``E[L]``, or 1 if it is 0."""
        expected = self.expected_loss
        return expected if expected > 0.0 else 1.0


def _scaled(data: DesignData, payout: StepPayout) -> _Scaled:
    """Scale the problem data and build the payout's basis matrix."""
    scale = data.scale
    n_units = data.loss.shape[1]
    basis = sp.csr_matrix(
        sp.hstack([payout.basis(data.index[:, j]) for j in range(n_units)])
    )
    return _Scaled(
        loss=data.total_loss / scale,
        caps=np.repeat(data.caps / scale, payout.n_steps),
        basis=basis,
        event_prob=data.measure.event_probability,
        group=data.measure.grouping_matrix,
        prob=data.measure.probability,
        monotone=(
            _monotone_matrix(n_units, payout.n_steps) if payout.n_steps > 1 else None
        ),
    )


def _objective_norm(
    objective: Objective, arr: _Scaled, measure: OutcomeMeasure
) -> float:
    """Return the objective of the zero cover in scaled units, or 1 if it is 0.

    The solver minimises the objective divided by this value, which is then
    at most 1 at the optimum, whatever the probabilities: with 10,000
    simulation years an outcome has probability ``1e-4``, and an
    unnormalised objective would be close to the solvers' tolerances.
    """
    value = objective.evaluate(arr.loss, np.zeros_like(arr.loss), measure)
    return value if value > 0.0 else 1.0


def _add_budget(
    program: _LinearProgram,
    arr: _Scaled,
    theta: int,
    pricing: Pricing,
    budget: float,
) -> None:
    """Add ``premium / E[L] <= budget`` (module docstring)."""
    unit = arr.budget_unit
    bound = np.array([budget * arr.expected_loss / unit])
    premium_row = sp.csr_matrix(arr.premium_row[None, :] / unit)
    if not (isinstance(pricing, CostOfCapitalLoad) and pricing.rate > 0.0):
        program.constrain({theta: premium_row}, bound)
        return
    rate, level = pricing.rate, pricing.q
    pays = sp.csr_matrix(arr.group @ arr.basis)
    eta, tail, weights = program.tail({theta: pays}, pays.getnnz(axis=1) > 0, arr.prob)
    program.constrain(
        {
            theta: (1.0 - rate) * premium_row,
            eta: sp.csr_matrix([[rate / unit]]),
            tail: sp.csr_matrix(rate / (1.0 - level) * weights[None, :] / unit),
        },
        bound,
    )


def _add_shortfall(
    program: _LinearProgram, arr: _Scaled, theta: int, lossy: NDArray[np.bool_]
) -> int:
    """Add the shortfall ``u`` of the events with loss and its rows; return its id."""
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
        zeta, excess, weights = program.tail(
            {short: group}, np.asarray(group.sum(axis=1)).ravel() > 0, arr.prob
        )
        program.cost[zeta][:] = alpha / norm
        program.cost[excess][:] = alpha / (1.0 - beta) * weights / norm
        program.cost[short][:] = (1.0 - alpha) * arr.event_prob[lossy] / norm
    else:
        program.cost[short][:] = arr.event_prob[lossy] / norm


def _solve_lp(
    data: DesignData,
    payout: StepPayout,
    objective: Objective,
    pricing: Pricing,
    budget: float | None,
) -> Solution:
    """Assemble the LP of the module docstring and solve it with HiGHS."""
    arr = _scaled(data, payout)
    norm = _objective_norm(objective, arr, data.measure)
    program = _LinearProgram()
    theta = program.add(arr.caps.size, upper=arr.caps)
    lossy = arr.loss > 0.0
    short = _add_shortfall(program, arr, theta, lossy)
    if arr.monotone is not None:
        program.constrain({theta: arr.monotone}, np.zeros(_n_rows(arr.monotone)))
    _set_cost(program, arr, objective, short=short, lossy=lossy, norm=norm)
    if budget is not None:
        _add_budget(program, arr, theta, pricing, budget)
    solution, value, info = program.solve(data.problem_id)
    levels = solution[program.offset(theta)].reshape(data.loss.shape[1], payout.n_steps)
    return Solution(levels * data.scale, value * norm * data.scale, info)


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
            "the mean squared error objective is a quadratic program and needs "
            "the optional dependency cvxpy, which is not installed. Install "
            "cvxpy, or use ExpectedUncovered or WeightedCVaR, which are linear "
            "programs solved by scipy's HiGHS"
        ) from exc


def _qp_budget(
    cvx: Any,
    arr: _Scaled,
    theta: Any,
    outcome_payout: Any,
    pricing: Pricing,
    *,
    budget: float,
) -> Any:
    """Return the QP's budget constraint ``premium / E[L] <= budget`` (cvxpy)."""
    expected_payout = arr.premium_row @ theta
    if isinstance(pricing, CostOfCapitalLoad) and pricing.rate > 0.0:
        eta = cvx.Variable()
        tail = eta + arr.prob @ cvx.pos(outcome_payout - eta) / (1.0 - pricing.q)
        premium = (1.0 - pricing.rate) * expected_payout + pricing.rate * tail
    else:
        premium = expected_payout
    unit = arr.budget_unit
    return premium / unit <= budget * arr.expected_loss / unit


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
        solver=f"cvxpy-{_QP_SOLVER}",
        status=str(problem.status),
        message=str(problem.status),
        iterations=None if iterations is None else int(iterations),
        seconds=time.perf_counter() - start,
    )


def _solve_qp(
    data: DesignData, payout: StepPayout, pricing: Pricing, budget: float | None
) -> Solution:
    """Solve the mean squared error design with cvxpy and Clarabel."""
    cvx = _import_cvxpy()
    start = time.perf_counter()
    arr = _scaled(data, payout)
    norm = _objective_norm(MeanSquaredError(), arr, data.measure)
    theta = cvx.Variable(arr.caps.size)
    outcome_payout = arr.group @ (arr.basis @ theta)
    residual = arr.group @ arr.loss - outcome_payout
    constraints = [theta >= 0, theta <= arr.caps]
    if arr.monotone is not None:
        constraints.append(arr.monotone @ theta <= 0)
    if budget is not None:
        constraints.append(
            _qp_budget(cvx, arr, theta, outcome_payout, pricing, budget=budget)
        )
    problem = cvx.Problem(
        cvx.Minimize((arr.prob / norm) @ cvx.square(residual)), constraints
    )
    info = _run_qp(cvx, problem, data.problem_id, start)
    if problem.status != "optimal" or theta.value is None:
        raise CalibrationError(
            f"cvxpy's {_QP_SOLVER} found no optimal solution: {problem.status}",
            problem_id=data.problem_id,
            status=str(problem.status),
        )
    levels = np.asarray(theta.value, dtype=np.float64).reshape(
        data.loss.shape[1], payout.n_steps
    )
    return Solution(
        levels * data.scale, float(problem.value) * norm * data.scale**2, info
    )


def solve_design(
    data: DesignData,
    payout: StepPayout,
    objective: Objective,
    pricing: Pricing,
    budget: float | None,
) -> Solution:
    """Solve one design problem (module docstring).

    Parameters
    ----------
    data : DesignData
        The problem's events, index, losses and measure.
    payout : StepPayout
        The payout structure of every unit.
    objective : Objective
        What is minimised.
    pricing : Pricing
        The premium the budget constrains.
    budget : float or None
        ``kappa`` in ``premium <= kappa E[L]``, or None for no budget.

    Returns
    -------
    Solution
        Levels and objective value in loss units, and the solver info.

    Raises
    ------
    CalibrationError
        If the solver finds no optimal solution.
    MissingDependencyError
        If the objective is the mean squared error and cvxpy is missing.
    """
    if isinstance(objective, MeanSquaredError):
        solution = _solve_qp(data, payout, pricing, budget)
    else:
        solution = _solve_lp(data, payout, objective, pricing, budget)
    LOGGER.debug(
        "design problem %r: %s %s in %.3f s",
        data.problem_id,
        solution.info.solver,
        solution.info.status,
        solution.info.seconds,
    )
    return solution
