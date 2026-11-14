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

The mean squared error's route: a quadratic program solved with ``SLSQP``.

The design problem is ``min sum_o w_o (L_o - A_o theta)^2`` over the
amounts ``theta``, with ``A = G B`` the outcomes' basis, under the amounts'
bounds and rows (such as monotone levels) and the premium budget. It has as
many variables as amounts that are not fixed, so
:func:`scipy.optimize.minimize` with ``method="SLSQP"`` works on it
directly, with the analytic gradient ``-2 A^T W (L - A theta)``; the fixed
amounts (``lower == upper``) enter as constants. Like the linear programs,
it is scaled: losses and amounts by the problem's largest loss in one row,
the objective by its value for the zero cover, the budget row by ``E[L]``.

* **Under the pure premium** the budget is one linear row,
  ``E[P] <= bound``, and one solve gives the design.
* **Under a cost-of-capital load** the premium
  ``(1 - r) E[P] + r CVaR_q(P)`` is the largest of finitely many linear
  functions of the amounts: ``CVaR_q(P)`` is the largest of
  ``sum_o lambda_o w_o P_o`` over the tail weights ``lambda`` of the
  outcomes (:func:`~climada_petals.engine.parametric.riskmeasures.tail_weights`).
  Each round solves the QP with the cuts so far and adds the linear piece
  of the premium at its solution, a cut, until the premium meets the budget
  within ``feasibility_tolerance``, relative to the larger of the bound and
  ``E[L]``. The first round's cut is the piece at the lower bounds. A cut
  never removes a design whose premium meets the budget, so the rounds
  approach the optimum from below. The QP needs no auxiliary variables.

**Stated limits** (:class:`QPLimits`, from ``CalibrationOptions``), each
raising :class:`~climada_petals.engine.parametric.errors.CalibrationLimitError`
that names it:

* ``max_slsqp_amounts=1_000``: ``SLSQP``'s cost grows about with the cube
  of the amounts, about 1 s at 335 amounts and minutes at 1,000. Individual
  designs are far below it.
* ``max_cut_rounds=100``: the most solves under a load. A loaded design of
  a few hundred amounts may need more.
* ``time_limit``, if given: the wall time of the whole design problem, all
  rounds included, checked at each evaluation of the objective.
* ``SLSQP``'s own iteration limit, :data:`SLSQP_OPTIONS`.

``SLSQP`` ending without success raises: its iteration limit
(``CalibrationLimitError``); rows and bounds that HiGHS finds infeasible
(:class:`~climada_petals.engine.parametric.errors.InfeasibleCalibrationError`),
since ``SLSQP`` often reports those as a failed line search; or anything
else (:class:`~climada_petals.engine.parametric.errors.CalibrationError`).

**Uniqueness.** The minimiser over the identified amounts (not fixed, and
reached by a row with a weight) is unique exactly when the Hessian
``2 A_I^T W A_I`` restricted to them is positive definite. The route
reports it as unique when the smallest eigenvalue is above
:data:`UNIQUE_RTOL` times the largest.
"""

from __future__ import annotations

import time
from typing import Any, Final, NamedTuple

import numpy as np
import scipy
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import Bounds, linprog, minimize

from ._problem import DesignProblem, Scaled, Solution, SolverInfo, objective_norm
from .errors import CalibrationError, CalibrationLimitError, InfeasibleCalibrationError
from .objectives import MSE
from .pricing import CostOfCapitalLoad, Pricing
from .riskmeasures import tail_weights

__all__ = ["SLSQP_OPTIONS", "UNIQUE_RTOL", "QPLimits", "solve_mse"]

SLSQP_OPTIONS: Final[dict[str, Any]] = {"ftol": 1e-12, "maxiter": 3_000}
"""Options of :func:`scipy.optimize.minimize` with ``method="SLSQP"``.

``ftol`` bounds, among others, the change of the scaled objective, which is
at most 1 at the optimum, and the constraints' violation in a converged
solve. Its default, ``1e-6``, leaves designs up to about ``1e-6`` away from
the optimum; at ``1e-13``, solves with many nearly parallel cuts sometimes
end in a failed line search at a point within ``1e-11`` of their cut.
``maxiter`` is ``SLSQP``'s iteration limit per solve."""

UNIQUE_RTOL: Final[float] = 1e-10
"""The uniqueness test's relative tolerance on the Hessian's eigenvalues."""

_SLSQP_ITERATIONS: Final[int] = 9
_HIGHS_INFEASIBLE: Final[int] = 2


class QPLimits(NamedTuple):
    """The stated limits of the mean squared error's route.

    Attributes
    ----------
    max_slsqp_amounts : int, default 1_000
        The most amounts ``SLSQP`` solves for in one design problem.
    max_cut_rounds : int, default 100
        The most ``SLSQP`` solves under a cost-of-capital load.
    feasibility_tolerance : float, default 1e-7
        Within it, relative to the larger of the bound and ``E[L]``, the
        loaded premium meets the budget and the cuts stop.
    """

    max_slsqp_amounts: int = 1_000
    max_cut_rounds: int = 100
    feasibility_tolerance: float = 1e-7


class _Deadline(NamedTuple):
    """The end of a design problem's ``time_limit``, checked at each evaluation."""

    end: float | None
    seconds: float | None
    problem_id: str

    def check(self) -> None:
        """Raise :class:`CalibrationLimitError` once the time limit is reached."""
        if self.end is not None and time.perf_counter() > self.end:
            raise CalibrationLimitError(
                f"the mean squared error's route reached time_limit={self.seconds:g} s",
                problem_id=self.problem_id,
                status="time_limit",
            )


class _Quadratic:
    """The scaled mean squared error over the amounts that are not fixed.

    ``f = sum_o w_o (l_o - A_o theta)^2 / norm``, with ``A = G B`` the
    outcomes' basis and ``theta`` the amounts: the solver's variables where
    ``lower < upper``, and the fixed amounts (``lower == upper``) at their
    value.
    """

    def __init__(self, arr: Scaled, norm: float, deadline: _Deadline) -> None:
        self.outcome_basis = sp.csr_matrix(arr.group @ arr.basis)
        self.free = arr.lower < arr.upper
        self.base = np.where(self.free, 0.0, arr.lower)
        self.design = sp.csr_matrix(sp.csc_matrix(self.outcome_basis)[:, self.free])
        self.target = np.asarray(
            arr.group @ arr.loss - self.outcome_basis @ self.base, dtype=np.float64
        )
        self.weights = arr.weights / norm
        self.deadline = deadline

    def value_and_gradient(
        self, values: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return ``f`` and its gradient at the solver's variables ``values``.

        Raises
        ------
        CalibrationLimitError
            Once the time limit is reached.
        """
        self.deadline.check()
        residual = self.residual(values)
        weighted = self.weights * residual
        return float(weighted @ residual), -2.0 * (self.design.T @ weighted)

    def residual(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``l_o - A_o theta`` per outcome at the solver's variables."""
        return np.asarray(self.target - self.design @ values, dtype=np.float64)

    def amounts(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return every amount, the fixed ones at their value."""
        out = self.base.copy()
        out[self.free] = values
        return out

    def restrict(
        self, rows: Any, rhs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the rows ``rows @ theta <= rhs`` on the solver's variables, dense."""
        matrix = sp.csc_matrix(rows)
        fixed = np.asarray(matrix[:, ~self.free] @ self.base[~self.free]).reshape(-1)
        return matrix[:, self.free].toarray(), rhs - fixed


class _Budget:
    """The budget of the mean squared error: one row, or cutting planes.

    Under the pure premium the budget is one linear row. Under a
    cost-of-capital load the premium is convex and piecewise linear in the
    amounts: ``CVaR_q(P)`` is the largest of ``sum_o lambda_o w_o P_o`` over
    finitely many tail weights ``lambda``, so the premium is at least its
    linear piece at any point, with equality there. :meth:`row` returns
    that piece, a cut.
    """

    def __init__(self, problem: DesignProblem, arr: Scaled, pricing: Pricing) -> None:
        self.arr = arr
        self.measure = problem.measure
        self.pricing = pricing
        self.bound = None if problem.bound is None else problem.bound / problem.scale
        loaded = isinstance(pricing, CostOfCapitalLoad) and pricing.rate > 0.0
        self.load = pricing if loaded else None

    def row(
        self, quadratic: _Quadratic, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the budget's row at ``theta`` on the solver's variables, over ``E[L]``."""
        slope = self.arr.premium_row
        if self.load is not None:
            level = self.load.q
            paid = quadratic.outcome_basis @ theta
            tail = tail_weights(paid, self.arr.weights, level) / (1.0 - level)
            piece = np.asarray(quadratic.outcome_basis.T @ tail, dtype=np.float64)
            slope = (1.0 - self.load.rate) * slope + self.load.rate * piece
        unit = self.arr.budget_unit
        return quadratic.restrict(
            slope[None, :] / unit, np.array([float(self.bound or 0.0) / unit])
        )

    def met(self, theta: NDArray[np.float64], tolerance: float) -> bool:
        """Whether the premium at ``theta`` meets the budget within the tolerance.

        Relative to the larger of the bound and ``E[L]`` (1 if both are 0).
        Always true without a load, whose single row needs no cut.
        """
        if self.load is None or self.bound is None:
            return True
        premium = self.pricing.premium(self.arr.basis @ theta, self.measure)
        reference = max(self.bound, self.arr.expected_loss) or 1.0
        return premium <= self.bound + tolerance * reference


class _Rows(NamedTuple):
    """Linear rows ``matrix @ x <= rhs`` and bounds on the solver's variables."""

    matrix: NDArray[np.float64]
    rhs: NDArray[np.float64]
    bounds: Bounds

    def infeasible(self) -> bool:
        """Whether HiGHS finds no point that meets the rows and bounds."""
        result = linprog(
            np.zeros(self.matrix.shape[1]),
            A_ub=self.matrix if self.rhs.size else None,
            b_ub=self.rhs if self.rhs.size else None,
            bounds=np.column_stack([self.bounds.lb, self.bounds.ub]),
            method="highs",
        )
        return int(result.status) == _HIGHS_INFEASIBLE


def _slsqp_error(
    result: Any, rows: _Rows, problem_id: str, rounds: int
) -> CalibrationError:
    """Return the typed error of an ``SLSQP`` result that is not a success.

    ``SLSQP`` often reports infeasible rows as a failed line search, so the
    rows are checked with HiGHS first.
    """
    status, message = int(result.status), str(result.message)
    if status == _SLSQP_ITERATIONS:
        kind, what = (
            CalibrationLimitError,
            f"SLSQP reached its iteration limit, {SLSQP_OPTIONS['maxiter']}",
        )
    elif rows.infeasible():
        kind, what = InfeasibleCalibrationError, "the problem is infeasible"
    else:
        kind, what = CalibrationError, "SLSQP found no optimal solution"
    return kind(
        f"{what} (round {rounds}): {message}",
        problem_id=problem_id,
        status=str(status),
        solver_message=message,
    )


def _slsqp(quadratic: _Quadratic, rows: _Rows, start: NDArray[np.float64]) -> Any:
    """Minimise the quadratic under the rows with ``SLSQP``."""
    matrix, rhs = rows.matrix, rows.rhs
    constraints = []
    if rhs.size:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values: rhs - matrix @ values,
                "jac": lambda _values: -matrix,
            }
        )
    return minimize(
        quadratic.value_and_gradient,
        start,
        jac=True,
        method="SLSQP",
        bounds=rows.bounds,
        constraints=constraints,
        options=SLSQP_OPTIONS,
    )


def _rounds(
    quadratic: _Quadratic, budget: _Budget, arr: Scaled, limits: QPLimits
) -> tuple[NDArray[np.float64], list[Any]]:
    """Solve with ``SLSQP``, adding a cut per round under a load; return x and results.

    Raises
    ------
    CalibrationLimitError, InfeasibleCalibrationError, CalibrationError
        As :func:`solve_mse`.
    """
    problem_id = quadratic.deadline.problem_id
    values = arr.lower[quadratic.free].copy()
    if not quadratic.free.any():
        return values, []
    rows = [quadratic.restrict(arr.rows, arr.rhs)]
    bounds = Bounds(arr.lower[quadratic.free], arr.upper[quadratic.free])
    results: list[Any] = []
    for _ in range(limits.max_cut_rounds if budget.load is not None else 1):
        if budget.bound is not None:
            rows.append(budget.row(quadratic, quadratic.amounts(values)))
        stacked = _Rows(
            np.vstack([part[0] for part in rows]),
            np.concatenate([part[1] for part in rows]),
            bounds,
        )
        result = _slsqp(quadratic, stacked, values)
        results.append(result)
        if not result.success:
            raise _slsqp_error(result, stacked, problem_id, len(results))
        values = np.asarray(result.x, dtype=np.float64)
        if budget.met(quadratic.amounts(values), limits.feasibility_tolerance):
            return values, results
    raise CalibrationLimitError(
        f"the premium is above the budget after max_cut_rounds={limits.max_cut_rounds} "
        "rounds of cutting planes (CalibrationOptions)",
        problem_id=problem_id,
        status="max_cut_rounds",
    )


def _unique(arr: Scaled, outcome_basis: sp.csr_matrix) -> bool:
    """Whether the mean squared error's identified amounts are unique.

    The identified amounts are those that are not fixed and that a row with
    a weight reaches. The minimiser over them is unique exactly when the
    objective's Hessian restricted to them, ``2 A_I^T W A_I``, is positive
    definite: here, when its smallest eigenvalue is above
    :data:`UNIQUE_RTOL` times its largest. True when no amount is
    identified.
    """
    weighted = (arr.event_weight > 0.0).astype(np.float64)
    reached = np.asarray(abs(arr.basis).T @ weighted).reshape(-1) > 0.0
    identified = (arr.lower < arr.upper) & reached
    if not identified.any():
        return True
    design = sp.csc_matrix(outcome_basis)[:, identified]
    hessian = np.asarray((design.T @ sp.diags(arr.weights) @ design).toarray())
    values = np.linalg.eigvalsh(hessian)
    return bool(values[-1] > 0.0 and values[0] > UNIQUE_RTOL * values[-1])


def solve_mse(
    problem: DesignProblem,
    pricing: Pricing,
    *,
    time_limit: float | None,
    limits: QPLimits,
) -> Solution:
    """Solve the mean squared error design with ``SLSQP`` (module docstring).

    Raises
    ------
    CalibrationLimitError
        Above ``max_slsqp_amounts``, after ``max_cut_rounds``, at
        ``time_limit`` or at ``SLSQP``'s iteration limit.
    InfeasibleCalibrationError
        If ``SLSQP`` fails and HiGHS finds no point that meets the rows and
        bounds.
    CalibrationError
        If ``SLSQP`` fails for another reason.
    """
    start = time.perf_counter()
    arr = problem.scaled()
    norm = objective_norm(MSE(), arr, problem.measure)
    deadline = _Deadline(
        None if time_limit is None else start + time_limit,
        time_limit,
        problem.problem_id,
    )
    quadratic = _Quadratic(arr, norm, deadline)
    count = int(quadratic.free.sum())
    if count > limits.max_slsqp_amounts:
        raise CalibrationLimitError(
            f"SLSQP would solve for {count} amounts, above max_slsqp_amounts="
            f"{limits.max_slsqp_amounts} (CalibrationOptions): its time grows about "
            "with the cube of the amounts. Calibrate fewer units jointly, or raise "
            "the option",
            problem_id=problem.problem_id,
            status="max_slsqp_amounts",
        )
    values, results = _rounds(quadratic, _Budget(problem, arr, pricing), arr, limits)
    last = results[-1] if results else None
    info = SolverInfo(
        route="qp",
        solver="scipy-slsqp",
        version=scipy.__version__,
        method="slsqp",
        status="0" if last is None else str(last.status),
        message="no amount to solve for" if last is None else str(last.message),
        iterations=sum(int(item.nit) for item in results),
        seconds=time.perf_counter() - start,
        rounds=len(results),
    )
    value = float(quadratic.weights @ quadratic.residual(values) ** 2)
    return Solution(
        quadratic.amounts(values) * problem.scale,
        value * norm * problem.scale**2,
        info,
        unique=_unique(arr, quadratic.outcome_basis),
    )
