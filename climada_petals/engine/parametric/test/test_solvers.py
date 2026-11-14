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

Tests of the solver routes that the calibration's tests do not reach.

The mean squared error is solved with cvxpy, an optional dependency. Where
cvxpy is installed, ``test_calibrate.py`` solves it for real. The tests here
replace cvxpy with :class:`_FakeCvxpy`, a stand-in that evaluates the
module's cvxpy expressions in numpy at the zero cover and reports a status
chosen by the test. They check the module's side of the route: the
problem it builds, the solver it asks for, how it rescales the result and
how it reports a failed solve. They say nothing about the optimum.
"""

from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import scipy.sparse as sp

from climada_petals.engine.parametric import (
    MSE,
    CalibrationError,
    CalibrationLimitError,
    CatalogueYears,
    CostOfCapitalLoad,
    ExpectedUncovered,
    InfeasibleCalibrationError,
    PurePremium,
    StepPayout,
    UnboundedCalibrationError,
    WeightedCVaR,
)
from climada_petals.engine.parametric.solvers import (
    CLARABEL_OPTIONS,
    DesignProblem,
    Solution,
    SolverInfo,
    check_amounts,
    lp_method,
    solve_design,
    verify_solution,
)

PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=2)
INDEX = np.array([[30.0, 10.0], [60.0, 50.0], [25.0, 65.0]])
LOSS = np.array([[4.0, 0.0], [10.0, 6.0], [1.0, 8.0]])


class _FakeVariable(np.ndarray):
    """A cvxpy ``Variable`` stand-in: zeros, with a ``value`` set by a solve."""

    value: Any = None


class _FakeProblem:
    """A cvxpy ``Problem`` stand-in that evaluates its objective at zero."""

    def __init__(self, fake: _FakeCvxpy, objective: Any, constraints: list) -> None:
        self.fake = fake
        self.objective = objective
        self.constraints = constraints
        self.status: str | None = None
        self.value: float | None = None
        self.solver_stats: Any = None

    def solve(self, **options: Any) -> None:
        """Record the options, then fail or set the status chosen by the test."""
        self.fake.solve_options = options
        if self.fake.error:
            raise self.fake.SolverError("numerical trouble")
        self.status = self.fake.status
        self.value = float(self.objective)
        self.solver_stats = SimpleNamespace(num_iters=7)
        for var in self.fake.variables:
            var.value = np.asarray(var).copy() if self.status == "optimal" else None
        self.fake.problems.append(self)


class _FakeCvxpy(ModuleType):
    """The part of cvxpy that the mean squared error route uses."""

    class SolverError(Exception):
        """Raised by a solve when the test asks for an error."""

    def __init__(self, *, status: str = "optimal", error: bool = False) -> None:
        super().__init__("cvxpy")
        self.__version__ = "0.0-fake"
        self.status = status
        self.error = error
        self.variables: list[_FakeVariable] = []
        self.problems: list[_FakeProblem] = []
        self.solve_options: dict[str, Any] = {}

    def Variable(self, size: int | None = None) -> _FakeVariable:
        """Return zeros of the variable's shape."""
        var = np.zeros(() if size is None else size).view(_FakeVariable)
        self.variables.append(var)
        return var

    @staticmethod
    def pos(expr: Any) -> Any:
        """The positive part."""
        return np.maximum(expr, 0.0)

    @staticmethod
    def square(expr: Any) -> Any:
        """The square."""
        return np.square(expr)

    @staticmethod
    def Minimize(expr: Any) -> Any:
        """The objective itself."""
        return expr

    def Problem(self, objective: Any, constraints: list) -> _FakeProblem:
        """A problem that evaluates its objective at the zero cover."""
        return _FakeProblem(self, objective, constraints)


def _problem(
    *,
    bound: float | None = None,
    payout: StepPayout = PAYOUT,
    loss: np.ndarray = LOSS,
    limits: tuple[float | None, float | None] = (10.0, 8.0),
) -> DesignProblem:
    """Two units, three events in years 0, 0 and 3 of 10, one joint problem."""
    years = CatalogueYears.from_labels([1, 2, 3], [0, 0, 3], n_years=10)
    measure = years.measure([1, 2, 3], [0.1] * 3)
    return DesignProblem(
        problem_id="joint",
        basis=sp.hstack([payout.basis(INDEX[:, j]) for j in range(2)], format="csr"),
        loss=loss.sum(axis=1),
        constraints=[payout.amount_constraints(limit) for limit in limits],
        measure=measure,
        bound=bound,
    )


class TestQuadraticProgramRoute(unittest.TestCase):
    """The mean squared error route, with cvxpy replaced by :class:`_FakeCvxpy`."""

    def _solve(self, fake: _FakeCvxpy, pricing: Any, problem: DesignProblem) -> Any:
        with mock.patch.dict(sys.modules, {"cvxpy": fake}):
            return solve_design(problem, MSE(), pricing, method="highs")

    def test_the_result_is_rescaled_to_loss_units(self) -> None:
        for pricing, bound, limits, n_constraints, n_variables in (
            (PurePremium(), None, (10.0, 8.0), 3, 1),
            (PurePremium(), 0.5, (10.0, 8.0), 4, 1),
            (PurePremium(), None, (None, None), 2, 1),
            (CostOfCapitalLoad(rate=0.1, q=0.9), 0.5, (10.0, 8.0), 5, 2),
        ):
            problem = _problem(bound=bound, limits=limits)
            zero_cover = MSE().evaluate(
                problem.loss, np.zeros_like(problem.loss), problem.measure
            )
            fake = _FakeCvxpy()
            solution = self._solve(fake, pricing, problem)
            with self.subTest(pricing=pricing, bound=bound, limits=limits):
                # At the zero cover the scaled objective is 1, so the value
                # in loss units is the zero cover's mean squared error.
                self.assertAlmostEqual(solution.objective_value, zero_cover)
                np.testing.assert_array_equal(solution.amounts, np.zeros(4))
                self.assertEqual(
                    (solution.info.route, solution.info.solver, solution.info.method),
                    ("qp", "cvxpy-CLARABEL", None),
                )
                self.assertEqual(solution.info.version, "0.0-fake")
                self.assertEqual(solution.info.status, "optimal")
                self.assertEqual(solution.info.iterations, 7)
                self.assertEqual(
                    fake.solve_options, {"solver": "CLARABEL", **CLARABEL_OPTIONS}
                )
                self.assertEqual(len(fake.variables), n_variables)
                # Bounds, finite upper bounds, monotonicity, the budget and the
                # load's level, each where it applies.
                (built,) = fake.problems
                self.assertEqual(len(built.constraints), n_constraints)
                self.assertTrue(all(np.all(c) for c in built.constraints))

    def test_a_solver_error_raises(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "CLARABEL failed"):
            self._solve(_FakeCvxpy(error=True), PurePremium(), _problem())

    def test_a_status_other_than_optimal_raises(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "infeasible") as caught:
            self._solve(_FakeCvxpy(status="infeasible"), PurePremium(), _problem())
        self.assertEqual(caught.exception.status, "infeasible")
        self.assertEqual(caught.exception.problem_id, "joint")
        self.assertEqual(caught.exception.solver_message, "infeasible")


class TestLinearProgramRoute(unittest.TestCase):
    """The method rule, the statuses and the time limit of the linear programs."""

    def test_the_method_rule(self) -> None:
        self.assertEqual(lp_method("auto", 1), "highs")
        self.assertEqual(lp_method("auto", 2), "highs-ipm")
        self.assertEqual(lp_method("highs", 67), "highs")
        self.assertEqual(lp_method("highs-ipm", 1), "highs-ipm")

    def test_both_methods_give_one_objective(self) -> None:
        problem = _problem(bound=0.5)
        objective = WeightedCVaR(alpha=0.9, beta=0.8)
        values = []
        for method in ("highs", "highs-ipm"):
            solution = solve_design(problem, objective, PurePremium(), method=method)
            values.append(solution.objective_value)
            self.assertEqual(solution.info.method, method)
            self.assertEqual(solution.info.route, "lp")
        self.assertAlmostEqual(values[0], values[1], places=9)

    def test_statuses_raise_typed_errors(self) -> None:
        cases = {
            1: (CalibrationLimitError, "reached a limit"),
            2: (InfeasibleCalibrationError, "infeasible"),
            3: (UnboundedCalibrationError, "unbounded"),
            4: (CalibrationError, "no optimal solution"),
        }
        for status, (kind, match) in cases.items():
            failed = SimpleNamespace(
                status=status, message="from HiGHS", x=None, fun=None, nit=0
            )
            with (
                self.subTest(status=status),
                mock.patch(
                    "climada_petals.engine.parametric.solvers.linprog",
                    return_value=failed,
                ),
                self.assertRaisesRegex(kind, match) as caught,
            ):
                solve_design(
                    _problem(), ExpectedUncovered(), PurePremium(), method="highs"
                )
            self.assertEqual(caught.exception.status, str(status))
            self.assertEqual(caught.exception.solver_message, "from HiGHS")
            self.assertIs(type(caught.exception), kind)

    def test_the_time_limit_passes_through(self) -> None:
        with mock.patch(
            "climada_petals.engine.parametric.solvers.linprog",
            wraps=__import__("scipy.optimize", fromlist=["linprog"]).linprog,
        ) as spy:
            solve_design(
                _problem(),
                ExpectedUncovered(),
                PurePremium(),
                method="highs-ipm",
                time_limit=30.0,
            )
        options = spy.call_args.kwargs["options"]
        self.assertEqual(options["time_limit"], 30.0)
        self.assertEqual(spy.call_args.kwargs["method"], "highs-ipm")

    def test_problems_without_loss_or_payout(self) -> None:
        # No loss: the tail has no outcome, and the objective is 0.
        problem = _problem(bound=0.5, loss=np.zeros_like(LOSS))
        solution = solve_design(
            problem, WeightedCVaR(alpha=0.9, beta=0.8), PurePremium(), method="highs"
        )
        self.assertEqual(solution.objective_value, 0.0)
        # A payout no event reaches: the load's tail has no outcome.
        never = StepPayout.uniform(attachment=100.0, exhaustion=120.0, n_steps=2)
        problem = _problem(bound=0.5, payout=never)
        solution = solve_design(
            problem,
            ExpectedUncovered(),
            CostOfCapitalLoad(rate=0.1, q=0.9),
            method="highs",
        )
        self.assertAlmostEqual(
            solution.objective_value, problem.measure.expectation(problem.loss)
        )


class TestChecks(unittest.TestCase):
    """The checks after a solve."""

    def _solution(self, amounts: np.ndarray, value: float) -> Solution:
        info = SolverInfo("lp", "scipy-highs", "1", "highs", "0", "", 1, 0.0)
        return Solution(amounts, value, info)

    def test_amounts_beyond_their_constraints_raise(self) -> None:
        problem = _problem()
        check_amounts(problem, np.array([1.0, 2.0, 0.0, 8.0]), 1e-7)
        # Rounding below a bound, or across the monotone order, is fine.
        check_amounts(problem, np.array([-1e-12, 2.0, 1.0, 1.0 - 1e-12]), 1e-7)
        for amounts in ([-1e-3, 2.0, 0.0, 0.0], [3.0, 2.0, 0.0, 0.0], [0, 0, 0, 9.0]):
            with (
                self.subTest(amounts=amounts),
                self.assertRaisesRegex(CalibrationError, "break their bounds or rows"),
            ):
                check_amounts(problem, np.array(amounts), 1e-7)

    def test_the_objective_and_the_budget_are_recomputed(self) -> None:
        problem = _problem(bound=0.5)
        paid = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 0.0]])
        objective = ExpectedUncovered()
        value = objective.evaluate(problem.loss, paid.sum(axis=1), problem.measure)
        found = verify_solution(
            problem,
            paid,
            objective=objective,
            pricing=PurePremium(),
            solution=self._solution(np.zeros(4), value * (1 + 1e-9)),
            tolerance=1e-7,
        )
        self.assertEqual(found, (value, 0.5))
        with self.assertRaisesRegex(CalibrationError, "the solver's"):
            verify_solution(
                problem,
                paid,
                objective=objective,
                pricing=PurePremium(),
                solution=self._solution(np.zeros(4), value * 1.01),
                tolerance=1e-7,
            )
        with self.assertRaisesRegex(CalibrationError, "above the budget"):
            verify_solution(
                problem,
                paid * 1.01,
                objective=objective,
                pricing=PurePremium(),
                solution=self._solution(
                    np.zeros(4),
                    objective.evaluate(
                        problem.loss, 1.01 * paid.sum(axis=1), problem.measure
                    ),
                ),
                tolerance=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
