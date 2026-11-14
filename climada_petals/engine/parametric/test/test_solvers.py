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

Tests of the solver routes that the other tests do not reach.

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

from climada_petals.engine.parametric import (
    CalibrationError,
    CatalogueYears,
    CostOfCapitalLoad,
    MeanSquaredError,
    PurePremium,
    StepPayout,
)
from climada_petals.engine.parametric.solvers import (
    CLARABEL_OPTIONS,
    DesignData,
    solve_design,
)

PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=2)


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


def _data() -> DesignData:
    """Two units, three events in years 0, 0 and 3 of 10."""
    years = CatalogueYears.from_labels([1, 2, 3], [0, 0, 3], n_years=10)
    measure = years.measure([1, 2, 3], [0.1] * 3)
    return DesignData(
        problem_id="portfolio",
        index=np.array([[30.0, 10.0], [60.0, 50.0], [25.0, 65.0]]),
        loss=np.array([[4.0, 0.0], [10.0, 6.0], [1.0, 8.0]]),
        measure=measure,
    )


class TestQuadraticProgramRoute(unittest.TestCase):
    """The mean squared error route, with cvxpy replaced by :class:`_FakeCvxpy`."""

    def _solve(self, fake: _FakeCvxpy, pricing: Any, budget: float | None) -> Any:
        with mock.patch.dict(sys.modules, {"cvxpy": fake}):
            return solve_design(_data(), PAYOUT, MeanSquaredError(), pricing, budget)

    def test_the_result_is_rescaled_to_loss_units(self) -> None:
        data = _data()
        zero_cover = MeanSquaredError().evaluate(
            data.total_loss, np.zeros_like(data.total_loss), data.measure
        )
        for pricing, budget, n_variables in (
            (PurePremium(), None, 1),
            (PurePremium(), 0.5, 1),
            (CostOfCapitalLoad(rate=0.1, q=0.9), 0.5, 2),
        ):
            fake = _FakeCvxpy()
            solution = self._solve(fake, pricing, budget)
            with self.subTest(pricing=pricing, budget=budget):
                # At the zero cover the scaled objective is 1, so the value
                # in loss units is the zero cover's mean squared error.
                self.assertAlmostEqual(solution.objective_value, zero_cover)
                np.testing.assert_array_equal(solution.levels, np.zeros((2, 2)))
                self.assertEqual(solution.info.solver, "cvxpy-CLARABEL")
                self.assertEqual(solution.info.status, "optimal")
                self.assertEqual(solution.info.iterations, 7)
                self.assertEqual(
                    fake.solve_options, {"solver": "CLARABEL", **CLARABEL_OPTIONS}
                )
                self.assertEqual(len(fake.variables), n_variables)
                # Levels, caps and monotonicity, and the budget if there is one.
                (problem,) = fake.problems
                self.assertEqual(len(problem.constraints), 3 + (budget is not None))
                self.assertTrue(all(np.all(c) for c in problem.constraints))

    def test_a_solver_error_raises(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "CLARABEL failed"):
            self._solve(_FakeCvxpy(error=True), PurePremium(), None)

    def test_a_status_other_than_optimal_raises(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "infeasible") as caught:
            self._solve(_FakeCvxpy(status="infeasible"), PurePremium(), 0.5)
        self.assertEqual(caught.exception.status, "infeasible")
        self.assertEqual(caught.exception.problem_id, "portfolio")


if __name__ == "__main__":
    unittest.main()
