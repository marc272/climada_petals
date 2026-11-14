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

The mean squared error's route is tested here in full: its result, its
cutting planes against the exact formulation with the load's auxiliary
variables, fixed amounts, its stated limits, failed solves, and the
uniqueness test. ``test_calibrate.py`` checks it on the golden cases and in
the scopes' property.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, minimize

from climada_petals.engine.parametric import (
    MSE,
    Budget,
    CalibrationError,
    CalibrationLimitError,
    CalibrationOptions,
    CatalogueYears,
    CostOfCapitalLoad,
    ExpectedUncovered,
    Free,
    InfeasibleCalibrationError,
    PurePremium,
    StepPayout,
    UnboundedCalibrationError,
    WeightedCVaR,
    calibrate,
)
from climada_petals.engine.parametric.solvers import (
    DesignProblem,
    QPLimits,
    Solution,
    SolverInfo,
    check_amounts,
    check_limit,
    lp_method,
    solve_design,
    verify_solution,
)
from climada_petals.engine.parametric.test._synthetic import (
    array_sample,
    random_portfolio,
    template,
)

PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=2)
INDEX = np.array([[30.0, 10.0], [60.0, 50.0], [25.0, 65.0]])
LOSS = np.array([[4.0, 0.0], [10.0, 6.0], [1.0, 8.0]])
LOAD = CostOfCapitalLoad(rate=0.3, q=0.9)
MINIMIZE = "climada_petals.engine.parametric._qp.minimize"


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


def _one_unit(bound: float | None = None) -> DesignProblem:
    """The first unit of :func:`_problem` alone."""
    years = CatalogueYears.from_labels([1, 2, 3], [0, 0, 3], n_years=10)
    return DesignProblem(
        problem_id="U0",
        basis=sp.csr_matrix(PAYOUT.basis(INDEX[:, 0])),
        loss=LOSS[:, 0],
        constraints=[PAYOUT.amount_constraints(10.0)],
        measure=years.measure([1, 2, 3], [0.1] * 3),
        bound=bound,
    )


def _exact_loaded(problem: DesignProblem, load: CostOfCapitalLoad) -> Any:
    """Solve the loaded design with the load's auxiliary variables, no cuts.

    The variables are the amounts, ``eta`` and one ``x_o`` per outcome:
    ``x_o >= P_o - eta``, ``x, eta >= 0`` and
    ``(1 - r) E[P] + r (eta + sum_o w_o x_o / (1 - q)) <= bound``.
    """
    group = problem.measure.grouping.toarray()
    weights = np.asarray(problem.measure.weights)
    outcome_basis = group @ problem.basis.toarray()
    target = group @ problem.loss
    n_amounts, n_outcomes = outcome_basis.shape[1], weights.size
    rows, rhs = problem.rows
    tail = np.hstack([outcome_basis, -np.ones((n_outcomes, 1)), -np.eye(n_outcomes)])
    budget = np.concatenate(
        [
            (1.0 - load.rate) * weights @ outcome_basis,
            [load.rate],
            load.rate * weights / (1.0 - load.q),
        ]
    )
    monotone = np.hstack([rows.toarray(), np.zeros((rhs.size, 1 + n_outcomes))])
    constraints = LinearConstraint(
        np.vstack([tail, budget, monotone]),
        -np.inf,
        np.concatenate([np.zeros(n_outcomes), [problem.bound], rhs]),
    )

    def value(variables: np.ndarray) -> tuple[float, np.ndarray]:
        residual = target - outcome_basis @ variables[:n_amounts]
        gradient = np.zeros_like(variables)
        gradient[:n_amounts] = -2.0 * outcome_basis.T @ (weights * residual)
        return float(weights @ residual**2), gradient

    lower = np.concatenate([problem.lower, np.zeros(1 + n_outcomes)])
    upper = np.concatenate([problem.upper, np.full(1 + n_outcomes, np.inf)])
    return minimize(
        value,
        lower,
        jac=True,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=[constraints],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )


class TestMeanSquaredErrorRoute(unittest.TestCase):
    """The mean squared error with ``SLSQP``, and cutting planes under a load."""

    def test_the_result_is_in_loss_units_and_matches_numpy(self) -> None:
        for pricing, bound, rounds in (
            (PurePremium(), None, 1),
            (PurePremium(), 0.5, 1),
            (LOAD, 0.5, 2),
            (LOAD, None, 1),
        ):
            problem = _problem(bound=bound)
            solution = solve_design(problem, MSE(), pricing, method="highs")
            paid = problem.basis @ solution.amounts
            info = solution.info
            with self.subTest(pricing=pricing, bound=bound):
                self.assertEqual(
                    (info.route, info.solver, info.method, info.status),
                    ("qp", "scipy-slsqp", "slsqp", "0"),
                )
                self.assertEqual(
                    (info.version, info.rounds), (scipy.__version__, rounds)
                )
                self.assertGreater(info.iterations or 0, 0)
                value = MSE().evaluate(problem.loss, paid, problem.measure)
                self.assertAlmostEqual(solution.objective_value, value, delta=1e-9)
                check_amounts(problem, solution.amounts, 1e-9)
                if bound is not None:
                    premium = pricing.premium(paid, problem.measure)
                    self.assertLessEqual(premium, bound * (1 + 1e-7))
                    self.assertAlmostEqual(premium, bound, delta=1e-7)

    def test_a_hand_worked_design(self) -> None:
        # Year 0 pays t0 + t1 against a loss of 14, year 3 pays t0 against 1,
        # each with weight 0.1. The fit t = (1, 13) is above the limit 10, so
        # t1 = 10 and t0 = 2.5 splits the rest: MSE = 0.1 (1.5^2 + 1.5^2).
        solution = solve_design(_one_unit(), MSE(), PurePremium(), method="highs")
        np.testing.assert_allclose(solution.amounts, [2.5, 10.0], atol=1e-6)
        self.assertAlmostEqual(solution.objective_value, 0.45, delta=1e-9)
        # E[P] = 0.2 t0 + 0.1 t1 <= 0.6 binds. A level of t0 costs twice as
        # much as one of t1 and fits less, so t0 = 0 and t1 = 6: the KKT
        # multipliers are 16 for the budget and 1.4 for t0 >= 0.
        bounded = solve_design(_one_unit(0.6), MSE(), PurePremium(), method="highs")
        np.testing.assert_allclose(bounded.amounts, [0.0, 6.0], atol=1e-6)
        self.assertAlmostEqual(
            bounded.objective_value, 0.1 * (8.0**2 + 1.0), delta=1e-8
        )

    def test_the_cuts_reach_the_exact_loaded_design(self) -> None:
        for bound in (0.2, 0.5, 1.0):
            problem = _problem(bound=bound)
            solution = solve_design(problem, MSE(), LOAD, method="highs")
            exact = _exact_loaded(problem, LOAD)
            with self.subTest(bound=bound):
                self.assertTrue(exact.success, exact.message)
                self.assertAlmostEqual(solution.objective_value, exact.fun, delta=1e-8)
                self.assertGreaterEqual(solution.info.rounds or 0, 1)

    def test_fixed_amounts_are_constants(self) -> None:
        # The fixed levels alone cost 0.7 * 1.5 + 0.3 * 10 = 4.05 under the load.
        fixed = StepPayout.uniform(20.0, 70.0, 2, levels=[Free(), 5.0])
        problem = _problem(bound=5.0, payout=fixed)
        solution = solve_design(
            problem, MSE(), LOAD, method="highs", limits=QPLimits(max_slsqp_amounts=2)
        )
        self.assertEqual(solution.amounts[[1, 3]].tolist(), [5.0, 5.0])
        # Every amount fixed: nothing to solve for.
        both = StepPayout.uniform(20.0, 70.0, 2, levels=[3.0, 5.0])
        solution = solve_design(_problem(payout=both), MSE(), LOAD, method="highs")
        self.assertEqual(solution.amounts.tolist(), [3.0, 5.0, 3.0, 5.0])
        self.assertEqual((solution.info.rounds, solution.info.iterations), (0, 0))
        self.assertTrue(solution.unique)

    def test_the_stated_limits_raise(self) -> None:
        cases = [
            (
                {"limits": QPLimits(max_slsqp_amounts=3)},
                PurePremium(),
                "max_slsqp_amounts",
            ),
            ({"limits": QPLimits(max_cut_rounds=1)}, LOAD, "max_cut_rounds"),
            ({"time_limit": 1e-9}, LOAD, "time_limit"),
        ]
        for options, pricing, name in cases:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(CalibrationLimitError, f"{name}=") as caught,
            ):
                solve_design(
                    _problem(bound=0.2), MSE(), pricing, method="highs", **options
                )
            self.assertEqual(caught.exception.status, name)
        # Two rounds are enough there; fixed amounts don't count.
        solution = solve_design(
            _problem(bound=0.2),
            MSE(),
            LOAD,
            method="highs",
            limits=QPLimits(max_slsqp_amounts=4, max_cut_rounds=2),
        )
        self.assertEqual(solution.info.rounds, 2)

    def test_failed_solves_raise_typed_errors(self) -> None:
        cases = {
            9: (CalibrationLimitError, "iteration limit"),
            8: (CalibrationError, "no optimal solution"),
        }
        for status, (kind, match) in cases.items():
            failed = SimpleNamespace(
                success=False, status=status, message="from SLSQP", x=np.zeros(4)
            )
            with (
                self.subTest(status=status),
                mock.patch(MINIMIZE, return_value=failed),
                self.assertRaisesRegex(kind, match) as caught,
            ):
                solve_design(_problem(bound=0.5), MSE(), PurePremium(), method="highs")
            self.assertIs(type(caught.exception), kind)
            self.assertEqual(caught.exception.status, str(status))
            self.assertEqual(caught.exception.solver_message, "from SLSQP")
        # A fixed level whose premium is above a zero budget: HiGHS confirms.
        fixed = StepPayout.uniform(20.0, 70.0, 2, levels=[Free(), 5.0])
        for pricing in (PurePremium(), LOAD):
            with (
                self.subTest(pricing=pricing),
                self.assertRaisesRegex(InfeasibleCalibrationError, "infeasible"),
            ):
                solve_design(
                    _problem(bound=0.0, payout=fixed), MSE(), pricing, method="highs"
                )

    def test_uniqueness_from_the_hessian(self) -> None:
        never = StepPayout.uniform(attachment=100.0, exhaustion=120.0, n_steps=2)
        cases = {
            # Two outcomes with a payout fit two levels of one unit.
            "one unit": (_one_unit(0.5), True),
            # Three identified levels, two outcomes: a direction is free.
            "two units": (_problem(bound=0.5), False),
            # No amount is identified.
            "no payout": (_problem(bound=0.5, payout=never), True),
        }
        for name, (problem, unique) in cases.items():
            with self.subTest(name=name):
                solution = solve_design(problem, MSE(), PurePremium(), method="highs")
                self.assertIs(solution.unique, unique)
        lp = solve_design(
            _one_unit(0.5), ExpectedUncovered(), PurePremium(), method="highs"
        )
        self.assertIsNone(lp.unique)


class TestMeanSquaredErrorDesigns(unittest.TestCase):
    """The mean squared error through :func:`calibrate`."""

    def test_a_design_pays_nothing_below_the_attachment(self) -> None:
        values, loss = [[30.0], [60.0], [5.0]], [[4.0], [10.0], [0.0]]
        sample = array_sample(values, loss, [0, 1, 2], 10)[0]
        result = calibrate(
            template(sample, StepPayout.uniform(20.0, 70.0, 2)),
            sample,
            MSE(),
            scope="individual",
            budget=Budget(ratio=5.0),
        )
        payout = result.cover.payout["U0"]
        assert isinstance(payout, StepPayout)
        np.testing.assert_allclose(payout.amounts(), [4.0, 10.0], atol=1e-6)
        self.assertAlmostEqual(result.objective_value, 0.0, places=9)
        self.assertEqual(result.problems[0].solver.route, "qp")

    def test_the_route_needs_no_cvxpy(self) -> None:
        sample, _ = random_portfolio(0)
        with mock.patch.dict("sys.modules", {"cvxpy": None}):
            result = calibrate(
                template(sample, PAYOUT),
                sample,
                MSE(),
                scope="joint",
                pricing=LOAD,
                budget=Budget(ratio=0.5),
                options=CalibrationOptions(time_limit=60.0),
            )
        self.assertEqual(result.problems[0].solver.solver, "scipy-slsqp")

    def test_the_record(self) -> None:
        sample, _ = random_portfolio(1)
        common = {"budget": Budget(ratio=0.5), "pricing": LOAD}
        result = calibrate(
            template(sample, PAYOUT), sample, MSE(), scope="individual", **common
        )
        summary = result.summary()
        self.assertEqual(
            summary["unique"].tolist(), [p.unique for p in result.problems]
        )
        self.assertEqual(result.unique, all(summary["unique"]))
        self.assertTrue((summary["rounds"] >= 1).all())
        self.assertEqual(set(summary["method"]), {"slsqp"})
        linear = calibrate(
            template(sample, PAYOUT),
            sample,
            ExpectedUncovered(),
            scope="joint",
            **common,
        )
        self.assertIsNone(linear.unique)
        self.assertIsNone(linear.summary()["rounds"].iloc[0])

    def test_the_options_reach_the_route(self) -> None:
        sample, _ = random_portfolio(2)
        cover = template(sample, PAYOUT)
        cases = {
            "max_slsqp_amounts": CalibrationOptions(max_slsqp_amounts=4),
            "max_cut_rounds": CalibrationOptions(max_cut_rounds=1),
        }
        for name, options in cases.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(CalibrationLimitError, name),
            ):
                calibrate(
                    cover,
                    sample,
                    MSE(),
                    scope="joint",
                    pricing=LOAD,
                    budget=Budget(ratio=0.3),
                    options=options,
                )


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

    def test_payouts_above_their_limit_raise(self) -> None:
        problem = _problem()
        paid = np.array([[4.0, 0.0], [10.0, 8.0 + 1e-9], [1.0, 8.0]])
        check_limit(problem, paid, [10.0, 8.0], 1e-7)
        check_limit(problem, paid * 2.0, [None, None], 1e-7)
        with self.assertRaisesRegex(CalibrationError, "above its limit 8"):
            check_limit(problem, paid + 0.01, [None, 8.0], 1e-7)

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
