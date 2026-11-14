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

Tests of the calibration: golden and hand-worked designs, the LP against
numpy, the scopes, the template route and the result record.

The golden cases are read with :mod:`._golden`.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import pickle
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from climada_petals.engine.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CalibrationError,
    CalibrationLimitError,
    CalibrationOptions,
    CalibrationResult,
    CostOfCapitalLoad,
    EventBasis,
    ExpectedUncovered,
    Free,
    IndexDefinition,
    Limit,
    LinearInAmounts,
    LinearPayout,
    OutcomeSample,
    ParametricCover,
    ParametricIndex,
    ParametricInputError,
    PiecewiseLinearPayout,
    PurePremium,
    StepPayout,
    WeightedCVaR,
    calibrate,
    var_cvar,
)
from climada_petals.engine.parametric.solvers import Solution
from climada_petals.engine.parametric.test._golden import (
    golden_budget,
    golden_cases,
    golden_objective,
    golden_pricing,
    golden_sample,
)
from climada_petals.engine.parametric.test._synthetic import (
    array_index,
    array_sample,
    random_portfolio,
    template,
)
from climada_petals.util.config import LOGGER

PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=4)
RTOL = 1e-7
SOLVE_DESIGN = "climada_petals.engine.parametric.solvers.solve_design"


def _hand_example(drop_quiet_event: bool = False) -> OutcomeSample:
    """One unit, 10 years: (index 30, loss 4), (60, 10) and (5, 0) in years 0-2."""
    values = [[30.0], [60.0], [5.0]]
    loss = [[4.0], [10.0], [0.0]]
    years = [0, 1, 2]
    if drop_quiet_event:
        values, loss, years = values[:2], loss[:2], years[:2]
    return array_sample(values, loss, years, 10)[0]


def _levels(result: CalibrationResult, unit: Any = "U0") -> list[float]:
    """Return a unit's amounts in the resolved cover."""
    payout = result.cover.payout[unit]
    assert isinstance(payout, LinearInAmounts)
    return payout.amounts().tolist()


@dataclass(frozen=True)
class _Double(LinearInAmounts):
    """A family whose basis is 2 above a trigger: it pays twice its amount."""

    kind: ClassVar[str] = "test_double"

    trigger: float
    amount: Any
    tail: str = "upper"

    def amount_names(self) -> tuple[str, ...]:
        """Return ``("amount",)``."""
        return ("amount",)

    def basis(self, index: ArrayLike) -> sp.csr_matrix:
        """Return 2 where the index reaches the trigger."""
        values = np.asarray(index, dtype=float)
        return sp.csr_matrix(2.0 * (values >= self.trigger)[:, None])


class TestGoldenDesigns(unittest.TestCase):
    """The golden design cases, within their stated tolerance."""

    def _check(self, case: dict[str, Any]) -> None:
        """Design a case and compare it with the case's values."""
        sample, scale = golden_sample(case)
        objective = golden_objective(case["objective"])
        result = calibrate(
            ParametricCover.from_dict(case["cover"]),
            sample,
            objective,
            scope=case["scope"],
            pricing=golden_pricing(case["pricing"]),
            budget=golden_budget(case["budget"]),
        )
        expected = case["expected"]
        squared = scale**2 if isinstance(objective, MSE) else scale
        if case["scope"] == "joint":
            pairs = [(result.problems[0], expected["objective_value"])]
            premiums = [(result.problems[0].premium, expected["premium"])]
        else:
            pairs = [
                (problem, expected["objective_value"][problem.unit_id[0]])
                for problem in result.problems
            ]
            premiums = [
                (problem.premium, expected["premium"][problem.unit_id[0]])
                for problem in result.problems
            ]
        for problem, value in pairs:
            self.assertLessEqual(
                abs(problem.objective_value - value),
                max(1e-6 * abs(value), 1e-9 * squared),
                f"objective of {problem.problem_id}",
            )
        self.assertEqual(
            {unit: set(names) for unit, names in result.unidentified.items()},
            {unit: set(names) for unit, names in expected["unidentified"].items()},
        )
        if isinstance(objective, MSE):
            # The Hessian test.
            self.assertEqual(result.unique, expected["unique"])
        else:
            # Without ranges=True the LP route doesn't determine it.
            self.assertIsNone(result.unique)
        if case["compare_parameters"] != "exact":
            return
        # Every amount, the filled ones included: the case fills them as
        # fill="lowest" does.
        for unit, parameters in expected["parameters"].items():
            found = result.cover.payout[unit].parameters()
            for name, value in parameters.items():
                self.assertLessEqual(abs(found[name] - value), 1e-6 * scale, name)
        for got, value in premiums:
            self.assertLessEqual(abs(got - value), max(1e-6 * abs(value), 1e-9 * scale))

    def _run(self, kinds: tuple[str, ...]) -> int:
        """Check the design cases whose objective is of ``kinds``; return the count."""
        cases = [
            case
            for case in golden_cases("design")
            if case["objective"]["kind"] in kinds
        ]
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self._check(case)
        return len(cases)

    def test_expected_uncovered_and_weighted_cvar(self) -> None:
        # Both scopes, both bases, both pricings, and the closed forms.
        self.assertEqual(self._run(("expected_uncovered", "weighted_cvar")), 21)

    def test_mean_squared_error(self) -> None:
        # Both scopes, both bases and both pricings from the core, and three
        # closed forms, one under a load; parameters and premiums as well.
        self.assertEqual(self._run(("mse",)), 11)

    def _check_ranges(self, scope: str) -> int:
        """Range the LP cases of one scope; compare ``unique`` and the parameters.

        Every identified amount of the case lies within the module's wide
        range, widened by 1e-6 of the scale: the rule of the two
        ``"within_ranges"`` cases, applied to the ``"exact"`` ones too.
        """
        cases = [
            case
            for case in golden_cases("design")
            if case["objective"]["kind"] != "mse" and case["scope"] == scope
        ]
        for case in cases:
            sample, scale = golden_sample(case)
            result = calibrate(
                ParametricCover.from_dict(case["cover"]),
                sample,
                golden_objective(case["objective"]),
                scope=scope,
                pricing=golden_pricing(case["pricing"]),
                budget=golden_budget(case["budget"]),
                options=CalibrationOptions(ranges=True),
            )
            expected = case["expected"]
            with self.subTest(case=case["case_id"]):
                self.assertIs(result.unique, expected["unique"])
                for unit, parameters in expected["parameters"].items():
                    filled = set(expected["unidentified"].get(unit, ()))
                    for name, value in parameters.items():
                        if name in filled:
                            continue
                        low, high = result.ranges[unit][name]
                        self.assertLessEqual(low - 1e-6 * scale, value, name)
                        self.assertLessEqual(value, high + 1e-6 * scale, name)
        return len(cases)

    def test_ranges_and_uniqueness_on_the_lp_route_individual(self) -> None:
        # step-eu-pure-annual-individual-s1 is "within_ranges", not unique.
        self.assertEqual(self._check_ranges("individual"), 12)

    def test_ranges_and_uniqueness_on_the_lp_route_joint(self) -> None:
        # step-eu-pure-annual-joint-s1 is "within_ranges", not unique.
        self.assertEqual(self._check_ranges("joint"), 9)


class TestHandWorkedDesigns(unittest.TestCase):
    """Designs small enough to solve by hand, one per family."""

    def test_a_step_payout(self) -> None:
        sample = _hand_example()
        cover = template(sample, StepPayout.uniform(20.0, 70.0, 2))
        # E[L] = 1.4, so the premium is at most 0.7 and E[U] at least 1.4 - 0.7.
        result = calibrate(
            cover,
            sample,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=0.5),
        )
        (problem,) = result.problems
        self.assertAlmostEqual(problem.objective_value, 0.7, places=9)
        self.assertAlmostEqual(problem.premium, 0.7, places=9)
        self.assertAlmostEqual(float(problem.budget_bound or 0.0), 0.7, places=12)
        self.assertAlmostEqual(float(problem.budget_use or 0.0), 1.0, places=9)
        # The worst year alone: max(4 - t0, 10 - t1) with t0 + t1 = 7 is
        # smallest at t0 = 0.5, t1 = 6.5.
        result = calibrate(
            cover,
            sample,
            WeightedCVaR(alpha=1.0, beta=0.9),
            scope="individual",
            budget=Budget(ratio=0.5),
        )
        self.assertAlmostEqual(result.objective_value, 3.5, places=9)
        np.testing.assert_allclose(_levels(result), [0.5, 6.5], atol=1e-9)
        self.assertEqual(result.problems[0].degenerate_tail, "single_outcome")

    def test_a_linear_payout(self) -> None:
        # Events (40, 5) and (60, 10): shares 0.5 and 1 of the limit. The
        # premium 0.15 l is at most 0.5 E[L] = 0.75, so l = 5.
        sample, _ = array_sample(
            [[40.0], [60.0], [10.0]], [[5.0], [10.0], [0.0]], [0, 1, 2], 10
        )
        cover = template(sample, LinearPayout(20.0, 60.0, Free()))
        result = calibrate(
            cover,
            sample,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=0.5),
        )
        np.testing.assert_allclose(_levels(result), [5.0], rtol=1e-9)
        self.assertAlmostEqual(result.objective_value, 0.1 * 2.5 + 0.1 * 5.0, places=9)
        self.assertAlmostEqual(result.premium, 0.75, places=9)
        self.assertEqual(result.cover.limit, Limit({"U0": 10.0}))

    def test_a_piecewise_linear_payout(self) -> None:
        # Events at 30, 50 and 70 with losses 2, 6 and 9; the worst two years
        # are 50 and 70. Per unit of premium, y2 cuts their CVaR most, so the
        # whole budget 0.85 goes to it: y2 = 0.85 / 0.15.
        sample, _ = array_sample(
            [[30.0], [50.0], [70.0], [10.0]],
            [[2.0], [6.0], [9.0], [0.0]],
            [0, 1, 2, 3],
            10,
        )
        cover = template(
            sample, PiecewiseLinearPayout([20.0, 40.0, 60.0], [Free(), Free(), Free()])
        )
        result = calibrate(
            cover,
            sample,
            WeightedCVaR(alpha=1.0, beta=0.8),
            scope="individual",
            budget=Budget(ratio=0.5),
        )
        top = 0.85 / 0.15
        np.testing.assert_allclose(_levels(result), [0.0, 0.0, top], atol=1e-9)
        self.assertAlmostEqual(
            result.objective_value, ((9.0 - top) + (6.0 - top / 2)) / 2, places=9
        )

    def test_a_binary_payout(self) -> None:
        # Two events reach the trigger: the premium 0.2 A is at most
        # 0.5 E[L] = 1.15, so A = 5.75.
        sample, _ = array_sample(
            [[40.0], [60.0], [80.0]], [[3.0], [8.0], [12.0]], [0, 1, 2], 10
        )
        cover = template(sample, BinaryPayout(50.0, Free()))
        result = calibrate(
            cover,
            sample,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=0.5),
        )
        np.testing.assert_allclose(_levels(result), [5.75], rtol=1e-9)
        self.assertAlmostEqual(
            result.objective_value, 0.1 * (3.0 + 2.25 + 6.25), places=9
        )

    def test_the_event_basis_above_one_per_year(self) -> None:
        # Frequencies 0.8, 0.7 and 0.5: the budget 0.25 E[L] = 3.5 buys
        # A = 3.5 / 1.2, and the worst 0.5 of frequency is the largest event.
        index, impact, units = array_index(
            [[40.0], [60.0], [80.0]], [[3.0], [8.0], [12.0]], frequency=[0.8, 0.7, 0.5]
        )
        sample = OutcomeSample.from_impact(
            index, impact, units=units, basis=EventBasis()
        )
        self.assertAlmostEqual(sample.measure.total_weight, 2.0)
        result = calibrate(
            template(sample, BinaryPayout(50.0, Free())),
            sample,
            WeightedCVaR(alpha=1.0, beta=0.5),
            scope="individual",
            budget=Budget(ratio=0.25),
        )
        amount = 3.5 / 1.2
        np.testing.assert_allclose(_levels(result), [amount], rtol=1e-9)
        self.assertAlmostEqual(result.objective_value, 12.0 - amount, places=9)
        self.assertEqual(result.problems[0].degenerate_tail, "single_outcome")

    def test_loss_free_events_change_nothing_at_twelve_per_year(self) -> None:
        # ADR 0006's spot check, small: frequencies summing to 12 per year.
        rng = np.random.default_rng(12)
        values = rng.uniform(25.0, 80.0, size=(40, 1))
        loss = (values - 25.0) ** 2
        quiet = np.full((30, 1), 5.0)
        results = []
        for index_values, losses in (
            (values, loss),
            (np.vstack([values, quiet]), None),
        ):
            losses = np.vstack([loss, np.zeros((30, 1))]) if losses is None else losses
            index, impact, units = array_index(
                index_values, losses, frequency=np.full(index_values.shape[0], 0.3)
            )
            sample = OutcomeSample.from_impact(
                index, impact, units=units, basis=EventBasis()
            )
            results.append(
                calibrate(
                    template(sample, PAYOUT),
                    sample,
                    WeightedCVaR(alpha=0.9, beta=0.99),
                    scope="individual",
                    budget=Budget(ratio=0.8),
                )
            )
        self.assertAlmostEqual(
            results[0].objective_value, results[1].objective_value, delta=1e-9
        )
        np.testing.assert_allclose(_levels(results[0]), _levels(results[1]), atol=1e-9)


class TestLinearProgram(unittest.TestCase):
    """The LP's optimum is feasible and equals the objective evaluated in numpy."""

    def test_optimum_is_feasible_and_matches_numpy(self) -> None:
        cases = itertools.product(
            (0, 1),
            ("individual", "joint"),
            ("annual", "event"),
            (ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9)),
            (PurePremium(), CostOfCapitalLoad(rate=0.1, q=0.9)),
        )
        for seed, scope, basis, objective, pricing in cases:
            sample, _ = random_portfolio(seed, basis=basis)
            result = calibrate(
                template(sample, PAYOUT),
                sample,
                objective,
                scope=scope,
                pricing=pricing,
                budget=Budget(ratio=0.6),
            )
            measure = sample.measure
            loss = sample.loss.toarray()
            paid = result.cover.payout_matrix(sample.index)
            caps = loss.max(axis=0)
            levels = np.array([_levels(result, unit) for unit in sample.unit_id])
            with self.subTest(seed=seed, scope=scope, basis=basis, objective=objective):
                self.assertTrue(np.all(np.diff(levels, axis=1) >= 0.0))
                self.assertTrue(np.all(levels >= 0.0))
                self.assertTrue(np.all(levels <= caps[:, None]))
                units = (
                    [[j] for j in range(3)] if scope == "individual" else [[0, 1, 2]]
                )
                for problem, cols in zip(result.problems, units, strict=True):
                    total_loss = loss[:, cols].sum(axis=1)
                    total_paid = paid[:, cols].sum(axis=1)
                    self.assertEqual(
                        problem.objective_value,
                        objective.evaluate(total_loss, total_paid, measure),
                    )
                    self.assertEqual(
                        problem.premium, pricing.premium(total_paid, measure)
                    )
                    self.assertLessEqual(
                        problem.premium,
                        float(problem.budget_bound or 0.0) * (1 + RTOL) + 1e-9,
                    )

    def test_alpha_zero_is_expected_uncovered(self) -> None:
        sample, _ = random_portfolio(2)
        common = {"scope": "joint", "budget": Budget(ratio=0.5)}
        cover = template(sample, PAYOUT)
        mean = calibrate(cover, sample, ExpectedUncovered(), **common)
        mixed = calibrate(cover, sample, WeightedCVaR(0.0, 0.95), **common)
        np.testing.assert_allclose(
            mixed.objective_value, mean.objective_value, rtol=1e-9
        )

    def test_a_tail_that_holds_every_loss_gives_the_expected_uncovered_design(
        self,
    ) -> None:
        sample, _ = random_portfolio(3)
        common = {"scope": "joint", "budget": Budget(ratio=0.5)}
        cover = template(sample, PAYOUT)
        mean = calibrate(cover, sample, ExpectedUncovered(), **common)
        zero = 1.0 - mean.problems[0].loss_weight
        self.assertGreater(zero, 0.1)
        alpha, beta = 0.9, zero / 2
        with self.assertLogs(LOGGER, "WARNING") as logs:
            tail = calibrate(cover, sample, WeightedCVaR(alpha, beta), **common)
        problem = tail.problems[0]
        self.assertEqual(problem.degenerate_tail, "expected_uncovered")
        self.assertIn("minimises E[U] only", "\n".join(logs.output))
        np.testing.assert_allclose(
            problem.objective_value,
            (alpha / (1 - beta) + 1 - alpha) * mean.objective_value,
            rtol=1e-7,
        )

    def test_a_tail_within_one_outcome_is_recorded_and_logged(self) -> None:
        # 60 years: one outcome has probability 1/60, more than 1 - beta.
        sample, _ = random_portfolio(2)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            result = calibrate(
                template(sample, PAYOUT),
                sample,
                WeightedCVaR(alpha=0.9, beta=0.99),
                scope="joint",
                budget=Budget(ratio=0.5),
            )
        self.assertEqual(result.degenerate_tail, {"joint": "single_outcome"})
        self.assertEqual(
            result.summary()["degenerate_tail"].tolist(), ["single_outcome"]
        )
        self.assertIn("single worst outcome", "\n".join(logs.output))

    def test_the_method_rule_and_the_time_limit(self) -> None:
        sample, _ = random_portfolio(0)
        cover = template(sample, PAYOUT)
        common = {"budget": Budget(ratio=0.5)}
        methods = {
            ("individual", "auto"): "highs",
            ("joint", "auto"): "highs-ipm",
            ("joint", "highs"): "highs",
        }
        for (scope, option), method in methods.items():
            result = calibrate(
                cover,
                sample,
                ExpectedUncovered(),
                scope=scope,
                options=CalibrationOptions(lp_method=option, time_limit=60.0),
                **common,
            )
            with self.subTest(scope=scope, option=option):
                self.assertEqual({p.solver.method for p in result.problems}, {method})
        stalled = SimpleNamespace(status=1, message="Time limit reached.", x=None)
        with (
            mock.patch(
                "climada_petals.engine.parametric.solvers.linprog", return_value=stalled
            ),
            self.assertRaisesRegex(CalibrationLimitError, "Time limit reached"),
        ):
            calibrate(cover, sample, ExpectedUncovered(), scope="joint", **common)

    def test_a_failed_solve_raises(self) -> None:
        sample, _ = random_portfolio(0)
        failed = SimpleNamespace(status=4, message="numerical trouble", x=None, nit=0)
        with (
            mock.patch(
                "climada_petals.engine.parametric.solvers.linprog", return_value=failed
            ),
            self.assertRaisesRegex(CalibrationError, "numerical trouble"),
        ):
            calibrate(
                template(sample, PAYOUT),
                sample,
                ExpectedUncovered(),
                scope="individual",
                budget=Budget(ratio=0.5),
            )

    def test_a_solution_that_fails_a_check_raises(self) -> None:
        sample = _hand_example()
        cover = template(sample, StepPayout.uniform(20.0, 70.0, 2))
        info = SimpleNamespace(method="highs")
        cases = {
            "break their bounds": Solution(np.array([-1.0, 7.0]), 0.7, info),
            "and the solver's": Solution(np.array([3.0, 4.0]), 0.5, info),
        }
        for match, solution in cases.items():
            with (
                self.subTest(match=match),
                mock.patch(SOLVE_DESIGN, return_value=solution),
                self.assertRaisesRegex(CalibrationError, match),
            ):
                calibrate(
                    cover,
                    sample,
                    ExpectedUncovered(),
                    scope="individual",
                    budget=Budget(ratio=0.5),
                )

    def test_a_payout_above_its_limit_raises(self) -> None:
        # A family whose basis rows sum above 1 pays more than its amount: an
        # amount of 8 within the limit 10 (the largest loss) pays 16.
        sample = _hand_example()
        cover = template(sample, _Double(25.0, 8.0))
        with self.assertRaisesRegex(CalibrationError, "above its limit"):
            calibrate(cover, sample, ExpectedUncovered(), scope="individual")


class TestScopes(unittest.TestCase):
    """At one total budget, the joint problem relaxes the individual ones."""

    def test_the_joint_optimum_is_at_most_the_individual_designs_on_the_total(
        self,
    ) -> None:
        cases = itertools.product(
            (0, 1, 2),
            ("annual", "event"),
            (ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9), MSE()),
            (PurePremium(), CostOfCapitalLoad(rate=0.1, q=0.9)),
            (0.3, 1.0),
        )
        for seed, basis, objective, pricing, ratio in cases:
            sample, _ = random_portfolio(seed, basis=basis)
            cover = template(sample, PAYOUT)
            common = {"pricing": pricing, "budget": Budget(ratio=ratio)}
            single = calibrate(cover, sample, objective, scope="individual", **common)
            joint = calibrate(cover, sample, objective, scope="joint", **common)
            measure = sample.measure
            total_loss = sample.loss.toarray().sum(axis=1)
            total_paid = single.cover.payout_matrix(sample.index).sum(axis=1)
            on_total = objective.evaluate(total_loss, total_paid, measure)
            optimum = joint.objective_value
            scale = max(abs(on_total), 1.0)
            with self.subTest(seed=seed, basis=basis, objective=objective, ratio=ratio):
                # The individual designs fit the joint problem's constraints.
                self.assertLessEqual(
                    pricing.premium(total_paid, measure),
                    float(joint.problems[0].budget_bound or 0.0) * (1 + RTOL) + 1e-9,
                )
                self.assertLessEqual(optimum, on_total + 1e-6 * scale)
                if not isinstance(objective, MSE):
                    self.assertLessEqual(
                        on_total, single.objective_value + 1e-6 * scale
                    )

    def test_a_budget_by_unit(self) -> None:
        sample, _ = random_portfolio(4)
        budget = Budget(amount={"U0": 100.0, "U1": 0.0, "U2": 50.0})
        result = calibrate(
            template(sample, PAYOUT),
            sample,
            ExpectedUncovered(),
            scope="individual",
            budget=budget,
        )
        self.assertEqual(
            [problem.budget_bound for problem in result.problems], [100.0, 0.0, 50.0]
        )
        self.assertTrue(math.isnan(result.problems[1].budget_use or 0.0))
        self.assertEqual(_levels(result, "U1"), [0.0] * 4)
        joint = calibrate(
            template(sample, PAYOUT),
            sample,
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(amount=150.0),
        )
        self.assertAlmostEqual(joint.premium, 150.0, places=6)


class TestTemplateRoute(unittest.TestCase):
    """Any payout linear in its amounts, with fixed shape parameters."""

    def setUp(self) -> None:
        self.sample, _ = random_portfolio(6)
        self.common = {"scope": "individual", "budget": Budget(ratio=0.8)}

    def test_payouts_the_old_guard_refused(self) -> None:
        payouts = {
            "not monotone": StepPayout.uniform(20.0, 70.0, 4, monotone=False),
            "fixed level": StepPayout.uniform(20.0, 70.0, 2, levels=[Free(), 50.0]),
            "bounded level": StepPayout.uniform(
                20.0, 70.0, 2, levels=[Free(), Free(0.0, 100.0)]
            ),
        }
        for name, payout in payouts.items():
            result = calibrate(
                template(self.sample, payout),
                self.sample,
                ExpectedUncovered(),
                **self.common,
            )
            levels = np.array([_levels(result, unit) for unit in self.sample.unit_id])
            with self.subTest(name=name):
                if name == "fixed level":
                    np.testing.assert_array_equal(levels[:, 1], 50.0)
                    self.assertEqual(
                        result.calibrated["U0"], ("levels[0]",), "fixed stays fixed"
                    )
                if name == "bounded level":
                    self.assertTrue(np.all(levels[:, 1] <= 100.0))

    def test_a_lower_tail(self) -> None:
        drought = IndexDefinition("table", unit="index", tail="lower")
        sample, _ = array_sample(
            [[-0.5], [-1.5], [-2.5], [0.0]],
            [[0.0], [3.0], [8.0], [0.0]],
            [0, 1, 2, 3],
            10,
            definition=drought,
        )
        self.assertEqual(sample.n_left_out, 0)
        cover = template(
            sample, StepPayout([-1.0, -2.0], [Free(), Free()], tail="lower")
        )
        result = calibrate(cover, sample, ExpectedUncovered(), **self.common)
        # E[L] = 1.1; the premium 0.1 (t0 + t1) is at most 0.88.
        self.assertAlmostEqual(result.objective_value, 1.1 - 0.88, places=9)

    def test_a_payout_per_unit_and_a_stated_limit(self) -> None:
        payouts = {
            "U0": StepPayout.uniform(20.0, 70.0, 3),
            "U1": LinearPayout(25.0, 75.0, Free()),
            "U2": BinaryPayout(40.0, Free()),
        }
        cover = template(self.sample, payouts, limit=Limit(300.0))
        result = calibrate(cover, self.sample, WeightedCVaR(0.9, 0.9), **self.common)
        self.assertEqual(result.cover.limit, Limit(300.0))
        frame = result.parameters_frame()
        amounts = frame[frame["role"] == "amount"]
        self.assertEqual(amounts["unit_id"].tolist(), ["U0"] * 3 + ["U1", "U2"])
        self.assertTrue(np.all(amounts["value"] <= 300.0))
        paid = result.cover.payout_matrix(self.sample.index)
        self.assertLessEqual(float(paid.max()), 300.0)

    def test_a_sample_with_draws(self) -> None:
        # Two identical draws of every event: the same outcomes, the same design.
        drawn = OutcomeSample(
            self.sample.index,
            np.vstack([self.sample.loss.toarray()] * 2),
            self.sample.frequency,
            self.sample.units,
            self.sample.basis,
            n_draws=2,
        )
        results = [
            calibrate(
                template(sample, PAYOUT),
                sample,
                WeightedCVaR(alpha=0.9, beta=0.9),
                scope="joint",
                budget=Budget(ratio=0.8),
                options=CalibrationOptions(lp_method="highs"),
            )
            for sample in (self.sample, drawn)
        ]
        self.assertAlmostEqual(
            results[0].objective_value, results[1].objective_value, delta=1e-9
        )
        self.assertEqual(results[1].design_sample.n_draws, 2)

    def test_without_a_budget_the_limit_bounds_the_amounts(self) -> None:
        sample = _hand_example()
        result = calibrate(
            template(sample, StepPayout.uniform(20.0, 70.0, 2)),
            sample,
            ExpectedUncovered(),
            scope="individual",
        )
        # Every loss is covered, up to the largest loss.
        self.assertAlmostEqual(result.objective_value, 0.0, places=12)
        self.assertIsNone(result.problems[0].budget_bound)
        self.assertIsNone(result.problems[0].budget_use)


class TestResult(unittest.TestCase):
    """What a result records and returns."""

    def setUp(self) -> None:
        self.sample, _ = random_portfolio(4)
        self.train = self.sample.select(np.arange(30))
        self.template = template(self.train, PAYOUT)
        self.result = calibrate(
            self.template,
            self.train,
            WeightedCVaR(alpha=0.5, beta=0.9),
            scope="joint",
            pricing=CostOfCapitalLoad(rate=0.1, q=0.9),
            budget=Budget(ratio=1.0),
        )

    def test_the_record(self) -> None:
        result = self.result
        self.assertTrue(result.cover.is_resolved)
        self.assertEqual(result.template_hash, self.template.definition_hash)
        self.assertNotEqual(result.cover.definition_hash, result.template_hash)
        self.assertEqual(result.objective_value, result.problems[0].objective_value)
        self.assertEqual(result.premium, result.problems[0].premium)
        self.assertEqual(result.expected_loss, result.problems[0].expected_loss)
        self.assertEqual((result.basis, result.route), ("annual", "catalogue"))
        self.assertEqual(result.fill, "lowest")
        # No grid, and no ranges without ranges=True.
        self.assertIsNone(result.ties)
        self.assertIsNone(result.profile)
        self.assertIsNone(result.ranges)
        self.assertEqual(result.summary()["n_candidates"].tolist(), [None])
        record = result.design_sample
        self.assertEqual(record.n_years, 30)
        self.assertEqual(record.n_events, self.train.n_events)
        self.assertEqual(record.n_outcomes, self.train.measure.n_outcomes)
        np.testing.assert_array_equal(record.event_id, self.train.event_id)
        np.testing.assert_array_equal(record.block, self.train.block)
        np.testing.assert_array_equal(record.years, np.arange(30))
        self.assertEqual(record.split, self.train.split)
        self.assertEqual(dict(result.calibrated)["U1"], PAYOUT.free_parameters())

    def test_summary_and_parameters(self) -> None:
        summary = self.result.summary()
        row = summary.iloc[0]
        self.assertEqual(row["problem_id"], "joint")
        self.assertEqual(row["objective"], "weighted_cvar(alpha=0.5, beta=0.9)")
        self.assertEqual((row["route"], row["solver"]), ("lp", "scipy-highs"))
        self.assertEqual(row["method"], "highs-ipm")
        self.assertAlmostEqual(row["budget_use"], row["premium"] / row["budget_bound"])
        frame = self.result.parameters_frame()
        self.assertEqual(len(frame), 3 * 2 * PAYOUT.n_steps)
        levels = frame[frame["role"] == "amount"]
        self.assertTrue(levels["calibrated"].all())
        shapes = frame[frame["role"] == "shape"]
        self.assertFalse(shapes["calibrated"].any())
        np.testing.assert_array_equal(
            shapes["value"].iloc[: PAYOUT.n_steps], PAYOUT.breakpoints
        )

    def test_filled_amounts(self) -> None:
        # One unit's index never reaches the last step in training.
        sample, _ = array_sample(
            [[30.0, 30.0], [60.0, 40.0], [65.0, 25.0]],
            [[4.0, 2.0], [10.0, 5.0], [12.0, 1.0]],
            [0, 1, 2],
            10,
        )
        cover = template(sample, StepPayout.uniform(20.0, 70.0, 2))
        result = calibrate(
            cover, sample, ExpectedUncovered(), scope="joint", budget=Budget(ratio=0.5)
        )
        self.assertEqual(dict(result.unidentified), {"U0": (), "U1": ("levels[1]",)})
        self.assertFalse(result.filled(sample.index).any())
        later = ParametricIndex(
            [[60.0, 60.0], [30.0, 30.0]],
            [7, 8],
            sample.unit_id,
            sample.index.definition,
        )
        np.testing.assert_array_equal(
            result.filled(later), [[False, True], [False, False]]
        )
        payout = result.cover.payout["U1"]
        assert isinstance(payout, StepPayout)
        self.assertEqual(payout.levels[1], payout.levels[0])
        highest = calibrate(
            cover,
            sample,
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=0.5),
            options=CalibrationOptions(fill="highest"),
        )
        self.assertEqual(_levels(highest, "U1")[1], 5.0)
        other = ParametricIndex(later.values, [7, 8], ("A", "B"), later.definition)
        with self.assertRaisesRegex(ParametricInputError, "definition and units"):
            result.filled(other)

    def test_a_result_pickles(self) -> None:
        for copied in (
            pickle.loads(pickle.dumps(self.result)),
            copy.deepcopy(self.result),
        ):
            self.assertEqual(copied.cover, self.result.cover)
            self.assertEqual(copied.problems, self.result.problems)
            self.assertEqual(dict(copied.unidentified), dict(self.result.unidentified))
            np.testing.assert_array_equal(
                copied.design_sample.event_id, self.result.design_sample.event_id
            )

    def test_the_scope_is_checked(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "scope must be one of"):
            dataclasses.replace(self.result, scope="portfolio")


class TestInputs(unittest.TestCase):
    """Invalid inputs raise before anything is solved."""

    def setUp(self) -> None:
        self.sample, _ = random_portfolio(0)
        self.cover = template(self.sample, PAYOUT)

    def test_invalid_arguments_raise(self) -> None:
        good = {
            "template": self.cover,
            "sample": self.sample,
            "objective": ExpectedUncovered(),
            "scope": "individual",
            "budget": Budget(ratio=0.5),
        }
        bad = {
            "template": self.cover.payout["U0"],
            "sample": self.sample.index,
            "objective": "mse",
            "scope": "portfolio",
            "pricing": 0.1,
            "budget": 0.5,
            "options": {"fill": "lowest"},
        }
        for name, value in bad.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ParametricInputError, name),
            ):
                calibrate(**{**good, name: value})

    def test_a_template_that_does_not_fit_the_sample_raises(self) -> None:
        cases = {
            "units are not": ParametricCover(
                self.cover.index,
                PAYOUT,
                unit_id=["U0", "U1"],
                payout_unit="USD",
            ),
            "the template pays on": ParametricCover(
                IndexDefinition("table", unit="m/s", source="elsewhere"),
                PAYOUT,
                unit_id=self.sample.unit_id,
                payout_unit="USD",
            ),
        }
        for match, cover in cases.items():
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                calibrate(cover, self.sample, ExpectedUncovered(), scope="joint")

    def test_invalid_options_raise(self) -> None:
        bad = [
            {"fill": "middle"},
            {"ranges": 1},
            {"lp_method": "simplex"},
            {"feasibility_tolerance": 0.0},
            {"time_limit": -1.0},
            {"max_slsqp_amounts": 0},
            {"max_cut_rounds": 2.5},
            {"max_grid_points": 0},
        ]
        for fields in bad:
            with self.subTest(fields=fields), self.assertRaises(ParametricInputError):
                CalibrationOptions(**fields)

    def test_a_linear_objective_needs_a_budget_or_bounds(self) -> None:
        unbounded = template(self.sample, PAYOUT, limit=None)
        with self.assertRaisesRegex(ParametricInputError, "needs a budget or bounded"):
            calibrate(unbounded, self.sample, ExpectedUncovered(), scope="individual")
        result = calibrate(
            unbounded,
            self.sample,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=0.5),
        )
        self.assertIsNone(result.cover.limit)

    def test_a_payout_that_pays_at_zero_on_left_out_events_is_refused(self) -> None:
        # The third event has no loss and an index of 0: the sample leaves it out.
        sample, _ = array_sample(
            [[30.0], [60.0], [0.0]], [[4.0], [10.0], [0.0]], [0, 1, 2], 10
        )
        self.assertEqual(sample.n_left_out, 1)
        common = {"scope": "individual", "budget": Budget(ratio=0.5)}
        at_zero = StepPayout.uniform(0.0, 70.0, 2)
        with self.assertRaisesRegex(ParametricInputError, "left out 1 events"):
            calibrate(template(sample, at_zero), sample, ExpectedUncovered(), **common)
        # A payout with nothing at an index of 0 goes ahead.
        result = calibrate(
            template(sample, StepPayout.uniform(20.0, 70.0, 2)),
            sample,
            ExpectedUncovered(),
            **common,
        )
        self.assertAlmostEqual(result.objective_value, 0.7, places=9)
        # So does one at 0 on a sample that left nothing out.
        whole = _hand_example()
        calibrate(template(whole, at_zero), whole, ExpectedUncovered(), **common)

    def test_years_without_events_raise(self) -> None:
        sample, _ = array_sample([[30.0]], [[1.0]], [0], 10)
        with self.assertRaisesRegex(ParametricInputError, "no event"):
            sample.select([5, 6])

    def test_a_loss_free_event_below_the_attachment_changes_nothing(self) -> None:
        values = []
        for drop in (False, True):
            sample = _hand_example(drop_quiet_event=drop)
            result = calibrate(
                template(sample, StepPayout.uniform(20.0, 70.0, 2)),
                sample,
                ExpectedUncovered(),
                scope="individual",
                budget=Budget(ratio=0.5),
            )
            values.append((result.objective_value, result.premium))
        np.testing.assert_allclose(values[0], values[1], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
