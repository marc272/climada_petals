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

Tests of the calibration: hand-worked designs, the LP against numpy, the scopes."""

from __future__ import annotations

import importlib.util
import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from climada.engine import Impact

from climada_petals.engine.parametric import (
    CalibrationError,
    CalibrationResult,
    CostOfCapitalLoad,
    EventBasis,
    ExpectedUncovered,
    Free,
    MeanSquaredError,
    MissingDependencyError,
    OutcomeSample,
    ParametricIndex,
    ParametricInputError,
    PurePremium,
    StepPayout,
    WeightedCVaR,
    calibrate,
)
from climada_petals.engine.parametric.test._synthetic import (
    array_index,
    array_sample,
    random_portfolio,
)
from climada_petals.util.config import LOGGER

HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None
PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=4)
RTOL = 1e-7


def _hand_example(drop_quiet_event: bool = False) -> OutcomeSample:
    """One unit, 10 years: (index 30, loss 4), (60, 10) and (5, 0) in years 0-2."""
    values = [[30.0], [60.0], [5.0]]
    loss = [[4.0], [10.0], [0.0]]
    years = [0, 1, 2]
    if drop_quiet_event:
        values, loss, years = values[:2], loss[:2], years[:2]
    return array_sample(values, loss, years, 10)[0]


class TestHandWorkedDesigns(unittest.TestCase):
    """Designs small enough to solve by hand."""

    def test_expected_uncovered_spends_the_budget(self) -> None:
        cover = calibrate(
            _hand_example(),
            payout=StepPayout.uniform(20.0, 70.0, 2),
            objective=ExpectedUncovered(),
            budget=0.5,
        )
        (problem,) = cover.problems
        # E[L] = 1.4, so the premium is at most 0.7 and E[U] at least 1.4 - 0.7.
        self.assertAlmostEqual(problem.objective_value, 0.7, places=9)
        self.assertAlmostEqual(problem.premium, 0.7, places=9)
        self.assertIsNotNone(problem.budget_bound)
        self.assertAlmostEqual(float(problem.budget_bound or 0.0), 0.7, places=12)

    def test_zero_mode_ignores_a_quiet_event_below_the_attachment(self) -> None:
        values = []
        for drop in (False, True):
            cover = calibrate(
                _hand_example(drop_quiet_event=drop),
                payout=StepPayout.uniform(20.0, 70.0, 2),
                objective=ExpectedUncovered(),
                budget=0.5,
            )
            values.append((cover.problems[0].objective_value, cover.premium))
        np.testing.assert_allclose(values[0], values[1], rtol=1e-9)

    @unittest.skipUnless(HAS_CVXPY, "the mean squared error needs cvxpy")
    def test_mean_squared_error_pays_nothing_below_the_attachment(self) -> None:
        cover = calibrate(
            _hand_example(),
            payout=StepPayout.uniform(20.0, 70.0, 2),
            objective=MeanSquaredError(),
            budget=5.0,
        )
        # An interior-point solution approaches a level at its cap only to
        # about the square root of the solver tolerance, because the objective
        # is flat at its optimum; the value is accurate.
        np.testing.assert_allclose(cover.levels[0], [4.0, 10.0], atol=1e-3)
        self.assertAlmostEqual(cover.problems[0].objective_value, 0.0, 6)


class TestLinearProgram(unittest.TestCase):
    """The LP's optimum is feasible and equals the objective evaluated in numpy."""

    def test_optimum_is_feasible_and_matches_numpy(self) -> None:
        cases = itertools.product(
            (0, 1),
            ("individual", "portfolio"),
            ("annual", "event"),
            (ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9)),
            (PurePremium(), CostOfCapitalLoad(rate=0.1, q=0.9)),
        )
        for seed, scope, basis, objective, pricing in cases:
            sample, _ = random_portfolio(seed, basis=basis)
            cover = calibrate(
                sample,
                payout=PAYOUT,
                objective=objective,
                scope=scope,
                pricing=pricing,
                budget=0.6,
            )
            measure = sample.measure.as_probability()
            loss = sample.loss.toarray()
            paid = cover.payout_matrix(sample.index)
            caps = loss.max(axis=0)
            with self.subTest(seed=seed, scope=scope, basis=basis, objective=objective):
                self.assertTrue(
                    np.all(np.diff(cover.levels, axis=1) >= -1e-9 * caps.max())
                )
                self.assertTrue(np.all(cover.levels >= -1e-12))
                self.assertTrue(np.all(cover.levels <= caps[:, None] * (1 + 1e-9)))
                units = (
                    [[j] for j in range(3)] if scope == "individual" else [[0, 1, 2]]
                )
                for problem, cols in zip(cover.problems, units, strict=True):
                    total_loss = loss[:, cols].sum(axis=1)
                    total_paid = paid[:, cols].sum(axis=1)
                    value = objective.evaluate(total_loss, total_paid, measure)
                    np.testing.assert_allclose(
                        problem.objective_value, value, rtol=RTOL
                    )
                    self.assertAlmostEqual(
                        problem.premium, pricing.premium(total_paid, measure), 9
                    )
                    self.assertLessEqual(
                        problem.premium,
                        float(problem.budget_bound or 0.0) * (1 + RTOL) + 1e-9,
                    )

    def test_alpha_zero_is_expected_uncovered(self) -> None:
        sample, _ = random_portfolio(2)
        common = {"payout": PAYOUT, "scope": "portfolio", "budget": 0.5}
        mean = calibrate(sample, objective=ExpectedUncovered(), **common)
        mixed = calibrate(sample, objective=WeightedCVaR(0.0, 0.95), **common)
        np.testing.assert_allclose(
            mixed.problems[0].objective_value,
            mean.problems[0].objective_value,
            rtol=1e-9,
        )

    def test_a_beta_below_the_loss_free_mass_gives_the_expected_uncovered_design(
        self,
    ) -> None:
        sample, _ = random_portfolio(3)
        common = {"payout": PAYOUT, "scope": "portfolio", "budget": 0.5}
        mean = calibrate(sample, objective=ExpectedUncovered(), **common)
        zero = mean.problems[0].diagnostics["zero_loss_probability"]
        self.assertGreater(zero, 0.1)
        alpha, beta = 0.9, zero / 2
        tail = calibrate(sample, objective=WeightedCVaR(alpha, beta), **common)
        problem = tail.problems[0]
        self.assertTrue(problem.diagnostics["degenerate_tail"])
        np.testing.assert_allclose(
            problem.objective_value,
            (alpha / (1 - beta) + 1 - alpha) * mean.problems[0].objective_value,
            rtol=1e-7,
        )

    def test_a_tail_within_one_outcome_is_recorded_and_logged(self) -> None:
        # 60 years: one outcome has probability 1/60, more than 1 - beta.
        sample, _ = random_portfolio(2)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            cover = calibrate(
                sample,
                payout=PAYOUT,
                objective=WeightedCVaR(alpha=0.9, beta=0.99),
                scope="portfolio",
                budget=0.5,
            )
        self.assertTrue(cover.problems[0].diagnostics["tail_within_one_outcome"])
        self.assertTrue(cover.summary()["tail_within_one_outcome"].all())
        self.assertIn("single worst outcome", "\n".join(logs.output))

    def test_a_failed_solve_raises(self) -> None:
        sample, _ = random_portfolio(0)
        failed = SimpleNamespace(status=4, message="numerical trouble", x=None, nit=0)
        with (
            mock.patch(
                "climada_petals.engine.parametric.solvers.linprog", return_value=failed
            ),
            self.assertRaisesRegex(CalibrationError, "numerical trouble"),
        ):
            calibrate(sample, payout=PAYOUT, objective=ExpectedUncovered())


class TestScopes(unittest.TestCase):
    """At one total budget, the joint problem relaxes the individual ones."""

    def test_portfolio_optimum_is_at_most_the_individual_designs_on_the_total(
        self,
    ) -> None:
        objectives = [ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9)]
        if HAS_CVXPY:
            objectives.append(MeanSquaredError())
        cases = itertools.product(
            (0, 1, 2),
            ("annual", "event"),
            objectives,
            (PurePremium(), CostOfCapitalLoad(rate=0.1, q=0.9)),
            (0.3, 1.0),
        )
        for seed, basis, objective, pricing, budget in cases:
            sample, _ = random_portfolio(seed, basis=basis)
            common = {
                "payout": PAYOUT,
                "objective": objective,
                "pricing": pricing,
                "budget": budget,
            }
            single = calibrate(sample, scope="individual", **common)
            joint = calibrate(sample, scope="portfolio", **common)
            measure = sample.measure.as_probability()
            total_loss = sample.loss.toarray().sum(axis=1)
            total_paid = single.payout_matrix(sample.index).sum(axis=1)
            on_total = objective.evaluate(total_loss, total_paid, measure)
            optimum = joint.problems[0].objective_value
            scale = max(abs(on_total), 1.0)
            with self.subTest(
                seed=seed, basis=basis, objective=objective, budget=budget
            ):
                # The individual designs fit the joint problem's constraints.
                self.assertLessEqual(
                    pricing.premium(total_paid, measure),
                    float(joint.problems[0].budget_bound or 0.0) * (1 + RTOL) + 1e-9,
                )
                self.assertLessEqual(optimum, on_total + 1e-6 * scale)
                if not isinstance(objective, MeanSquaredError):
                    parts = sum(p.objective_value for p in single.problems)
                    self.assertLessEqual(on_total, parts + 1e-6 * scale)


class TestCover(unittest.TestCase):
    """What a cover returns."""

    def setUp(self) -> None:
        self.sample, self.impact = random_portfolio(4)
        self.cover = calibrate(
            self.sample,
            payout=PAYOUT,
            objective=WeightedCVaR(alpha=0.5, beta=0.9),
            scope="portfolio",
            budget=1.0,
        )

    def test_payout_impact_is_a_climada_impact(self) -> None:
        index, units = self.sample.index, self.sample.units
        paid = self.cover.payout_impact(index, self.impact, units=units)
        self.assertIsInstance(paid, Impact)
        matrix = self.cover.payout_matrix(index)
        np.testing.assert_allclose(paid.imp_mat.toarray(), matrix)
        np.testing.assert_allclose(paid.at_event, matrix.sum(axis=1))
        np.testing.assert_allclose(paid.eai_exp, self.impact.frequency @ matrix)
        self.assertAlmostEqual(
            paid.aai_agg, float((self.impact.frequency @ matrix).sum())
        )
        np.testing.assert_array_equal(paid.event_id, self.impact.event_id)
        _, _, others = array_index(np.zeros((1, 2)), np.zeros((1, 2)))
        for wrong in (others, index):
            with (
                self.subTest(units=wrong),
                self.assertRaisesRegex(ParametricInputError, "cover's unit_id"),
            ):
                self.cover.payout_impact(index, self.impact, units=wrong)

    def test_tables(self) -> None:
        frame = self.cover.levels_frame()
        self.assertEqual(len(frame), 3 * PAYOUT.n_steps)
        np.testing.assert_allclose(frame["level"], self.cover.levels.reshape(-1))
        summary = self.cover.summary()
        self.assertEqual(summary["problem_id"].tolist(), ["portfolio"])
        self.assertIn("degenerate_tail", summary.columns)

    def test_a_cover_checks_its_levels_and_scope(self) -> None:
        fields = {
            "payout": PAYOUT,
            "objective": ExpectedUncovered(),
            "pricing": PurePremium(),
            "scope": "individual",
            "basis": "annual",
            "budget": None,
            "unit_id": ("U0", "U1"),
            "levels": np.zeros((2, PAYOUT.n_steps)),
            "problems": (),
            "calibration_years": np.arange(3),
        }
        cover = CalibrationResult(**fields)
        self.assertFalse(cover.levels.flags.writeable)
        self.assertEqual(cover.premium, 0.0)
        self.assertIsNone(
            CalibrationResult(**{**fields, "calibration_years": None}).calibration_years
        )
        with self.assertRaisesRegex(ParametricInputError, "levels must have shape"):
            CalibrationResult(**{**fields, "levels": np.zeros((2, 1))})
        with self.assertRaisesRegex(ParametricInputError, "scope must be one of"):
            CalibrationResult(**{**fields, "scope": "cluster"})

    def test_payouts_need_the_covers_units(self) -> None:
        other, _, _ = array_index(np.zeros((2, 3)), np.zeros((2, 3)))
        other = ParametricIndex(
            other.values, other.event_id, ("A", "B", "C"), other.definition
        )
        with self.assertRaisesRegex(ParametricInputError, "not the cover's"):
            self.cover.payout_matrix(other)

    def test_training_years_only(self) -> None:
        train = np.arange(20)
        cover = calibrate(
            self.sample.select(train), payout=PAYOUT, objective=ExpectedUncovered()
        )
        np.testing.assert_array_equal(cover.calibration_years, train)
        on_events, _ = random_portfolio(4, basis="event")
        cover = calibrate(on_events, payout=PAYOUT, objective=ExpectedUncovered())
        self.assertIsNone(cover.calibration_years)
        self.assertEqual(cover.basis, "event")


class TestInputs(unittest.TestCase):
    """Invalid inputs raise before anything is solved."""

    def test_invalid_arguments_raise(self) -> None:
        sample, _ = random_portfolio(0)
        good = {"payout": PAYOUT, "objective": ExpectedUncovered()}
        bad = [
            {"scope": "cluster"},
            {"budget": -1.0},
            {"objective": "mse"},
            {"pricing": 0.1},
        ]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ParametricInputError):
                calibrate(sample, **{**good, **change})

    def test_sample_and_payout_types_are_checked(self) -> None:
        sample, _ = random_portfolio(0)
        good = {"payout": PAYOUT, "objective": ExpectedUncovered()}
        for wrong in (sample.index, sample.index.values):
            with (
                self.subTest(sample=type(wrong).__name__),
                self.assertRaisesRegex(ParametricInputError, "OutcomeSample"),
            ):
                calibrate(wrong, **good)
        with self.assertRaisesRegex(ParametricInputError, "StepPayout"):
            calibrate(sample, **{**good, "payout": (20.0, 70.0, 4)})

    def test_a_payout_the_solver_cannot_take_is_refused(self) -> None:
        sample, _ = random_portfolio(0)
        good = {"objective": ExpectedUncovered()}
        bad = {
            "lower tail": StepPayout.uniform(70.0, 20.0, 4, tail="lower"),
            "not monotone": StepPayout.uniform(20.0, 70.0, 4, monotone=False),
            "fixed level": StepPayout.uniform(20.0, 70.0, 2, levels=[Free(), 1.0e6]),
            "bounded level": StepPayout.uniform(
                20.0, 70.0, 2, levels=[Free(), Free(0.0, 1.0e6)]
            ),
            "free breakpoint": StepPayout(
                [20.0, Free(30.0, 50.0, grid=3)], [Free(), Free()]
            ),
        }
        for name, payout in bad.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ParametricInputError, "Free()"),
            ):
                calibrate(sample, payout=payout, **good)

    def test_a_payout_that_pays_at_zero_on_left_out_events_is_refused(self) -> None:
        # The third event has no loss and an index of 0: the sample leaves it out.
        sample, _ = array_sample(
            [[30.0], [60.0], [0.0]], [[4.0], [10.0], [0.0]], [0, 1, 2], 10
        )
        self.assertEqual(sample.n_left_out, 1)
        good = {"objective": ExpectedUncovered(), "budget": 0.5}
        at_zero = StepPayout.uniform(0.0, 70.0, 2)
        with self.assertRaisesRegex(ParametricInputError, "left out 1 events"):
            calibrate(sample, payout=at_zero, **good)
        # A payout with nothing at an index of 0 goes ahead.
        cover = calibrate(sample, payout=StepPayout.uniform(20.0, 70.0, 2), **good)
        self.assertAlmostEqual(cover.problems[0].objective_value, 0.7, places=9)
        # So does one at 0 on a sample that left nothing out.
        calibrate(_hand_example(), payout=at_zero, **good)

    def test_years_without_events_raise(self) -> None:
        sample, _ = array_sample([[30.0]], [[1.0]], [0], 10)
        with self.assertRaisesRegex(ParametricInputError, "no event"):
            sample.select([5, 6])

    def test_frequencies_above_one_per_year_on_the_event_basis_raise(self) -> None:
        # The event basis holds them; the calibration reads it as a
        # probability over one year, which needs at most 1.
        index, impact, units = array_index(
            [[30.0], [60.0]], [[4.0], [10.0]], frequency=[0.8, 0.7]
        )
        sample = OutcomeSample.from_impact(
            index, impact, units=units, basis=EventBasis()
        )
        self.assertAlmostEqual(sample.measure.total_weight, 1.5)
        with self.assertRaisesRegex(ParametricInputError, "sum to 1.5 per year"):
            calibrate(sample, payout=PAYOUT, objective=ExpectedUncovered())

    def test_mean_squared_error_without_cvxpy_raises(self) -> None:
        sample, _ = random_portfolio(0)
        with mock.patch.dict("sys.modules", {"cvxpy": None}):
            with self.assertRaisesRegex(MissingDependencyError, "cvxpy"):
                calibrate(sample, payout=PAYOUT, objective=MeanSquaredError())
            # The linear objectives need no cvxpy.
            calibrate(
                sample, payout=PAYOUT, objective=WeightedCVaR(alpha=0.9, beta=0.9)
            )


if __name__ == "__main__":
    unittest.main()
