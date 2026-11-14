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

import copy
import importlib.util
import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import scipy.sparse as sp
from climada.engine import Impact
from numpy.typing import ArrayLike, NDArray

from climada_petals.engine.parametric import (
    CalibrationError,
    CalibrationResult,
    CostOfCapitalLoad,
    ExpectedUncovered,
    Free,
    MeanSquaredError,
    MissingDependencyError,
    ParametricIndex,
    ParametricInputError,
    PurePremium,
    StepPayout,
    WeightedCVaR,
    calibrate,
    outcome_measure,
)
from climada_petals.engine.parametric.calibrate import loss_matrix
from climada_petals.engine.parametric.measure import OutcomeMeasure
from climada_petals.engine.parametric.test._synthetic import (
    array_index,
    random_portfolio,
)
from climada_petals.util.config import LOGGER

HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None
PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=4)
RTOL = 1e-7


def _hand_example(drop_quiet_event: bool = False) -> tuple[ParametricIndex, Impact]:
    """One unit, 10 years: (index 30, loss 4), (60, 10) and (5, 0) in years 0-2."""
    values = [[30.0], [60.0], [5.0]]
    loss = [[4.0], [10.0], [0.0]]
    years = [0, 1, 2]
    if drop_quiet_event:
        values, loss, years = values[:2], loss[:2], years[:2]
    return array_index(values, loss, years, 10)


def _measure(
    index: ParametricIndex, cover: CalibrationResult, years: ArrayLike | None = None
) -> tuple[NDArray[np.bool_], OutcomeMeasure]:
    mask = index.event_mask(years)
    return mask, outcome_measure(
        index.event_year[mask],
        index.frequency[mask],
        basis=cover.basis,
        years=index.check_years(years),
        n_catalogue_years=index.years.size,
    )


class TestHandWorkedDesigns(unittest.TestCase):
    """Designs small enough to solve by hand."""

    def test_expected_uncovered_spends_the_budget(self) -> None:
        index, impact = _hand_example()
        cover = calibrate(
            index,
            impact,
            payout=StepPayout.uniform(20.0, 70.0, 2),
            objective=ExpectedUncovered(),
            basis="annual",
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
            index, impact = _hand_example(drop_quiet_event=drop)
            cover = calibrate(
                index,
                impact,
                payout=StepPayout.uniform(20.0, 70.0, 2),
                objective=ExpectedUncovered(),
                basis="annual",
                budget=0.5,
            )
            values.append((cover.problems[0].objective_value, cover.premium))
        np.testing.assert_allclose(values[0], values[1], rtol=1e-9)

    @unittest.skipUnless(HAS_CVXPY, "the mean squared error needs cvxpy")
    def test_mean_squared_error_pays_nothing_below_the_attachment(self) -> None:
        index, impact = _hand_example()
        cover = calibrate(
            index,
            impact,
            payout=StepPayout.uniform(20.0, 70.0, 2),
            objective=MeanSquaredError(),
            basis="annual",
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
            index, impact = random_portfolio(seed)
            cover = calibrate(
                index,
                impact,
                payout=PAYOUT,
                objective=objective,
                basis=basis,
                scope=scope,
                pricing=pricing,
                budget=0.6,
            )
            mask, measure = _measure(index, cover)
            loss = impact.imp_mat.toarray()[mask]
            paid = cover.payout_matrix(index)[mask]
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
        index, impact = random_portfolio(2)
        common = {
            "payout": PAYOUT,
            "basis": "annual",
            "scope": "portfolio",
            "budget": 0.5,
        }
        mean = calibrate(index, impact, objective=ExpectedUncovered(), **common)
        mixed = calibrate(index, impact, objective=WeightedCVaR(0.0, 0.95), **common)
        np.testing.assert_allclose(
            mixed.problems[0].objective_value,
            mean.problems[0].objective_value,
            rtol=1e-9,
        )

    def test_a_beta_below_the_loss_free_mass_gives_the_expected_uncovered_design(
        self,
    ) -> None:
        index, impact = random_portfolio(3)
        common = {
            "payout": PAYOUT,
            "basis": "annual",
            "scope": "portfolio",
            "budget": 0.5,
        }
        mean = calibrate(index, impact, objective=ExpectedUncovered(), **common)
        zero = mean.problems[0].diagnostics["zero_loss_probability"]
        self.assertGreater(zero, 0.1)
        alpha, beta = 0.9, zero / 2
        tail = calibrate(index, impact, objective=WeightedCVaR(alpha, beta), **common)
        problem = tail.problems[0]
        self.assertTrue(problem.diagnostics["degenerate_tail"])
        np.testing.assert_allclose(
            problem.objective_value,
            (alpha / (1 - beta) + 1 - alpha) * mean.problems[0].objective_value,
            rtol=1e-7,
        )

    def test_a_tail_within_one_outcome_is_recorded_and_logged(self) -> None:
        # 60 years: one outcome has probability 1/60, more than 1 - beta.
        index, impact = random_portfolio(2)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            cover = calibrate(
                index,
                impact,
                payout=PAYOUT,
                objective=WeightedCVaR(alpha=0.9, beta=0.99),
                basis="annual",
                scope="portfolio",
                budget=0.5,
            )
        self.assertTrue(cover.problems[0].diagnostics["tail_within_one_outcome"])
        self.assertTrue(cover.summary()["tail_within_one_outcome"].all())
        self.assertIn("single worst outcome", "\n".join(logs.output))

    def test_a_failed_solve_raises(self) -> None:
        index, impact = random_portfolio(0)
        failed = SimpleNamespace(status=4, message="numerical trouble", x=None, nit=0)
        with (
            mock.patch(
                "climada_petals.engine.parametric.solvers.linprog", return_value=failed
            ),
            self.assertRaisesRegex(CalibrationError, "numerical trouble"),
        ):
            calibrate(
                index,
                impact,
                payout=PAYOUT,
                objective=ExpectedUncovered(),
                basis="annual",
            )


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
            index, impact = random_portfolio(seed)
            common = {
                "payout": PAYOUT,
                "objective": objective,
                "basis": basis,
                "pricing": pricing,
                "budget": budget,
            }
            single = calibrate(index, impact, scope="individual", **common)
            joint = calibrate(index, impact, scope="portfolio", **common)
            mask, measure = _measure(index, single)
            total_loss = impact.imp_mat.toarray()[mask].sum(axis=1)
            total_paid = single.payout_matrix(index)[mask].sum(axis=1)
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
        self.index, self.impact = random_portfolio(4)
        self.cover = calibrate(
            self.index,
            self.impact,
            payout=PAYOUT,
            objective=WeightedCVaR(alpha=0.5, beta=0.9),
            basis="annual",
            scope="portfolio",
            budget=1.0,
        )

    def test_payout_impact_is_a_climada_impact(self) -> None:
        paid = self.cover.payout_impact(self.index, self.impact)
        self.assertIsInstance(paid, Impact)
        matrix = self.cover.payout_matrix(self.index)
        np.testing.assert_allclose(paid.imp_mat.toarray(), matrix)
        np.testing.assert_allclose(paid.at_event, matrix.sum(axis=1))
        np.testing.assert_allclose(paid.eai_exp, self.index.frequency @ matrix)
        self.assertAlmostEqual(
            paid.aai_agg, float((self.index.frequency @ matrix).sum())
        )
        np.testing.assert_array_equal(paid.event_id, self.impact.event_id)

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
        with self.assertRaisesRegex(ParametricInputError, "levels must have shape"):
            CalibrationResult(**{**fields, "levels": np.zeros((2, 1))})
        with self.assertRaisesRegex(ParametricInputError, "scope must be one of"):
            CalibrationResult(**{**fields, "scope": "cluster"})

    def test_payouts_need_the_covers_units(self) -> None:
        other, _ = array_index(np.zeros((2, 3)), np.zeros((2, 3)), [0, 1], 10)
        other = ParametricIndex(
            values=other.values,
            event_id=other.event_id,
            event_name=other.event_name,
            frequency=other.frequency,
            event_year=other.event_year,
            years=other.years,
            unit_id=("A", "B", "C"),
            coords=other.coords,
        )
        with self.assertRaisesRegex(ParametricInputError, "not the cover's"):
            self.cover.payout_matrix(other)

    def test_training_years_only(self) -> None:
        train = np.arange(20)
        cover = calibrate(
            self.index,
            self.impact,
            payout=PAYOUT,
            objective=ExpectedUncovered(),
            basis="annual",
            years=train,
        )
        np.testing.assert_array_equal(cover.calibration_years, train)


class TestInputs(unittest.TestCase):
    """Invalid inputs raise before anything is solved."""

    def test_invalid_arguments_raise(self) -> None:
        index, impact = random_portfolio(0)
        good = {"payout": PAYOUT, "objective": ExpectedUncovered(), "basis": "annual"}
        bad = [
            {"scope": "cluster"},
            {"basis": "event_conditional"},
            {"budget": -1.0},
            {"objective": "mse"},
            {"pricing": 0.1},
            {"years": [99]},
        ]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ParametricInputError):
                calibrate(index, impact, **{**good, **change})

    def test_index_and_payout_types_are_checked(self) -> None:
        index, impact = random_portfolio(0)
        good = {"payout": PAYOUT, "objective": ExpectedUncovered(), "basis": "annual"}
        with self.assertRaisesRegex(ParametricInputError, "ParametricIndex"):
            calibrate(index.values, impact, **good)
        with self.assertRaisesRegex(ParametricInputError, "StepPayout"):
            calibrate(index, impact, **{**good, "payout": (20.0, 70.0, 4)})

    def test_a_payout_the_solver_cannot_take_is_refused(self) -> None:
        index, impact = random_portfolio(0)
        good = {"objective": ExpectedUncovered(), "basis": "annual"}
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
                calibrate(index, impact, payout=payout, **good)

    def test_an_impact_that_does_not_match_the_index_is_refused(self) -> None:
        index, impact = array_index([[30.0], [60.0]], [[4.0], [10.0]], [0, 1], 10)
        np.testing.assert_array_equal(loss_matrix(index, impact), [[4.0], [10.0]])
        changes = {
            "event_id": ({"event_id": np.array([1, 3])}, "event_id"),
            "frequency": ({"frequency": np.array([0.1, 0.2])}, "frequencies"),
            "coord_exp": ({"coord_exp": np.array([[25.0, -70.0]])}, "coord_exp"),
            "loss": ({"imp_mat": sp.csr_matrix([[4.0], [-1.0]])}, "negative"),
        }
        for name, (change, match) in changes.items():
            other = copy.deepcopy(impact)
            for attribute, value in change.items():
                setattr(other, attribute, value)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                loss_matrix(index, other)

    def test_years_without_events_raise(self) -> None:
        index, impact = array_index([[30.0]], [[1.0]], [0], 10)
        with self.assertRaisesRegex(ParametricInputError, "no event"):
            calibrate(
                index,
                impact,
                payout=PAYOUT,
                objective=ExpectedUncovered(),
                basis="annual",
                years=[5, 6],
            )

    def test_mean_squared_error_without_cvxpy_raises(self) -> None:
        index, impact = random_portfolio(0)
        with mock.patch.dict("sys.modules", {"cvxpy": None}):
            with self.assertRaisesRegex(MissingDependencyError, "cvxpy"):
                calibrate(
                    index,
                    impact,
                    payout=PAYOUT,
                    objective=MeanSquaredError(),
                    basis="annual",
                )
            # The linear objectives need no cvxpy.
            calibrate(
                index,
                impact,
                payout=PAYOUT,
                objective=WeightedCVaR(alpha=0.9, beta=0.9),
                basis="annual",
            )


if __name__ == "__main__":
    unittest.main()
