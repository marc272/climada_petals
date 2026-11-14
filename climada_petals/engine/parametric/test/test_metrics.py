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

Tests of the basis-risk metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from climada_petals.engine.parametric import (
    CalibrationResult,
    CostOfCapitalLoad,
    ExpectedUncovered,
    ParametricInputError,
    StepPayout,
    WeightedCVaR,
    basis_risk,
    calibrate,
    outcome_measure,
    split_years,
)
from climada_petals.engine.parametric.metrics import METRICS, subject_metrics
from climada_petals.engine.parametric.pricing import Pricing
from climada_petals.engine.parametric.test._synthetic import (
    array_index,
    random_portfolio,
)


class TestSubjectMetrics(unittest.TestCase):
    """Metrics of one subject, worked out by hand."""

    def test_two_years_of_ten(self) -> None:
        index, _ = array_index(
            [[30.0], [60.0], [65.0]], [[4.0], [10.0], [2.0]], [0, 1, 1], 10
        )
        measure = outcome_measure(
            index.event_year,
            index.frequency,
            basis="annual",
            years=index.years,
            n_catalogue_years=10,
        )
        loss = np.array([4.0, 10.0, 2.0])
        paid = np.array([1.0, 12.0, 0.0])
        out = subject_metrics(loss, paid, measure, beta=0.85)
        # Year 0: L = 4, P = 1, U = 3. Year 1: L = 12, P = 12, U = 0 + 2 = 2.
        # The 8 other years: U = 0. The worst 15 % are U = 3 (10 %) and U = 2.
        self.assertAlmostEqual(out["expected_loss"], 1.6)
        self.assertAlmostEqual(out["expected_payout"], 1.3)
        self.assertAlmostEqual(out["expected_uncovered"], 0.5)
        self.assertAlmostEqual(out["var_uncovered"], 2.0)
        self.assertAlmostEqual(out["cvar_uncovered"], (0.1 * 3 + 0.05 * 2) / 0.15)
        self.assertAlmostEqual(out["rmse_residual"], math.sqrt(0.9))
        self.assertAlmostEqual(out["payout_to_loss_ratio"], 1.3 / 1.6)
        self.assertEqual(set(out), set(METRICS))

    def test_flat_outcomes_have_no_correlation(self) -> None:
        measure = outcome_measure(
            [0, 1], [0.1, 0.1], basis="annual", years=range(10), n_catalogue_years=10
        )
        out = subject_metrics([4.0, 10.0], [0.0, 0.0], measure, beta=0.85)
        self.assertTrue(math.isnan(out["correlation"]))
        self.assertEqual(out["payout_to_loss_ratio"], 0.0)
        out = subject_metrics([0.0, 0.0], [1.0, 2.0], measure, beta=0.85)
        self.assertTrue(math.isnan(out["payout_to_loss_ratio"]))


class TestBasisRisk(unittest.TestCase):
    """Out-of-sample metrics of a cover."""

    def setUp(self) -> None:
        self.index, self.impact = random_portfolio(5, n_events=60, n_years=90)
        self.train, self.test = split_years(self.index, train_fraction=0.7)

    def _cover(self, scope: str, pricing: Pricing | None = None) -> CalibrationResult:
        kwargs: dict[str, Pricing] = {} if pricing is None else {"pricing": pricing}
        return calibrate(
            self.index,
            self.impact,
            payout=StepPayout.uniform(20.0, 70.0, 4),
            objective=WeightedCVaR(alpha=0.9, beta=0.9),
            basis="annual",
            scope=scope,
            budget=1.0,
            years=self.train,
            **kwargs,
        )

    def test_individual_cover_on_the_test_years(self) -> None:
        cover = self._cover("individual")
        frame = basis_risk(self.index, self.impact, cover, beta=0.9, years=self.test)
        self.assertEqual(frame["level"].tolist(), ["unit"] * 3 + ["portfolio"])
        self.assertTrue(frame["split_identified"].all())
        self.assertEqual(int(frame["n_years"].iloc[0]), self.test.size)
        units = frame[frame["level"] == "unit"]
        portfolio = frame[frame["level"] == "portfolio"].iloc[0]
        for name in ("expected_loss", "expected_payout", "premium"):
            self.assertAlmostEqual(units[name].sum(), portfolio[name], places=6)
        # Shortfalls net across the units within an event, so the total's E[U]
        # is at most the sum of the units'.
        self.assertLessEqual(
            portfolio["expected_uncovered"], units["expected_uncovered"].sum() + 1e-9
        )

    def test_portfolio_cover_flags_its_unit_rows(self) -> None:
        cover = self._cover("portfolio", CostOfCapitalLoad(rate=0.1, q=0.9))
        frame = basis_risk(self.index, self.impact, cover, beta=0.9, years=self.test)
        units = frame[frame["level"] == "unit"]
        self.assertFalse(units["split_identified"].any())
        self.assertTrue(np.isnan(np.asarray(units["premium"], dtype=float)).all())
        self.assertTrue(np.isfinite(frame["premium"].iloc[-1]))

    def test_an_unknown_basis_raises(self) -> None:
        cover = self._cover("individual")
        with self.assertRaisesRegex(ParametricInputError, "basis must be one of"):
            basis_risk(self.index, self.impact, cover, beta=0.9, basis="season")

    def test_in_sample_metric_is_the_objective(self) -> None:
        cover = calibrate(
            self.index,
            self.impact,
            payout=StepPayout.uniform(20.0, 70.0, 4),
            objective=ExpectedUncovered(),
            basis="annual",
            scope="portfolio",
            budget=0.8,
            years=self.train,
        )
        frame = basis_risk(self.index, self.impact, cover, beta=0.9, years=self.train)
        self.assertAlmostEqual(
            frame["expected_uncovered"].iloc[-1],
            cover.problems[0].objective_value,
            delta=1e-7 * cover.problems[0].objective_value,
        )


if __name__ == "__main__":
    unittest.main()
