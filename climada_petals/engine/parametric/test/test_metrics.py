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

Tests of the metrics, their options and the setup text, worked out by hand."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from climada_petals.engine.parametric import (
    CatalogueYears,
    CostOfCapitalLoad,
    EventBasis,
    IndexDefinition,
    MetricOptions,
    OutcomeMeasure,
    ParametricCover,
    ParametricInputError,
    PurePremium,
    StepPayout,
    evaluate,
    setup_text,
)
from climada_petals.engine.parametric.metrics import (
    SETUP_COLUMNS,
    index_text,
    label,
    pricing_text,
    subject_metrics,
)
from climada_petals.engine.parametric.test._synthetic import TABLE_WIND, array_sample


def _weighted_correlation(x: list[float], y: list[float], w: list[float]) -> float:
    """Pearson correlation with weights, written out."""
    xs, ys, ws = np.array(x), np.array(y), np.array(w) / sum(w)
    mx, my = ws @ xs, ws @ ys
    cov = ws @ ((xs - mx) * (ys - my))
    return float(cov / math.sqrt((ws @ (xs - mx) ** 2) * (ws @ (ys - my) ** 2)))


class TestOptions(unittest.TestCase):
    """Metric options, labels and columns."""

    def test_labels(self) -> None:
        self.assertEqual(
            [label(x) for x in (0.99, 100.0, 2.5, 0.0, 1e-7, 250)],
            ["0_99", "100", "2_5", "0", "0_0000001", "250"],
        )

    def test_default_columns(self) -> None:
        columns = MetricOptions().columns()
        self.assertEqual(
            columns[:7],
            (
                "expected_loss",
                "expected_payout",
                "expected_uncovered",
                "premium",
                "premium_repriced",
                "coverage_ratio",
                "payout_to_loss_ratio",
            ),
        )
        for name in (
            "tail_coverage_b0_9",
            "cvar_uncovered_to_loss_b0_99",
            "payout_frequency",
            "unidentified_share",
            "false_negative_rate_l0",
            "false_positive_rate_l0",
            "rmse",
            "correlation",
            "var_uncovered_b0_95",
            "cvar_net_loss_b0_99",
            "var_loss_b0_9",
            "loss_rp250",
            "net_loss_rp2",
        ):
            self.assertIn(name, columns)
        # 7 + 2 x 3 tail ratios + 2 + 2 rates + 2 + 6 x 3 + 4 x 7.
        self.assertEqual(len(columns), 7 + 6 + 2 + 2 + 2 + 18 + 28)
        self.assertEqual(len(set(columns)), len(columns))

    def test_values_are_stored_as_tuples_of_floats(self) -> None:
        options = MetricOptions(betas=[0.5], return_periods=np.array([10]))
        self.assertEqual(options.betas, (0.5,))
        self.assertEqual(options.return_periods, (10.0,))

    def test_invalid_options_raise(self) -> None:
        cases = [
            ({"betas": (1.0,)}, "betas"),
            ({"betas": ()}, "at least one value"),
            ({"betas": "0.9"}, "sequence"),
            ({"betas": (0.9, 0.90)}, "distinct labels"),
            ({"return_periods": (1.0,)}, "return_periods"),
            ({"loss_thresholds": (-1.0,)}, "loss_thresholds"),
            ({"payout_threshold": -0.1}, "payout_threshold"),
            ({"allocation": "share"}, "allocation"),
        ]
        for kwargs, match in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                MetricOptions(**kwargs)


class TestSubjectMetrics(unittest.TestCase):
    """Metrics of one subject, worked out by hand."""

    options = MetricOptions(
        betas=(0.85,), return_periods=(5,), loss_thresholds=(0.0, 5.0)
    )

    def test_two_years_of_ten(self) -> None:
        # Year 0: L = 4, P = 1, U = 3. Year 1: two events, L = 10 + 2 = 12,
        # P = 12 + 0, U = 0 + 2 = 2. The 8 other years: nothing.
        years = CatalogueYears.from_labels([1, 2, 3], [0, 1, 1], n_years=10)
        measure = years.measure([1, 2, 3], [0.1] * 3)
        out = subject_metrics(
            [4.0, 10.0, 2.0],
            [1.0, 12.0, 0.0],
            measure,
            premium=1.3,
            filled=[False, True, False],
            options=self.options,
        )
        expected = {
            "expected_loss": 1.6,
            "expected_payout": 1.3,
            "expected_uncovered": 0.5,
            "coverage_ratio": 1.0 - 0.5 / 1.6,
            "payout_to_loss_ratio": 1.3 / 1.6,
            # The worst 15 % of the loss: year 1 (12, U = 2) and half of year 0
            # (4, U = 3).
            "tail_coverage_b0_85": (0.1 * 10.0 + 0.05 * 1.0) / (0.1 * 12 + 0.05 * 4),
            "cvar_uncovered_to_loss_b0_85": (0.1 * 3 + 0.05 * 2)
            / (0.1 * 12 + 0.05 * 4),
            "payout_frequency": 0.2,
            "unidentified_share": 0.1,
            "false_negative_rate_l0": 0.0,
            "false_positive_rate_l0": 0.0,
            "false_negative_rate_l5": 0.0,
            "false_positive_rate_l5": 0.1 / 0.9,
            "rmse": math.sqrt(0.9),
            "correlation": _weighted_correlation(
                [4.0, 12.0, 0.0], [1.0, 12.0, 0.0], [0.1, 0.1, 0.8]
            ),
            "var_uncovered_b0_85": 2.0,
            "cvar_uncovered_b0_85": (0.1 * 3 + 0.05 * 2) / 0.15,
            "var_net_loss_b0_85": 1.3,
            "cvar_net_loss_b0_85": (0.1 * 4.3 + 0.05 * 1.3) / 0.15,
            "var_loss_b0_85": 4.0,
            "cvar_loss_b0_85": (0.1 * 12 + 0.05 * 4) / 0.15,
            "loss_rp5": 0.0,
            "payout_rp5": 0.0,
            "uncovered_rp5": 0.0,
            "net_loss_rp5": 1.3,
        }
        self.assertEqual(set(out), set(expected))
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(out[name], value, places=12)

    def test_event_basis_above_one_per_year(self) -> None:
        # Frequencies 2.0, 0.6, 0.3, 0.1 per year (3.0 in all).
        measure = EventBasis().measure([1, 2, 3, 4], [2.0, 0.6, 0.3, 0.1])
        options = MetricOptions(betas=(0.75,), return_periods=(4,))
        out = subject_metrics(
            [0.0, 1.0, 5.0, 12.0],
            [1.0, 0.0, 5.0, 10.0],
            measure,
            premium=4.5,
            options=options,
        )
        self.assertAlmostEqual(out["expected_loss"], 3.3)
        self.assertAlmostEqual(out["expected_payout"], 4.5)
        self.assertAlmostEqual(out["expected_uncovered"], 0.8)
        self.assertAlmostEqual(out["payout_frequency"], 2.4)
        self.assertAlmostEqual(out["false_negative_rate_l0"], 0.6)
        self.assertAlmostEqual(out["false_positive_rate_l0"], 1.0)
        self.assertEqual(out["var_loss_b0_75"], 5.0)
        self.assertAlmostEqual(out["cvar_loss_b0_75"], (0.1 * 12 + 0.15 * 5) / 0.25)
        self.assertEqual(out["loss_rp4"], 5.0)
        self.assertAlmostEqual(out["rmse"], math.sqrt(2.0 + 0.6 + 0.1 * 4))
        for name in ("var_net_loss_b0_75", "cvar_net_loss_b0_75", "net_loss_rp4"):
            self.assertTrue(math.isnan(out[name]), name)
        self.assertTrue(math.isnan(out["unidentified_share"]))

    def test_missing_mass_counts_as_an_outcome_without_loss(self) -> None:
        # Frequencies summing to 0.5: the metrics equal those with the missing
        # mass as an outcome of its own, but the false-positive rates, whose
        # denominator holds the sample's events alone.
        rng = np.random.default_rng(3)
        loss = rng.uniform(0.0, 10.0, 8) * (rng.random(8) < 0.7)
        paid = rng.uniform(0.0, 10.0, 8) * (rng.random(8) < 0.6)
        measure = EventBasis().measure(np.arange(8), np.full(8, 0.5 / 8))
        padded = OutcomeMeasure(
            sp.vstack([measure.grouping, sp.csr_matrix((1, 8))]),
            np.append(measure.weights, 0.5),
            "event",
            "frequency",
        )
        out = subject_metrics(loss, paid, measure)
        full = subject_metrics(loss, paid, padded)
        self.assertEqual(padded.n_outcomes, 9)
        for name, value in out.items():
            with self.subTest(name=name):
                if name.startswith("false_positive_rate") or math.isnan(value):
                    continue
                self.assertAlmostEqual(value, full[name], places=12)
        no_loss = loss <= 0.0
        self.assertAlmostEqual(
            out["false_positive_rate_l0"],
            float(np.sum(no_loss & (paid > 0))) / float(np.sum(no_loss)),
        )

    def test_null_metrics(self) -> None:
        years = CatalogueYears.from_labels([1, 2], [0, 1], n_years=10)
        measure = years.measure([1, 2], [0.1, 0.1])
        out = subject_metrics([4.0, 10.0], [0.0, 0.0], measure)
        self.assertTrue(math.isnan(out["correlation"]))
        self.assertEqual(out["payout_to_loss_ratio"], 0.0)
        self.assertTrue(math.isnan(out["cvar_net_loss_b0_99"]))
        out = subject_metrics([0.0, 0.0], [1.0, 2.0], measure)
        for name in (
            "payout_to_loss_ratio",
            "coverage_ratio",
            "tail_coverage_b0_9",
            "cvar_uncovered_to_loss_b0_9",
            "false_negative_rate_l0",
        ):
            self.assertTrue(math.isnan(out[name]), name)
        self.assertEqual(out["false_positive_rate_l0"], 0.2)

    def test_arrays_must_fit_the_measure(self) -> None:
        measure = EventBasis().measure([1, 2], [0.1, 0.1])
        for loss, paid in (([1.0], [1.0, 2.0]), ([1.0, np.nan], [1.0, 2.0])):
            with self.assertRaisesRegex(ParametricInputError, "one finite value"):
                subject_metrics(loss, paid, measure)


class TestSetupText(unittest.TestCase):
    """The setup columns as text."""

    def test_index_text(self) -> None:
        wind = IndexDefinition(
            "hazard",
            unit="m/s",
            index_stat="mean",
            weight="value",
            source="TC Jamaica",
            parameters={"haz_type": "TC"},
        )
        self.assertEqual(
            index_text(wind),
            "hazard (TC Jamaica): mean, value-weighted, in m/s; haz_type=TC",
        )
        drought = IndexDefinition("hazard", unit="mm", tail="lower", index_stat=10.0)
        self.assertEqual(
            index_text(drought), "hazard: percentile 10, in mm, lower tail"
        )
        self.assertEqual(index_text(TABLE_WIND), "table (synthetic): in m/s")

    def test_pricing_text(self) -> None:
        self.assertEqual(pricing_text(PurePremium()), "pure")
        self.assertEqual(
            pricing_text(CostOfCapitalLoad(rate=0.06, q=0.99)),
            "cost_of_capital(q=0.99, rate=0.06)",
        )

    def test_a_table_states_its_setup(self) -> None:
        sample, _ = array_sample(
            [[30.0, 10.0], [60.0, 40.0]], [[4.0, 1.0], [10.0, 3.0]], [0, 1], 10
        )
        cover = ParametricCover(
            TABLE_WIND,
            StepPayout([25.0, 50.0], [2.0, 8.0]),
            unit_id=sample.unit_id,
            payout_unit="USD",
            label="steps",
        )
        table = evaluate(cover, sample, split="out_of_sample")
        self.assertEqual(
            table.columns[: len(SETUP_COLUMNS)].tolist(), list(SETUP_COLUMNS)
        )
        text = setup_text(table)
        for phrase in (
            "Split: out of sample, fold 0",
            "Basis: annual basis on 10 catalogue years (route catalogue)",
            "Scenario: baseline",
            "Index: table (synthetic): in m/s",
            "Pricing: pure; premium: priced on this sample",
            "Scope: a cover given as it is",
            "Rows: each unit's own cover; the owner's total over the units; units U0, U1",
            "Payout threshold: 0",
        ):
            self.assertIn(phrase, text)
        self.assertEqual(setup_text(table.iloc[-1]).count("\n"), 7)
        on_events, _ = array_sample(
            [[30.0, 10.0]], [[4.0, 1.0]], [0], 10, basis="event"
        )
        mixed = pd.concat([table, evaluate(cover, on_events, split="in_sample")])
        text = setup_text(mixed)
        self.assertIn("Split: out of sample, fold 0; in sample, fold 0", text)
        self.assertIn("; event basis (frequencies per year as weights)", text)

    def test_rows_without_a_setup_raise(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "rows of evaluate"):
            setup_text(pd.DataFrame())
        with self.assertRaisesRegex(ParametricInputError, "setup columns"):
            setup_text(pd.DataFrame({"split": ["in_sample"]}))


if __name__ == "__main__":
    unittest.main()
