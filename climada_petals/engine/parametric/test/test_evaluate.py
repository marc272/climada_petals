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

Tests of the evaluation: tables, premiums, rows, and payouts as an ``Impact``.

The golden cases are read with :mod:`._golden`.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from climada.engine import Impact
from climada.entity import Exposures
from climada.util.yearsets import impact_yearset_from_sampling_vect

from climada_petals.engine.parametric import (
    Budget,
    CalibrationResult,
    CostOfCapitalLoad,
    EventAlignmentError,
    EventBasis,
    ExpectedUncovered,
    IndexDefinition,
    InsuredUnits,
    LeakageError,
    MetricOptions,
    OutcomeSample,
    ParametricCover,
    ParametricIndex,
    ParametricInputError,
    PurePremium,
    StepPayout,
    UnresolvedParameterError,
    WeightedCVaR,
    YearSets,
    calibrate,
    check_no_leakage,
    compare,
    evaluate,
    holdout,
    payout_impact,
    uncovered_impact,
    var_cvar,
    year_block_kfold,
)
from climada_petals.engine.parametric.sample import check_out_of_sample
from climada_petals.engine.parametric.test._golden import (
    DATA,
    GOLDEN,
    golden_cases,
    golden_pricing,
    golden_sample,
    metric_floor,
    within,
)
from climada_petals.engine.parametric.test._synthetic import (
    TABLE_WIND,
    array_sample,
    random_portfolio,
    template,
)

PAYOUT = StepPayout.uniform(20.0, 70.0, 4)


def _golden_cover(case: dict[str, Any]) -> ParametricCover:
    """Return a case's cover, a design's template resolved with its parameters.

    A design's limit is the largest-loss rule, which the calibration resolves;
    the metrics don't read it, so the cover is evaluated without one.
    """
    cover = ParametricCover.from_dict({**case["cover"], "limit": None})
    if case["kind"] == "design":
        cover = cover.with_values(case["expected"]["parameters"])
    return cover


class TestGolden(unittest.TestCase):
    """The golden cases, within their stated tolerance."""

    def _check_case(self, case: dict[str, Any]) -> int:
        """Compare a case's metrics; return how many values were compared."""
        sample, scale = golden_sample(case)
        table = evaluate(
            _golden_cover(case),
            sample,
            split="in_sample",
            pricing=golden_pricing(case["pricing"]),
            options=GOLDEN,
        )
        metrics = case["expected"]["metrics"]
        if case["scope"] == "joint" or "expected_loss" in metrics:
            targets = {None: metrics}
        else:
            targets = metrics
        compared = 0
        for unit, values in targets.items():
            level = (
                table["level"] == "total" if unit is None else table["unit_id"] == unit
            )
            row = table[level].iloc[0]
            for name, expected in values.items():
                with self.subTest(case=case["case_id"], unit=unit, metric=name):
                    self._check_value(expected, float(row[name]), scale, name)
                compared += 1
        return compared

    def _check_value(self, expected: Any, got: float, scale: float, name: str) -> None:
        """Compare one value with the case's tolerances: 1e-6 relative, with floors."""
        if expected is None:
            self.assertTrue(math.isnan(got), f"{name} should be null, got {got}")
            return
        self.assertTrue(
            within(expected, got, metric_floor(name, scale)),
            f"{name}: {got} is not within tolerance of {expected}",
        )

    def test_the_copies_are_small_and_licensed(self) -> None:
        files = [path for path in DATA.rglob("*") if path.is_file()]
        self.assertLessEqual(sum(path.stat().st_size for path in files), 200_000)
        for path in sorted((DATA / "samples").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["licence"], "CC0-1.0", path.name)

    def test_evaluate_cases(self) -> None:
        cases = golden_cases("evaluate")
        self.assertEqual(len(cases), 16)
        compared = sum(self._check_case(case) for case in cases)
        self.assertGreater(compared, 600)

    def test_design_cases_at_their_parameters(self) -> None:
        cases = golden_cases("design")
        self.assertEqual(len(cases), 32)
        compared = sum(self._check_case(case) for case in cases)
        self.assertGreater(compared, 1200)


class TestRows(unittest.TestCase):
    """Rows, premiums and setup of covers and designs."""

    def setUp(self) -> None:
        sample, _ = random_portfolio(5, n_events=60, n_years=90)
        self.train, self.test = holdout(sample, train_fraction=0.7)
        self.cover = ParametricCover(
            TABLE_WIND,
            StepPayout.uniform(20.0, 70.0, 4, levels=[50.0, 200.0, 600.0, 1500.0]),
            unit_id=self.test.unit_id,
            payout_unit="USD",
            label="as is",
        )

    def _design(self, scope: str, pricing: Any = None) -> CalibrationResult:
        return calibrate(
            template(self.train, PAYOUT),
            self.train,
            WeightedCVaR(alpha=0.9, beta=0.9),
            scope=scope,
            pricing=pricing or PurePremium(),
            budget=Budget(ratio=1.0),
        )

    def test_a_cover_given_as_it_is(self) -> None:
        table = evaluate(self.cover, self.test, split="out_of_sample")
        self.assertEqual(table["level"].tolist(), ["unit"] * 3 + ["total"])
        self.assertEqual(table["unit_id"].tolist()[:3], ["U0", "U1", "U2"])
        self.assertTrue(pd.isna(table["unit_id"].iloc[-1]))
        self.assertTrue(table["scope"].isna().all())
        self.assertTrue((table["premium_basis"] == "evaluation").all())
        self.assertTrue((table["cover_label"] == "as is").all())
        self.assertTrue((table["cover_hash"] == self.cover.definition_hash).all())
        self.assertEqual(table["n_years"].tolist(), [27] * 4)
        self.assertEqual(set(table["route"]), {"catalogue"})
        self.assertEqual(set(table["index_definition"]), {"table (synthetic): in m/s"})
        self.assertTrue(table[["unique", "ties", "degenerate_tail"]].isna().all().all())
        # Under the pure premium the premium is the expected payout, and the unit
        # premiums add up to the total's.
        np.testing.assert_allclose(table["premium"], table["expected_payout"])
        np.testing.assert_allclose(table["premium_repriced"], table["premium"])
        units, total = table.iloc[:3], table.iloc[3]
        for name in ("expected_loss", "expected_payout", "premium"):
            self.assertAlmostEqual(units[name].sum(), total[name], places=6)
        # Shortfalls net across the units within an event.
        self.assertLessEqual(
            total["expected_uncovered"], units["expected_uncovered"].sum() + 1e-9
        )

    def test_a_cover_is_one_contract_on_its_total(self) -> None:
        loaded = CostOfCapitalLoad(rate=0.1, q=0.9)
        table = evaluate(self.cover, self.test, split="in_sample", pricing=loaded)
        measure = self.test.measure
        paid = self.cover.payout_matrix(self.test.index)
        self.assertAlmostEqual(
            table["premium"].iloc[-1], loaded.premium(paid.sum(axis=1), measure)
        )
        self.assertAlmostEqual(
            table["premium"].iloc[0], loaded.premium(paid[:, 0], measure)
        )
        self.assertEqual(table["pricing"].iloc[0], "cost_of_capital(q=0.9, rate=0.1)")
        # The net loss of a year is its loss less its payout plus the premium.
        net = measure.aggregate(
            np.asarray(self.test.loss.sum(axis=1)).ravel() - paid.sum(axis=1)
        )
        _, cvar = var_cvar(net + table["premium"].iloc[-1], measure.weights, 0.95)
        self.assertAlmostEqual(table["cvar_net_loss_b0_95"].iloc[-1], cvar)

    def test_a_stated_premium(self) -> None:
        table = evaluate(self.cover, self.test, split="out_of_sample", premium=900.0)
        self.assertEqual(table["premium"].iloc[-1], 900.0)
        self.assertEqual(
            table["premium_basis"].tolist(), ["evaluation"] * 3 + ["stated"]
        )
        self.assertAlmostEqual(
            table["premium_repriced"].iloc[-1], table["expected_payout"].iloc[-1]
        )
        one, _ = array_sample([[30.0], [60.0]], [[4.0], [10.0]], [0, 1], 10)
        single = ParametricCover(
            TABLE_WIND, StepPayout([25.0], [5.0]), unit_id=["U0"], payout_unit="USD"
        )
        table = evaluate(single, one, split="in_sample", premium=2.0)
        self.assertEqual(table["premium_basis"].tolist(), ["stated", "stated"])
        self.assertEqual(table["premium"].tolist(), [2.0, 2.0])

    def test_an_individual_design(self) -> None:
        design = self._design("individual")
        table = evaluate(design, self.test, split="out_of_sample")
        self.assertEqual(table["level"].tolist(), ["unit"] * 3 + ["total"])
        self.assertTrue((table["scope"] == "individual").all())
        self.assertTrue((table["premium_basis"] == "design").all())
        np.testing.assert_allclose(
            table["premium"].iloc[:3], [problem.premium for problem in design.problems]
        )
        self.assertAlmostEqual(table["premium"].iloc[-1], design.premium)
        # Repriced on the test years with the design's own pricing.
        np.testing.assert_allclose(table["premium_repriced"], table["expected_payout"])
        self.assertTrue(pd.isna(table["degenerate_tail"].iloc[-1]))
        self.assertEqual(
            table["cover_hash"].nunique(), 1, "one design, one cover document"
        )

    def test_a_joint_design(self) -> None:
        loaded = CostOfCapitalLoad(rate=0.1, q=0.9)
        design = self._design("joint", loaded)
        table = evaluate(design, self.test, split="out_of_sample")
        self.assertEqual(table["level"].tolist(), ["total"])
        self.assertEqual(table["scope"].iloc[0], "joint")
        self.assertEqual(table["premium"].iloc[0], design.premium)
        paid = design.cover.payout_matrix(self.test.index).sum(axis=1)
        self.assertAlmostEqual(
            table["premium_repriced"].iloc[0], loaded.premium(paid, self.test.measure)
        )
        for rule in ("value", "loss"):
            options = MetricOptions(allocation=rule)
            rows = evaluate(design, self.test, split="out_of_sample", options=options)
            with self.subTest(rule=rule):
                self.assertEqual(
                    rows["level"].tolist(), ["total"] + [f"unit_by_{rule}"] * 3
                )
                shares, total = rows.iloc[1:], rows.iloc[0]
                sums = ["expected_loss", "expected_payout"]
                sums += ["expected_uncovered"] if rule == "loss" else []
                for name in sums:
                    self.assertAlmostEqual(shares[name].sum(), total[name], places=6)
                for name in ("premium", "premium_repriced", "cvar_net_loss_b0_9"):
                    self.assertTrue(shares[name].isna().all(), name)
                self.assertTrue(shares["premium_basis"].isna().all())

    def test_value_shares_of_a_joint_design(self) -> None:
        # Units of value 1 and 3: a quarter and three quarters of each payout.
        sample, _ = array_sample(
            [[30.0, 30.0], [60.0, 10.0]], [[4.0, 0.0], [10.0, 0.0]], [0, 1], 10
        )
        exp = sample.units.exposures.copy()
        exp.gdf["value"] = [1.0, 3.0]
        units = InsuredUnits.points(exp)
        valued = OutcomeSample(
            sample.index, sample.loss, sample.frequency, units, sample.basis
        )
        design = calibrate(
            template(valued, StepPayout.uniform(20.0, 70.0, 2)),
            valued,
            ExpectedUncovered(),
            scope="joint",
        )
        table = evaluate(
            design, valued, split="in_sample", options=MetricOptions(allocation="value")
        )
        total = table["expected_payout"].iloc[0]
        np.testing.assert_allclose(
            table["expected_payout"].iloc[1:], [0.25 * total, 0.75 * total]
        )

    def test_degenerate_tails_are_recorded(self) -> None:
        # 1 - beta below one year's probability, and beta below P(L = 0).
        expected = {0.999: "single_outcome", 0.05: "expected_uncovered"}
        for beta, flag in expected.items():
            design = calibrate(
                template(self.train, PAYOUT),
                self.train,
                WeightedCVaR(alpha=0.9, beta=beta),
                scope="joint",
                budget=Budget(ratio=1.0),
            )
            table = evaluate(design, self.test, split="out_of_sample")
            with self.subTest(beta=beta):
                self.assertEqual(table["degenerate_tail"].iloc[0], flag)

    def test_in_sample_metrics_are_the_objective(self) -> None:
        for objective in (ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9)):
            design = calibrate(
                template(self.train, PAYOUT),
                self.train,
                objective,
                scope="joint",
                budget=Budget(ratio=0.8),
            )
            row = evaluate(
                design,
                self.train,
                split="in_sample",
                options=MetricOptions(betas=(0.9,)),
            ).iloc[0]
            value = design.problems[0].objective_value
            if isinstance(objective, WeightedCVaR):
                found = (
                    0.9 * row["cvar_uncovered_b0_9"] + 0.1 * row["expected_uncovered"]
                )
            else:
                found = row["expected_uncovered"]
            with self.subTest(objective=objective):
                self.assertAlmostEqual(found, value, delta=1e-7 * value)

    def test_on_the_event_basis(self) -> None:
        on_events, _ = random_portfolio(5, n_events=60, n_years=90, basis="event")
        table = evaluate(self.cover, on_events, split="in_sample")
        self.assertTrue((table["basis"] == "event").all())
        self.assertTrue((table["route"] == "frequency").all())
        self.assertTrue(table["n_years"].isna().all())
        self.assertTrue(table["net_loss_rp100"].isna().all())

    def test_folds(self) -> None:
        folds = year_block_kfold(self.train, 3)
        test = folds[1][1]
        table = evaluate(self.cover, test, split="out_of_sample")
        self.assertTrue((table["fold"] == 1).all())
        self.assertEqual(table["fold"].dtype, np.int64)
        with self.assertRaisesRegex(ParametricInputError, "fold 0 is not"):
            evaluate(self.cover, test, split="out_of_sample", fold=0)

    def test_invalid_arguments_raise(self) -> None:
        design = self._design("individual")
        unresolved = ParametricCover(
            TABLE_WIND, PAYOUT, unit_id=self.test.unit_id, payout_unit="USD"
        )
        other = ParametricCover(
            IndexDefinition("table", unit="m/s", source="elsewhere"),
            self.cover.payout[self.cover.unit_id[0]],
            unit_id=self.test.unit_id,
            payout_unit="USD",
        )
        few = ParametricCover(
            TABLE_WIND, self.cover.payout["U0"], unit_id=["U0"], payout_unit="USD"
        )
        cases = [
            ((self.cover, self.test), {"split": "test"}, "split must be"),
            ((self.cover, self.test.index), {"split": "in_sample"}, "OutcomeSample"),
            ((self.cover, self.test), {"split": "in_sample", "fold": -1}, "fold"),
            ((self.test, self.test), {"split": "in_sample"}, "an entry is"),
            ((few, self.test), {"split": "in_sample"}, "units are not"),
            ((other, self.test), {"split": "in_sample"}, "definition"),
            (
                (self.cover, self.test),
                {"split": "in_sample", "pricing": 0.1},
                "pricing",
            ),
            (
                (self.cover, self.test),
                {"split": "in_sample", "premium": -1.0},
                "premium",
            ),
            ((self.cover, self.test), {"split": "in_sample", "options": {}}, "options"),
            (
                (design, self.test),
                {"split": "in_sample", "pricing": CostOfCapitalLoad(rate=0.1, q=0.9)},
                "its own pricing",
            ),
        ]
        for allocated in (self.cover, design):
            cases.append(
                (
                    (allocated, self.test),
                    {"split": "in_sample", "options": MetricOptions(allocation="loss")},
                    "joint designs only",
                )
            )
        for args, kwargs, match in cases:
            with (
                self.subTest(kwargs=kwargs, match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                evaluate(*args, **kwargs)
        with self.assertRaises(UnresolvedParameterError):
            evaluate(unresolved, self.test, split="in_sample")
        # The design's own pricing, stated, is accepted.
        evaluate(design, self.test, split="in_sample", pricing=PurePremium())

    def test_a_cover_that_pays_at_zero_on_left_out_events_is_refused(self) -> None:
        # The third event has no loss and an index of 0: the sample leaves it out.
        sample, _ = array_sample(
            [[30.0], [60.0], [0.0]], [[4.0], [10.0], [0.0]], [0, 1, 2], 10
        )
        self.assertEqual(sample.n_left_out, 1)
        paying = ParametricCover(
            TABLE_WIND,
            StepPayout([0.0, 35.0], [1.0, 5.0]),
            unit_id=["U0"],
            payout_unit="USD",
        )
        for function in (evaluate, compare):
            entry = paying if function is evaluate else [paying]
            with self.assertRaisesRegex(ParametricInputError, "left out 1 events"):
                function(entry, sample, split="in_sample")
        for function in (payout_impact, uncovered_impact):
            with self.assertRaisesRegex(ParametricInputError, "left out 1 events"):
                function(paying, sample)
        # A design whose first step holds an index of 0 and both events, made
        # on a sample that left nothing out, pays there too.
        whole, _ = array_sample([[30.0], [60.0]], [[4.0], [10.0]], [0, 1], 10)
        design = calibrate(
            template(whole, StepPayout.uniform(0.0, 140.0, 2)),
            whole,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=0.5),
        )
        self.assertGreater(design.cover.payout["U0"].levels[0], 0.0)
        with self.assertRaisesRegex(ParametricInputError, "left out 1 events"):
            evaluate(design, sample, split="in_sample")
        # A first level of 0 pays nothing at an index of 0.
        unpaid = ParametricCover(
            TABLE_WIND,
            StepPayout([0.0, 35.0], [0.0, 5.0]),
            unit_id=["U0"],
            payout_unit="USD",
        )
        table = evaluate(unpaid, sample, split="in_sample")
        self.assertAlmostEqual(table["expected_payout"].iloc[-1], 0.5)


class TestDesigns(unittest.TestCase):
    """What the evaluation reads from a design: its fill rule and design sample."""

    def setUp(self) -> None:
        # Ten catalogue years: the training years 0 to 6 hold indices 30 and
        # 35, in the first step; the test years 7 to 9 hold 60, in the second
        # step, which no training event reaches, and 25.
        self.sample, self._impact = array_sample(
            [[30.0], [35.0], [60.0], [25.0]],
            [[4.0], [6.0], [10.0], [2.0]],
            [0, 1, 7, 8],
            10,
        )
        self.train, self.test = holdout(self.sample, train_fraction=0.7)
        self.design = calibrate(
            template(self.train, StepPayout.uniform(20.0, 70.0, 2)),
            self.train,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=0.5),
        )

    def test_unidentified_share(self) -> None:
        self.assertEqual(dict(self.design.unidentified), {"U0": ("levels[1]",)})
        in_sample = evaluate(self.design, self.train, split="in_sample")
        np.testing.assert_array_equal(in_sample["unidentified_share"], [0.0, 0.0])
        # The test year 7 of three lands on the filled level.
        table = evaluate(self.design, self.test, split="out_of_sample")
        np.testing.assert_allclose(table["unidentified_share"], [1 / 3, 1 / 3])
        cover = self.design.cover
        self.assertTrue(
            evaluate(cover, self.test, split="out_of_sample")["unidentified_share"]
            .isna()
            .all()
        )

    def test_an_overlapping_sample_raises(self) -> None:
        with self.assertRaisesRegex(LeakageError, "events are in both samples"):
            evaluate(self.design, self.train, split="out_of_sample")
        whole = calibrate(
            template(self.sample, StepPayout.uniform(20.0, 70.0, 2)),
            self.sample,
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=0.5),
        )
        with self.assertRaisesRegex(LeakageError, "events are in both samples"):
            evaluate(whole, self.test, split="out_of_sample")
        # In sample, the samples are not compared.
        evaluate(self.design, self.train, split="in_sample")
        # Disjoint sides of one split pass.
        evaluate(self.design, self.test, split="out_of_sample")
        check_no_leakage(self.design.design_sample, self.test)
        with self.assertRaisesRegex(ParametricInputError, "design must be"):
            check_no_leakage(self.design, self.test)

    def test_the_uncut_sample_of_a_design_raises(self) -> None:
        # The design sample, the training side, keeps the token of the whole
        # sample it was cut from; so does its record.
        self.assertEqual(self.design.design_sample.source, self.sample.source)
        with self.assertRaisesRegex(LeakageError, "2 events are in both samples"):
            evaluate(self.design, self.sample, split="out_of_sample")
        with self.assertRaisesRegex(LeakageError, "2 events are in both samples"):
            compare([self.design], self.sample, split="out_of_sample")
        # The same events built again from the index and the Impact, on
        # another basis, have the same token.
        index, impact = self.sample.index, self._impact
        rebuilt = OutcomeSample.from_impact(
            index, impact, units=self.sample.units, basis=EventBasis()
        )
        self.assertEqual(rebuilt.source, self.sample.source)
        with self.assertRaisesRegex(LeakageError, "events are in both samples"):
            evaluate(self.design, rebuilt, split="out_of_sample")
        evaluate(self.design, self.sample, split="in_sample")

    def test_another_hazard_set_with_the_same_ids_passes(self) -> None:
        # Another hazard set: the same event ids 1 to 4 in the same years,
        # with other index values and losses, so another source.
        other, _ = array_sample(
            [[32.0], [36.0], [61.0], [26.0]],
            [[5.0], [6.0], [9.0], [1.0]],
            [0, 1, 7, 8],
            10,
        )
        np.testing.assert_array_equal(
            other.event_id, self.train.event_id.tolist() + [3, 4]
        )
        self.assertNotEqual(other.source, self.sample.source)
        table = evaluate(self.design, other, split="out_of_sample")
        self.assertEqual(list(table["split"]), ["out_of_sample"] * 2)
        # Its training side shares the design sample's event ids and years by
        # chance, and passes too: only samples of one source are compared.
        other_train, _ = holdout(other, train_fraction=0.7)
        np.testing.assert_array_equal(other_train.event_id, self.train.event_id)
        evaluate(self.design, other_train, split="out_of_sample")
        # A record without a token is compared with no sample.
        record = dataclasses.replace(self.design.design_sample, source=None)
        check_out_of_sample(record, self.train)
        with self.assertRaisesRegex(ParametricInputError, "design must be"):
            check_out_of_sample(self.design, self.train)
        with self.assertRaisesRegex(ParametricInputError, "sample must be"):
            check_out_of_sample(record, record)


class TestCompare(unittest.TestCase):
    """Several entries in one table."""

    def setUp(self) -> None:
        sample, _ = random_portfolio(7, n_events=50, n_years=80)
        self.train, self.test = holdout(sample, train_fraction=0.7)
        self.designs = [
            calibrate(
                template(self.train, PAYOUT),
                self.train,
                ExpectedUncovered(),
                scope=scope,
                budget=Budget(ratio=1.0),
            )
            for scope in ("individual", "joint")
        ]
        self.cover = ParametricCover(
            TABLE_WIND,
            StepPayout([30.0, 50.0], [100.0, 900.0]),
            unit_id=self.test.unit_id,
            payout_unit="USD",
        )

    def test_rows_of_every_entry(self) -> None:
        table = compare([self.cover, *self.designs], self.test, split="out_of_sample")
        self.assertEqual(len(table), 4 + 4 + 1)
        self.assertTrue(pd.isna(table["scope"].iloc[0]))
        self.assertEqual(table["scope"].iloc[[4, 8]].tolist(), ["individual", "joint"])
        pd.testing.assert_frame_equal(
            table.iloc[4:8].reset_index(drop=True),
            evaluate(self.designs[0], self.test, split="out_of_sample"),
        )

    def test_allocation_takes_joint_designs_only(self) -> None:
        options = MetricOptions(allocation="value")
        table = compare(
            [self.designs[1]], self.test, split="out_of_sample", options=options
        )
        self.assertEqual(table["level"].tolist(), ["total"] + ["unit_by_value"] * 3)
        with self.assertRaisesRegex(ParametricInputError, r"at \[0, 1\] are not"):
            compare(
                [self.cover, *self.designs],
                self.test,
                split="in_sample",
                options=options,
            )

    def test_invalid_entries_raise(self) -> None:
        for entries, match in ((self.cover, "sequence"), ([], "at least one")):
            with self.assertRaisesRegex(ParametricInputError, match):
                compare(entries, self.test, split="in_sample")


class TestImpacts(unittest.TestCase):
    """Payouts and uncovered losses as CLIMADA ``Impact`` objects."""

    def setUp(self) -> None:
        self.sample, self.impact = random_portfolio(4)
        self.design = calibrate(
            template(self.sample, PAYOUT),
            self.sample,
            WeightedCVaR(alpha=0.5, beta=0.9),
            scope="joint",
            budget=Budget(ratio=1.0),
        )

    def test_payouts_are_a_climada_impact(self) -> None:
        paid = payout_impact(self.design, self.sample)
        self.assertIsInstance(paid, Impact)
        matrix = self.design.cover.payout_matrix(self.sample.index)
        np.testing.assert_allclose(paid.imp_mat.toarray(), matrix)
        np.testing.assert_allclose(paid.at_event, matrix.sum(axis=1))
        np.testing.assert_allclose(paid.eai_exp, self.sample.frequency @ matrix)
        np.testing.assert_array_equal(paid.event_id, self.sample.event_id)
        self.assertEqual(paid.unit, "USD")
        table = evaluate(self.design, self.sample, split="in_sample")
        self.assertAlmostEqual(paid.aai_agg, table["expected_payout"].iloc[0])
        curve = paid.calc_freq_curve()
        self.assertAlmostEqual(float(np.max(curve.impact)), float(paid.at_event.max()))
        exceedance = paid.local_exceedance_impact(return_periods=(10, 50))[0]
        self.assertEqual(len(exceedance), self.sample.units.n_points)

    def test_aai_agg_is_the_expected_payout_from_the_frequencies(self) -> None:
        def expected(sample: OutcomeSample) -> float:
            table = evaluate(self.design, sample, split="in_sample")
            return float(table["expected_payout"].iloc[0])

        def frequency_weighted(sample: OutcomeSample) -> float:
            paid = self.design.cover.payout_matrix(sample.index).sum(axis=1)
            return float(sample.frequency @ paid)

        events = OutcomeSample.from_impact(
            self.sample.index, self.impact, units=self.sample.units, basis=EventBasis()
        )
        on_years = OutcomeSample.from_impact(
            self.sample.index,
            self.impact,
            units=self.sample.units,
            basis=YearSets.from_frequencies(
                self.impact, n_years=50, rng=np.random.default_rng(393)
            ),
        )
        train, _ = holdout(self.sample, train_fraction=0.7)
        for sample in (events, self.sample, on_years, train):
            with self.subTest(route=sample.route, split=sample.split):
                self.assertAlmostEqual(
                    payout_impact(self.design, sample).aai_agg
                    / frequency_weighted(sample),
                    1.0,
                    places=12,
                )
        # The event basis, and catalogue years on a whole sample, whose
        # events have frequency 1 / N: the evaluation's expected payout.
        for sample in (events, self.sample):
            self.assertAlmostEqual(
                payout_impact(self.design, sample).aai_agg / expected(sample),
                1.0,
                places=12,
            )
        # Year sets: the mean over 50 sampled years differs from it.
        self.assertGreater(
            abs(payout_impact(self.design, on_years).aai_agg / expected(on_years) - 1),
            1e-3,
        )
        # A side of a catalogue split: its years weigh 1 / N_side, its events
        # keep 1 / N.
        self.assertAlmostEqual(
            payout_impact(self.design, train).aai_agg / expected(train),
            train.years.size / self.sample.years.size,
            places=12,
        )

    def test_the_loss_impact_is_read_without_deprecated_attributes(self) -> None:
        logger = logging.getLogger("climada")
        found = []
        with self.assertLogs(logger, "WARNING") as logs:
            for function in (payout_impact, uncovered_impact):
                found.append(function(self.design, self.sample, impact=self.impact))
            logger.warning("done")
        self.assertEqual([line for line in logs.output if "tot_value" in line], [])
        for impact in (*found, payout_impact(self.design, self.sample)):
            self.assertEqual(impact.tot_value, float(self.sample.units.value.sum()))

    def test_a_write_hdf5_round_trip(self) -> None:
        cases = itertools.product(
            (payout_impact, uncovered_impact), (None, self.impact)
        )
        for function, impact in cases:
            written = function(self.design, self.sample, impact=impact)
            with (
                tempfile.TemporaryDirectory() as folder,
                self.subTest(function=function, impact=impact is not None),
            ):
                path = Path(folder) / "impact.hdf5"
                written.write_hdf5(path)
                read = Impact.from_hdf5(path)
                np.testing.assert_allclose(
                    read.imp_mat.toarray(), written.imp_mat.toarray()
                )
                np.testing.assert_allclose(read.eai_exp, written.eai_exp)
                np.testing.assert_array_equal(read.event_id, written.event_id)
                np.testing.assert_array_equal(read.date, written.date)
                np.testing.assert_allclose(read.frequency, written.frequency)
                self.assertEqual(read.unit, written.unit)

    def test_uncovered_losses(self) -> None:
        shortfall = uncovered_impact(self.design, self.sample)
        loss = self.sample.loss.toarray()
        paid = self.design.cover.payout_matrix(self.sample.index)
        np.testing.assert_allclose(
            shortfall.imp_mat.toarray(), np.maximum(loss - paid, 0)
        )
        self.assertTrue(np.all(shortfall.imp_mat.data >= 0))
        cover = ParametricCover(
            TABLE_WIND,
            PAYOUT.with_values(
                {f"levels[{k}]": v for k, v in enumerate((50.0, 200.0, 600.0, 900.0))}
            ),
            unit_id=self.sample.unit_id,
            payout_unit="USD",
        )
        table = evaluate(cover, self.sample, split="in_sample")
        self.assertAlmostEqual(
            uncovered_impact(cover, self.sample).aai_agg,
            table["expected_uncovered"].iloc[:3].sum(),
        )

    def test_the_loss_impact_gives_events_dates_and_order(self) -> None:
        self.impact.date = np.arange(1, self.impact.event_id.size + 1) * 1000
        paid = payout_impact(self.design, self.sample, impact=self.impact)
        np.testing.assert_array_equal(paid.event_id, self.impact.event_id)
        np.testing.assert_array_equal(paid.date, self.impact.date)
        self.assertEqual(paid.event_name, list(self.impact.event_name))
        # The same sampling_vect gives the payouts' annual totals.
        sampling_vect = [np.array([0, 3]), np.array([], dtype=int), np.array([5, 5, 7])]
        years = impact_yearset_from_sampling_vect(
            paid, [2001, 2002, 2003], sampling_vect, correction_fac=False
        )
        at_event = paid.at_event
        np.testing.assert_allclose(
            years.at_event,
            [at_event[0] + at_event[3], 0.0, 2 * at_event[5] + at_event[7]],
        )
        on_years = OutcomeSample.from_impact(
            self.sample.index,
            self.impact,
            units=self.sample.units,
            basis=YearSets.from_sampling_vect(self.impact.event_id, sampling_vect),
        )
        table = evaluate(self.design, on_years, split="in_sample")
        self.assertAlmostEqual(table["expected_payout"].iloc[0], years.at_event.mean())

    def test_left_out_events_pay_nothing_in_the_loss_impact(self) -> None:
        sample, impact = array_sample(
            [[30.0], [60.0], [0.0]], [[4.0], [10.0], [0.0]], [0, 1, 2], 10
        )
        cover = ParametricCover(
            TABLE_WIND,
            StepPayout([25.0, 50.0], [1.0, 5.0]),
            unit_id=["U0"],
            payout_unit="USD",
        )
        paid = payout_impact(cover, sample, impact=impact)
        np.testing.assert_allclose(paid.at_event, [1.0, 5.0, 0.0])
        np.testing.assert_allclose(
            uncovered_impact(cover, sample, impact=impact).at_event, [3.0, 5.0, 0.0]
        )
        self.assertEqual(payout_impact(cover, sample).event_id.tolist(), [1, 2])

    def test_an_impact_that_does_not_fit_raises(self) -> None:
        train, _ = holdout(self.sample, train_fraction=0.7)
        with self.assertRaisesRegex(ParametricInputError, "whole sample"):
            payout_impact(self.design, train, impact=self.impact)
        other, other_impact = random_portfolio(4, n_events=41)
        with self.assertRaisesRegex(EventAlignmentError, "beyond the sample's"):
            payout_impact(self.design, self.sample, impact=other_impact)
        fewer = self.impact.select(event_ids=self.impact.event_id[1:])
        fewer.imp_mat = self.impact.imp_mat[1:]
        with self.assertRaisesRegex(EventAlignmentError, "in the impact once"):
            payout_impact(self.design, self.sample, impact=fewer)
        rescaled = self.impact.select(event_ids=self.impact.event_id)
        rescaled.frequency = self.impact.frequency * 2
        with self.assertRaisesRegex(ParametricInputError, "frequencies"):
            payout_impact(self.design, self.sample, impact=rescaled)
        self.assertEqual(other.n_events, 41)

    def test_draws_need_a_draw_for_their_uncovered_impact(self) -> None:
        drawn = OutcomeSample(
            self.sample.index,
            np.vstack([self.sample.loss.toarray()] * 2),
            self.sample.frequency,
            self.sample.units,
            self.sample.basis,
            n_draws=2,
        )
        self.assertEqual(payout_impact(self.design, drawn).imp_mat.shape[0], 40)
        with self.assertRaisesRegex(ParametricInputError, "2 draws"):
            uncovered_impact(self.design, drawn)
        table = evaluate(self.design, drawn, split="in_sample")
        self.assertAlmostEqual(
            table["expected_uncovered"].iloc[0],
            evaluate(self.design, self.sample, split="in_sample")[
                "expected_uncovered"
            ].iloc[0],
        )


if __name__ == "__main__":
    unittest.main()
