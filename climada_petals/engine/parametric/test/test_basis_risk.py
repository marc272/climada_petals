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

Tests of basis risk: the index diagnostics and their warnings, impact-function
draws, samples with draws, and the limit of the draws.

The CLIMADA case is a flood on four centroids, one exposure point at each,
an index read at each point's centroid and one impact function that does no
harm up to 1 m and grows linearly to the whole value at 5 m: the degenerate
set-up, where the index determines the loss.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.engine import Impact, ImpactCalc
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from scipy.stats import spearmanr

from climada_petals.engine.parametric import (
    MSE,
    Budget,
    CalibrationOptions,
    CatalogueYears,
    EventAlignmentError,
    EventBasis,
    ExpectedUncovered,
    Free,
    ImpactInputError,
    IndexDefinition,
    InsuredUnits,
    Limit,
    MetricOptions,
    OutcomeSample,
    ParametricIndex,
    ParametricInputError,
    StepPayout,
    YearSets,
    calibrate,
    calibrate_frontier,
    compare,
    evaluate,
    holdout,
    impact_function_draws,
    index_diagnostics,
    uncovered_impact,
)
from climada_petals.engine.parametric.basis_risk import (
    DIAGNOSTIC_COLUMNS,
    MONOTONE_TOLERANCE,
    comonotone_flags,
    comonotone_of,
    diagnose,
)
from climada_petals.engine.parametric.test._synthetic import (
    TABLE_WIND,
    array_index,
    array_sample,
    grid_hazard,
    random_portfolio,
    template,
)
from climada_petals.util.config import LOGGER

#: Flood depths (m) of six events at the four centroids of ``grid_hazard``.
DEPTH = (
    (0.5, 2.0, 0.0, 3.0),
    (1.5, 0.8, 2.5, 0.0),
    (3.0, 0.0, 1.2, 4.0),
    (0.0, 3.5, 0.6, 1.8),
    (4.5, 1.1, 3.8, 0.9),
    (2.2, 4.2, 0.0, 2.6),
)


def _impf(intensity: Any, mdd: Any, *, haz_type: str = "FL", fun_id: int = 1) -> Any:
    """Return an impact function with ``paa`` 1."""
    return ImpactFunc(
        haz_type=haz_type,
        id=fun_id,
        intensity=np.asarray(intensity, dtype=float),
        mdd=np.asarray(mdd, dtype=float),
        paa=np.ones(len(intensity)),
        intensity_unit="m",
        name="test",
    )


#: No harm up to 1 m, then linear to the whole value at 5 m.
FLOOD_IMPF = ImpactFuncSet([_impf([0.0, 1.0, 5.0], [0.0, 0.0, 1.0])])


class _Flood:
    """The CLIMADA case of the module docstring."""

    def __init__(self) -> None:
        self.hazard = grid_hazard(
            DEPTH, event_id=range(1, 7), event_name=[f"e{k}" for k in range(1, 7)]
        )
        self.exposures = Exposures(
            lat=[10.0, 10.0, 11.0, 11.0],
            lon=[20.0, 21.0, 20.0, 21.0],
            value=[1, 2, 3, 4],
        )
        self.exposures.gdf["impf_FL"] = 1
        self.units = InsuredUnits.points(self.exposures)
        self.index = ParametricIndex.from_hazard(self.hazard, self.units)

    def impact(self, impf_set: ImpactFuncSet = FLOOD_IMPF) -> Impact:
        """Return the loss with ``impf_set``."""
        return ImpactCalc(self.exposures.copy(), impf_set, self.hazard).impact(
            save_mat=True
        )

    def sample(self, impacts: list[Impact] | None = None) -> OutcomeSample:
        """Return the sample on the event basis, with draws if ``impacts`` is given."""
        if impacts is None:
            return OutcomeSample.from_impact(
                self.index, self.impact(), units=self.units, basis=EventBasis()
            )
        return OutcomeSample.from_impact_draws(
            self.index, impacts, units=self.units, basis=EventBasis()
        )


def _flagged(test: unittest.TestCase, call: Callable[[], Any]) -> tuple[Any, list[str]]:
    """Return what ``call`` returns, and the index diagnostics' warnings it logged."""
    with test.assertLogs(LOGGER, "INFO") as logs:
        found = call()
        LOGGER.info("done")
    return found, [line for line in logs.output if "one-to-one" in line]


def _impact_like(
    reference: Impact, event_id: Any, losses: Any, **changes: Any
) -> Impact:
    """Return an ``Impact`` on ``reference``'s points with other events and losses."""
    losses = np.asarray(losses, dtype=float)
    frequency = np.asarray(changes.get("frequency", np.full(len(event_id), 0.1)))
    return Impact(
        event_id=np.asarray(event_id),
        event_name=[f"e{k}" for k in event_id],
        date=np.ones(len(event_id), dtype=int),
        frequency=frequency,
        coord_exp=np.asarray(changes.get("coord_exp", reference.coord_exp)),
        eai_exp=frequency @ losses,
        at_event=losses.sum(axis=1),
        aai_agg=float((frequency @ losses).sum()),
        imp_mat=sp.csr_matrix(losses),
        unit=changes.get("unit", "USD"),
    )


class TestIndexDiagnostics(unittest.TestCase):
    """The diagnostics' values by hand, the weights, the tail and the draws."""

    def test_three_units_by_hand(self) -> None:
        # U0: the loss rises with the index; event 5 has neither and is left out.
        # U1: a loss of 5 at an index of 0, and the loss falls as the index rises.
        # U2: the index takes one value where there is a loss or an index.
        index, impact, units = array_index(
            [[10, 0, 5], [20, 30, 5], [30, 20, 5], [40, 10, 0], [0, 50, 0]],
            [[0, 5, 1], [1, 1, 2], [4, 2, 3], [9, 3, 0], [0, 0, 0]],
        )
        sample = OutcomeSample.from_impact(
            index, impact, units=units, basis=EventBasis()
        )
        table = index_diagnostics(sample)
        self.assertEqual(list(table.columns), list(DIAGNOSTIC_COLUMNS))
        self.assertEqual(table["unit_id"].tolist(), ["U0", "U1", "U2"])
        self.assertEqual(table["n_events"].tolist(), [4, 5, 3])
        self.assertAlmostEqual(table["spearman"].iloc[0], 1.0, places=12)
        self.assertAlmostEqual(table["spearman"].iloc[1], -1.0, places=12)
        self.assertTrue(math.isnan(table["spearman"].iloc[2]))
        self.assertEqual(table["comonotone_index"].tolist(), [True, False, None])
        np.testing.assert_allclose(
            table["zero_index_loss_share"], [0.0, 5.0 / 11.0, 0.0], atol=1e-15
        )
        self.assertEqual(table["threshold"].tolist(), [0.99] * 3)
        # U1: from the loss of 5 at index 0 to 0 at index 50, over 5. U2: the
        # losses 1, 2 and 3 at one index value, a fall of 2, over 3.
        np.testing.assert_allclose(
            table["monotone_gap"], [0.0, 1.0, 2.0 / 3.0], atol=1e-15
        )
        self.assertEqual(table["tolerance"].tolist(), [MONOTONE_TOLERANCE] * 3)
        self.assertEqual(table["comonotone_index"].dtype, object)

    def test_a_non_decreasing_loss_is_flagged_whatever_its_ranks(self) -> None:
        # The loss is a non-decreasing function of the index, but two events
        # below the impact function's threshold lose nothing: their tied ranks
        # take the Spearman correlation to 0.9747, below the threshold.
        index, impact, units = array_index(
            [[10.0], [20.0], [30.0], [40.0], [50.0]], [[0], [0], [1], [2], [3]]
        )
        sample = OutcomeSample.from_impact(
            index, impact, units=units, basis=EventBasis()
        )
        expected = spearmanr([10, 20, 30, 40, 50], [0, 0, 1, 2, 3]).statistic
        table = index_diagnostics(sample)
        self.assertAlmostEqual(table["spearman"].iloc[0], float(expected), places=12)
        self.assertAlmostEqual(table["spearman"].iloc[0], 0.9747, places=4)
        self.assertEqual(table["monotone_gap"].iloc[0], 0.0)
        self.assertIs(table["comonotone_index"].iloc[0], True)
        # A fall of 0.5 at the top is no function any more, and the ranks
        # stay below the threshold.
        index, impact, units = array_index(
            [[10.0], [20.0], [30.0], [40.0], [50.0]], [[0], [0], [1], [3], [2.5]]
        )
        table = index_diagnostics(
            OutcomeSample.from_impact(index, impact, units=units, basis=EventBasis())
        )
        self.assertAlmostEqual(table["monotone_gap"].iloc[0], 0.5 / 3.0, places=12)
        self.assertIs(table["comonotone_index"].iloc[0], False)

    def test_the_tolerance_is_stated(self) -> None:
        # Falls of about 6e-8 and 1e-6 of the largest loss (rounding, say),
        # within the default tolerance of 1e-6, and of 1.5e-5, beyond it.
        for fall, flag in ((2.0**-24, True), (2.0**-20, True), (2.0**-16, False)):
            values = [[10.0], [20.0], [30.0], [40.0]]
            loss = [[0.5], [1.0], [1.0 - fall], [1.0]]
            index, _, units = array_index(values, loss)
            sample = OutcomeSample(index, loss, [0.1] * 4, units, EventBasis())
            table = index_diagnostics(sample)
            with self.subTest(fall=fall):
                self.assertEqual(table["monotone_gap"].iloc[0], fall)
                self.assertLess(table["spearman"].iloc[0], 0.99)
                self.assertIs(table["comonotone_index"].iloc[0], flag)
        table = index_diagnostics(sample, tolerance=1e-4)
        self.assertIs(table["comonotone_index"].iloc[0], True)
        self.assertEqual(table["tolerance"].iloc[0], 1e-4)
        for tolerance in (-1e-6, 1.0, "small", math.nan):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ParametricInputError, "tolerance"):
                    index_diagnostics(sample, tolerance=tolerance)

    def test_one_point_per_unit_against_a_group(self) -> None:
        # Flood depths drawn at four centroids, one exposure point at each and
        # one impact function with no harm up to 1 m: about half of the rows
        # with a depth lose nothing. Each point's loss is a function of
        # its depth, flagged though its Spearman correlation is below 0.99;
        # the four points as one unit, on their value-weighted mean depth,
        # are not.
        rng = np.random.default_rng(393)
        depth = np.round(rng.uniform(0.0, 2.0, size=(40, 4)), 2)
        depth[rng.uniform(size=depth.shape) < 0.2] = 0.0
        hazard = grid_hazard(
            depth, event_id=range(1, 41), event_name=[f"e{k}" for k in range(40)]
        )
        exposures = _Flood().exposures
        impact = ImpactCalc(exposures.copy(), FLOOD_IMPF, hazard).impact(save_mat=True)
        points = InsuredUnits.points(exposures)
        one = index_diagnostics(
            OutcomeSample.from_impact(
                ParametricIndex.from_hazard(hazard, points),
                impact,
                units=points,
                basis=EventBasis(),
            )
        )
        np.testing.assert_array_equal(one["monotone_gap"], 0.0)
        self.assertTrue(np.all(one["spearman"] < 0.99))
        self.assertEqual(one["comonotone_index"].tolist(), [True] * 4)
        whole = InsuredUnits.whole(exposures, unit_id="all")
        group = index_diagnostics(
            OutcomeSample.from_impact(
                ParametricIndex.from_hazard(
                    hazard, whole, index_stat="mean", weight="value"
                ),
                impact,
                units=whole,
                basis=EventBasis(),
            )
        )
        self.assertGreater(group["monotone_gap"].iloc[0], 0.01)
        self.assertLess(group["spearman"].iloc[0], 0.99)
        self.assertIs(group["comonotone_index"].iloc[0], False)

    def test_the_weights_are_the_frequencies(self) -> None:
        # Mid-ranks of the index 0.05, 0.2, 0.45 and of the loss 0.25, 0.1, 0.45
        # under the weights 0.1, 0.2, 0.3: their weighted Pearson is 0.8. With
        # equal weights the ranks give 1 - 6 * 2 / 24 = 0.5.
        values, loss = [[1.0], [2.0], [3.0]], [[2.0], [1.0], [3.0]]
        index, _, units = array_index(values, loss)
        weighted = OutcomeSample(index, loss, [0.1, 0.2, 0.3], units, EventBasis())
        equal = OutcomeSample(index, loss, [0.1, 0.1, 0.1], units, EventBasis())
        self.assertAlmostEqual(
            index_diagnostics(weighted)["spearman"].iloc[0], 0.8, places=12
        )
        self.assertAlmostEqual(
            index_diagnostics(equal)["spearman"].iloc[0], 0.5, places=12
        )
        # Integer weights are repeated rows, and a row of weight 0 counts for none.
        repeated_values = [[1.0], [2.0], [2.0], [3.0], [3.0], [3.0], [4.0]]
        repeated_loss = [[2.0], [1.0], [1.0], [3.0], [3.0], [3.0], [0.5]]
        index, _, units = array_index(repeated_values, repeated_loss)
        repeated = OutcomeSample(
            index, repeated_loss, [0.1] * 6 + [0.0], units, EventBasis()
        )
        expected = spearmanr(np.ravel(repeated_values[:6]), np.ravel(repeated_loss[:6]))
        self.assertAlmostEqual(
            index_diagnostics(repeated)["spearman"].iloc[0],
            float(expected.statistic),
            places=12,
        )
        self.assertAlmostEqual(float(expected.statistic), 0.8, places=12)

    def test_equal_weights_give_scipys_spearman_with_ties(self) -> None:
        sample, _ = random_portfolio(3, n_events=60, n_units=2)
        table = index_diagnostics(sample)
        loss = sample.loss.toarray()
        for j in range(2):
            rows = (loss[:, j] > 0) | (sample.index.values[:, j] != 0)
            expected = spearmanr(sample.index.values[rows, j], loss[rows, j])
            with self.subTest(unit=j):
                self.assertAlmostEqual(
                    table["spearman"].iloc[j], float(expected.statistic), places=12
                )
                self.assertFalse(table["comonotone_index"].iloc[j])

    def test_a_lower_tail_index_is_oriented(self) -> None:
        # A drought index: the loss rises as the index falls.
        values, loss = [[10.0], [8.0], [5.0], [2.0]], [[0.0], [1.0], [3.0], [6.0]]
        lower = IndexDefinition("table", unit="1", source="synthetic", tail="lower")
        flagged, gaps = [], []
        for definition in (lower, TABLE_WIND):
            index, impact, units = array_index(values, loss, definition=definition)
            sample = OutcomeSample.from_impact(
                index, impact, units=units, basis=EventBasis()
            )
            table = index_diagnostics(sample)
            self.assertAlmostEqual(table["spearman"].iloc[0], -1.0, places=12)
            flagged.append(table["comonotone_index"].iloc[0])
            gaps.append(table["monotone_gap"].iloc[0])
        self.assertEqual(flagged, [True, False])
        # The upper tail reads a fall from 6 at index 2 to 0 at index 10.
        self.assertEqual(gaps, [0.0, 1.0])

    def test_draws_are_rows_of_their_event(self) -> None:
        # Two events in two draws. Event 3 has no index and a loss in draw 1
        # only: it counts, with both its draws. Event 4 has neither.
        values = [[10.0], [20.0], [0.0], [0.0]]
        loss = [[1.0], [2.0], [0.0], [0.0], [3.0], [0.5], [2.0], [0.0]]
        index, _, units = array_index(values, np.zeros((4, 1)))
        sample = OutcomeSample(index, loss, [0.1] * 4, units, EventBasis(), n_draws=2)
        table = index_diagnostics(sample)
        self.assertEqual(table["n_events"].iloc[0], 3)
        rows = [0, 1, 2, 4, 5, 6]
        expected = spearmanr(
            np.tile(np.ravel(values), 2)[rows], np.ravel(loss)[rows]
        ).statistic
        self.assertAlmostEqual(table["spearman"].iloc[0], float(expected), places=12)
        # E[L] = 0.05 (1 + 2 + 3 + 0.5 + 2); on index 0: 0.05 * 2.
        self.assertAlmostEqual(
            table["zero_index_loss_share"].iloc[0], 2.0 / 8.5, places=12
        )
        # Two losses at one index value are a fall: event 3's 0 and 2 at an
        # index of 0, event 1's 1 and 3 at 10, the largest from 3 to 0.5 at 20.
        self.assertAlmostEqual(table["monotone_gap"].iloc[0], 2.5 / 3.0, places=12)
        self.assertIs(table["comonotone_index"].iloc[0], False)

    def test_a_unit_without_loss_has_no_share(self) -> None:
        index, _, units = array_index([[10.0, 1.0], [20.0, 2.0]], np.zeros((2, 2)))
        loss = [[1.0, 0.0], [2.0, 0.0]]
        table = index_diagnostics(
            OutcomeSample(index, loss, [0.1, 0.1], units, EventBasis())
        )
        self.assertTrue(math.isnan(table["zero_index_loss_share"].iloc[1]))
        self.assertIsNone(table["comonotone_index"].iloc[1])

    def test_the_threshold_is_stated(self) -> None:
        sample, _ = random_portfolio(3, n_units=1)
        rho = index_diagnostics(sample)["spearman"].iloc[0]
        self.assertGreater(index_diagnostics(sample)["monotone_gap"].iloc[0], 0.01)
        for threshold, flag in ((rho, True), (rho + 1e-6, False), (1.0, False)):
            table = index_diagnostics(sample, threshold=threshold)
            with self.subTest(threshold=threshold):
                self.assertEqual(table["comonotone_index"].iloc[0], flag)
                self.assertEqual(table["threshold"].iloc[0], threshold)
        for threshold in (0.0, 1.5, "high", math.nan):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ParametricInputError, "threshold"):
                    index_diagnostics(sample, threshold=threshold)
        with self.assertRaisesRegex(ParametricInputError, "OutcomeSample"):
            index_diagnostics("sample")

    def test_flags_combine_as_and_with_an_unknown(self) -> None:
        table = pd.DataFrame(
            {
                "unit_id": ["a", "b", "c", "d"],
                "comonotone_index": [True, False, None, math.nan],
            }
        )
        self.assertEqual(
            comonotone_flags(table), {"a": True, "b": False, "c": None, "d": None}
        )
        cases = {("a",): True, ("a", "b"): False, ("a", "c"): None, ("b", "c"): False}
        for units, expected in cases.items():
            with self.subTest(units=units):
                self.assertIs(comonotone_of(table, units), expected)
        self.assertIsNone(comonotone_of(None, ["a"]))

    def test_diagnose_warns_only_of_flagged_units(self) -> None:
        sample, _ = random_portfolio(3, n_units=2)
        table, warnings = _flagged(self, lambda: diagnose(sample, "here"))
        self.assertEqual(warnings, [])
        pd.testing.assert_frame_equal(table, index_diagnostics(sample))
        flood = _Flood().sample()
        _, warnings = _flagged(self, lambda: diagnose(flood, "here"))
        self.assertEqual(len(warnings), 1)
        self.assertIn("here: at 4 of 4 units", warnings[0])
        self.assertIn("comonotone_index", warnings[0])


class TestWarningsAndColumns(unittest.TestCase):
    """calibrate, calibrate_frontier, evaluate and compare run the diagnostics."""

    def setUp(self) -> None:
        self.flood = _Flood()
        self.sample = self.flood.sample()
        self.template = template(self.sample, StepPayout.uniform(1.0, 4.0, 3))

    def _design(self, scope: str = "individual") -> Any:
        return calibrate(
            self.template,
            self.sample,
            ExpectedUncovered(),
            scope=scope,
            budget=Budget(ratio=1.0),
        )

    def test_the_degenerate_case_warns_in_calibration(self) -> None:
        result, warnings = _flagged(self, self._design)
        self.assertEqual(len(warnings), 1)
        self.assertIn("calibrate: at 4 of 4 units", warnings[0])
        pd.testing.assert_frame_equal(
            result.index_diagnostics, index_diagnostics(self.sample)
        )
        self.assertEqual(result.summary()["comonotone_index"].tolist(), [True] * 4)
        joint, _ = _flagged(self, lambda: self._design("joint"))
        self.assertEqual(joint.summary()["comonotone_index"].tolist(), [True])

    def test_the_degenerate_case_warns_in_evaluation(self) -> None:
        design = self._design()
        table, warnings = _flagged(
            self, lambda: evaluate(design, self.sample, split="in_sample")
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("evaluate: at 4 of 4 units", warnings[0])
        self.assertEqual(table["level"].tolist(), ["unit"] * 4 + ["total"])
        self.assertEqual(table["comonotone_index"].tolist(), [True] * 5)

    def test_compare_and_the_frontier_warn_once(self) -> None:
        design = self._design()
        table, warnings = _flagged(
            self,
            lambda: compare([design, design.cover], self.sample, split="in_sample"),
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("compare:", warnings[0])
        self.assertEqual(len(table), 10)
        frontier, warnings = _flagged(
            self,
            lambda: calibrate_frontier(
                self.template,
                self.sample,
                ExpectedUncovered(),
                budgets=[Budget(ratio=0.5), Budget(ratio=1.0)],
                scope="individual",
            ),
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("calibrate_frontier:", warnings[0])
        self.assertEqual(frontier["comonotone_index"].tolist(), [True] * 8)

    def test_rows_of_mixed_units(self) -> None:
        # U0 flagged, U1 not (the loss falls as the index rises), U2 undefined.
        sample, _ = array_sample(
            [[30, 30, 40], [40, 50, 40], [50, 40, 40], [60, 60, 0]],
            [[1, 6, 1], [2, 2, 2], [3, 4, 3], [4, 1, 0]],
            [0, 1, 2, 3],
            10,
        )
        cover = template(sample, StepPayout([35.0], [2.0]), limit=None)
        table, warnings = _flagged(
            self, lambda: evaluate(cover, sample, split="in_sample")
        )
        self.assertIn("at 1 of 3 units, e.g. ['U0']", warnings[0])
        self.assertEqual(table["comonotone_index"].tolist(), [True, False, None, False])
        design = calibrate(
            template(sample, StepPayout([35.0], [Free()])),
            sample,
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        self.assertEqual(design.summary()["comonotone_index"].tolist(), [False])
        allocated = evaluate(
            design, sample, split="in_sample", options=MetricOptions(allocation="value")
        )
        self.assertEqual(
            allocated["comonotone_index"].tolist(), [False, True, False, None]
        )

    def test_no_warning_without_a_flagged_unit(self) -> None:
        sample, _ = random_portfolio(5)
        cover = template(sample, StepPayout.uniform(30.0, 70.0, 3))
        result, warnings = _flagged(
            self,
            lambda: calibrate(
                cover, sample, MSE(), scope="joint", budget=Budget(ratio=1.0)
            ),
        )
        self.assertEqual(warnings, [])
        table = evaluate(result, sample, split="in_sample")
        self.assertEqual(table["comonotone_index"].tolist(), [False])


class TestImpactFunctionDraws(unittest.TestCase):
    """The helper: the stretch above the threshold, by hand, and its checks."""

    def setUp(self) -> None:
        self.rising = _impf([0.0, 10.0, 20.0, 30.0], [0.0, 0.0, 0.5, 1.0])
        self.at_once = _impf([0.0, 1.0, 2.0], [0.2, 0.5, 1.0], haz_type="DR")
        self.harmless = _impf([0.0, 5.0], [0.0, 0.0], fun_id=2)
        self.impf_set = ImpactFuncSet([self.rising, self.at_once, self.harmless])

    def test_a_fixed_factor_by_hand(self) -> None:
        (drawn,) = impact_function_draws(
            self.impf_set, n_draws=1, scale=(2.0, 2.0), rng=np.random.default_rng(1)
        )
        # The thresholds: 10, the first intensity 0, and the last intensity 5.
        np.testing.assert_array_equal(
            drawn.get_func("FL", 1).intensity, [0.0, 10.0, 30.0, 50.0]
        )
        np.testing.assert_array_equal(
            drawn.get_func("DR", 1).intensity, [0.0, 2.0, 4.0]
        )
        np.testing.assert_array_equal(drawn.get_func("FL", 2).intensity, [0.0, 5.0])
        # At 30 the drawn function takes the original's value at 10 + 20 / 2.
        self.assertAlmostEqual(float(drawn.get_func("FL", 1).calc_mdr(30.0)), 0.5)
        np.testing.assert_array_equal(drawn.get_func("FL", 1).mdd, self.rising.mdd)
        # The input is not changed.
        np.testing.assert_array_equal(self.rising.intensity, [0.0, 10.0, 20.0, 30.0])

    def test_one_factor_per_draw_for_every_function(self) -> None:
        draws = impact_function_draws(
            self.impf_set, n_draws=3, scale=(0.5, 1.5), rng=np.random.default_rng(7)
        )
        factors = np.random.default_rng(7).uniform(0.5, 1.5, 3)
        self.assertEqual(len(draws), 3)
        for drawn, factor in zip(draws, factors):
            np.testing.assert_allclose(
                drawn.get_func("FL", 1).intensity,
                [0.0, 10.0, 10.0 + 10.0 * factor, 10.0 + 20.0 * factor],
                rtol=1e-15,
            )
            np.testing.assert_allclose(
                drawn.get_func("DR", 1).intensity, [0.0, factor, 2.0 * factor]
            )

    def test_the_threshold_and_which_intensities_harm_dont_move(self) -> None:
        intensity = np.linspace(-5.0, 60.0, 1301)
        state = np.random.get_state()
        draws = impact_function_draws(
            self.impf_set, n_draws=20, scale=(0.3, 3.0), rng=np.random.default_rng(2)
        )
        after = np.random.get_state()
        self.assertTrue(np.array_equal(state[1], after[1]))
        self.assertEqual(state[2], after[2])
        harm = self.rising.calc_mdr(intensity) > 0
        for drawn in draws:
            np.testing.assert_array_equal(
                drawn.get_func("FL", 1).calc_mdr(intensity) > 0, harm
            )
        again = impact_function_draws(
            self.impf_set, n_draws=20, scale=(0.3, 3.0), rng=np.random.default_rng(2)
        )
        self.assertTrue(all(a == b for a, b in zip(draws, again)))

    def test_invalid_arguments_raise(self) -> None:
        rng = np.random.default_rng(0)
        valid = {"n_draws": 2, "scale": (0.5, 1.5), "rng": rng}
        cases: list[tuple[Any, dict[str, Any], str]] = [
            ("impfs", {}, "ImpactFuncSet"),
            (ImpactFuncSet(), {}, "no impact function"),
            (self.impf_set, {"n_draws": 0}, "n_draws"),
            (self.impf_set, {"scale": 1.2}, "pair"),
            (self.impf_set, {"scale": (0.5, 1.0, 1.5)}, "pair"),
            (self.impf_set, {"scale": (0.0, 1.0)}, "scale's low"),
            (self.impf_set, {"scale": (1.0, 0.5)}, "scale's high"),
            (self.impf_set, {"rng": 3}, "Generator"),
            (
                ImpactFuncSet([_impf([0.0, 2.0, 1.0], [0.0, 0.5, 1.0])]),
                {},
                "don't fall",
            ),
            (
                ImpactFuncSet([_impf([0.0, 1.0], [0.0, -0.5])]),
                {},
                "mdd \\* paa >= 0",
            ),
        ]
        for impf_set, changes, message in cases:
            with self.subTest(message=message, changes=changes):
                with self.assertRaisesRegex(ParametricInputError, message):
                    impact_function_draws(impf_set, **{**valid, **changes})
        short = _impf([0.0, 1.0], [0.0, 1.0])
        short.mdd = np.array([0.0])
        with self.assertRaisesRegex(ParametricInputError, "of one size"):
            impact_function_draws(ImpactFuncSet([short]), **valid)


class TestFromImpactDraws(unittest.TestCase):
    """Samples with draws: rows, alignment, and the uncovered loss by hand."""

    def setUp(self) -> None:
        # Events 1, 2, 3 at the index 40, 60 and 20; losses in two draws.
        self.index, self.impact, self.units = array_index(
            [[40.0], [60.0], [20.0]], [[4.0], [10.0], [2.0]]
        )
        self.draws = [
            self.impact,
            _impact_like(self.impact, [3, 1, 2], [[1.0], [8.0], [6.0]]),
        ]
        self.sample = OutcomeSample.from_impact_draws(
            self.index, self.draws, units=self.units, basis=EventBasis()
        )
        self.cover = template(
            self.sample, StepPayout([30.0, 50.0], [5.0, 7.0]), limit=None
        )

    def test_rows_are_ordered_by_draw_then_event(self) -> None:
        self.assertEqual(self.sample.n_draws, 2)
        np.testing.assert_array_equal(self.sample.event_id, [1, 2, 3])
        # The second impact holds the events in another order: aligned on event_id.
        np.testing.assert_array_equal(
            self.sample.loss.toarray().ravel(), [4.0, 10.0, 2.0, 8.0, 6.0, 1.0]
        )
        np.testing.assert_allclose(self.sample.measure.weights, [0.05] * 6)
        one = OutcomeSample.from_impact_draws(
            self.index, [self.impact], units=self.units, basis=EventBasis()
        )
        whole = OutcomeSample.from_impact(
            self.index, self.impact, units=self.units, basis=EventBasis()
        )
        self.assertEqual(one.n_draws, 1)
        self.assertEqual((one.loss != whole.loss).nnz, 0)
        pd.testing.assert_frame_equal(
            evaluate(self.cover, one, split="in_sample"),
            evaluate(self.cover, whole, split="in_sample"),
        )

    def test_draws_change_the_uncovered_loss_as_computed_by_hand(self) -> None:
        # The payouts are 5, 7 and 0. Draw 0 leaves 0, 3 and 2 uncovered, draw 1
        # 3, 0 and 1: E[U] = 0.05 (0 + 3 + 2 + 3 + 0 + 1) = 0.45. The mean loss
        # per event, 6, 8 and 1.5, leaves 1, 1 and 1.5: E[U] = 0.1 * 3.5 = 0.35.
        # Event 3 loses and doesn't pay in either draw, and every event has a
        # loss, so the false-negative rate at 0 is 1 / 3 either way.
        table = evaluate(self.cover, self.sample, split="in_sample")
        unit = table.iloc[0]
        self.assertAlmostEqual(unit["expected_uncovered"], 0.45, places=12)
        self.assertAlmostEqual(unit["expected_loss"], 1.55, places=12)
        self.assertAlmostEqual(unit["false_negative_rate_l0"], 1.0 / 3.0, places=12)
        mean, impact, units = array_index(
            [[40.0], [60.0], [20.0]], [[6.0], [8.0], [1.5]]
        )
        averaged = OutcomeSample.from_impact(
            mean, impact, units=units, basis=EventBasis()
        )
        plain = evaluate(self.cover, averaged, split="in_sample").iloc[0]
        self.assertAlmostEqual(plain["expected_uncovered"], 0.35, places=12)
        self.assertAlmostEqual(plain["false_negative_rate_l0"], 1.0 / 3.0, places=12)
        # One Impact per draw, with the events' own frequencies: aai_agg 0.5
        # and 0.4, whose mean is the table's E[U].
        found = []
        for draw, expected in ((0, [0.0, 3.0, 2.0]), (1, [3.0, 0.0, 1.0])):
            shortfall = uncovered_impact(self.cover, self.sample, draw=draw)
            np.testing.assert_allclose(shortfall.at_event, expected)
            np.testing.assert_allclose(shortfall.frequency, [0.1] * 3)
            found.append(shortfall.aai_agg)
        np.testing.assert_allclose(found, [0.5, 0.4])
        self.assertAlmostEqual(float(np.mean(found)), unit["expected_uncovered"])
        # With the draw's own Impact, its events in its order: 3, 1, 2.
        given = uncovered_impact(self.cover, self.sample, impact=self.draws[1], draw=1)
        np.testing.assert_allclose(given.at_event, [1.0, 3.0, 0.0])
        self.assertEqual(given.event_id.tolist(), [3, 1, 2])

    def test_draws_on_catalogue_years(self) -> None:
        # Events 1 and 2 in year 0, event 3 in year 1, of 10: outcomes are
        # years in a draw, of weight 1 / 20. Year 0 leaves 3 uncovered in both
        # draws, year 1 leaves 2 and 1.
        years = CatalogueYears.from_labels([1, 2, 3], [0, 0, 1], n_years=10)
        index, impact, units = array_index(
            [[40.0], [60.0], [20.0]], [[4.0], [10.0], [2.0]], frequency=0.1
        )
        draws = [impact, _impact_like(impact, [3, 1, 2], [[1.0], [8.0], [6.0]])]
        sample = OutcomeSample.from_impact_draws(index, draws, units=units, basis=years)
        row = evaluate(self.cover, sample, split="in_sample").iloc[0]
        self.assertAlmostEqual(
            row["expected_uncovered"], (3 + 2 + 3 + 1) / 20, places=12
        )
        self.assertEqual(row["n_years"], 10)

    def test_a_design_on_draws_by_hand(self) -> None:
        # One event, index 50, losses 4 and 8, frequency 0.1: E[L] = 0.6. Under
        # the budget the level is at most 6. E[U] = 0.05 ((4 - x)+ + (8 - x)+)
        # falls up to 8, so x = 6 and E[U] = 0.1; without draws (the mean loss
        # 6) it would be 0. MSE: x = 6, the mean, and 0.05 (2^2 + 2^2) = 0.4.
        index, impact, units = array_index([[50.0]], [[4.0]])
        other = _impact_like(impact, [1], [[8.0]])
        sample = OutcomeSample.from_impact_draws(
            index, [impact, other], units=units, basis=EventBasis()
        )
        cover = template(sample, StepPayout([30.0], [Free()]))
        for objective, value in ((ExpectedUncovered(), 0.1), (MSE(), 0.4)):
            result = calibrate(
                cover,
                sample,
                objective,
                scope="individual",
                budget=Budget(ratio=1.0),
                options=CalibrationOptions(lp_method="highs"),
            )
            with self.subTest(objective=objective.name):
                self.assertAlmostEqual(result.objective_value, value, places=7)
                levels = result.parameters_frame()
                self.assertAlmostEqual(
                    float(levels[levels["parameter"] == "levels[0]"]["value"].iloc[0]),
                    6.0,
                    places=6,
                )
                self.assertEqual(result.cover.limit, Limit({"U0": 8.0}))
                self.assertEqual(result.design_sample.n_draws, 2)

    def test_alignment_across_draws_is_checked(self) -> None:
        def build(impacts: Any) -> OutcomeSample:
            return OutcomeSample.from_impact_draws(
                self.index, impacts, units=self.units, basis=EventBasis()
            )

        missing = _impact_like(self.impact, [1, 2], [[1.0], [8.0]])
        with self.assertRaisesRegex(EventAlignmentError, "event_id") as caught:
            build([self.impact, missing])
        self.assertIn("raised by the impact of draw 1", caught.exception.__notes__)
        rates = _impact_like(
            self.impact, [1, 2, 3], [[1.0], [8.0], [6.0]], frequency=[0.1, 0.2, 0.1]
        )
        with self.assertRaisesRegex(EventAlignmentError, "frequencies"):
            build([self.impact, rates])
        euros = _impact_like(self.impact, [1, 2, 3], [[1.0], [8.0], [6.0]], unit="EUR")
        with self.assertRaisesRegex(ImpactInputError, "'EUR'"):
            build([self.impact, euros])
        moved = _impact_like(
            self.impact, [1, 2, 3], [[1.0], [8.0], [6.0]], coord_exp=[[0.0, 0.0]]
        )
        with self.assertRaisesRegex(ImpactInputError, "coord_exp"):
            build([self.impact, moved])
        for impacts in (self.impact, [], "impacts"):
            with self.subTest(impacts=type(impacts).__name__):
                with self.assertRaisesRegex(ParametricInputError, "non-empty sequence"):
                    build(impacts)

    def test_an_event_with_a_loss_in_one_draw_is_kept(self) -> None:
        index, impact, units = array_index(
            [[40.0], [0.0], [0.0]], [[4.0], [0.0], [0.0]]
        )
        other = _impact_like(impact, [1, 2, 3], [[5.0], [1.0], [0.0]])
        sample = OutcomeSample.from_impact_draws(
            index, [impact, other], units=units, basis=EventBasis()
        )
        np.testing.assert_array_equal(sample.event_id, [1, 2])
        self.assertEqual(sample.n_left_out, 1)
        np.testing.assert_array_equal(
            sample.loss.toarray().ravel(), [4.0, 0.0, 5.0, 1.0]
        )

    def test_one_draw_must_be_chosen_for_an_impact(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "pass draw="):
            uncovered_impact(self.cover, self.sample)
        for draw, message in (
            (2, "below the sample's 2 draws"),
            (-1, ">= 0"),
            (1.0, "int"),
        ):
            with self.subTest(draw=draw):
                with self.assertRaisesRegex(ParametricInputError, message):
                    uncovered_impact(self.cover, self.sample, draw=draw)
        whole = OutcomeSample.from_impact(
            self.index, self.impact, units=self.units, basis=EventBasis()
        )
        np.testing.assert_allclose(
            uncovered_impact(self.cover, whole, draw=0).at_event,
            uncovered_impact(self.cover, whole).at_event,
        )

    def test_draws_split_and_evaluate_out_of_sample(self) -> None:
        sample, impact = random_portfolio(6, n_events=40, n_units=2, basis="event")
        noisy = _impact_like(
            impact,
            impact.event_id,
            impact.imp_mat.toarray()
            * np.random.default_rng(1).uniform(0.5, 1.5, (40, 2)),
            frequency=impact.frequency,
        )
        basis = YearSets.from_frequencies(
            impact, n_years=200, rng=np.random.default_rng(2)
        )
        drawn = OutcomeSample.from_impact_draws(
            sample.index, [impact, noisy], units=sample.units, basis=basis
        )
        train, test = holdout(drawn, train_fraction=0.7, rng=np.random.default_rng(3))
        self.assertEqual((train.n_draws, test.n_draws), (2, 2))
        result = calibrate(
            template(train, StepPayout.uniform(30.0, 70.0, 3)),
            train,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=1.0),
        )
        table = evaluate(result, test, split="out_of_sample")
        self.assertEqual(table["comonotone_index"].tolist(), [False, False, False])


class TestTheLimitOfDraws(unittest.TestCase):
    """Draws of the helper widen the uncovered loss; they don't change which events lose."""

    def test_draws_on_climada_objects(self) -> None:
        flood = _Flood()
        plain = flood.sample()
        draws = impact_function_draws(
            FLOOD_IMPF, n_draws=8, scale=(0.5, 2.0), rng=np.random.default_rng(320)
        )
        drawn = flood.sample([flood.impact(impf_set) for impf_set in draws])
        self.assertEqual(drawn.n_draws, 8)
        self.assertTrue(all(index_diagnostics(plain)["comonotone_index"]))
        self.assertFalse(any(index_diagnostics(drawn)["comonotone_index"]))
        # A cover from 0.5 m pays on depths up to 1 m that do no harm (false
        # positives); one from 2.5 m doesn't pay on depths from 1 m that do
        # (false negatives).
        covers = [
            template(plain, StepPayout([0.5, 3.0], [0.2, 0.5]), limit=None),
            template(plain, StepPayout([2.5], [0.5]), limit=None),
        ]
        rates = ["false_negative_rate_l0", "false_positive_rate_l0"]
        before = compare(covers, plain, split="in_sample")
        after = compare(covers, drawn, split="in_sample")
        pd.testing.assert_frame_equal(before[rates], after[rates])
        self.assertGreater(before["false_positive_rate_l0"].iloc[0], 0.0)
        self.assertGreater(before["false_negative_rate_l0"].iloc[5], 0.0)
        self.assertFalse(
            np.allclose(before["expected_uncovered"], after["expected_uncovered"])
        )
        self.assertEqual(before["comonotone_index"].tolist(), [True] * 10)
        self.assertEqual(after["comonotone_index"].tolist(), [False] * 10)
        unit = (drawn.loss.toarray().reshape(8, 6, 4) > 0).any(axis=0)
        np.testing.assert_array_equal(unit, plain.loss.toarray() > 0)


if __name__ == "__main__":
    unittest.main()
