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

Tests of the outcome measures and the risk measures.Tests of the outcome measures, the bases and the risk measures.Tests of the outcome samples, the routes on CLIMADA objects and the splits.

The catalogue tests use a 10,000-year catalogue in the IRIS layout
(:data:`STORMS`): storms named ``NA_<year>_<number>`` in years 0 to 9,999,
frequency 1 / 10,000, and dates that carry year ``y`` as 1 January of
``y + 1``, so that no date holds year 9,999.
"""

from __future__ import annotations

import copy
import datetime as dt
import itertools
import unittest
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.engine import ImpactCalc
from climada.entity import ImpactFuncSet, ImpfTropCyclone

from climada_petals.engine.parametric import (
    ENSEMBLE_PARENT,
    CatalogueYears,
    EventAlignmentError,
    EventBasis,
    ExpectedUncovered,
    ImpactInputError,
    IndexDefinition,
    InsuredUnits,
    LeakageError,
    OutcomeSample,
    ParametricIndex,
    ParametricInputError,
    SplitRecord,
    StepPayout,
    WeightedCVaR,
    YearRouteError,
    YearSets,
    basis_risk,
    calibrate,
    check_no_leakage,
    holdout,
    labels_from_names,
    sample_years,
    year_block_kfold,
)
from climada_petals.engine.parametric.test._synthetic import (
    array_index,
    array_sample,
    catalogue_tracks,
    exposures,
    grid_exposures,
    grid_hazard,
    grid_impact,
    hazard_and_impact,
    random_portfolio,
    three_tracks,
)
from climada_petals.util.config import LOGGER

PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=4)

#: A 10,000-year catalogue with storms in its first and last year.
N_CATALOGUE = 10_000
STORMS: tuple[tuple[int, int, float, float], ...] = (
    (0, 0, 25.75, 120.5),
    (2500, 0, 25.8, 100.0),
    (5000, 0, 25.9, 110.0),
    (6999, 0, 25.76, 90.0),
    (7000, 0, 25.8, 115.0),
    (9999, 0, 25.85, 120.0),
    (9999, 1, 26.0, 95.0),
)

#: The year of IRIS's event names, ``NA_<year>_<number>``.
IRIS = r"NA_(?P<year>[0-9]+)_[0-9]+"


def _year_dates(years: list[int]) -> list[int]:
    """Date year ``y`` on 1 January of ``y + 1``; 0 (no date) above 9998."""
    return [dt.date(y + 1, 1, 1).toordinal() if y < dt.MAXYEAR else 0 for y in years]


def _ensemble(
    n_parents: int, n_members: int, frequency: float
) -> tuple[ParametricIndex, Any, InsuredUnits, list[str]]:
    """Return one unit's index, impact and units for an ensemble, and its parents.

    Event ``p n_members + m + 1`` is member ``m`` of parent ``P<p>``, named
    ``P<p>`` for ``m = 0`` and ``P<p>_gen<m>`` otherwise, as CLIMADA names
    perturbed tracks. Every event has index 30 and loss 1.
    """
    n_events = n_parents * n_members
    index, impact, units = array_index(
        np.full((n_events, 1), 30.0), np.ones((n_events, 1)), frequency=frequency
    )
    names = [
        f"P{p}" if m == 0 else f"P{p}_gen{m}"
        for p in range(n_parents)
        for m in range(n_members)
    ]
    return index, impact, units, labels_from_names(names, ENSEMBLE_PARENT)["block"]


class TestFromImpact(unittest.TestCase):
    """A sample from an index and an Impact: the kept events and the checks."""

    def test_events_without_loss_or_index_are_left_out(self) -> None:
        values = [[30.0, 0.0], [0.0, 0.0], [5.0, 1.0]]
        index, impact, units = array_index(values, [[4.0, 0.0], [0.0] * 2, [0.0] * 2])
        with self.assertLogs(LOGGER, "INFO") as logs:
            sample = OutcomeSample.from_impact(
                index, impact, units=units, basis=EventBasis()
            )
        self.assertIn("1 of 3 events", logs.output[-1])
        self.assertEqual(sample.n_left_out, 1)
        np.testing.assert_array_equal(sample.event_id, [1, 3])
        np.testing.assert_array_equal(sample.loss.toarray(), [[4.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(sample.index_values, [[30.0, 0.0], [5.0, 1.0]])
        np.testing.assert_array_equal(sample.frequency, [0.1, 0.1])
        self.assertFalse(sample.loss.data.flags.writeable)
        self.assertEqual((sample.n_events, sample.n_rows, sample.n_units), (2, 2, 2))
        self.assertEqual(sample.unit_id, ("U0", "U1"))
        self.assertEqual((sample.kind, sample.route), ("event", "frequency"))
        np.testing.assert_array_equal(sample.event_row, [0, 1])
        np.testing.assert_array_equal(sample.draw, [0, 0])
        np.testing.assert_array_equal(sample.block, [1, 3])
        for none in (sample.n_years, sample.event_year, sample.correction_factor):
            self.assertIsNone(none)
        self.assertEqual((sample.scenario, sample.split), ("baseline", None))
        # On a lower tail an index of 0 can be severe, so every event stays.
        lower = IndexDefinition("table", unit="mm", tail="lower")
        dry = ParametricIndex(index.values, index.event_id, index.unit_id, lower)
        kept = OutcomeSample.from_impact(dry, impact, units=units, basis=EventBasis())
        self.assertEqual((kept.n_events, kept.n_left_out), (3, 0))
        quiet, still, _ = array_index(np.zeros((2, 2)), np.zeros((2, 2)))
        with self.assertRaisesRegex(ParametricInputError, "no event of the impact"):
            OutcomeSample.from_impact(quiet, still, units=units, basis=EventBasis())

    def test_an_impact_that_does_not_match_is_refused(self) -> None:
        index, impact, units = array_index([[30.0], [60.0]], [[4.0], [10.0]])
        years = CatalogueYears.from_labels([1, 2], [0, 1], n_years=10)
        sample = OutcomeSample.from_impact(index, impact, units=units, basis=years)
        np.testing.assert_array_equal(sample.loss.toarray(), [[4.0], [10.0]])
        changes: dict[str, tuple[dict[str, Any], type[Exception], str]] = {
            "event_id": ({"event_id": np.array([1, 3])}, EventAlignmentError, "event"),
            "frequency": ({"frequency": np.array([0.1, 0.2])}, YearRouteError, "x N"),
            "coord_exp": (
                {"coord_exp": np.array([[25.0, -70.0]])},
                ImpactInputError,
                "coord_exp",
            ),
            "loss": (
                {"imp_mat": sp.csr_matrix([[4.0], [-1.0]])},
                ImpactInputError,
                "negative",
            ),
        }
        for name, (change, error, match) in changes.items():
            other = copy.deepcopy(impact)
            for attribute, value in change.items():
                setattr(other, attribute, value)
            with self.subTest(name=name), self.assertRaisesRegex(error, match):
                OutcomeSample.from_impact(index, other, units=units, basis=years)
        unlabelled = CatalogueYears.from_labels([1], [0], n_years=10)
        with self.assertRaisesRegex(EventAlignmentError, "have no year"):
            OutcomeSample.from_impact(index, impact, units=units, basis=unlabelled)

    def test_invalid_parts_raise(self) -> None:
        index, impact, units = array_index([[30.0], [60.0]], [[4.0], [10.0]])
        _, _, others = array_index(np.ones((1, 2)), np.ones((1, 2)))
        parts: list[tuple[dict[str, Any], str]] = [
            ({"index": index.values}, "ParametricIndex"),
            ({"units": others}, "index's unit_id, in order"),
            ({"basis": "annual"}, "no default"),
        ]
        good: dict[str, Any] = {"index": index, "units": units, "basis": EventBasis()}
        for change, match in parts:
            arguments = {**good, **change}
            with (
                self.subTest(change=list(change)),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                OutcomeSample.from_impact(
                    arguments["index"],
                    impact,
                    units=arguments["units"],
                    basis=arguments["basis"],
                )
        fields: dict[str, Any] = {
            "index": index,
            "loss": np.array([[4.0], [10.0]]),
            "frequency": [0.1, 0.1],
            "units": units,
            "basis": EventBasis(),
        }
        self.assertEqual(OutcomeSample(**fields).n_events, 2)
        nothing = ParametricIndex(np.zeros((0, 1)), [], ("U0",), index.definition)
        bad: list[tuple[dict[str, Any], str]] = [
            ({"loss": np.ones((3, 1))}, "loss must have shape"),
            ({"loss": [[4.0], [-1.0]]}, "finite and >= 0"),
            ({"loss": [["a"], ["b"]]}, "numeric"),
            ({"frequency": [0.1]}, "one value >= 0 per event"),
            ({"frequency": [0.1, -0.1]}, "one value >= 0 per event"),
            ({"scenario": ""}, "non-empty str"),
            ({"split": "train"}, "SplitRecord"),
            ({"years": [0, 1]}, "catalogue years only"),
            ({"n_draws": 0}, "n_draws must be >= 1"),
            ({"n_left_out": -1}, "n_left_out must be >= 0"),
            ({"index": nothing, "loss": np.zeros((0, 1)), "frequency": []}, "no event"),
        ]
        for change, match in bad:
            with (
                self.subTest(change=list(change)),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                OutcomeSample(**{**fields, **change})


class TestRoutes(unittest.TestCase):
    """One set on two annual routes, and a frequency-scaled hazard on all three."""

    def test_a_uniform_catalogue_and_its_sampling_vect_give_identical_outcomes(
        self,
    ) -> None:
        catalogue, impact = random_portfolio(1)
        event_year = catalogue.basis.year_of(impact.event_id)
        vect = [np.flatnonzero(event_year == year) for year in range(60)]
        sets = YearSets.from_sampling_vect(impact.event_id, vect)
        sampled = OutcomeSample.from_impact(
            catalogue.index, impact, units=catalogue.units, basis=sets
        )
        np.testing.assert_array_equal(
            sampled.measure.grouping.toarray(), catalogue.measure.grouping.toarray()
        )
        np.testing.assert_array_equal(
            sampled.measure.weights, catalogue.measure.weights
        )
        self.assertAlmostEqual(sampled.correction_factor or 0.0, 1.0, places=12)
        cases = itertools.product(
            ("individual", "portfolio"),
            (ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.95)),
        )
        for scope, objective in cases:
            covers = [
                calibrate(
                    sample, payout=PAYOUT, objective=objective, scope=scope, budget=0.8
                )
                for sample in (catalogue, sampled)
            ]
            tables = [
                basis_risk(sample, covers[0], beta=0.95)
                for sample in (catalogue, sampled)
            ]
            with self.subTest(scope=scope, objective=objective):
                np.testing.assert_array_equal(covers[0].levels, covers[1].levels)
                self.assertEqual(covers[0].premium, covers[1].premium)
                pd.testing.assert_frame_equal(tables[0], tables[1])

    def test_a_frequency_scaled_hazard(self) -> None:
        exp = exposures()
        hazard, _ = hazard_and_impact(three_tracks(), exp, frequency=1 / 3)
        hazard.category = np.array([4, 1, 0])
        future = hazard.apply_climate_scenario_knu(
            percentile="50", scenario="8.5", target_year=2080
        )
        # Category 4 becomes more frequent and the others less: 1 / N no more.
        self.assertGreater(future.frequency[0] * 3, 1.1)
        self.assertLess(future.frequency[1] * 3, 0.9)
        impfset = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])
        units = InsuredUnits.points(exp)
        impacts = [
            ImpactCalc(exp.copy(), impfset, haz).impact(save_mat=True)
            for haz in (hazard, future)
        ]
        indices = [ParametricIndex.from_hazard(haz, units) for haz in (hazard, future)]
        # The dates carry the baseline's three years, 2001 to 2003.
        dated = CatalogueYears.from_dates(hazard)
        base = OutcomeSample.from_impact(
            indices[0], impacts[0], units=units, basis=dated
        )
        self.assertEqual(base.n_years, 3)
        with self.assertRaisesRegex(YearRouteError, "frequency x N between"):
            OutcomeSample.from_impact(
                indices[1],
                impacts[1],
                units=units,
                basis=CatalogueYears.from_dates(future),
            )
        on_events = OutcomeSample.from_impact(
            indices[1], impacts[1], units=units, basis=EventBasis()
        )
        loss = on_events.loss.toarray()
        # Storm C has no wind at the points and is left out; E[L] is aai_agg.
        self.assertEqual(on_events.n_left_out, 1)
        for unit in range(3):
            self.assertAlmostEqual(
                on_events.measure.expectation(loss[:, unit]),
                impacts[1].eai_exp[unit],
                delta=1e-12 * impacts[1].aai_agg,
            )
        sets = YearSets.from_frequencies(
            impacts[1], n_years=10_000, rng=np.random.default_rng(311)
        )
        on_years = OutcomeSample.from_impact(
            indices[1], impacts[1], units=units, basis=sets
        )
        # Each kept storm occurs at its own frequency, and the mean annual loss
        # is near aai_agg.
        np.testing.assert_allclose(
            [on_years.measure.expectation(row) for row in np.eye(2)],
            future.frequency[:2],
            rtol=0.05,
        )
        self.assertAlmostEqual(on_years.correction_factor or 0.0, 1.0, delta=0.06)
        for sample in (on_events, on_years):
            cover = calibrate(
                sample,
                payout=PAYOUT,
                objective=WeightedCVaR(alpha=0.9, beta=0.95),
                scope="portfolio",
                budget=1.0,
            )
            table = basis_risk(sample, cover, beta=0.95)
            with self.subTest(route=sample.route):
                self.assertGreater(cover.premium, 0.0)
                self.assertEqual(table["basis"].iloc[-1], sample.kind)

    def test_loss_free_events_below_the_attachment_change_nothing(self) -> None:
        rng = np.random.default_rng(6)
        values = rng.uniform(25.0, 80.0, size=(20, 2))
        loss = (values - 25.0) ** 2
        years = np.sort(rng.integers(0, 40, size=20))
        # Ten events below the attachment, kept, and five with an index of 0,
        # left out; none has a loss.
        quiet = np.vstack([np.full((10, 2), 5.0), np.zeros((5, 2))])
        more = (
            np.vstack([values, quiet]),
            np.vstack([loss, np.zeros((15, 2))]),
            np.r_[years, rng.integers(0, 40, size=15)],
        )
        cases = itertools.product(("annual", "event"), ("individual", "portfolio"))
        for basis, scope in cases:
            samples = [
                array_sample(*arrays, 40, basis=basis)[0]
                for arrays in ((values, loss, years), more)
            ]
            covers = [
                calibrate(
                    sample,
                    payout=PAYOUT,
                    objective=WeightedCVaR(alpha=0.9, beta=0.9),
                    scope=scope,
                    budget=0.5,
                )
                for sample in samples
            ]
            with self.subTest(basis=basis, scope=scope):
                self.assertEqual((samples[0].n_events, samples[1].n_events), (20, 30))
                self.assertEqual(samples[1].n_left_out, 5)
                for first, second in zip(covers[0].problems, covers[1].problems):
                    np.testing.assert_allclose(
                        [second.objective_value, second.premium],
                        [first.objective_value, first.premium],
                        rtol=1e-9,
                    )


class TestCatalogueOnClimada(unittest.TestCase):
    """Catalogue years from the names and the dates of a TropCyclone hazard."""

    def setUp(self) -> None:
        self.exposures = exposures()
        self.units = InsuredUnits.points(self.exposures)
        self.tracks = catalogue_tracks(STORMS)
        self.years = [year for year, *_ in STORMS]
        self.hazard, self.impact = hazard_and_impact(
            self.tracks,
            self.exposures,
            frequency=1 / N_CATALOGUE,
            date=_year_dates(self.years),
        )
        self.index = ParametricIndex.from_hazard(self.hazard, self.units)

    def test_a_10000_year_catalogue_calibrates_from_its_event_names(self) -> None:
        # The last year has no date, so the date route refuses the hazard.
        with self.assertRaisesRegex(YearRouteError, "one ordinal date"):
            CatalogueYears.from_dates(self.hazard)
        basis = CatalogueYears.from_event_names(self.hazard, IRIS)
        np.testing.assert_array_equal(basis.year, self.years)
        np.testing.assert_array_equal(basis.years, np.arange(N_CATALOGUE))
        sample = OutcomeSample.from_impact(
            self.index, self.impact, units=self.units, basis=basis
        )
        circle = ParametricIndex.from_cat_in_circle(
            self.tracks, self.units, hazard=self.hazard, radius_km=50.0
        )
        on_circle = OutcomeSample.from_impact(
            circle, self.impact, units=self.units, basis=basis
        )
        np.testing.assert_array_equal(on_circle.event_year, self.years)
        train, test = holdout(sample, train_fraction=0.7)
        np.testing.assert_array_equal(train.years, np.arange(7000))
        np.testing.assert_array_equal(test.years, np.arange(7000, N_CATALOGUE))
        np.testing.assert_array_equal(test.event_year, [7000, 9999, 9999])
        payout = StepPayout.uniform(attachment=20, exhaustion=70, n_steps=5)
        for scope in ("individual", "portfolio"):
            cover = calibrate(
                train,
                payout=payout,
                objective=ExpectedUncovered(),
                scope=scope,
                budget=1.0,
            )
            with self.subTest(scope=scope):
                np.testing.assert_array_equal(cover.calibration_years, train.years)
                self.assertGreater(cover.premium, 0.0)
                metrics = basis_risk(test, cover, beta=0.99)
                self.assertEqual(len(metrics), self.exposures.gdf.shape[0] + 1)

    def test_years_the_data_do_not_carry_raise(self) -> None:
        # No storm in year 0: the names span 2500 to 9999, 7,500 years.
        storms = [s for s in STORMS if s[0] != 0]
        hazard, impact = hazard_and_impact(
            catalogue_tracks(storms), self.exposures, frequency=1 / N_CATALOGUE
        )
        index = ParametricIndex.from_hazard(hazard, self.units)
        spanned = CatalogueYears.from_event_names(hazard, IRIS)
        self.assertEqual((spanned.n_years, spanned.first_year), (7500, 2500))
        with self.assertRaisesRegex(YearRouteError, "frequency x N between 0.75"):
            OutcomeSample.from_impact(index, impact, units=self.units, basis=spanned)
        with self.assertRaisesRegex(YearRouteError, "pass first_year"):
            CatalogueYears.from_event_names(hazard, IRIS, n_years=N_CATALOGUE)
        stated = CatalogueYears.from_event_names(
            hazard, IRIS, n_years=N_CATALOGUE, first_year=0
        )
        sample = OutcomeSample.from_impact(
            index, impact, units=self.units, basis=stated
        )
        train, _ = holdout(sample, train_fraction=0.7)
        np.testing.assert_array_equal(train.years, np.arange(7000))
        # Every track starts on one date: the dates give one year, not 10,000.
        # This used to be accepted with a logged warning; it raises now.
        dated = CatalogueYears.from_dates(hazard)
        self.assertEqual(dated.n_years, 1)
        with self.assertRaisesRegex(YearRouteError, "calendar years from 2000"):
            OutcomeSample.from_impact(index, impact, units=self.units, basis=dated)

    def test_dates_with_years_missing_need_the_years_stated(self) -> None:
        # Storms in years 0 to 7,000, dated year y on 1 January of y + 1.
        storms = [s for s in STORMS if s[0] != 9999]
        years = [year for year, *_ in storms]
        hazard, impact = hazard_and_impact(
            catalogue_tracks(storms),
            self.exposures,
            frequency=1 / N_CATALOGUE,
            date=_year_dates(years),
        )
        index = ParametricIndex.from_hazard(hazard, self.units)
        dated = CatalogueYears.from_dates(hazard)
        self.assertEqual((dated.n_years, dated.first_year), (7001, 1))
        with self.assertRaisesRegex(YearRouteError, "frequency x N"):
            OutcomeSample.from_impact(index, impact, units=self.units, basis=dated)
        with self.assertRaisesRegex(YearRouteError, "pass first_year"):
            CatalogueYears.from_dates(hazard, n_years=N_CATALOGUE)
        stated = CatalogueYears.from_dates(hazard, n_years=N_CATALOGUE, first_year=1)
        sample = OutcomeSample.from_impact(
            index, impact, units=self.units, basis=stated
        )
        np.testing.assert_array_equal(sample.years, np.arange(1, N_CATALOGUE + 1))

    def test_a_flood_on_a_date_route_with_stated_years(self) -> None:
        exp = grid_exposures()
        with self.assertLogs(LOGGER, "WARNING"):
            units = InsuredUnits.from_column(exp, "region_id")
        hazard = grid_hazard()
        impact = grid_impact(hazard, exp)
        index = ParametricIndex.from_hazard(hazard, units, index_stat="max")
        # Dates in 2001 to 2003 and frequency 1 / 10: the dates don't carry
        # the ten years.
        with self.assertRaisesRegex(YearRouteError, "frequency x N"):
            OutcomeSample.from_impact(
                index, impact, units=units, basis=CatalogueYears.from_dates(hazard)
            )
        stated = CatalogueYears.from_dates(hazard, n_years=10, first_year=2001)
        sample = OutcomeSample.from_impact(index, impact, units=units, basis=stated)
        payout = StepPayout.uniform(attachment=0.5, exhaustion=2.5, n_steps=2)
        for scope in ("individual", "portfolio"):
            cover = calibrate(
                sample,
                payout=payout,
                objective=ExpectedUncovered(),
                scope=scope,
                budget=1.0,
            )
            table = basis_risk(sample, cover, beta=0.9)
            with self.subTest(scope=scope):
                self.assertEqual(table["subject"].tolist(), [1, 2, "portfolio"])
                np.testing.assert_allclose(
                    table["expected_loss"].iloc[:2], [0.32, 0.56], rtol=1e-12
                )


class TestCatalogueSplits(unittest.TestCase):
    """Contiguous blocks of years; ensemble members stay with their parent."""

    def test_holdout_and_folds_are_contiguous_years(self) -> None:
        sample, _ = array_sample(np.full((3, 1), 30.0), np.ones((3, 1)), [0, 4, 9], 10)
        train, test = holdout(sample, train_fraction=0.7)
        np.testing.assert_array_equal(train.years, np.arange(7))
        np.testing.assert_array_equal(test.years, [7, 8, 9])
        np.testing.assert_array_equal(train.event_id, [1, 2])
        np.testing.assert_array_equal(test.event_id, [3])
        self.assertEqual(train.split, SplitRecord("holdout", "train", 0, 1, 0.7))
        self.assertEqual(test.split, SplitRecord("holdout", "test", 0, 1, 0.3))
        self.assertEqual((train.n_years, test.n_years), (7, 3))
        with self.assertRaisesRegex(ParametricInputError, "no rng"):
            holdout(sample, rng=np.random.default_rng(0))
        with self.assertRaisesRegex(ParametricInputError, "without a year"):
            holdout(sample, train_fraction=0.01)
        folds = year_block_kfold(sample, 3)
        tested = [fold_test.years.tolist() for _, fold_test in folds]
        self.assertEqual(tested, [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]])
        for fold, (fold_train, fold_test) in enumerate(folds):
            with self.subTest(fold=fold):
                self.assertEqual(
                    sorted([*fold_train.years, *fold_test.years]), list(range(10))
                )
                self.assertEqual(fold_test.split.fold, fold)
                self.assertEqual(fold_train.split.n_folds, 3)
        for k, match in ((11, "at least 11 years"), (1, "k must be >= 2")):
            with self.subTest(k=k), self.assertRaisesRegex(ParametricInputError, match):
                year_block_kfold(sample, k)
        with self.assertRaisesRegex(ParametricInputError, "no rng"):
            year_block_kfold(sample, 2, rng=np.random.default_rng(0))

    def test_ensemble_members_stay_with_their_parent(self) -> None:
        # Two members of a storm in each of 6 calendar years: 12 member-years.
        index, impact, units, _ = _ensemble(6, 2, 1 / 12)
        parent = np.repeat(np.arange(6), 2)
        member = np.tile([0, 1], 6)
        # Labels y + 6 m put a parent's members 6 years apart.
        apart = CatalogueYears.from_labels(
            impact.event_id, parent + 6 * member, n_years=12, block=parent
        )
        sample = OutcomeSample.from_impact(index, impact, units=units, basis=apart)
        for split in (lambda: holdout(sample), lambda: year_block_kfold(sample, 3)):
            with self.assertRaisesRegex(LeakageError, "every cut"):
                split()
        # Numbered next to each other, 2 y + m, a cut between parents works.
        beside = CatalogueYears.from_labels(
            impact.event_id, 2 * parent + member, n_years=12, block=parent
        )
        sample = OutcomeSample.from_impact(index, impact, units=units, basis=beside)
        with self.assertLogs(LOGGER, "INFO") as logs:
            train, test = holdout(sample, train_fraction=0.75)
        self.assertIn("[(9, 8)]", logs.output[-1])
        np.testing.assert_array_equal(np.unique(train.block), [0, 1, 2, 3])
        np.testing.assert_array_equal(np.unique(test.block), [4, 5])
        self.assertAlmostEqual(train.split.share, 8 / 12)
        cuts = [fold_test.years[0] for _, fold_test in year_block_kfold(sample, 4)]
        self.assertEqual(cuts, [0, 2, 6, 8])
        with self.assertRaisesRegex(LeakageError, "too few valid cuts"):
            year_block_kfold(sample, 7)
        # A parent on both sides, with no event on both, is a leak too.
        with self.assertRaisesRegex(LeakageError, "1 blocks are in both"):
            check_no_leakage(sample.select(range(7)), sample.select(range(7, 12)))
        with self.assertRaisesRegex(LeakageError, "1 events are in both"):
            check_no_leakage(sample.select(range(6)), sample.select(range(5, 12)))
        with self.assertRaisesRegex(ParametricInputError, "OutcomeSample"):
            check_no_leakage(sample, sample.index)
        with self.assertRaisesRegex(ParametricInputError, "no rng"):
            sample.select([0], rng=np.random.default_rng(0))


class TestPoolSplits(unittest.TestCase):
    """Pools of blocks on the event basis and on year sets: no leak, lambda kept."""

    def setUp(self) -> None:
        self.index, self.impact, self.units, self.parents = _ensemble(10, 3, 0.01)

    def _samples(self) -> list[OutcomeSample]:
        bases = [
            EventBasis(event_id=self.impact.event_id, block=self.parents),
            YearSets.from_frequencies(
                self.impact,
                n_years=1_000,
                rng=np.random.default_rng(1),
                block=self.parents,
            ),
        ]
        return [
            OutcomeSample.from_impact(
                self.index, self.impact, units=self.units, basis=basis
            )
            for basis in bases
        ]

    def test_members_of_a_parent_stay_on_one_side(self) -> None:
        for sample in self._samples():
            train, test = holdout(
                sample, train_fraction=0.7, rng=np.random.default_rng(2)
            )
            with self.subTest(route=sample.route):
                self.assertEqual(np.intersect1d(train.event_id, test.event_id).size, 0)
                self.assertFalse(set(train.block) & set(test.block))
                for side in (train, test):
                    _, counts = np.unique(side.block, return_counts=True)
                    self.assertTrue(np.all(counts == 3))
                    # Each side keeps lambda: its frequencies are divided by its share.
                    self.assertAlmostEqual(side.frequency.sum(), 0.3, places=12)
                self.assertAlmostEqual(train.split.share + test.split.share, 1.0)
                self.assertAlmostEqual(train.split.share, 0.7, delta=0.1)

    def test_years_are_sampled_from_each_side_alone(self) -> None:
        _, sample = self._samples()
        train, test = holdout(sample, train_fraction=0.7, rng=np.random.default_rng(2))
        for side in (train, test):
            np.testing.assert_array_equal(
                np.sort(side.basis.event_id), np.sort(side.event_id)
            )
            self.assertEqual(side.n_years, round(side.split.share * 1_000))
        folds = year_block_kfold(sample, 5, rng=np.random.default_rng(3))
        tested = np.concatenate([fold_test.event_id for _, fold_test in folds])
        np.testing.assert_array_equal(np.sort(tested), sample.event_id)
        for fold_train, fold_test in folds:
            self.assertEqual(
                np.intersect1d(fold_train.event_id, fold_test.event_id).size, 0
            )

    def test_selections_and_their_checks(self) -> None:
        on_events, on_years = self._samples()
        part = on_events.select(["P0"])
        self.assertEqual(
            part.split, SplitRecord("select", "selected", 0, 1, part.split.share)
        )
        self.assertAlmostEqual(part.split.share, 0.1)
        np.testing.assert_allclose(part.frequency, [0.1, 0.1, 0.1])
        with self.assertRaisesRegex(ParametricInputError, "not blocks of the sample"):
            on_events.select(["Q"])
        with self.assertRaisesRegex(ParametricInputError, "not random"):
            on_events.select(["P0"], rng=np.random.default_rng(0))
        with self.assertRaisesRegex(ParametricInputError, "pass rng"):
            on_years.select(["P0"])
        with self.assertRaisesRegex(ParametricInputError, "pass rng"):
            holdout(on_years)
        drawn = on_years.select(["P0", "P1"], rng=np.random.default_rng(4))
        self.assertEqual(drawn.n_years, 200)
        alone = OutcomeSample.from_impact(
            self.index,
            self.impact,
            units=self.units,
            basis=EventBasis(event_id=self.impact.event_id, block=["P"] * 30),
        )
        with self.assertRaisesRegex(ParametricInputError, "too few blocks"):
            holdout(alone, rng=np.random.default_rng(0))

    def test_pools_without_frequency_raise(self) -> None:
        silent = copy.deepcopy(self.impact)
        silent.frequency = np.r_[np.zeros(3), np.full(27, 0.01)]
        on_events = OutcomeSample.from_impact(
            self.index,
            silent,
            units=self.units,
            basis=EventBasis(event_id=silent.event_id, block=self.parents),
        )
        with self.assertRaisesRegex(ParametricInputError, "holds no frequency"):
            on_events.select(["P0"])
        silent.frequency = np.zeros(30)
        given = YearSets.from_sampling_vect(silent.event_id, [[0], [1, 2]])
        on_years = OutcomeSample.from_impact(
            self.index, silent, units=self.units, basis=given
        )
        with self.assertRaisesRegex(ParametricInputError, "sum to 0"):
            holdout(on_years, rng=np.random.default_rng(0))


class TestDrawsAndRandomState(unittest.TestCase):
    """The draws dimension, and numpy's global random state."""

    def test_draws_of_an_event_stay_together(self) -> None:
        index, _, units = array_index([[30.0], [60.0]], [[4.0], [10.0]])
        loss = [[4.0], [10.0], [6.0], [8.0]]
        years = CatalogueYears.from_labels([1, 2], [0, 5], n_years=10)
        sample = OutcomeSample(index, loss, [0.1, 0.1], units, years, n_draws=2)
        np.testing.assert_array_equal(sample.event_row, [0, 1, 0, 1])
        np.testing.assert_array_equal(sample.draw, [0, 0, 1, 1])
        np.testing.assert_array_equal(sample.index_values[:, 0], [30, 60, 30, 60])
        np.testing.assert_allclose(sample.measure.weights, [0.05] * 4 + [0.8])
        # E[L] per year is the mean of the draws': (4 + 10) / 10 and (6 + 8) / 10.
        self.assertAlmostEqual(sample.measure.expectation(np.ravel(loss)), 1.4)
        train, test = holdout(sample, train_fraction=0.5)
        np.testing.assert_array_equal(train.loss.toarray(), [[4.0], [6.0]])
        np.testing.assert_array_equal(test.loss.toarray(), [[10.0], [8.0]])
        on_events = OutcomeSample(
            index, loss, [0.1, 0.1], units, EventBasis(), n_draws=2
        )
        np.testing.assert_allclose(on_events.measure.weights, [0.05] * 4)

    def test_numpy_global_random_state_does_not_move(self) -> None:
        index, impact, units, parents = _ensemble(4, 2, 0.1)
        state = np.random.get_state()
        rng = np.random.default_rng(5)
        sets = YearSets.from_frequencies(impact, n_years=500, rng=rng, block=parents)
        sample = OutcomeSample.from_impact(index, impact, units=units, basis=sets)
        holdout(sample, rng=rng)
        year_block_kfold(sample, 2, rng=rng)
        sample_years([0.1, 0.2], 100, rng)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
