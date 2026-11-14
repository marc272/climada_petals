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

Tests of the index builders, the event matching, the CLIMADA specifics and
the simulation years, on small synthetic CLIMADA objects of three hazard
types: tropical cyclones
(``TC``, from the tracks of
:func:`~climada_petals.engine.parametric.test._synthetic.three_tracks`), a
flood (``FL``) and a drought index (``DR``, lower tail) on the grid of
:func:`~climada_petals.engine.parametric.test._synthetic.grid_hazard`.

On the grid, region 1 holds the points of values 1 and 3 at centroid 0 and
the point of value 2 at centroid 1; region 2 the points of value 4 at
centroids 2 and 3. The flood depths of the events 30, 10 and 20 at the four
centroids are (0, 2, 1, 0), (1, 3, 0, 0) and (0.5, 0, 4, 2).

The simulation-year tests use a 10,000-year catalogue in the IRIS layout
(:data:`STORMS`): storms named ``NA_<year>_<number>`` in years 0 to 9,999,
frequency 1 / 10,000, and dates that carry year ``y`` as 1 January of
``y + 1``, so that no date holds year 9,999.
"""

from __future__ import annotations

import datetime as dt
import unittest
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.hazard import Hazard, TCTracks

from climada_petals.engine.parametric import (
    CoverDefinitionError,
    EventAlignmentError,
    ExpectedUncovered,
    ImpactInputError,
    IndexDefinition,
    InsuredUnits,
    ParametricIndex,
    ParametricInputError,
    StepPayout,
    UnitMappingError,
    basis_risk,
    calibrate,
)
from climada_petals.engine.parametric import index as index_module
from climada_petals.engine.parametric import split_years
from climada_petals.engine.parametric.calibrate import loss_matrix
from climada_petals.engine.parametric.test._synthetic import (
    FLOOD_DEPTH,
    POINTS,
    array_index,
    catalogue_tracks,
    exposures,
    grid_exposures,
    grid_hazard,
    grid_impact,
    hazard_and_impact,
    shuffled,
    three_tracks,
)
from climada_petals.util.config import LOGGER

#: Each statistic over the two regions, worked out by hand from the depths.
BY_HAND: dict[tuple[Any, Any], list[list[float]]] = {
    # Region 1: mean of centroids 0 and 1; region 2: of centroids 2 and 3.
    ("mean", None): [[1.0, 0.5], [2.0, 0.0], [0.25, 3.0]],
    # Region 1 weighs centroid 0 by 1 + 3 and centroid 1 by 2.
    ("mean", "value"): [[4 / 6, 0.5], [10 / 6, 0.0], [2 / 6, 3.0]],
    ("max", None): [[2.0, 1.0], [3.0, 0.0], [0.5, 4.0]],
    # Two values a <= b: the 25th percentile is a + (b - a) / 4.
    (25.0, None): [[0.5, 0.25], [1.5, 0.0], [0.125, 2.5]],
    (100.0, None): [[2.0, 1.0], [3.0, 0.0], [0.5, 4.0]],
}


def _regions(exp: Any = None) -> InsuredUnits:
    """Return the two regions of the grid points; the last point is in none."""
    with unittest.TestCase().assertLogs(LOGGER, "WARNING"):
        return InsuredUnits.from_column(
            grid_exposures() if exp is None else exp, "region_id"
        )


def _dated(hazard: Any, exp: Any, **years: Any) -> ParametricIndex:
    """Return the intensity at each point, with years from ``hazard``."""
    index = ParametricIndex.from_hazard(hazard, InsuredUnits.points(exp))
    return index.with_years(hazard, **years)


class TestFromClimada(unittest.TestCase):
    """The hazard builder on a TropCyclone hazard."""

    def setUp(self) -> None:
        self.tracks = three_tracks()
        self.exposures = exposures()
        self.hazard, self.impact = hazard_and_impact(self.tracks, self.exposures)

    def test_centroid_intensity(self) -> None:
        units = InsuredUnits.points(self.exposures)
        index = ParametricIndex.from_hazard(self.hazard, units).with_years(self.hazard)
        np.testing.assert_array_equal(index.values, self.hazard.intensity.toarray())
        self.assertEqual(
            index.definition,
            IndexDefinition(
                "hazard",
                unit="m/s",
                index_stat="at_centroid",
                parameters={"haz_type": "TC"},
            ),
        )
        self.assertEqual(index.event_name, ("A", "B", "C"))
        np.testing.assert_array_equal(index.event_id, self.hazard.event_id)
        self.assertIs(index.units, units)
        self.assertNotIn("centr_TC", self.exposures.gdf.columns)
        # Three years, 2001 to 2003, from the dates and frequency 1 / 3.
        np.testing.assert_array_equal(index.years, [2001, 2002, 2003])
        np.testing.assert_array_equal(index.event_year, [2001, 2003, 2003])
        np.testing.assert_array_equal(
            loss_matrix(index, self.impact), self.impact.imp_mat.toarray()
        )

    def test_years_can_be_given(self) -> None:
        index = _dated(self.hazard, self.exposures, years=range(2000, 2005))
        np.testing.assert_array_equal(index.years, np.arange(2000, 2005))
        np.testing.assert_array_equal(index.event_year, [2001, 2003, 2003])

    def test_an_exposure_point_without_a_centroid_raises(self) -> None:
        far = exposures([*POINTS, (0.0, 0.0, 1.0e9)])
        with self.assertRaisesRegex(ParametricInputError, "no hazard centroid"):
            ParametricIndex.from_hazard(self.hazard, InsuredUnits.points(far))

    def test_a_hazard_without_events_raises(self) -> None:
        empty = Hazard(
            haz_type="TC",
            units="m/s",
            centroids=self.hazard.centroids,
            event_id=np.array([], dtype=int),
            event_name=[],
            frequency=np.array([]),
            date=np.array([], dtype=int),
            intensity=sp.csr_matrix((0, self.hazard.centroids.size)),
            fraction=sp.csr_matrix((0, self.hazard.centroids.size)),
        )
        with self.assertRaisesRegex(ParametricInputError, "the hazard has no events"):
            ParametricIndex.from_hazard(empty, InsuredUnits.points(self.exposures))

    def test_an_impact_of_other_events_is_refused(self) -> None:
        index = _dated(self.hazard, self.exposures)
        _, other = hazard_and_impact(TCTracks(self.tracks.data[:2]), self.exposures)
        with self.assertRaises(ParametricInputError):
            loss_matrix(index, other)
        no_matrix = self.impact.__class__(
            event_id=self.impact.event_id,
            event_name=self.impact.event_name,
            date=self.impact.date,
            frequency=self.impact.frequency,
            coord_exp=self.impact.coord_exp,
            eai_exp=self.impact.eai_exp,
            at_event=self.impact.at_event,
        )
        with self.assertRaisesRegex(ParametricInputError, "save_mat"):
            loss_matrix(index, no_matrix)


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


def _year_dates(years: list[int]) -> list[int]:
    """Date year ``y`` on 1 January of ``y + 1``; 0 (no date) above 9998.

    A date holds the calendar years 1 to 9999, so a hazard dated this way
    cannot date the last year of a 10,000-year catalogue.
    """
    return [dt.date(y + 1, 1, 1).toordinal() if y < dt.MAXYEAR else 0 for y in years]


class TestSimulationYears(unittest.TestCase):
    """Years from the dates, the event names, or n_years and first_year."""

    def setUp(self) -> None:
        self.exposures = exposures()
        self.tracks = catalogue_tracks(STORMS)
        self.years = [year for year, *_ in STORMS]
        self.hazard, self.impact = hazard_and_impact(
            self.tracks,
            self.exposures,
            frequency=1 / N_CATALOGUE,
            date=_year_dates(self.years),
        )

    def test_a_10000_year_catalogue_calibrates_from_its_event_names(self) -> None:
        # The last year has no date, so the default route refuses the hazard.
        with self.assertRaisesRegex(ParametricInputError, "pass event_year"):
            _dated(self.hazard, self.exposures)
        index = _dated(self.hazard, self.exposures, event_year="event_name")
        np.testing.assert_array_equal(index.event_year, self.years)
        np.testing.assert_array_equal(index.years, np.arange(N_CATALOGUE))
        stated = _dated(
            self.hazard,
            self.exposures,
            event_year="event_name",
            n_years=N_CATALOGUE,
            first_year=0,
        )
        np.testing.assert_array_equal(stated.years, index.years)
        circle = ParametricIndex.from_cat_in_circle(
            self.tracks,
            InsuredUnits.points(self.exposures),
            hazard=self.hazard,
            radius_km=50.0,
        ).with_years(self.hazard, event_year="event_name", first_year=0)
        np.testing.assert_array_equal(circle.event_year, self.years)
        np.testing.assert_array_equal(circle.years, index.years)
        train, test = split_years(index, train_fraction=0.7)
        np.testing.assert_array_equal(train, np.arange(7000))
        np.testing.assert_array_equal(test, np.arange(7000, N_CATALOGUE))
        np.testing.assert_array_equal(index.event_mask(test), [False] * 4 + [True] * 3)
        payout = StepPayout.uniform(attachment=20, exhaustion=70, n_steps=5)
        for scope in ("individual", "portfolio"):
            cover = calibrate(
                index,
                self.impact,
                payout=payout,
                objective=ExpectedUncovered(),
                basis="annual",
                scope=scope,
                budget=1.0,
                years=train,
            )
            with self.subTest(scope=scope):
                np.testing.assert_array_equal(cover.calibration_years, train)
                self.assertGreater(cover.premium, 0.0)
                metrics = basis_risk(index, self.impact, cover, beta=0.99, years=test)
                self.assertEqual(len(metrics), self.exposures.gdf.shape[0] + 1)

    def test_the_span_is_stated_not_taken_from_the_first_event(self) -> None:
        storms = [s for s in STORMS if s[0] != 0]  # no storm in year 0
        tracks = catalogue_tracks(storms)
        hazard, _ = hazard_and_impact(tracks, self.exposures, frequency=1 / N_CATALOGUE)
        # The years 2500-9999 are 7,500 of 10,000: where the rest lie is unknown.
        with self.assertRaisesRegex(ParametricInputError, "Pass first_year"):
            _dated(hazard, self.exposures, event_year="event_name")
        index = _dated(hazard, self.exposures, event_year="event_name", first_year=0)
        np.testing.assert_array_equal(index.years, np.arange(N_CATALOGUE))
        train, _ = split_years(index, train_fraction=0.7)
        np.testing.assert_array_equal(train, np.arange(7000))
        # Every track starts on one date: the dates give one year, not 10,000.
        # The date route keeps its earlier result, but no longer silently.
        with self.assertLogs(LOGGER, "WARNING") as logs:
            dated = _dated(hazard, self.exposures)
        self.assertIn("run from 2000 to 2000", logs.output[0])
        self.assertIn("pass first_year", logs.output[0])
        np.testing.assert_array_equal(dated.years, np.arange(2000, 12_000))

    def test_valid_dates_with_years_missing_keep_the_earlier_result(self) -> None:
        # A hazard with storms in years 0 to 9,997, all
        # dated (year y on 1 January of y + 1), 9,998 of 10,000 years.
        storms = [s for s in STORMS if s[0] != 9999]
        years = [year for year, *_ in storms]
        hazard, _ = hazard_and_impact(
            catalogue_tracks(storms),
            self.exposures,
            frequency=1 / N_CATALOGUE,
            date=_year_dates(years),
        )
        with self.assertLogs(LOGGER, "WARNING") as logs:
            dated = _dated(hazard, self.exposures)
        self.assertIn("could lie before 1 or after 7001", logs.output[0])
        np.testing.assert_array_equal(dated.event_year, np.add(years, 1))
        np.testing.assert_array_equal(dated.years, np.arange(1, N_CATALOGUE + 1))
        stated = _dated(hazard, self.exposures, first_year=1)
        np.testing.assert_array_equal(stated.years, dated.years)
        with self.assertRaisesRegex(ParametricInputError, "Pass first_year"):
            _dated(hazard, self.exposures, event_year="event_name")
        named = _dated(hazard, self.exposures, event_year="event_name", first_year=0)
        np.testing.assert_array_equal(named.years, np.arange(N_CATALOGUE))

    def test_the_date_route_is_the_default(self) -> None:
        tracks = three_tracks()
        hazard, _ = hazard_and_impact(tracks, self.exposures)
        default = _dated(hazard, self.exposures)
        dated = _dated(hazard, self.exposures, event_year="date")
        np.testing.assert_array_equal(dated.event_year, [2001, 2003, 2003])
        np.testing.assert_array_equal(dated.years, default.years)
        stated = _dated(hazard, self.exposures, n_years=5, first_year=2000)
        np.testing.assert_array_equal(stated.years, np.arange(2000, 2005))

    def test_n_years_and_first_year_with_explicit_event_years(self) -> None:
        # An ensemble of two members per storm of a 3-year catalogue: member
        # m of year y has the year y + 3 m.
        hazard, _ = hazard_and_impact(three_tracks(), self.exposures)
        index = _dated(
            hazard, self.exposures, event_year=[0, 2, 5], n_years=6, first_year=0
        )
        np.testing.assert_array_equal(index.years, np.arange(6))
        np.testing.assert_array_equal(index.event_year, [0, 2, 5])
        # n_years does not need the frequencies to agree on N.
        hazard.frequency = np.array([0.1, 0.2, 0.3])
        index = _dated(
            hazard, self.exposures, event_year=[0, 2, 5], n_years=6, first_year=0
        )
        np.testing.assert_array_equal(index.years, np.arange(6))

    def test_ambiguous_or_invalid_years_raise(self) -> None:
        hazard, exp = self.hazard, self.exposures
        cases: list[tuple[dict[str, Any], str]] = [
            ({"years": np.arange(N_CATALOGUE), "n_years": N_CATALOGUE}, "not both"),
            ({"years": np.arange(N_CATALOGUE), "first_year": 0}, "not both"),
            ({"event_year": "calendar"}, "'date', 'event_name'"),
            ({"event_year": "event_name", "n_years": 0}, "n_years must be >= 1"),
            ({"event_year": "event_name", "n_years": 2.5}, "n_years must be an int"),
            ({"event_year": "event_name", "n_years": True}, "n_years must be an int"),
            ({"event_year": "event_name", "first_year": "0"}, "first_year must be"),
            ({"event_year": "event_name", "first_year": 1}, "outside the 10000"),
            ({"event_year": "event_name", "n_years": 5000}, "more than the 5000"),
        ]
        for kwargs, match in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                _dated(hazard, exp, **kwargs)
        uneven = hazard.select(event_names=list(hazard.event_name))
        uneven.frequency = np.linspace(1e-4, 2e-4, uneven.size)
        with self.assertRaisesRegex(ParametricInputError, "pass n_years or years"):
            _dated(uneven, exp, event_year="event_name")

    def test_event_names_without_a_year_raise(self) -> None:
        for names in (["A", "B", "C"], ["NA_5_0_gen1", "NA_5_0", "NA_x_0"]):
            tracks = catalogue_tracks(STORMS[:3])
            for track, name in zip(tracks.data, names, strict=True):
                track.attrs["sid"] = track.attrs["name"] = name
            hazard, _ = hazard_and_impact(tracks, self.exposures)
            with (
                self.subTest(names=names),
                self.assertRaisesRegex(
                    ParametricInputError, "<prefix>_<year>_<number>"
                ),
            ):
                _dated(hazard, self.exposures, event_year="event_name")


class TestFromHazard(unittest.TestCase):
    """Each statistic of a footprint, by hand, on three hazard types."""

    def test_each_statistic_over_the_regions_of_a_flood(self) -> None:
        units = _regions()
        for (stat, weight), expected in BY_HAND.items():
            index = ParametricIndex.from_hazard(
                grid_hazard(), units, index_stat=stat, weight=weight
            )
            with self.subTest(stat=stat, weight=weight):
                np.testing.assert_allclose(index.values, expected, rtol=1e-12)
                self.assertEqual(index.definition.unit, "m")
                self.assertEqual(index.definition.index_stat, stat)
                self.assertEqual(index.definition.weight, weight)

    def test_blocks_of_events_give_the_same_values(self) -> None:
        units = _regions()
        for stat, weight in BY_HAND:
            whole = ParametricIndex.from_hazard(
                grid_hazard(), units, index_stat=stat, weight=weight
            )
            with mock.patch.object(index_module, "_CHUNK", 3):
                blocks = ParametricIndex.from_hazard(
                    grid_hazard(), units, index_stat=stat, weight=weight
                )
            with self.subTest(stat=stat):
                np.testing.assert_array_equal(blocks.values, whole.values)

    def test_absent_readings_take_the_stated_value(self) -> None:
        units = _regions()
        index = ParametricIndex.from_hazard(
            grid_hazard(), units, index_stat="mean", absent_value=-1.0
        )
        # Every zero depth reads as -1 before the mean.
        np.testing.assert_allclose(
            index.values, [[0.5, 0.0], [2.0, -1.0], [-0.25, 3.0]]
        )
        self.assertEqual(index.definition.parameters["absent_value"], -1.0)

    def test_a_lower_tail_index_states_absent_readings(self) -> None:
        # A drought index: negative is dry, and zero is no reading.
        spi = [[-2.0, -0.5, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0], [-3.0, -1.0, -0.2, 0.0]]
        drought = grid_hazard(spi, haz_type="DR", units="1")
        units = _regions()
        with self.assertRaisesRegex(ParametricInputError, "absent_value"):
            ParametricIndex.from_hazard(drought, units, index_stat="max", tail="lower")
        index = ParametricIndex.from_hazard(
            drought, units, index_stat="max", tail="lower", absent_value=0.0
        )
        np.testing.assert_allclose(index.values, [[-0.5, 0.0], [0.0, 0.0], [-1.0, 0.0]])
        drier = ParametricIndex.from_hazard(
            drought, units, index_stat=50.0, tail="lower", absent_value=-9.0
        )
        np.testing.assert_allclose(
            drier.values, [[-1.25, -5.0], [-9.0, -9.0], [-2.0, -4.6]]
        )
        self.assertEqual(drier.definition.tail, "lower")

    def test_two_points_at_one_centroid_count_once_unweighted(self) -> None:
        # Region 1 holds two points at centroid 0 and one at centroid 1.
        units = InsuredUnits.whole(grid_exposures())
        index = ParametricIndex.from_hazard(grid_hazard(), units, index_stat="mean")
        np.testing.assert_allclose(index.values[:, 0], [0.75, 1.0, 1.625])
        weighted = ParametricIndex.from_hazard(
            grid_hazard(), units, index_stat="mean", weight="value"
        )
        # Values 1 + 3, 2, 4 and 4 + 0 at the four centroids, of 14.
        np.testing.assert_allclose(
            weighted.values[:, 0],
            [(4 * 0 + 2 * 2 + 4 * 1) / 14, (4 * 1 + 2 * 3) / 14, (2 + 16 + 8) / 14],
        )

    def test_invalid_arguments_raise(self) -> None:
        units = _regions()
        hazard = grid_hazard()
        cases: list[tuple[dict[str, Any], type[Exception], str]] = [
            ({}, ParametricInputError, "one point each"),
            ({"index_stat": "median"}, CoverDefinitionError, "index_stat"),
            ({"index_stat": 101.0}, CoverDefinitionError, "percentile"),
            ({"index_stat": "max", "weight": "value"}, CoverDefinitionError, "weight"),
            ({"index_stat": "max", "tail": "left"}, CoverDefinitionError, "tail"),
            ({"index_stat": "max", "absent_value": "none"}, ParametricInputError, ""),
        ]
        for kwargs, error, match in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(error, match):
                ParametricIndex.from_hazard(hazard, units, **kwargs)
        with self.assertRaisesRegex(ParametricInputError, "InsuredUnits"):
            ParametricIndex.from_hazard(hazard, grid_exposures(), index_stat="max")
        unitless = grid_hazard(units="")
        with self.assertRaisesRegex(ParametricInputError, "hazard.units"):
            ParametricIndex.from_hazard(unitless, units, index_stat="max")
        empty = grid_hazard(np.zeros((0, 4)), event_id=[], event_name=[])
        with self.assertRaisesRegex(ParametricInputError, "no events"):
            ParametricIndex.from_hazard(empty, units, index_stat="max")


class TestEvents(unittest.TestCase):
    """Events are matched on event_id, or on a stated key, never by position."""

    def setUp(self) -> None:
        self.exposures = grid_exposures()
        self.units = _regions(self.exposures)
        self.hazard = grid_hazard()
        self.index = ParametricIndex.from_hazard(
            self.hazard, self.units, index_stat="mean"
        ).with_years(self.hazard, event_year=[0, 1, 1], years=range(10))

    def test_a_shuffled_impact_is_read_in_the_index_order(self) -> None:
        impact = grid_impact(self.hazard, self.exposures)
        reordered = grid_impact(shuffled(self.hazard, [2, 0, 1]), self.exposures)
        np.testing.assert_array_equal(reordered.event_id, [20, 30, 10])
        expected = [[0.8, 0.8], [2.0, 0.0], [0.4, 4.8]]
        np.testing.assert_allclose(loss_matrix(self.index, impact), expected)
        np.testing.assert_allclose(loss_matrix(self.index, reordered), expected)
        np.testing.assert_array_equal(self.index.rows_in(reordered), [1, 2, 0])
        payout = StepPayout.uniform(attachment=0.5, exhaustion=2.5, n_steps=2)
        covers = [
            calibrate(
                self.index,
                loss,
                payout=payout,
                objective=ExpectedUncovered(),
                basis="annual",
                budget=0.5,
            )
            for loss in (impact, reordered)
        ]
        np.testing.assert_allclose(covers[0].levels, covers[1].levels, atol=1e-12)
        paid = covers[0].payout_impact(self.index, reordered)
        np.testing.assert_array_equal(paid.event_id, reordered.event_id)
        unit_paid = covers[0].payout_matrix(self.index)[[2, 0, 1]]
        points = paid.imp_mat.toarray()
        np.testing.assert_allclose(points[:, :3].sum(axis=1), unit_paid[:, 0])
        np.testing.assert_allclose(points[:, 3:5].sum(axis=1), unit_paid[:, 1])
        # Region 1 shares its payout 1 : 3 : 2 by value, and region 2 evenly.
        np.testing.assert_allclose(points[:, :3], unit_paid[:, [0]] * [[1, 3, 2]] / 6)
        np.testing.assert_allclose(points[:, 3], points[:, 4])
        np.testing.assert_array_equal(points[:, 5], 0.0)

    def test_an_impact_of_other_events_is_refused(self) -> None:
        fewer = grid_impact(
            grid_hazard(FLOOD_DEPTH[:2], event_id=[30, 10], event_name=["a", "b"]),
            self.exposures,
        )
        with self.assertRaisesRegex(EventAlignmentError, "matched_to"):
            loss_matrix(self.index, fewer)
        twice = grid_impact(grid_hazard(event_id=[30, 10, 30]), self.exposures)
        with self.assertRaisesRegex(EventAlignmentError, "repeat"):
            self.index.rows_in(twice)

    def test_an_index_from_another_hazard_set_is_matched_on_names(self) -> None:
        impact = grid_impact(self.hazard, self.exposures)
        # Another set of the same storms: other ids, another order, an extra
        # event, and a coarser grid of two centroids (one per row).
        coarse = Hazard(
            haz_type="FL",
            units="m",
            centroids=grid_hazard().centroids.select(sel_cen=np.array([0, 2])),
            event_id=np.array([7, 5, 6, 8]),
            event_name=["c", "a", "x", "b"],
            frequency=np.full(4, 0.1),
            date=np.ones(4, dtype=int),
            intensity=sp.csr_matrix([[0.5, 4.0], [0.0, 1.0], [9.0, 9.0], [1.0, 0.0]]),
            fraction=sp.csr_matrix((4, 2)),
        )
        other = ParametricIndex.from_hazard(coarse, self.units, index_stat="max")
        with self.assertRaisesRegex(EventAlignmentError, "have no index value"):
            other.matched_to(impact)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            matched = other.matched_to(impact, on="event_name")
        self.assertIn("1 of the index's 4 events", logs.output[0])
        np.testing.assert_array_equal(matched.event_id, [30, 10, 20])
        self.assertEqual(matched.event_name, ("a", "b", "c"))
        np.testing.assert_array_equal(
            matched.values, [[0.0, 1.0], [1.0, 0.0], [0.5, 4.0]]
        )
        self.assertIs(matched.units, self.units)
        self.assertEqual(matched.definition, other.definition)
        with self.assertRaisesRegex(ParametricInputError, "no simulation years"):
            loss_matrix(matched, impact)
        dated = matched.with_years(impact, event_year=[0, 1, 2], years=range(3))
        np.testing.assert_allclose(
            loss_matrix(dated, impact), [[0.8, 0.8], [2.0, 0.0], [0.4, 4.8]]
        )

    def test_invalid_matching_raises(self) -> None:
        impact = grid_impact(self.hazard, self.exposures)
        with self.assertRaisesRegex(ParametricInputError, "on must be"):
            self.index.matched_to(impact, on="date")
        nameless = ParametricIndex(
            self.index.values, self.index.event_id, (1, 2), self.index.definition
        )
        with self.assertRaisesRegex(ParametricInputError, "no event names"):
            nameless.matched_to(impact, on="event_name")
        named_twice = ParametricIndex(
            self.index.values,
            self.index.event_id,
            (1, 2),
            self.index.definition,
            event_name=["a", "a", "c"],
        )
        with self.assertRaisesRegex(EventAlignmentError, "repeat"):
            named_twice.matched_to(impact, on="event_name")
        dated = nameless.with_years(impact, event_year=[0, 1, 1], years=range(10))
        with self.assertRaisesRegex(ParametricInputError, "no InsuredUnits"):
            loss_matrix(dated, impact)


class TestFromTable(unittest.TestCase):
    """A user's table, keyed on event_id and unit."""

    DEFINITION = IndexDefinition("table", unit="mm", source="rain gauges")

    def setUp(self) -> None:
        self.units = _regions()
        self.table = pd.DataFrame(
            {
                "event_id": [10, 30, 10, 30, 20, 20],
                "unit_id": ["2", "1", "1", "2", "1", "2"],
                "value": [5.0, 1.0, 2.0, 3.0, 4.0, 6.0],
            }
        )

    def test_a_long_table_in_any_order(self) -> None:
        index = ParametricIndex.from_table(
            self.table, self.units, definition=self.DEFINITION
        )
        np.testing.assert_array_equal(index.event_id, [10, 20, 30])
        self.assertEqual(index.unit_id, (1, 2))
        np.testing.assert_array_equal(
            index.values, [[2.0, 5.0], [4.0, 6.0], [1.0, 3.0]]
        )
        self.assertEqual(index.definition, self.DEFINITION)
        self.assertIsNone(index.event_name)
        renamed = self.table.rename(
            columns={"event_id": "storm", "unit_id": "parish", "value": "rain"}
        )
        same = ParametricIndex.from_table(
            renamed,
            self.units,
            definition=self.DEFINITION,
            event_id="storm",
            unit_id="parish",
            value="rain",
        )
        np.testing.assert_array_equal(same.values, index.values)

    def test_invalid_tables_raise(self) -> None:
        table = self.table
        cases: dict[str, tuple[Any, type[Exception], str]] = {
            "not a table": (table.to_numpy(), ParametricInputError, "DataFrame"),
            "a missing column": (
                table.drop(columns="value"),
                ParametricInputError,
                "DataFrame",
            ),
            "no rows": (table.iloc[:0], ParametricInputError, "no rows"),
            "an unknown unit": (
                table.replace({"unit_id": {"2": "3"}}),
                UnitMappingError,
                "not in units",
            ),
            "a pair twice": (
                pd.concat([table, table.iloc[:1]]),
                ParametricInputError,
                "appear twice",
            ),
            "a pair missing": (table.iloc[1:], ParametricInputError, "not at all"),
            "a value missing": (
                table.replace({"value": {6.0: np.nan}}),
                ParametricInputError,
                "finite",
            ),
            "a fractional id": (
                table.replace({"event_id": {20: 20.5}}),
                ParametricInputError,
                "whole numbers",
            ),
        }
        for name, (frame, error, match) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(error, match):
                ParametricIndex.from_table(
                    frame, self.units, definition=self.DEFINITION
                )


class TestFromDefinition(unittest.TestCase):
    """A cover's index, rebuilt on another hazard set."""

    def test_a_hazard_index_on_another_hazard_set(self) -> None:
        units = _regions()
        built = ParametricIndex.from_hazard(
            grid_hazard(),
            units,
            index_stat="mean",
            weight="value",
            absent_value=0.0,
            source="flood model, present",
        )
        definition = IndexDefinition.from_dict(built.definition.to_dict())
        future = grid_hazard(np.asarray(FLOOD_DEPTH) * 2.0, event_id=[1, 2, 3])
        rebuilt = ParametricIndex.from_definition(definition, future, units)
        self.assertEqual(rebuilt.definition, built.definition)
        np.testing.assert_allclose(rebuilt.values, built.values * 2.0)
        np.testing.assert_array_equal(rebuilt.event_id, [1, 2, 3])

    def test_a_definition_that_does_not_fit_raises(self) -> None:
        units = _regions()
        hazard = grid_hazard()
        stated = {"kind": "hazard", "unit": "m", "index_stat": "max"}
        cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            "another hazard type": ({"parameters": {"haz_type": "TC"}}, {}),
            "another unit": ({"unit": "cm", "parameters": {"haz_type": "FL"}}, {}),
            "no hazard type": ({"parameters": {"absent_value": 0.0}}, {}),
            "an unknown parameter": (
                {"parameters": {"haz_type": "FL", "radius_km": 5}},
                {},
            ),
            "no statistic": (
                {"index_stat": None, "parameters": {"haz_type": "FL"}},
                {},
            ),
            "tracks": ({"parameters": {"haz_type": "FL"}}, {"tracks": three_tracks()}),
            "a table": ({"kind": "table"}, {}),
            "a cat-in-a-circle without tracks": ({"kind": "cat_in_circle"}, {}),
        }
        for name, (change, extra) in cases.items():
            definition = IndexDefinition.from_dict({**stated, **change})
            with (
                self.subTest(name=name),
                self.assertRaises(ParametricInputError),
            ):
                ParametricIndex.from_definition(definition, hazard, units, **extra)
        with self.assertRaisesRegex(ParametricInputError, "IndexDefinition"):
            ParametricIndex.from_definition(stated, hazard, units)


class TestWithYears(unittest.TestCase):
    """with_years, the transitional source of the annual basis's years."""

    def setUp(self) -> None:
        self.exposures = grid_exposures()
        self.units = _regions(self.exposures)
        self.hazard = grid_hazard()
        self.index = ParametricIndex.from_hazard(
            self.hazard, self.units, index_stat="max"
        )

    def test_years_come_from_the_hazard_or_the_impact(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "no simulation years"):
            _ = self.index.event_year
        dated = self.index.with_years(self.hazard)
        np.testing.assert_array_equal(dated.years, np.arange(2001, 2011))
        np.testing.assert_array_equal(dated.event_year, [2001, 2002, 2003])
        np.testing.assert_array_equal(dated.frequency, [0.1, 0.1, 0.1])
        reordered = grid_impact(shuffled(self.hazard, [1, 2, 0]), self.exposures)
        from_impact = self.index.with_years(reordered, event_year=[5, 9, 0], n_years=10)
        # The years follow the index's events, matched on event_id.
        np.testing.assert_array_equal(from_impact.event_year, [0, 5, 9])
        np.testing.assert_array_equal(from_impact.years, np.arange(10))
        self.assertFalse(dated.years.flags.writeable)

    def test_invalid_sources_raise(self) -> None:
        per_day = grid_hazard()
        per_day.frequency_unit = "1/day"
        with self.assertRaisesRegex(ImpactInputError, "frequency_unit"):
            self.index.with_years(per_day)
        fewer = grid_hazard(FLOOD_DEPTH[:2], event_id=[30, 10], event_name=["a", "b"])
        with self.assertRaisesRegex(EventAlignmentError, "not in the source"):
            self.index.with_years(fewer)
        cases: list[tuple[dict[str, Any], str]] = [
            ({"event_year": [0, 1]}, "event_year has 2 entries for 3 events"),
            ({"event_year": [0, 1, 2], "years": [0, 1]}, "event years outside"),
            ({"event_year": [0, 1, 1], "years": [0, 1, 1]}, "non-empty and unique"),
        ]
        for kwargs, match in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                self.index.with_years(self.hazard, **kwargs)
        empty = grid_hazard(np.zeros((0, 4)), event_id=[], event_name=[])
        with self.assertRaisesRegex(ParametricInputError, "has no events"):
            self.index.with_years(empty)
        never = grid_hazard()
        never.frequency = np.array([0.1, 0.0, 0.1])
        with self.assertRaisesRegex(ParametricInputError, "frequency must be > 0"):
            self.index.with_years(never, event_year=[0, 1, 2], years=range(10))

    def test_calibration_needs_the_years(self) -> None:
        impact = grid_impact(self.hazard, self.exposures)
        payout = StepPayout.uniform(attachment=0.5, exhaustion=2.5, n_steps=2)
        arguments: dict[str, Any] = {
            "payout": payout,
            "objective": ExpectedUncovered(),
            "basis": "annual",
        }
        with self.assertRaisesRegex(ParametricInputError, "with_years"):
            calibrate(self.index, impact, **arguments)
        dated = self.index.with_years(self.hazard)
        for scope in ("individual", "portfolio"):
            cover = calibrate(dated, impact, scope=scope, budget=1.0, **arguments)
            table = basis_risk(dated, impact, cover, beta=0.9)
            with self.subTest(scope=scope):
                self.assertEqual(table["subject"].tolist(), [1, 2, "portfolio"])
                np.testing.assert_allclose(
                    table["expected_loss"].iloc[:2], [0.32, 0.56], rtol=1e-12
                )


class TestParametricIndex(unittest.TestCase):
    """Validation and year splits."""

    def test_split_years_is_two_contiguous_blocks(self) -> None:
        index, _ = array_index(np.zeros((3, 1)), np.zeros((3, 1)), [0, 4, 9], 10)
        train, test = split_years(index, train_fraction=0.7)
        np.testing.assert_array_equal(train, np.arange(7))
        np.testing.assert_array_equal(test, [7, 8, 9])
        np.testing.assert_array_equal(index.event_mask(test), [False, False, True])
        with self.assertRaises(ParametricInputError):
            split_years(index, train_fraction=0.01)

    def test_invalid_arrays_raise(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "event years outside"):
            array_index(np.zeros((2, 1)), np.zeros((2, 1)), [0, 10], 10)
        with self.assertRaisesRegex(ParametricInputError, "finite"):
            array_index(np.full((2, 1), np.nan), np.zeros((2, 1)), [0, 1], 10)
        index, _ = array_index(np.zeros((2, 1)), np.zeros((2, 1)), [0, 1], 10)
        with self.assertRaises(ParametricInputError):
            index.check_years([3, 3])
        with self.assertRaises(ParametricInputError):
            index.check_years([42])

    def test_inconsistent_fields_raise(self) -> None:
        definition = IndexDefinition("table", unit="m")
        units = _regions()
        good: dict[str, Any] = {
            "values": np.zeros((2, 2)),
            "event_id": [1, 2],
            "unit_id": (1, 2),
            "definition": definition,
        }
        index = ParametricIndex(**good, event_name=["a", "b"], units=units)
        self.assertEqual((index.n_events, index.n_units), (2, 2))
        self.assertFalse(index.values.flags.writeable)
        bad: list[tuple[dict[str, Any], str]] = [
            ({"values": np.full((2, 2), np.inf)}, "finite"),
            ({"event_id": [1]}, "2 unique ids"),
            ({"event_id": [1, 1]}, "2 unique ids"),
            ({"unit_id": (1, 1)}, "unique ids"),
            ({"definition": "wind"}, "IndexDefinition"),
            ({"event_name": ["a"]}, "event_name has 1 entries"),
            ({"units": grid_exposures()}, "InsuredUnits"),
            ({"unit_id": (2, 1), "units": units}, "in order"),
        ]
        for change, match in bad:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                ParametricIndex(**{**good, **change})


if __name__ == "__main__":
    unittest.main()
