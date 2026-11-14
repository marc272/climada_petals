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

Tests of the index and its simulation years, on small synthetic CLIMADA objects.

The tracks are those of
:func:`~climada_petals.engine.parametric.test._synthetic.three_tracks`.

The simulation-year tests use a 10,000-year catalogue in the IRIS layout
(:data:`STORMS`): storms named ``NA_<year>_<number>`` in years 0 to 9,999,
frequency 1 / 10,000, and dates that carry year ``y`` as 1 January of
``y + 1``, so that no date holds year 9,999.
"""

from __future__ import annotations

import datetime as dt
import unittest
from typing import Any

import numpy as np
import scipy.sparse as sp
from climada.hazard import Hazard, TCTracks

from climada_petals.engine.parametric import (
    ExpectedUncovered,
    ParametricIndex,
    ParametricInputError,
    StepPayout,
    basis_risk,
    calibrate,
    split_years,
)
from climada_petals.engine.parametric.calibrate import loss_matrix
from climada_petals.engine.parametric.test._synthetic import (
    POINTS,
    array_index,
    catalogue_tracks,
    exposures,
    hazard_and_impact,
    three_tracks,
)
from climada_petals.util.config import LOGGER


class TestFromClimada(unittest.TestCase):
    """The hazard builder on a TropCyclone hazard."""

    def setUp(self) -> None:
        self.tracks = three_tracks()
        self.exposures = exposures()
        self.hazard, self.impact = hazard_and_impact(self.tracks, self.exposures)

    def test_centroid_intensity(self) -> None:
        index = ParametricIndex.from_hazard(self.hazard, self.exposures)
        np.testing.assert_array_equal(index.values, self.hazard.intensity.toarray())
        self.assertEqual(index.unit, "m/s")
        self.assertEqual(index.event_name, ("A", "B", "C"))
        self.assertNotIn("centr_TC", self.exposures.gdf.columns)
        # Three years, 2001 to 2003, from the dates and frequency 1 / 3.
        np.testing.assert_array_equal(index.years, [2001, 2002, 2003])
        np.testing.assert_array_equal(index.event_year, [2001, 2003, 2003])
        np.testing.assert_array_equal(
            loss_matrix(index, self.impact), self.impact.imp_mat.toarray()
        )

    def test_years_can_be_given(self) -> None:
        index = ParametricIndex.from_hazard(
            self.hazard, self.exposures, years=range(2000, 2005)
        )
        np.testing.assert_array_equal(index.years, np.arange(2000, 2005))
        np.testing.assert_array_equal(index.event_year, [2001, 2003, 2003])

    def test_an_exposure_point_without_a_centroid_raises(self) -> None:
        far = exposures([*POINTS, (0.0, 0.0, 1.0e9)])
        with self.assertRaisesRegex(ParametricInputError, "no hazard centroid"):
            ParametricIndex.from_hazard(self.hazard, far)

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
            ParametricIndex.from_hazard(empty, self.exposures)

    def test_an_impact_of_other_events_is_refused(self) -> None:
        index = ParametricIndex.from_hazard(self.hazard, self.exposures)
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
            ParametricIndex.from_hazard(self.hazard, self.exposures)
        index = ParametricIndex.from_hazard(
            self.hazard, self.exposures, event_year="event_name"
        )
        np.testing.assert_array_equal(index.event_year, self.years)
        np.testing.assert_array_equal(index.years, np.arange(N_CATALOGUE))
        stated = ParametricIndex.from_hazard(
            self.hazard,
            self.exposures,
            event_year="event_name",
            n_years=N_CATALOGUE,
            first_year=0,
        )
        np.testing.assert_array_equal(stated.years, index.years)
        circle = ParametricIndex.from_cat_in_circle(
            self.tracks,
            self.exposures,
            hazard=self.hazard,
            radius_km=50.0,
            event_year="event_name",
            first_year=0,
        )
        np.testing.assert_array_equal(circle.event_year, self.years)
        np.testing.assert_array_equal(circle.years, index.years)
        train, test = split_years(index, train_fraction=0.7)
        np.testing.assert_array_equal(train, np.arange(7000))
        np.testing.assert_array_equal(test, np.arange(7000, N_CATALOGUE))
        np.testing.assert_array_equal(index.event_mask(test), [False] * 4 + [True] * 3)
        payout = StepPayout(attachment=20, limit=70, n_steps=5)
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
            ParametricIndex.from_hazard(hazard, self.exposures, event_year="event_name")
        index = ParametricIndex.from_hazard(
            hazard, self.exposures, event_year="event_name", first_year=0
        )
        np.testing.assert_array_equal(index.years, np.arange(N_CATALOGUE))
        train, _ = split_years(index, train_fraction=0.7)
        np.testing.assert_array_equal(train, np.arange(7000))
        # Every track starts on one date: the dates give one year, not 10,000.
        # The date route keeps its earlier result, but no longer silently.
        with self.assertLogs(LOGGER, "WARNING") as logs:
            dated = ParametricIndex.from_hazard(hazard, self.exposures)
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
            dated = ParametricIndex.from_hazard(hazard, self.exposures)
        self.assertIn("could lie before 1 or after 7001", logs.output[0])
        np.testing.assert_array_equal(dated.event_year, np.add(years, 1))
        np.testing.assert_array_equal(dated.years, np.arange(1, N_CATALOGUE + 1))
        stated = ParametricIndex.from_hazard(hazard, self.exposures, first_year=1)
        np.testing.assert_array_equal(stated.years, dated.years)
        with self.assertRaisesRegex(ParametricInputError, "Pass first_year"):
            ParametricIndex.from_hazard(hazard, self.exposures, event_year="event_name")
        named = ParametricIndex.from_hazard(
            hazard, self.exposures, event_year="event_name", first_year=0
        )
        np.testing.assert_array_equal(named.years, np.arange(N_CATALOGUE))

    def test_the_date_route_is_the_default(self) -> None:
        tracks = three_tracks()
        hazard, _ = hazard_and_impact(tracks, self.exposures)
        default = ParametricIndex.from_hazard(hazard, self.exposures)
        dated = ParametricIndex.from_hazard(hazard, self.exposures, event_year="date")
        np.testing.assert_array_equal(dated.event_year, [2001, 2003, 2003])
        np.testing.assert_array_equal(dated.years, default.years)
        stated = ParametricIndex.from_hazard(
            hazard, self.exposures, n_years=5, first_year=2000
        )
        np.testing.assert_array_equal(stated.years, np.arange(2000, 2005))

    def test_n_years_and_first_year_with_explicit_event_years(self) -> None:
        # An ensemble of two members per storm of a 3-year catalogue: member
        # m of year y has the year y + 3 m.
        hazard, _ = hazard_and_impact(three_tracks(), self.exposures)
        index = ParametricIndex.from_hazard(
            hazard, self.exposures, event_year=[0, 2, 5], n_years=6, first_year=0
        )
        np.testing.assert_array_equal(index.years, np.arange(6))
        np.testing.assert_array_equal(index.event_year, [0, 2, 5])
        # n_years does not need the frequencies to agree on N.
        hazard.frequency = np.array([0.1, 0.2, 0.3])
        index = ParametricIndex.from_hazard(
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
                ParametricIndex.from_hazard(hazard, exp, **kwargs)
        uneven = hazard.select(event_names=list(hazard.event_name))
        uneven.frequency = np.linspace(1e-4, 2e-4, uneven.size)
        with self.assertRaisesRegex(ParametricInputError, "pass n_years or years"):
            ParametricIndex.from_hazard(uneven, exp, event_year="event_name")

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
                ParametricIndex.from_hazard(
                    hazard, self.exposures, event_year="event_name"
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
        index, _ = array_index(np.zeros((2, 2)), np.zeros((2, 2)), [0, 1], 10)
        fields = {
            "values": index.values,
            "event_id": index.event_id,
            "event_name": index.event_name,
            "frequency": index.frequency,
            "event_year": index.event_year,
            "years": index.years,
            "unit_id": index.unit_id,
            "coords": index.coords,
        }
        self.assertEqual((index.n_events, index.n_units), (2, 2))
        bad = [
            ({"event_id": [1]}, "event_id has 1 entries for 2 events"),
            ({"event_id": [1, 1]}, "event_id must be unique"),
            ({"frequency": [0.1, 0.0]}, "frequency must be > 0"),
            ({"years": []}, "years must be non-empty and unique"),
            ({"years": [0, 1, 1]}, "years must be non-empty and unique"),
            ({"unit_id": ("U0", "U0")}, "unique ids"),
            ({"coords": np.zeros((2, 3))}, "coords must have shape"),
        ]
        for change, match in bad:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                ParametricIndex(**{**fields, **change})


if __name__ == "__main__":
    unittest.main()
