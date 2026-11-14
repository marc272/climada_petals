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

Tests of the cat-in-a-circle index and its track geometry.

The tracks are those of
:func:`~climada_petals.engine.parametric.test._synthetic.three_tracks`.
"""

from __future__ import annotations

import unittest

import numpy as np
from climada.hazard import TCTracks

from climada_petals.engine.parametric import (
    SAFFIR_SIMPSON_KN,
    ParametricIndex,
    ParametricInputError,
)
from climada_petals.engine.parametric.index_tracks import category_of, densify_track
from climada_petals.engine.parametric.test._synthetic import (
    exposures,
    hazard_and_impact,
    three_tracks,
)


class TestCatInCircleHelpers(unittest.TestCase):
    """Categories and track interpolation."""

    def test_category_counts_the_breakpoints_reached(self) -> None:
        wind = np.array([0.0, 63.99, 64.0, 82.9, 83.0, 96.0, 113.0, 137.0, 500.0])
        np.testing.assert_array_equal(
            category_of(wind, SAFFIR_SIMPSON_KN), [0, 0, 1, 1, 2, 3, 4, 5, 5]
        )

    def test_densify_keeps_the_points(self) -> None:
        lat = np.array([20.0, 21.0, 23.0])
        lon = np.array([-70.0, -71.0, -71.0])
        wind = np.array([60.0, 90.0, 30.0])
        out = densify_track(lat, lon, wind, np.array([0.0, 3.0, 6.0]), 0.5)
        self.assertEqual(len(out[0]), 13)
        np.testing.assert_array_equal(out[0][[0, 6, 12]], lat)
        np.testing.assert_allclose(out[2][6:], np.linspace(90, 30, 7), atol=1e-12)


class TestCatInCircle(unittest.TestCase):
    """The cat-in-a-circle builder on a TropCyclone hazard and its tracks."""

    def setUp(self) -> None:
        self.tracks = three_tracks()
        self.exposures = exposures()
        self.hazard, self.impact = hazard_and_impact(self.tracks, self.exposures)

    def test_cat_in_circle_by_radius(self) -> None:
        expected = {
            30.0: [[4, 4, 4], [0, 0, 0], [0, 0, 0]],
            110.0: [[4, 4, 4], [0, 0, 1], [0, 0, 0]],
            150.0: [[4, 4, 4], [1, 1, 1], [0, 0, 0]],
        }
        for radius, values in expected.items():
            index = ParametricIndex.from_cat_in_circle(
                self.tracks, self.exposures, hazard=self.hazard, radius_km=radius
            )
            with self.subTest(radius=radius):
                np.testing.assert_array_equal(index.values, values)
                self.assertEqual(index.unit, "Saffir-Simpson category")

    def test_interpolation_catches_a_storm_between_track_points(self) -> None:
        raw = ParametricIndex.from_cat_in_circle(
            self.tracks,
            self.exposures,
            hazard=self.hazard,
            radius_km=10.0,
            track_step_hours=None,
        )
        fine = ParametricIndex.from_cat_in_circle(
            self.tracks, self.exposures, hazard=self.hazard, radius_km=10.0
        )
        np.testing.assert_array_equal(raw.values, np.zeros((3, 3)))
        np.testing.assert_array_equal(fine.values[0], [4, 4, 0])

    def test_max_wind_metric_is_in_metres_per_second(self) -> None:
        index = ParametricIndex.from_cat_in_circle(
            self.tracks,
            self.exposures,
            hazard=self.hazard,
            radius_km=30.0,
            metric="max_wind",
        )
        np.testing.assert_allclose(index.values[0], 120.5 / 1.9438444924406)
        self.assertEqual(index.unit, "m/s")

    def test_events_need_their_track(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "no track"):
            ParametricIndex.from_cat_in_circle(
                TCTracks(self.tracks.data[:2]),
                self.exposures,
                hazard=self.hazard,
                radius_km=30.0,
            )
        knots = self.tracks.data[0].copy()
        knots.attrs["max_sustained_wind_unit"] = "m/s"
        with self.assertRaisesRegex(ParametricInputError, "kn"):
            ParametricIndex.from_cat_in_circle(
                TCTracks([knots, *self.tracks.data[1:]]),
                self.exposures,
                hazard=self.hazard,
                radius_km=30.0,
            )


if __name__ == "__main__":
    unittest.main()
