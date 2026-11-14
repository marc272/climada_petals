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

Tests of the insured units, on small synthetic CLIMADA objects.

The points and the hazard are those of
:func:`~climada_petals.engine.parametric.test._synthetic.grid_exposures` and
:func:`~climada_petals.engine.parametric.test._synthetic.grid_hazard`: six
points, of which the first three are region 1 (values 1, 3 and 2; the first
two at one centroid), the next two region 2 (values 4 and 4), and the last,
of value 0, in no region.
"""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
from climada.hazard import Centroids, TropCyclone
from shapely.geometry import LineString, MultiPolygon, box

from climada_petals.engine.parametric import (
    ImpactInputError,
    InsuredUnits,
    ParametricInputError,
    UnitMappingError,
)
from climada_petals.engine.parametric.test._synthetic import (
    POINTS,
    exposures,
    grid_exposures,
    grid_hazard,
    grid_impact,
    three_tracks,
)
from climada_petals.util.config import LOGGER

#: The loss of three events at the six points, and its sum per region.
IMP_MAT = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)
BY_REGION = [[6.0, 9.0], [0.5, 0.0], [7.0, 1.0]]


def _regions() -> InsuredUnits:
    """Return the two regions of the grid points."""
    with unittest.TestCase().assertLogs(LOGGER, "WARNING"):
        return InsuredUnits.from_column(grid_exposures(), "region_id")


class TestBuilders(unittest.TestCase):
    """Points, columns, polygons and the whole set."""

    def test_one_unit_per_point(self) -> None:
        exp = exposures()
        units = InsuredUnits.points(exp)
        self.assertEqual(units.unit_id, (0, 1, 2))
        self.assertEqual((units.n_points, units.n_units), (3, 3))
        np.testing.assert_array_equal(units.membership.toarray(), np.eye(3))
        np.testing.assert_array_equal(units.value, [p[2] for p in POINTS])
        np.testing.assert_array_equal(units.unit_value, [p[2] for p in POINTS])
        np.testing.assert_allclose(units.coords, [p[:2] for p in POINTS])
        np.testing.assert_array_equal(units.point_unit, [0, 1, 2])
        self.assertFalse(units.value.flags.writeable)

    def test_a_column_groups_the_points(self) -> None:
        with self.assertLogs(LOGGER, "WARNING") as logs:
            units = InsuredUnits.from_column(grid_exposures(), "region_id")
        self.assertIn("1 of 6 exposure points are in no insured unit", logs.output[0])
        self.assertEqual(units.unit_id, (1, 2))
        np.testing.assert_array_equal(units.point_unit, [0, 0, 0, 1, 1, -1])
        np.testing.assert_array_equal(units.unit_value, [6.0, 8.0])
        np.testing.assert_array_equal(units.value, [1.0, 3.0, 2.0, 4.0, 4.0, 0.0])
        exp = grid_exposures()
        exp.gdf["parish"] = ["west", "west", "east", "east", "west", None]
        units = InsuredUnits.from_column(exp, "parish")
        self.assertEqual(units.unit_id, ("east", "west"))
        np.testing.assert_array_equal(units.point_unit, [1, 1, 0, 0, 1, -1])

    def test_invalid_columns_raise(self) -> None:
        exp = grid_exposures()
        exp.gdf["half"] = [1.5, 1.0, 1.0, 2.0, 2.0, 2.0]
        exp.gdf["flag"] = [True] * 6
        exp.gdf["nothing"] = np.nan
        cases = {
            "missing": ("no column", "parish"),
            "a fraction": ("integers or non-empty strings", "half"),
            "a boolean": ("integers or non-empty strings", "flag"),
            "no value": ("at least one unit", "nothing"),
        }
        for name, (match, column) in cases.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(UnitMappingError, match),
            ):
                InsuredUnits.from_column(exp, column)

    def test_polygons_group_the_points(self) -> None:
        # Region 1 in a polygon; region 2 in two boxes that leave out the
        # last point, 0.001 degrees east of the fifth.
        polygons = gpd.GeoDataFrame(
            {"name": ["r1", "r2"]},
            geometry=[
                box(19.5, 9.5, 21.5, 10.5),
                MultiPolygon(
                    [box(19.9, 10.9, 20.1, 11.1), box(20.9, 10.9, 21.0005, 11.1)]
                ),
            ],
            crs="EPSG:4326",
        )
        for crs in ("EPSG:4326", "EPSG:3857"):
            with (
                self.subTest(crs=crs),
                self.assertLogs(LOGGER, "WARNING") as logs,
            ):
                units = InsuredUnits.from_geometries(
                    grid_exposures(), polygons.to_crs(crs), id_column="name"
                )
            self.assertIn("1 of 6", logs.output[0])
            self.assertEqual(units.unit_id, ("r1", "r2"))
            np.testing.assert_array_equal(units.point_unit, [0, 0, 0, 1, 1, -1])

    def test_invalid_polygons_raise(self) -> None:
        region = box(19.5, 9.5, 21.5, 10.5)
        empty = box(0.0, 0.0, 1.0, 1.0)
        everything = box(19.0, 9.0, 22.0, 12.0)
        cases: dict[str, tuple[Any, str]] = {
            "a polygon without points": ([region, empty], "hold no point"),
            "a point in two polygons": ([region, everything], "two or more units"),
            "a line": ([region, LineString([(0, 0), (1, 1)])], "polygons"),
        }
        for name, (shapes, match) in cases.items():
            gdf = gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=shapes, crs=4326)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(UnitMappingError, match),
            ):
                InsuredUnits.from_geometries(grid_exposures(), gdf, id_column="name")
        gdf = gpd.GeoDataFrame({"name": ["a", "a"]}, geometry=[region, everything])
        for change, match in (
            ({}, "stated crs"),
            ({"id_column": "id"}, "column 'id'"),
            ({"gdf": "polygons"}, "GeoDataFrame"),
            ({"gdf": gdf.set_crs(4326)}, "unique"),
        ):
            arguments: dict[str, Any] = {"gdf": gdf, "id_column": "name", **change}
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(UnitMappingError, match),
            ):
                InsuredUnits.from_geometries(grid_exposures(), **arguments)

    def test_the_whole_set_is_one_unit(self) -> None:
        units = InsuredUnits.whole(grid_exposures())
        self.assertEqual(units.unit_id, ("total",))
        np.testing.assert_array_equal(units.unit_value, [14.0])
        np.testing.assert_array_equal(units.point_unit, [0] * 6)
        self.assertEqual(InsuredUnits.whole(exposures(), unit_id=7).unit_id, (7,))

    def test_the_units_keep_a_copy_of_the_exposures(self) -> None:
        exp = grid_exposures()
        units = InsuredUnits.whole(exp)
        exp.gdf["value"] = 0.0
        np.testing.assert_array_equal(units.unit_value, [14.0])
        self.assertIsNot(units.exposures, exp)
        self.assertEqual(units.exposures.gdf["value"].sum(), 14.0)

    def test_invalid_groupings_raise(self) -> None:
        exp = grid_exposures()
        good = sp.csr_matrix(np.column_stack([np.ones(6), np.zeros(6)]))
        good[0, 0], good[0, 1] = 0.0, 1.0
        two = good.toarray()
        two[1, 1] = 1.0
        cases: dict[str, tuple[Any, Any, str]] = {
            "a two": (good * 2, ["a", "b"], "only 0 and 1"),
            "too few rows": (good[:5], ["a", "b"], "6 rows"),
            "an empty unit": (
                np.column_stack([np.ones(6), np.zeros(6)]),
                [1, 2],
                "no point",
            ),
            "a unit of zero value": (np.eye(6)[:, [0, 5]], [1, 2], "zero value"),
            "a point in two units": (two, ["a", "b"], "two or more units"),
            "more ids than units": (good, ["a", "b", "c"], "unit_id 3"),
            "ids equal as strings": (good, [1, "1"], "unique, also as a string"),
            "no ids": (np.zeros((6, 0)), [], "at least one unit"),
        }
        for name, (membership, unit_id, match) in cases.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(UnitMappingError, match),
            ):
                InsuredUnits(exp, membership, unit_id)

    def test_invalid_exposures_raise(self) -> None:
        for column in ("cover", "deductible"):
            exp = grid_exposures()
            exp.gdf[column] = 1.0
            with (
                self.subTest(column=column),
                self.assertRaisesRegex(ImpactInputError, "drop the columns"),
            ):
                InsuredUnits.whole(exp)
        bad_values = {
            "negative": [-1.0, 3.0, 2.0, 4.0, 4.0, 0.0],
            "not finite": [np.nan, 3.0, 2.0, 4.0, 4.0, 0.0],
        }
        for name, values in bad_values.items():
            exp = grid_exposures()
            exp.gdf["value"] = values
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(UnitMappingError, "negative or not finite"),
            ):
                InsuredUnits.whole(exp)
        exp = grid_exposures()
        exp.gdf["value"] = ["a", "b", "c", "d", "e", "f"]
        with self.assertRaisesRegex(UnitMappingError, "must be numbers"):
            InsuredUnits.whole(exp)
        exp = grid_exposures()
        exp.gdf.drop(columns="value", inplace=True)
        with self.assertRaisesRegex(UnitMappingError, "no value column"):
            InsuredUnits.whole(exp)
        exp = grid_exposures()
        exp.gdf.geometry = [box(0, 0, 1, 1)] * 6
        with self.assertRaisesRegex(UnitMappingError, "must be points"):
            InsuredUnits.whole(exp)
        with self.assertRaisesRegex(ParametricInputError, "climada Exposures"):
            InsuredUnits.points(SimpleNamespace(data=None))


class TestLossesAndPayouts(unittest.TestCase):
    """Losses summed into units and payouts shared back by value."""

    def setUp(self) -> None:
        self.units = _regions()

    def test_aggregate_sums_each_units_points(self) -> None:
        for matrix in (IMP_MAT, sp.csr_matrix(IMP_MAT)):
            out = self.units.aggregate(matrix)
            self.assertTrue(sp.issparse(out))
            np.testing.assert_array_equal(out.toarray(), BY_REGION)
        with self.assertRaisesRegex(ParametricInputError, "imp_mat must have shape"):
            self.units.aggregate(IMP_MAT[:, :5])

    def test_payouts_map_back_by_value_share(self) -> None:
        payout = np.array([[6.0, 8.0], [0.0, 2.0]])
        out = self.units.to_points(sp.csr_matrix(payout)).toarray()
        # Region 1 has the values 1, 3 and 2 (total 6); region 2 has 4 and 4.
        np.testing.assert_allclose(
            out, [[1.0, 3.0, 2.0, 4.0, 4.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]
        )
        np.testing.assert_allclose(out.sum(axis=1), payout.sum(axis=1))
        with self.assertRaisesRegex(ParametricInputError, "payout must have shape"):
            self.units.to_points(np.ones(2))


class TestImpactChecks(unittest.TestCase):
    """An Impact must be the loss on the units' points, per year, gross."""

    def setUp(self) -> None:
        self.exposures = grid_exposures()
        self.units = _regions()
        self.impact = grid_impact(grid_hazard(), self.exposures)

    def test_the_grid_impact_passes(self) -> None:
        self.units.check_impact(self.impact)
        np.testing.assert_allclose(
            self.units.aggregate(self.impact.imp_mat).toarray(),
            [[0.8, 0.8], [2.0, 0.0], [0.4, 4.8]],
        )

    def test_invalid_impacts_raise(self) -> None:
        cases: dict[str, tuple[dict[str, Any], str]] = {
            "no imp_mat": ({"imp_mat": sp.csr_matrix((0, 0))}, "save_mat=True"),
            "none": ({"imp_mat": None}, "save_mat=True"),
            "per day": ({"frequency_unit": "1/day"}, "frequency_unit"),
            "other points": (
                {"imp_mat": sp.csr_matrix(IMP_MAT[:, :5])},
                "one column per exposure point",
            ),
            "other coordinates": (
                {"coord_exp": self.impact.coord_exp + 0.5},
                "coord_exp",
            ),
            "a negative loss": ({"imp_mat": -sp.csr_matrix(IMP_MAT)}, "negative"),
            "a dense NaN": (
                {"imp_mat": np.where(IMP_MAT > 6, np.nan, IMP_MAT)},
                "not finite",
            ),
        }
        for name, (change, match) in cases.items():
            other = copy.deepcopy(self.impact)
            for attribute, value in change.items():
                setattr(other, attribute, value)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ImpactInputError, match),
            ):
                self.units.check_impact(other)


class TestCentroids(unittest.TestCase):
    """Each point's centroid, matched as ImpactCalc matches it."""

    def test_centroids_are_assigned_on_a_copy(self) -> None:
        exp = grid_exposures()
        # A column from another hazard of the same type is not read.
        exp.gdf["centr_FL"] = 3
        with self.assertLogs(LOGGER, "WARNING"):
            units = InsuredUnits.from_column(exp, "region_id")
        np.testing.assert_array_equal(
            units.centroids(grid_hazard()), [0, 0, 1, 2, 3, 3]
        )
        np.testing.assert_array_equal(exp.gdf["centr_FL"], [3] * 6)

    def test_a_point_without_a_centroid(self) -> None:
        far = [(10.0, 20.0, 1.0, 1), (40.0, 60.0, 1.0, np.nan)]
        with self.assertLogs(LOGGER, "WARNING"):
            units = InsuredUnits.from_column(grid_exposures(far), "region_id")
        # A point in no unit may lie far from every centroid.
        np.testing.assert_array_equal(units.centroids(grid_hazard()), [0, -1])
        far[1] = (40.0, 60.0, 1.0, 2)
        units = InsuredUnits.from_column(grid_exposures(far), "region_id")
        with self.assertRaisesRegex(UnitMappingError, "no hazard centroid"):
            units.centroids(grid_hazard())

    def test_duplicate_centroids_from_tracks_raise(self) -> None:
        # Two insured points at one location, and centroids built from the
        # points: TropCyclone.from_tracks returns column indices of -1.
        twice = [POINTS[0], POINTS[0], POINTS[2]]
        exp = exposures(twice)
        centroids = Centroids(lat=exp.latitude, lon=exp.longitude)
        hazard = TropCyclone.from_tracks(
            three_tracks(), centroids=centroids, ignore_distance_to_coast=True
        )
        units = InsuredUnits.whole(exp)
        with self.assertRaisesRegex(UnitMappingError, "unique coordinates"):
            units.centroids(hazard)
        unique = Centroids(lat=exp.latitude[1:], lon=exp.longitude[1:])
        hazard = TropCyclone.from_tracks(
            three_tracks(), centroids=unique, ignore_distance_to_coast=True
        )
        np.testing.assert_array_equal(units.centroids(hazard), [0, 0, 1])

    def test_invalid_hazards_raise(self) -> None:
        units = InsuredUnits.whole(grid_exposures())
        dense = grid_hazard()
        dense.intensity = dense.intensity.toarray()
        csc = grid_hazard()
        csc.intensity = csc.intensity.tocsc()
        narrow = grid_hazard()
        narrow.intensity = narrow.intensity[:, :3]
        for name, hazard in {"dense": dense, "csc": csc, "narrow": narrow}.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(UnitMappingError, "one valid column"),
            ):
                units.centroids(hazard)
        projected = grid_hazard()
        projected.centroids = Centroids(
            lat=projected.centroids.lat, lon=projected.centroids.lon, crs="EPSG:3857"
        )
        with self.assertRaisesRegex(UnitMappingError, "cannot match"):
            units.centroids(projected)


if __name__ == "__main__":
    unittest.main()
