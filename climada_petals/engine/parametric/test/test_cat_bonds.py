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

Tests of working with ``cat_bonds``: its subareas as insured units, its
linear payout, and its calibration by COBYLA.

The data, built by :func:`flood_case`: twelve exposure points on a 4 x 3
grid, 0.1 degrees apart, in two subareas of six points each, ``A`` (the
two western columns) and ``B``; a flood hazard of 120 events of frequency
1/60 each, whose centroids are the exposure points; and an impact function
linear from 1 m (0 %) to 4 m (60 %) and on to 8 m (100 %). With the
centroids on the points, the index ``"mean"`` is the same in both modules:
``cat_bonds`` averages over the centroids inside a subarea's polygon, this
module over the centroids of a unit's points.
"""

from __future__ import annotations

import datetime as dt
import unittest
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.engine import ImpactCalc
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Centroids, Hazard
from numpy.typing import NDArray
from shapely.geometry import box

from climada_petals.engine.cat_bonds.subarea_calculations import (
    SubareaCalculations,
    calc_payout,
)
from climada_petals.engine.cat_bonds.subareas import Subareas
from climada_petals.engine.parametric import (
    MSE,
    BinaryPayout,
    CalibrationResult,
    EventBasis,
    Free,
    InsuredUnits,
    Limit,
    LinearPayout,
    OutcomeSample,
    ParametricCover,
    ParametricIndex,
    ParametricInputError,
    UnitMappingError,
    calibrate,
)
from climada_petals.util.config import LOGGER

#: The number of years the 120 events represent: each has frequency 1/60.
N_YEARS = 60

#: The two subareas: the two western and the two eastern columns of points.
SUBAREAS = (box(19.95, 9.95, 20.15, 10.25), box(20.15, 9.95, 20.35, 10.25))

#: The grid of triggers of the designs, in metres of the index: the
#: attachment from 0 to 2 by 0.1, the exhaustion from 3 to 8 by 0.2. They
#: span the damage function's range (1 to 8 m at the points); 546 ordered
#: candidates per subarea.
ATTACHMENTS = np.round(np.arange(0.0, 2.05, 0.1), 6)
EXHAUSTIONS = np.round(np.arange(3.0, 8.1, 0.2), 6)

#: How far above ``cat_bonds``' COBYLA objective a design on the grid may
#: land: 5 %. On twelve other seeds of :func:`flood_case`, the largest gap
#: of the 24 subareas' grid designs was 3.4 %.
TOLERANCE = 0.05


class FloodCase(NamedTuple):
    """The objects of :func:`flood_case`."""

    exposures: Exposures
    hazard: Hazard
    impfset: ImpactFuncSet
    subareas: Subareas


def flood_case(seed: int = 22) -> FloodCase:
    """Return the data of the module docstring, from a seeded generator."""
    rng = np.random.default_rng(seed)
    lon, lat = np.meshgrid(20.0 + 0.1 * np.arange(4), 10.0 + 0.1 * np.arange(3))
    lon, lat = lon.ravel(), lat.ravel()
    exposures = Exposures(lat=lat, lon=lon, value=rng.uniform(1e6, 3e6, lat.size))
    exposures.gdf["impf_FL"] = 1
    n_events = 120
    depth = rng.gamma(2.0, 1.0, n_events)[:, None] * rng.uniform(
        0.5, 1.5, (n_events, lat.size)
    )
    depth[rng.random(depth.shape) < 0.2] = 0.0
    hazard = Hazard(
        haz_type="FL",
        units="m",
        centroids=Centroids(lat=lat.copy(), lon=lon.copy()),
        event_id=np.arange(1, n_events + 1),
        event_name=[f"e{k}" for k in range(n_events)],
        frequency=np.full(n_events, 1.0 / N_YEARS),
        date=np.array(
            [dt.date(2000 + k // 2, 1 + k % 12, 1).toordinal() for k in range(n_events)]
        ),
        intensity=sp.csr_matrix(depth),
        fraction=sp.csr_matrix(depth.shape),
    )
    impfset = ImpactFuncSet(
        [
            ImpactFunc(
                haz_type="FL",
                id=1,
                intensity=np.array([0.0, 1.0, 4.0, 8.0]),
                mdd=np.array([0.0, 0.0, 0.6, 1.0]),
                paa=np.ones(4),
                intensity_unit="m",
                name="linear",
            )
        ]
    )
    gdf = gpd.GeoDataFrame(geometry=list(SUBAREAS), crs="EPSG:4326")
    subareas = Subareas.from_geodataframe(hazard, impfset, exposures, gdf)
    return FloodCase(exposures, hazard, impfset, subareas)


def outcome_sample(case: FloodCase) -> OutcomeSample:
    """Return the module's sample of a case: index ``"mean"``, event basis."""
    units = InsuredUnits.from_subareas(case.subareas)
    index = ParametricIndex.from_hazard(case.hazard, units, index_stat="mean")
    impact = ImpactCalc(case.exposures, case.impfset, case.hazard).impact(save_mat=True)
    return OutcomeSample.from_impact(index, impact, units=units, basis=EventBasis())


def cobyla(case: FloodCase, initial_guess: tuple[float, float]) -> SubareaCalculations:
    """Return ``cat_bonds``' calibration of a case, from an initial guess.

    The attachment point is 0, so no event's loss is set to 0, and the
    principal is far above every loss, so each subarea's ``max_pay`` is its
    largest event loss.
    """
    calculations = SubareaCalculations(
        case.subareas, index_stat="mean", initial_guess=initial_guess
    )
    calculations.create_pay_vs_dam(attachment_point=0.0, exhaustion_point=1e15)
    return calculations


def squared_error(
    loss: NDArray[np.float64], paid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``cat_bonds``' objective per column: the sum over events of the squared error."""
    return np.asarray(((loss - paid) ** 2).sum(axis=0), dtype=np.float64)


def objectives(result: CalibrationResult) -> NDArray[np.float64]:
    """Return a design's objective per subarea, times the years.

    On the event basis the mean squared error weights each event by its
    frequency, 1/60 here, so 60 times it is ``cat_bonds``' sum.
    """
    by_problem = {
        problem.problem_id: problem.objective_value for problem in result.problems
    }
    return np.array([by_problem[unit] for unit in result.cover.unit_id]) * N_YEARS


def single_subarea(*polygons: Any, values: Any = None) -> Subareas:
    """Return ``Subareas`` with the given polygons on :func:`flood_case`'s points."""
    case = flood_case()
    exposures = case.exposures.copy()
    if values is not None:
        exposures.gdf["value"] = values
    gdf = gpd.GeoDataFrame(geometry=list(polygons), crs="EPSG:4326")
    return Subareas.from_geodataframe(case.hazard, case.impfset, exposures, gdf)


class TestUnitsFromSubareas(unittest.TestCase):
    """``InsuredUnits.from_subareas`` on ``cat_bonds``' ``Subareas``."""

    def test_one_unit_per_subarea(self) -> None:
        case = flood_case()
        units = InsuredUnits.from_subareas(case.subareas)
        self.assertEqual(units.unit_id, ("A", "B"))
        np.testing.assert_array_equal(units.point_unit, [0, 0, 1, 1] * 3)
        given = InsuredUnits.from_subareas(case.subareas, case.exposures)
        np.testing.assert_array_equal(given.point_unit, units.point_unit)
        doubled = case.exposures.copy()
        doubled.gdf["value"] *= 2.0
        np.testing.assert_allclose(
            InsuredUnits.from_subareas(case.subareas, doubled).unit_value,
            2.0 * units.unit_value,
        )

    def test_subareas_from_a_resolution(self) -> None:
        case = flood_case()
        subareas = Subareas.from_resolution(
            case.hazard, case.impfset, case.exposures, 0.2
        )
        units = InsuredUnits.from_subareas(subareas)
        self.assertEqual(units.n_units, subareas.count_subareas())
        self.assertEqual(units.unit_id, tuple(subareas.subareas_gdf["subarea_letter"]))
        self.assertTrue(np.all(units.point_unit >= 0))
        self.assertEqual(int(units.membership.sum()), case.exposures.gdf.shape[0])

    def test_labels_stay_unique_past_26_subareas(self) -> None:
        # One subarea around each of 30 points: cat_bonds labels the 27th
        # to 30th "[", "\", "]" and "^".
        lat = 10.0 + 0.1 * (np.arange(30) // 6)
        lon = 20.0 + 0.1 * (np.arange(30) % 6)
        exposures = Exposures(lat=lat, lon=lon, value=np.ones(30))
        cells = [box(x - 0.05, y - 0.05, x + 0.05, y + 0.05) for x, y in zip(lon, lat)]
        gdf = gpd.GeoDataFrame(geometry=cells, crs="EPSG:4326")
        # Only the polygons and the exposures are read.
        subareas = Subareas.from_geodataframe(None, None, exposures, gdf)
        units = InsuredUnits.from_subareas(subareas)
        self.assertEqual(units.unit_id, tuple(chr(65 + k) for k in range(30)))
        self.assertEqual(units.unit_id[26:], ("[", "\\", "]", "^"))
        np.testing.assert_array_equal(units.point_unit, np.arange(30))

    def test_the_subareas_must_cover_the_points_of_positive_value(self) -> None:
        # Subarea A leaves out its northern row, two points; cat_bonds'
        # from_geodataframe accepts it.
        short = box(19.95, 9.95, 20.15, 10.15)
        subareas = single_subarea(short, SUBAREAS[1])
        with (
            self.assertLogs(LOGGER, "WARNING"),
            self.assertRaisesRegex(UnitMappingError, "2 exposure points of positive"),
        ):
            InsuredUnits.from_subareas(subareas)
        values = flood_case().exposures.gdf["value"].to_numpy().copy()
        values[[8, 9]] = 0.0
        with self.assertLogs(LOGGER, "WARNING") as logs:
            units = InsuredUnits.from_subareas(
                single_subarea(short, SUBAREAS[1], values=values)
            )
        self.assertIn("2 of 12 exposure points are in no insured unit", logs.output[0])
        np.testing.assert_array_equal(units.point_unit[8:], [-1, -1, 1, 1])

    def test_invalid_subareas_raise(self) -> None:
        # Two subareas whose border runs through the third column of points,
        # and a third subarea with no point in it.
        border = (box(19.95, 9.95, 20.2, 10.25), box(20.2, 9.95, 20.35, 10.25))
        empty = box(30.0, 30.0, 31.0, 31.0)
        for polygons, match in (
            (border, "two or more units"),
            ((*SUBAREAS, empty), "hold no point"),
        ):
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(UnitMappingError, match),
            ):
                InsuredUnits.from_subareas(single_subarea(*polygons))
        subareas = flood_case().subareas
        subareas.subareas_gdf = subareas.subareas_gdf.drop(columns="subarea_letter")
        with self.assertRaisesRegex(UnitMappingError, "subarea_letter"):
            InsuredUnits.from_subareas(subareas)
        with self.assertRaisesRegex(ParametricInputError, "Subareas"):
            InsuredUnits.from_subareas(flood_case().exposures)


class TestPayoutAgreement(unittest.TestCase):
    """``LinearPayout`` pays what ``calc_payout`` pays.

    On float index values only: ``calc_payout`` builds its payouts with
    ``numpy.zeros_like`` on the index, so on an integer-typed index it
    truncates them, and ``LinearPayout`` does not
    (``test_payout.TestLinearPayout`` shows both). The index here is a mean
    over each subarea's centroids, a float, as in ``cat_bonds``.
    """

    def test_linear_payout_equals_calc_payout_on_the_subarea_index(self) -> None:
        index = outcome_sample(flood_case()).index
        values = index.values
        # Triggers between the grid's, and triggers on index values, where
        # the payout is 0 at min_trig and max_pay at max_trig.
        on_values = (float(np.sort(values[:, 0])[40]), float(np.sort(values[:, 0])[90]))
        for min_trig, max_trig, max_pay in (
            (0.63, 5.92, 1.2e7),
            (*on_values, 3.5e6),
            (1e-3, 0.5, 1.0),
        ):
            cover = ParametricCover(
                index.definition,
                LinearPayout.from_triggers(min_trig, max_trig, max_pay),
                unit_id=index.unit_id,
                payout_unit="USD",
            )
            paid = cover.payout_matrix(index)
            for column in range(index.n_units):
                expected = calc_payout(
                    min_trig, max_trig, pd.DataFrame(values[:, column]), max_pay
                )
                with self.subTest(triggers=(min_trig, max_trig), column=column):
                    np.testing.assert_allclose(
                        paid[:, column], expected, rtol=1e-12, atol=0.0
                    )

    def test_inverted_triggers_pay_as_a_binary_payout(self) -> None:
        # With min_trig > max_trig, calc_payout pays max_pay at or above
        # max_trig and nothing below it.
        values = np.concatenate(
            [np.random.default_rng(22).uniform(0.0, 8.0, 200), [3.0, 4.0]]
        )
        inverted = calc_payout(4.0, 3.0, pd.DataFrame(values), 2.5e6)
        np.testing.assert_array_equal(
            inverted, BinaryPayout(trigger=3.0, amount=2.5e6).evaluate(values)
        )


class TestCalibrationAgainstCobyla(unittest.TestCase):
    """An MSE design on a grid of triggers against ``cat_bonds``' COBYLA fit.

    ``cat_bonds`` fits each subarea's ``min_trig`` and ``max_trig`` by
    COBYLA, minimising the sum over events of the squared error between
    the subarea's loss and ``calc_payout``, with ``max_pay`` fixed at the
    subarea's largest loss. The module's :class:`MSE` on the event basis
    is that sum times the events' common frequency, 1/60.
    """

    case: FloodCase
    sample: OutcomeSample
    loss: NDArray[np.float64]
    max_pay: NDArray[np.float64]
    fitted: SubareaCalculations
    on_the_grid: CalibrationResult

    @classmethod
    def setUpClass(cls) -> None:
        """Run ``cat_bonds``' fit and the design with its ``max_pay``, once."""
        cls.case = flood_case()
        cls.sample = outcome_sample(cls.case)
        cls.loss = np.asarray(sp.csr_matrix(cls.sample.loss).toarray())
        cls.max_pay = cls.loss.max(axis=0)
        cls.fitted = cobyla(cls.case, initial_guess=(1.0, 4.0))
        # The same family as cat_bonds': triggers on the grid, max_pay fixed.
        payouts = {
            unit: LinearPayout.from_triggers(
                Free(0.0, 2.0, grid=ATTACHMENTS),
                Free(3.0, 8.0, grid=EXHAUSTIONS),
                float(cls.max_pay[column]),
            )
            for column, unit in enumerate(cls.sample.unit_id)
        }
        template = ParametricCover(
            cls.sample.index.definition,
            payouts,
            unit_id=cls.sample.unit_id,
            payout_unit="USD",
        )
        cls.on_the_grid = calibrate(template, cls.sample, MSE(), scope="individual")

    def paid_by(self, fitted: SubareaCalculations) -> NDArray[np.float64]:
        """Return what ``cat_bonds`` pays per event and subarea after a fit.

        These are ``calc_payout``'s payouts at the thresholds that
        ``create_pay_vs_dam`` keeps (COBYLA's, or its fallback after a
        failed fit), whatever their order.
        """
        values = self.sample.index.values
        return np.column_stack(
            [
                calc_payout(
                    fitted.opt_min_thresh[column],
                    fitted.opt_max_thresh[column],
                    pd.DataFrame(values[:, column]),
                    self.max_pay[column],
                )
                for column in range(values.shape[1])
            ]
        )

    def test_the_objectives_agree_at_cobylas_triggers(self) -> None:
        # The index, the loss per subarea, the payout and the objective are
        # the same in both modules: the module's MSE at COBYLA's triggers,
        # times 60, is the value COBYLA reports.
        self.assertEqual(self.sample.unit_id, ("A", "B"))
        for column, result in self.fitted.results.items():
            min_trig, max_trig = result.x
            with self.subTest(subarea=self.sample.unit_id[column]):
                self.assertTrue(result.success)
                self.assertLess(min_trig, max_trig)
                paid = LinearPayout.from_triggers(
                    min_trig, max_trig, self.max_pay[column]
                ).evaluate(self.sample.index.values[:, column])
                value = MSE().evaluate(self.loss[:, column], paid, self.sample.measure)
                self.assertAlmostEqual(value * N_YEARS / result.fun, 1.0, places=9)
        np.testing.assert_allclose(
            squared_error(self.loss, self.paid_by(self.fitted)),
            [result.fun for result in self.fitted.results.values()],
            rtol=1e-9,
        )

    def test_a_design_on_the_grid_reaches_cobyla(self) -> None:
        # On this data the grid's designs are 1.4 % (A) and 0.4 % (B) above
        # COBYLA's objective.
        reached = squared_error(self.loss, self.paid_by(self.fitted))
        designed = objectives(self.on_the_grid)
        np.testing.assert_array_less(designed, reached * (1.0 + TOLERANCE))
        for problem in self.on_the_grid.problems:
            self.assertEqual(problem.solver.route, "qp")
            self.assertEqual(problem.n_candidates, 546)

    def test_a_free_limit_does_at_least_as_well(self) -> None:
        # The module's own design: the limit is free up to each subarea's
        # largest loss (Limit.largest_loss), which is cat_bonds' max_pay
        # here, so at each candidate it is solved over a range that holds
        # max_pay. On this data it is 1.7 % below COBYLA's objective in A
        # and 0.4 % above it in B.
        template = ParametricCover(
            self.sample.index.definition,
            LinearPayout(
                Free(0.0, 2.0, grid=ATTACHMENTS),
                Free(3.0, 8.0, grid=EXHAUSTIONS),
                Free(0.0),
            ),
            unit_id=self.sample.unit_id,
            payout_unit="USD",
            limit=Limit.largest_loss(),
        )
        free = calibrate(template, self.sample, MSE(), scope="individual")
        np.testing.assert_array_less(
            objectives(free), objectives(self.on_the_grid) * (1.0 + 1e-9)
        )
        limits = free.parameters_frame().query("parameter == 'limit'")["value"]
        self.assertTrue(np.all(limits.to_numpy() <= self.max_pay * (1.0 + 1e-9)))
        reached = squared_error(self.loss, self.paid_by(self.fitted))
        np.testing.assert_array_less(objectives(free), reached * (1.0 + TOLERANCE))

    def test_against_a_fit_with_inverted_triggers(self) -> None:
        # COBYLA is unconstrained. From the initial guess (5, 1) it keeps
        # min_trig > max_trig on this data, where calc_payout pays max_pay
        # from max_trig on (test_inverted_triggers_pay_as_a_binary_payout),
        # and the design on the grid is about 97 % below what it pays.
        # Whichever way the fit lands, the design reaches what it pays.
        inverted = cobyla(self.case, initial_guess=(5.0, 1.0))
        reached = squared_error(self.loss, self.paid_by(inverted))
        np.testing.assert_array_less(
            objectives(self.on_the_grid), reached * (1.0 + TOLERANCE)
        )


if __name__ == "__main__":
    unittest.main()
