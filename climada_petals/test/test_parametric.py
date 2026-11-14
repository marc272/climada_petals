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

Integration tests of :mod:`climada_petals.engine.parametric` on hazard sets
and exposures of CLIMADA's Data API: two hazard types end to end, from the
``Hazard``, the ``Exposures`` and ``ImpactCalc`` to ``calibrate``,
``evaluate`` and ``compare``; the index diagnostics on real hazards; and
insured units from ``cat_bonds``' subareas. The scenarios and the lineage
of samples are in :mod:`climada_petals.test.test_parametric_scenarios`,
which uses the data and the helpers of this module.

The data, public datasets of the Data API (CC BY 4.0), read with
``climada.util.api_client.Client``, with no credentials. The client keeps
the files in CLIMADA's data folder, so only the first run downloads them,
about 205 MB, 164 MB of which are STORM's:

* tropical cyclones for Jamaica: the ``random_walk`` set of 1980 to 2020
  (:data:`TC_PRESENT`), its rcp 4.5 sets at 2040, 2060 and 2080
  (:data:`TC_RCP45`), the observed set of 1980 to 2020 (:data:`TC_OBSERVED`)
  and the STORM set (:data:`TC_STORM`), with Jamaica's LitPop and Emanuel's
  impact function for the USA;
* European windstorm for Luxembourg (:data:`WS_LUX`), with Luxembourg's
  LitPop and Schwierz's impact function.

Year sets are sampled from the frequencies, :data:`N_YEARS` years per
hazard set, so that each side of a 70/30 split holds at least 10,000.
Splits keep each ensemble's members with their parent storm.

Run them with the network open to the Data API only::

    python -m pytest -q -p no:cacheprovider \\
      --allow-hosts=climada.ethz.ch,data.iac.ethz.ch --allow-unix-socket \\
      climada_petals/test/test_parametric*.py
"""

from __future__ import annotations

import contextlib
import functools
import unittest
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
from climada.engine import Impact, ImpactCalc
from climada.entity import Exposures, ImpactFuncSet
from climada.entity.impact_funcs.storm_europe import ImpfStormEurope
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
from climada.hazard import Hazard
from climada.util.api_client import Client
from numpy.typing import NDArray
from shapely.geometry import box

from climada_petals.engine.cat_bonds.subareas import Subareas
from climada_petals.engine.parametric import (
    ENSEMBLE_PARENT,
    MSE,
    Budget,
    CalibrationResult,
    CatalogueYears,
    CostOfCapitalLoad,
    EventBasis,
    ExpectedUncovered,
    Free,
    InsuredUnits,
    LeakageError,
    Limit,
    LinearPayout,
    OutcomeSample,
    ParametricCover,
    ParametricIndex,
    PurePremium,
    StepPayout,
    WeightedCVaR,
    YearRouteError,
    YearSets,
    calibrate,
    compare,
    evaluate,
    holdout,
    index_diagnostics,
    labels_from_names,
    payout_impact,
    setup_text,
    uncovered_impact,
)
from climada_petals.util.config import LOGGER

#: The Data API's types of the two hazards.
TC, WS = "tropical_cyclone", "storm_europe"

#: Jamaica's ``random_walk`` set: 3,960 storms of 1980 to 2020 with ten
#: perturbed members each, every storm of the global set.
TC_PRESENT = "tropical_cyclone_10synth_tracks_150arcsec_JAM_1980_2020"

#: The same storms under rcp 4.5, at the reference years 2040, 2060, 2080.
TC_RCP45 = tuple(
    f"tropical_cyclone_10synth_tracks_150arcsec_rcp45_JAM_{year}"
    for year in (2040, 2060, 2080)
)

#: Jamaica's observed storms of 1980 to 2020, 41 calendar years.
TC_OBSERVED = "tropical_cyclone_0synth_tracks_150arcsec_historical_JAM_1980_2020"

#: Jamaica's STORM set: 10,000 simulated years, at 300 arcsec.
TC_STORM = "TC_JAM_0300as_STORM"

#: Luxembourg's windstorm set: 142 storms of 1940 to 2013, 30 members each.
WS_LUX = "storm_europe_150arcsec_LUX"

#: Years sampled per hazard set on year sets. A 70/30 split of parents
#: leaves each side at least 10,000, as the docs ask for tail measures.
N_YEARS = 40_000

#: The grid of ``Subareas.from_resolution`` for Jamaica, in degrees.
SUBAREA_RESOLUTION = 0.5

#: Two parts of Luxembourg, west and east of 6.1 degrees.
LUX_PARTS = gpd.GeoDataFrame(
    {"part": ["west", "east"]},
    geometry=[box(5.5, 49.3, 6.1, 50.3), box(6.1, 49.3, 6.7, 50.3)],
    crs="EPSG:4326",
)

#: The metrics every row of an annual evaluation fills.
METRICS = (
    "expected_loss",
    "expected_payout",
    "expected_uncovered",
    "premium",
    "coverage_ratio",
    "tail_coverage_b0_99",
    "cvar_uncovered_to_loss_b0_99",
    "payout_frequency",
    "rmse",
    "var_uncovered_b0_99",
    "cvar_uncovered_b0_99",
    "cvar_net_loss_b0_99",
    "loss_rp100",
)


@contextlib.contextmanager
def cat_bonds_deprecations() -> Iterator[None]:
    """Ignore the deprecation warnings raised by ``cat_bonds``' own code."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"climada_petals\.engine\.cat_bonds\.",
        )
        yield


@functools.cache
def client() -> Client:
    """Return one Data API client for the run."""
    return Client()


@functools.cache
def data_api_hazard(data_type: str, name: str) -> Hazard:
    """Return a hazard set of the Data API, in its newest active version."""
    return client().get_hazard(data_type, name=name)


@functools.cache
def litpop(country: str, haz_type: str) -> Exposures:
    """Return a country's LitPop, with impact function 1 for ``haz_type``."""
    exposures = client().get_litpop(country)
    exposures.gdf[f"impf_{haz_type}"] = 1
    return exposures


def tc_impfset() -> ImpactFuncSet:
    """Return Emanuel's impact function for the USA, the default of CLIMADA."""
    return ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])


def ws_impfset() -> ImpactFuncSet:
    """Return Schwierz's impact function for European windstorm."""
    return ImpactFuncSet([ImpfStormEurope.from_schwierz()])


def impact_of(exposures: Exposures, impfset: ImpactFuncSet, hazard: Hazard) -> Impact:
    """Return the ``Impact`` with ``imp_mat``, on a copy of the exposures."""
    return ImpactCalc(exposures.copy(), impfset, hazard).impact(save_mat=True)


def tc_parents(hazard: Hazard) -> list[str]:
    """Return each event's parent storm: its name without ``_gen<m>``."""
    return labels_from_names(hazard.event_name, ENSEMBLE_PARENT)["block"]


def ws_parents(hazard: Hazard) -> NDArray[np.int64]:
    """Return each windstorm event's parent: its id is ``100 parent + member``."""
    return np.asarray(hazard.event_id) // 100


def with_wind(hazard: Hazard) -> Hazard:
    """Return the events with wind at some centroid of the set.

    The others have no loss and an index of 0 at every unit, so a sample on
    the event basis leaves them out anyway.
    """
    return hazard.select(event_id=hazard.event_id[hazard.intensity.getnnz(axis=1) > 0])


def year_sets(hazard: Hazard, rng: np.random.Generator, block: Any) -> YearSets:
    """Return :data:`N_YEARS` years sampled from the hazard's frequencies."""
    return YearSets.from_frequencies(hazard, n_years=N_YEARS, rng=rng, block=block)


class Case(NamedTuple):
    """One hazard set end to end: the CLIMADA objects and the samples."""

    hazard: Hazard
    exposures: Exposures
    impfset: ImpactFuncSet
    impact: Impact
    units: InsuredUnits
    index: ParametricIndex
    sample: OutcomeSample
    train: OutcomeSample
    test: OutcomeSample

    def event_sample(self) -> OutcomeSample:
        """Return the sample on the event basis."""
        return OutcomeSample.from_impact(
            self.index, self.impact, units=self.units, basis=EventBasis()
        )

    def template(
        self,
        attachment: float,
        exhaustion: float,
        n_steps: int,
        *,
        label: str | None = None,
        unit_id: tuple[str, ...] | None = None,
    ) -> ParametricCover:
        """Return a step template with free levels, at most the largest loss.

        Its units are the case's, or the ones given.
        """
        return ParametricCover(
            self.index.definition,
            StepPayout.uniform(
                attachment=attachment, exhaustion=exhaustion, n_steps=n_steps
            ),
            unit_id=self.units.unit_id if unit_id is None else unit_id,
            payout_unit=self.impact.unit,
            limit=Limit.largest_loss(),
            label=label,
        )


def case_of(
    hazard: Hazard,
    exposures: Exposures,
    impfset: ImpactFuncSet,
    units: InsuredUnits,
    *,
    block: Any,
    seed: int,
) -> Case:
    """Return a case: the index ``"mean"`` weighted by value, on year sets."""
    impact = impact_of(exposures, impfset, hazard)
    index = ParametricIndex.from_hazard(
        hazard, units, index_stat="mean", weight="value"
    )
    rng = np.random.default_rng(seed)
    sample = OutcomeSample.from_impact(
        index, impact, units=units, basis=year_sets(hazard, rng, block)
    )
    train, test = holdout(sample, train_fraction=0.7, rng=rng)
    return Case(hazard, exposures, impfset, impact, units, index, sample, train, test)


@functools.cache
def jamaica_subareas() -> Subareas:
    """Return ``cat_bonds``' subareas of Jamaica's LitPop, on a 0.5 degree grid."""
    exposures = litpop("JAM", "TC")
    with cat_bonds_deprecations():
        return Subareas.from_resolution(
            data_api_hazard(TC, TC_PRESENT),
            tc_impfset(),
            exposures.copy(),
            SUBAREA_RESOLUTION,
        )


@functools.cache
def jamaica() -> Case:
    """Return the tropical cyclone case: ``random_walk``, Jamaica's subareas."""
    hazard = data_api_hazard(TC, TC_PRESENT)
    exposures = litpop("JAM", "TC")
    units = InsuredUnits.from_subareas(jamaica_subareas(), exposures)
    return case_of(
        hazard, exposures, tc_impfset(), units, block=tc_parents(hazard), seed=3911
    )


@functools.cache
def luxembourg() -> Case:
    """Return the windstorm case: Luxembourg in two parts."""
    hazard = data_api_hazard(WS, WS_LUX)
    exposures = litpop("LUX", "WS")
    units = InsuredUnits.from_geometries(exposures, LUX_PARTS, id_column="part")
    return case_of(
        hazard, exposures, ws_impfset(), units, block=ws_parents(hazard), seed=3912
    )


def designs_of(
    case: Case, template: ParametricCover, runs: Sequence[tuple[Any, str, Any]]
) -> list[CalibrationResult]:
    """Calibrate ``template`` on the training side: (objective, scope, pricing)."""
    return [
        calibrate(
            template,
            case.train,
            objective,
            scope=scope,
            pricing=pricing or PurePremium(),
            budget=Budget(ratio=1.0),
        )
        for objective, scope, pricing in runs
    ]


def total(table: pd.DataFrame, **match: Any) -> pd.Series:
    """Return the one total row of a table that matches the columns given."""
    rows = table[table["level"] == "total"]
    for column, value in match.items():
        rows = rows[rows[column] == value]
    assert len(rows) == 1, f"{len(rows)} total rows match {match}"
    return rows.iloc[0]


class EndToEnd(unittest.TestCase):
    """Checks that both hazard types share; see the subclasses."""

    #: The attachment, exhaustion and number of steps of a step template.
    thresholds: tuple[float, float, int] = (0.0, 1.0, 1)

    def check_the_sample_holds_the_impact(self, case: Case) -> None:
        """The losses and the expected loss are the ``Impact``'s."""
        sample = case.event_sample()
        # Every point of positive value is in a unit, so the units hold the
        # whole loss: per event, and per year on the event basis.
        self.assertTrue(np.all(case.units.point_unit >= 0))
        kept = np.isin(case.impact.event_id, sample.event_id)
        np.testing.assert_allclose(
            np.asarray(sample.loss.sum(axis=1)).ravel(),
            case.impact.at_event[kept],
            rtol=1e-12,
        )
        self.assertEqual(np.count_nonzero(case.impact.at_event[~kept]), 0)
        self.assertEqual(sample.n_left_out, case.hazard.size - sample.n_events)
        cover = calibrate(
            case.template(*self.thresholds, label="on the event basis"),
            sample,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=1.0),
        ).cover
        table = evaluate(cover, sample, split="in_sample")
        self.assertAlmostEqual(
            total(table)["expected_loss"] / case.impact.aai_agg, 1.0, delta=1e-12
        )
        per_unit = case.units.membership.T @ case.impact.eai_exp
        units = table[table["level"] == "unit"].set_index("unit_id")
        np.testing.assert_allclose(
            units.loc[list(case.units.unit_id), "expected_loss"], per_unit, rtol=1e-9
        )
        # The payouts and the uncovered losses as Impacts, per point: their
        # aai_agg is the sum of the unit rows' expectations on the event basis.
        for impact, column in (
            (payout_impact(cover, sample), "expected_payout"),
            (uncovered_impact(cover, sample), "expected_uncovered"),
        ):
            with self.subTest(column=column):
                self.assertEqual(
                    impact.imp_mat.shape, (sample.n_events, case.units.n_points)
                )
                self.assertAlmostEqual(
                    impact.aai_agg / units[column].sum(), 1.0, delta=1e-9
                )
                curve = impact.calc_freq_curve()
                self.assertTrue(np.all(np.diff(curve.impact) >= 0))
        # On year sets, the mean of the sampled years is near it.
        yearly = evaluate(cover, case.sample, split="in_sample")
        self.assertAlmostEqual(
            total(yearly)["expected_loss"] / case.impact.aai_agg, 1.0, delta=0.05
        )

    def check_designs(self, case: Case, designs: list[CalibrationResult]) -> None:
        """The designs hold their budgets, and compare out of sample."""
        for design in designs:
            summary = design.summary()
            with self.subTest(design=design.cover.label):
                self.assertTrue(np.all(summary["budget_use"] <= 1.0 + 1e-6))
                self.assertEqual(set(summary["status"]), {"0"})
                self.assertLessEqual(
                    design.premium, design.expected_loss * (1.0 + 1e-6)
                )
                self.assertEqual(design.route, "yearsets")
        table = compare(designs, case.test, split="out_of_sample")
        n_rows = [
            1 if design.scope == "joint" else case.units.n_units + 1
            for design in designs
        ]
        self.assertEqual(len(table), sum(n_rows))
        self.assertEqual(set(table["split"]), {"out_of_sample"})
        self.assertEqual(set(table["route"]), {"yearsets"})
        self.assertEqual(set(table["n_years"]), {case.test.n_years})
        self.assertEqual(set(table["premium_basis"]), {"design"})
        totals = table[table["level"] == "total"]
        self.assertEqual(len(totals), len(designs))
        self.assertTrue(np.all(np.isfinite(totals[list(METRICS)].astype(float))))
        self.assertTrue(
            np.all(totals["expected_loss"] == totals["expected_loss"].iloc[0])
        )
        self.assertTrue(
            np.all((totals["coverage_ratio"] >= 0) & (totals["coverage_ratio"] <= 1))
        )
        for design, (_, row) in zip(designs, totals.iterrows()):
            with self.subTest(design=design.cover.label):
                self.assertAlmostEqual(row["premium"], design.premium)
                self.assertEqual(row["scope"], design.scope)
                self.assertEqual(
                    row["comonotone_index"],
                    bool(index_diagnostics(case.test)["comonotone_index"].all()),
                )
        text = setup_text(totals.iloc[0])
        self.assertIn(f"annual basis on {case.test.n_years} sampled years", text)
        self.assertIn("out of sample", text)


class TestTropicalCycloneJamaica(EndToEnd):
    """Tropical cyclones: Jamaica's ``random_walk`` set and LitPop, in subareas."""

    #: Steps from 30 m/s (Emanuel's threshold is 25.7 m/s) to 70 m/s.
    thresholds = (30.0, 70.0, 5)

    def test_the_sample_holds_the_impact(self) -> None:
        self.check_the_sample_holds_the_impact(jamaica())

    def test_designs_out_of_sample(self) -> None:
        case = jamaica()
        template = case.template(*self.thresholds, label="step")
        designs = designs_of(
            case,
            template,
            [
                (WeightedCVaR(alpha=0.9, beta=0.99), "joint", None),
                (ExpectedUncovered(), "individual", None),
                (MSE(), "individual", None),
            ],
        )
        self.assertEqual(
            [design.summary()["route"].unique().tolist() for design in designs],
            [["lp"], ["lp"], ["qp"]],
        )
        self.check_designs(case, designs)

    def test_a_design_on_the_training_side_raises_on_the_whole_sample(self) -> None:
        case = jamaica()
        (design,) = designs_of(
            case,
            case.template(*self.thresholds, label="step"),
            [(ExpectedUncovered(), "joint", None)],
        )
        self.assertEqual(design.design_sample.source, case.sample.source)
        for sample in (case.sample, case.event_sample()):
            with self.subTest(basis=sample.kind), self.assertRaises(LeakageError):
                evaluate(design, sample, split="out_of_sample")
        with self.assertRaises(LeakageError):
            compare([design], case.sample, split="out_of_sample")
        table = evaluate(design, case.test, split="out_of_sample")
        self.assertEqual(len(table), 1)

    def test_units_from_subareas(self) -> None:
        case = jamaica()
        subareas = jamaica_subareas()
        letters = tuple(subareas.subareas_gdf["subarea_letter"])
        self.assertEqual(case.units.unit_id, letters)
        self.assertGreater(len(letters), 1)
        # Every point is in one subarea, and the units hold all the value.
        np.testing.assert_array_equal(
            np.asarray(case.units.membership.sum(axis=1)).ravel(), 1
        )
        self.assertAlmostEqual(
            case.units.unit_value.sum() / case.exposures.gdf["value"].sum(),
            1.0,
            delta=1e-12,
        )
        # A unit's index is the value-weighted mean of the wind at the
        # centroids of its points.
        centroid = case.units.centroids(case.hazard)
        unit = int(np.argmax(case.units.unit_value))
        points = np.flatnonzero(case.units.point_unit == unit)
        wind = case.hazard.intensity[:, centroid[points]].toarray()
        value = case.units.value[points]
        np.testing.assert_allclose(
            case.index.values[:, unit], wind @ value / value.sum(), rtol=1e-9, atol=1e-9
        )

    def test_one_unit_of_the_sample(self) -> None:
        case = jamaica()
        unit_id = case.units.unit_id[int(np.argmax(case.units.unit_value))]
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            train, test = case.train.unit(unit_id), case.test.unit(unit_id)
        self.assertIn("are in no insured unit", logs.output[0])
        for part, whole in ((train, case.train), (test, case.test)):
            self.assertEqual(part.unit_id, (unit_id,))
            np.testing.assert_array_equal(part.event_id, whole.event_id)
            self.assertEqual(part.source, case.sample.source)
            self.assertEqual(part.n_years, whole.n_years)
        design = calibrate(
            case.template(*self.thresholds, unit_id=(unit_id,)),
            train,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=1.0),
        )
        # In the individual scope, the unit's problem is the same in a
        # design of every unit.
        (every,) = designs_of(
            case,
            case.template(*self.thresholds, label="every unit"),
            [(ExpectedUncovered(), "individual", None)],
        )
        self.assertAlmostEqual(
            design.objective_value
            / every.summary().set_index("problem_id").loc[unit_id, "objective_value"],
            1.0,
            delta=1e-6,
        )
        table = evaluate(design, test, split="out_of_sample")
        self.assertEqual(
            table.loc[table["level"] == "unit", "unit_id"].tolist(), [unit_id]
        )
        with self.assertRaises(LeakageError):
            evaluate(design, case.sample.unit(unit_id), split="out_of_sample")
        # The payouts map back to the unit's points, and to no other.
        paid = payout_impact(design, test)
        self.assertEqual(paid.imp_mat.shape[1], case.units.n_points)
        elsewhere = case.units.point_unit != case.units.unit_id.index(unit_id)
        self.assertEqual(paid.imp_mat[:, elsewhere].nnz, 0)
        self.assertGreater(paid.imp_mat.nnz, 0)


class TestWindstormLuxembourg(EndToEnd):
    """European windstorm: Luxembourg's set and LitPop, in two parts."""

    #: Steps from 25 m/s (Schwierz's function starts at 20 m/s) to 40 m/s.
    thresholds = (25.0, 40.0, 3)

    def test_the_sample_holds_the_impact(self) -> None:
        self.check_the_sample_holds_the_impact(luxembourg())

    def test_designs_out_of_sample(self) -> None:
        case = luxembourg()
        step = case.template(*self.thresholds, label="step")
        designs = designs_of(
            case,
            step,
            [
                (MSE(), "individual", None),
                (
                    WeightedCVaR(alpha=0.9, beta=0.99),
                    "joint",
                    CostOfCapitalLoad(rate=0.1, q=0.99),
                ),
            ],
        )
        # A linear payout with its attachment on a grid.
        linear = ParametricCover(
            case.index.definition,
            LinearPayout(Free(20.0, 30.0, grid=11), 40.0, Free(0.0)),
            unit_id=case.units.unit_id,
            payout_unit=case.impact.unit,
            limit=Limit.largest_loss(),
            label="linear",
        )
        designs += designs_of(case, linear, [(ExpectedUncovered(), "individual", None)])
        routes = [design.summary()["route"].unique().tolist() for design in designs]
        self.assertEqual(routes, [["qp"], ["lp"], ["lp"]])
        self.assertEqual(
            designs[2].summary()["n_candidates"].tolist(), [11] * case.units.n_units
        )
        self.check_designs(case, designs)

    def test_catalogue_years_from_the_dates_raise(self) -> None:
        # The dates span 74 seasons, and the frequency is 1 / (73 x 30).
        case = luxembourg()
        with self.assertRaisesRegex(YearRouteError, "frequency x N"):
            OutcomeSample.from_impact(
                case.index,
                case.impact,
                units=case.units,
                basis=CatalogueYears.from_dates(case.hazard, season_start_month=7),
            )


class TestIndexDiagnostics(unittest.TestCase):
    """``index_diagnostics`` on the real sets, on the event basis.

    The diagnostics read the relation of index and loss per event. Their
    values on these sets are observations; the tests check what holds
    whatever they are.
    """

    def diagnose(self, sample: OutcomeSample) -> pd.DataFrame:
        """Return the diagnostics, checked against their definition."""
        table = index_diagnostics(sample)
        self.assertEqual(table["unit_id"].tolist(), list(sample.unit_id))
        self.assertTrue(np.all(np.abs(table["spearman"]) <= 1.0 + 1e-12))
        np.testing.assert_array_equal(
            table["comonotone_index"].astype(bool),
            (table["monotone_gap"] <= table["tolerance"])
            | (table["spearman"] >= table["threshold"] - 1e-12),
        )
        # An event with a loss at a unit has wind at one of its centroids, so
        # an index of 0 carries no loss.
        np.testing.assert_array_equal(table["zero_index_loss_share"], 0.0)
        loss = sample.loss.toarray()
        index = sample.index_values
        np.testing.assert_array_equal(
            table["n_events"], ((loss > 0) | (index != 0)).sum(axis=0)
        )
        return table

    def one_point_per_unit(
        self, hazard: Hazard, exposures: Exposures, impfset: ImpactFuncSet
    ) -> OutcomeSample:
        """Return the sample of one unit per point, indexed at its centroid."""
        hazard = with_wind(hazard)
        units = InsuredUnits.points(exposures)
        return OutcomeSample.from_impact(
            ParametricIndex.from_hazard(hazard, units),
            impact_of(exposures, impfset, hazard),
            units=units,
            basis=EventBasis(),
        )

    def test_one_point_per_unit(self) -> None:
        # One point, the wind at its centroid, one impact function: the loss
        # is a non-decreasing function of the index at every unit, and every
        # unit is flagged, whatever its Spearman correlation: the storms with
        # wind below the impact function's threshold lose nothing.
        for label, hazard, exposures, impfset in (
            ("TC", data_api_hazard(TC, TC_PRESENT), litpop("JAM", "TC"), tc_impfset()),
            ("WS", data_api_hazard(WS, WS_LUX), litpop("LUX", "WS"), ws_impfset()),
        ):
            with self.subTest(hazard=label):
                sample = self.one_point_per_unit(hazard, exposures, impfset)
                self.assertEqual(sample.n_units, exposures.gdf.shape[0])
                loss, index = sample.loss.toarray(), sample.index_values
                order = np.argsort(index, axis=0, kind="stable")
                ordered = np.take_along_axis(loss, order, axis=0)
                self.assertTrue(
                    np.all(np.diff(ordered, axis=0) >= -1e-6 * ordered.max())
                )
                table = self.diagnose(sample)
                self.assertTrue(np.all(table["monotone_gap"] <= table["tolerance"]))
                self.assertEqual(
                    table["comonotone_index"].tolist(), [True] * sample.n_units
                )

    def test_groups_of_points(self) -> None:
        for label, case in (("TC", jamaica()), ("WS", luxembourg())):
            whole = InsuredUnits.whole(case.exposures, unit_id=label)
            whole_index = ParametricIndex.from_hazard(
                case.hazard, whole, index_stat="mean", weight="value"
            )
            for units, sample in (
                ("groups", case.event_sample()),
                (
                    "whole",
                    OutcomeSample.from_impact(
                        whole_index, case.impact, units=whole, basis=EventBasis()
                    ),
                ),
            ):
                with self.subTest(hazard=label, units=units):
                    # A group's loss is no function of its mean wind.
                    table = self.diagnose(sample)
                    self.assertTrue(np.all(table["spearman"] < 1.0))
                    self.assertTrue(np.all(table["monotone_gap"] > table["tolerance"]))
                    self.assertEqual(
                        table["comonotone_index"].tolist(), [False] * sample.n_units
                    )

    def test_designs_carry_the_flags(self) -> None:
        case = jamaica()
        (design,) = designs_of(
            case,
            case.template(30.0, 70.0, 5, label="step"),
            [(ExpectedUncovered(), "individual", None)],
        )
        flags = index_diagnostics(case.train).set_index("unit_id")["comonotone_index"]
        pd.testing.assert_frame_equal(
            design.index_diagnostics, index_diagnostics(case.train)
        )
        summary = design.summary().set_index("problem_id")
        np.testing.assert_array_equal(
            summary.loc[list(case.units.unit_id), "comonotone_index"].astype(bool),
            flags.loc[list(case.units.unit_id)].astype(bool),
        )


if __name__ == "__main__":
    unittest.main()
