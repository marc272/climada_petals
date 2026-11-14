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

Tests of scenario sets: a fixed cover on each scenario, one design per
scenario, mixtures of windows or members, and the route they share.
"""

from __future__ import annotations

import datetime as dt
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.engine import ImpactCalc
from climada.entity import ImpactFunc, ImpactFuncSet, ImpfTropCyclone
from climada.hazard import Centroids, TropCyclone

from climada_petals.engine.parametric import (
    Budget,
    CatalogueYears,
    EventBasis,
    ExpectedUncovered,
    Free,
    IndexDefinition,
    InfeasibleCalibrationError,
    InsuredUnits,
    LeakageError,
    MetricOptions,
    OutcomeSample,
    ParametricCover,
    ParametricIndex,
    ParametricInputError,
    ScenarioError,
    ScenarioSet,
    StepPayout,
    UnresolvedParameterError,
    WeightedCVaR,
    YearRouteError,
    YearSets,
    calibrate,
    calibrate_per_scenario,
    designs_table,
    evaluate,
    evaluate_scenarios,
    holdout,
    setup_text,
    year_block_kfold,
)
from climada_petals.engine.parametric._mixture import MixtureCatalogue, MixtureYears
from climada_petals.engine.parametric.test._synthetic import (
    array_index,
    exposures,
    grid_exposures,
    grid_hazard,
    random_portfolio,
    template,
)

#: The index, loss and frequencies of the baseline of the hand-worked tests.
BASE = ([[30.0], [50.0], [70.0]], [[2.0], [6.0], [10.0]], [0.1, 0.1, 0.1])

#: The same events with other frequencies: a frequency-scaled scenario.
SCALED = (BASE[0], BASE[1], [0.05, 0.1, 0.3])

#: Two hazard sets of one scenario, as two windows of a pathway.
FIRST = ([[30.0], [50.0]], [[2.0], [6.0]], [0.1, 0.2])
SECOND = ([[40.0], [60.0], [10.0]], [[3.0], [8.0], [0.0]], [0.3, 0.1, 0.5])


def _sample(data: tuple[Any, Any, Any], basis: Any = None) -> OutcomeSample:
    """Return a sample of :func:`array_index`'s objects on ``basis``.

    ``basis`` is a basis, a function of the impact, or None for the event
    basis.
    """
    index, impact, units = array_index(data[0], data[1], frequency=data[2])
    if basis is None:
        basis = EventBasis()
    elif callable(basis):
        basis = basis(impact)
    return OutcomeSample.from_impact(index, impact, units=units, basis=basis)


def _years(vect: list[list[int]]) -> Callable[[Any], YearSets]:
    """Return the year sets of a given ``sampling_vect`` on an impact's events."""
    return lambda impact: YearSets.from_sampling_vect(impact.event_id, vect)


def _cover(
    sample: OutcomeSample, breakpoints: list[float], levels: list[float]
) -> ParametricCover:
    """Return a fixed step cover on a sample's index and units."""
    return ParametricCover(
        sample.index.definition,
        StepPayout(breakpoints, levels),
        unit_id=sample.unit_id,
        payout_unit="USD",
    )


def _total(table: pd.DataFrame, label: str) -> pd.Series:
    """Return the total row of a scenario."""
    rows = table[(table["scenario"] == label) & (table["level"] == "total")]
    return rows.iloc[0]


def _shifted(sample: OutcomeSample, offset: int) -> OutcomeSample:
    """Return a whole event-basis sample of the same rows, with ids moved by ``offset``."""
    index = ParametricIndex(
        sample.index.values,
        sample.event_id + offset,
        sample.unit_id,
        sample.index.definition,
    )
    return OutcomeSample(
        index, sample.loss, sample.frequency, sample.units, EventBasis()
    )


def _with_values(sample: OutcomeSample, values: list[float]) -> OutcomeSample:
    """Return the sample on units of the same points with other values."""
    exp = sample.units.exposures.copy()
    exp.gdf["value"] = values
    exp.gdf.index = list(sample.unit_id)
    return OutcomeSample(
        sample.index,
        sample.loss,
        sample.frequency,
        InsuredUnits.points(exp),
        sample.basis,
    )


def _year_sets_sample(seed: int, n_years: int) -> OutcomeSample:
    """Return a random portfolio on year sets sampled from its frequencies."""
    sample, impact = random_portfolio(seed, basis="event")
    sets = YearSets.from_frequencies(
        impact, n_years=n_years, rng=np.random.default_rng(seed)
    )
    return OutcomeSample.from_impact(
        sample.index, impact, units=sample.units, basis=sets
    )


class TestFrequencyScaled(unittest.TestCase):
    """A frequency-scaled scenario changes the metrics as computed by hand."""

    options = MetricOptions(betas=(0.9,))

    def test_on_the_event_basis(self) -> None:
        base, scaled = _sample(BASE), _sample(SCALED)
        cover = _cover(base, [40.0, 60.0], [5.0, 9.0])  # pays 0, 5, 9
        table = evaluate_scenarios(
            cover,
            ScenarioSet({"baseline": base, "scaled": scaled}),
            split="out_of_sample",
            options=self.options,
        )
        self.assertEqual(list(table["scenario"]), ["baseline"] * 2 + ["scaled"] * 2)
        # Uncovered: 2, 1, 1 per event.
        expected = {
            "baseline": {
                "expected_loss": 0.1 * (2 + 6 + 10),
                "expected_payout": 0.1 * (5 + 9),
                "expected_uncovered": 0.1 * (2 + 1 + 1),
                "payout_frequency": 0.2,
                "var_uncovered_b0_9": 1.0,
                "cvar_uncovered_b0_9": 2.0,
            },
            "scaled": {
                "expected_loss": 0.05 * 2 + 0.1 * 6 + 0.3 * 10,
                "expected_payout": 0.1 * 5 + 0.3 * 9,
                "expected_uncovered": 0.05 * 2 + 0.1 * 1 + 0.3 * 1,
                "payout_frequency": 0.4,
                "var_uncovered_b0_9": 1.0,
                # The worst 0.1 of frequency: 0.05 at 2 and 0.05 at 1.
                "cvar_uncovered_b0_9": (0.05 * 2 + 0.05 * 1) / 0.1,
            },
        }
        for label, columns in expected.items():
            row = _total(table, label)
            for column, value in columns.items():
                with self.subTest(label=label, column=column):
                    self.assertAlmostEqual(row[column], value, places=12)
            self.assertAlmostEqual(row["premium"], row["expected_payout"], places=12)
            self.assertEqual(row["route"], "frequency")

    def test_on_year_sets(self) -> None:
        empty: list[list[int]] = [[] for _ in range(7)]
        base = _sample(BASE, _years([[0], [1], [2], *empty]))
        # Event 2 in three years, event 1 once, event 0 never.
        scaled = _sample(SCALED, _years([[2], [2, 1], [2], *empty]))
        cover = _cover(base, [40.0, 60.0], [5.0, 9.0])
        table = evaluate_scenarios(
            cover,
            ScenarioSet({"baseline": base, "scaled": scaled}),
            split="out_of_sample",
            options=self.options,
        )
        # Years of the scaled set: (L, P, U) = (10, 9, 1), (16, 14, 2), (10, 9, 1).
        expected = {
            "baseline": (1.8, 1.4, 0.4, 0.2, 2.0),
            "scaled": (3.6, 3.2, 0.4, 0.3, 2.0),
        }
        for label, values in expected.items():
            row = _total(table, label)
            found = row[
                [
                    "expected_loss",
                    "expected_payout",
                    "expected_uncovered",
                    "payout_frequency",
                    "cvar_uncovered_b0_9",
                ]
            ]
            np.testing.assert_allclose(found.astype(float), values, rtol=1e-12)
            self.assertEqual(row["n_years"], 10)

    def test_a_mismatch_of_routes_raises(self) -> None:
        on_events = _sample(BASE)
        on_years = _sample(SCALED, _years([[2], [2, 1], [2]]))
        with self.assertRaisesRegex(ScenarioError, "take one route"):
            ScenarioSet({"baseline": on_events, "scaled": on_years})
        with self.assertRaisesRegex(ScenarioError, "scenario 'w', member 1"):
            ScenarioSet({"w": [on_events, on_years]})
        catalogue = _sample(
            BASE,
            lambda imp: CatalogueYears.from_labels(imp.event_id, [0, 1, 2], n_years=10),
        )
        with self.assertRaisesRegex(ScenarioError, "route catalogue"):
            ScenarioSet({"baseline": catalogue, "scaled": on_events})

    def test_catalogue_years_refuse_a_scaled_set_and_name_it(self) -> None:
        pairs = {}
        for label, data in (("baseline", BASE), ("scaled", SCALED)):
            index, impact, units = array_index(data[0], data[1], frequency=data[2])
            pairs[label] = (index, impact)
        with self.assertRaises(YearRouteError) as found:
            ScenarioSet.from_impacts(
                pairs,
                units=units,
                basis=lambda imp: CatalogueYears.from_labels(
                    imp.event_id, [0, 1, 2], n_years=10
                ),
            )
        self.assertIn(
            "raised by ScenarioSet.from_impacts at scenario 'scaled'",
            found.exception.__notes__,
        )


class TestClimateScenarioKnu(unittest.TestCase):
    """``TropCyclone.apply_climate_scenario_knu`` on a small synthetic set."""

    def setUp(self) -> None:
        self.exp = exposures()
        intensity = sp.csr_matrix(
            np.array([[60.0, 55.0, 45.0], [40.0, 38.0, 30.0], [20.0, 18.0, 15.0]])
        )
        self.hazard = TropCyclone(
            category=np.array([4, 1, 0]),
            basin=["NA"] * 3,
            centroids=Centroids(
                lat=np.array(self.exp.latitude, dtype=float),
                lon=np.array(self.exp.longitude, dtype=float),
            ),
            units="m/s",
            event_id=np.array([1, 2, 3]),
            event_name=["A", "B", "C"],
            frequency=np.full(3, 1 / 3),
            date=np.array([dt.date(2001 + k, 9, 1).toordinal() for k in range(3)]),
            intensity=intensity,
            fraction=sp.csr_matrix(intensity.shape),
        )
        self.future = self.hazard.apply_climate_scenario_knu(
            percentile="50", scenario="8.5", target_year=2080
        )
        self.units = InsuredUnits.points(self.exp)
        self.impfset = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])
        self.definition = ParametricIndex.from_hazard(
            self.hazard, self.units
        ).definition

    def _set(self, basis: Any) -> ScenarioSet:
        return ScenarioSet.from_hazards(
            {"baseline": self.hazard, "rcp85, 2080": self.future},
            definition=self.definition,
            units=self.units,
            impact_funcs=self.impfset,
            basis=basis,
        )

    def test_the_scaled_frequencies_weigh_the_same_footprints(self) -> None:
        # Category 4 becomes more frequent and the others less.
        self.assertGreater(self.future.frequency[0], 1 / 3)
        self.assertLess(self.future.frequency[1], 1 / 3)
        scenarios = self._set(EventBasis())
        loss = ImpactCalc(self.exp.copy(), self.impfset, self.hazard).impact(
            save_mat=True
        )
        per_event = loss.imp_mat.toarray()
        levels = [1.0e8, 3.0e8]
        cover = ParametricCover(
            self.definition,
            StepPayout([35.0, 50.0], levels),
            unit_id=self.units.unit_id,
            payout_unit="USD",
        )
        wind = self.hazard.intensity.toarray()
        paid = np.where(wind >= 50.0, levels[1], np.where(wind >= 35.0, levels[0], 0))
        table = evaluate_scenarios(cover, scenarios, split="out_of_sample")
        for label, frequency in (
            ("baseline", self.hazard.frequency),
            ("rcp85, 2080", self.future.frequency),
        ):
            row = _total(table, label)
            self.assertAlmostEqual(
                row["expected_loss"],
                float(frequency @ per_event.sum(axis=1)),
                delta=1e-12 * row["expected_loss"],
            )
            self.assertAlmostEqual(
                row["expected_payout"],
                float(frequency @ paid.sum(axis=1)),
                delta=1e-12 * row["expected_payout"],
            )

    def test_a_design_keeps_its_premium_under_the_scenario(self) -> None:
        scenarios = self._set(EventBasis())
        result = calibrate(
            template(
                scenarios.sample("baseline"),
                StepPayout([35.0, 50.0], [Free(), Free()]),
            ),
            scenarios.sample("baseline"),
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        table = evaluate_scenarios(
            result,
            scenarios,
            split={"baseline": "in_sample", "rcp85, 2080": "out_of_sample"},
        )
        self.assertEqual(list(table["split"]), ["in_sample", "out_of_sample"])
        future = _total(table, "rcp85, 2080")
        self.assertAlmostEqual(future["premium"], result.premium)
        self.assertEqual(future["premium_basis"], "design")
        paid = result.cover.payout_matrix(scenarios.sample("rcp85, 2080").index)
        self.assertAlmostEqual(
            future["premium_repriced"],
            float(scenarios.sample("rcp85, 2080").frequency @ paid.sum(axis=1)),
            delta=1e-12 * future["premium_repriced"],
        )

    def test_routes(self) -> None:
        with self.assertRaises(YearRouteError) as found:
            self._set(CatalogueYears.from_dates)
        self.assertIn(
            "raised by ScenarioSet.from_hazards at scenario 'rcp85, 2080'",
            found.exception.__notes__,
        )
        rng = np.random.default_rng(321)
        scenarios = self._set(
            lambda hazard: YearSets.from_frequencies(hazard, n_years=50, rng=rng)
        )
        self.assertEqual(scenarios.route, "yearsets")
        self.assertEqual(list(scenarios.summary()["n_years"]), [50, 50])


class TestMixtures(unittest.TestCase):
    """Several samples under one label: one scenario, an equal-weight mixture."""

    def test_the_event_basis_divides_the_frequencies(self) -> None:
        first, second = _sample(FIRST), _sample(SECOND)
        mixture = ScenarioSet({"window": [first, second]}).sample("window")
        np.testing.assert_array_equal(mixture.event_id, [-3, -5, -4, -6, -8])
        self.assertEqual(
            mixture.index.event_name, ("0:e0", "0:e1", "1:e0", "1:e1", "1:e2")
        )
        np.testing.assert_allclose(mixture.frequency, [0.05, 0.1, 0.15, 0.05, 0.25])
        self.assertEqual(list(mixture.block), ["0:1", "0:2", "1:1", "1:2", "1:3"])
        self.assertEqual((mixture.scenario, mixture.split), ("window", None))
        cover = _cover(first, [35.0, 55.0], [4.0, 7.0])  # 0, 4 and 4, 7, 0
        options = MetricOptions(betas=(0.8,))
        row = evaluate(cover, mixture, split="out_of_sample", options=options).iloc[-1]
        self.assertAlmostEqual(row["expected_loss"], (1.4 + 1.7) / 2)
        self.assertAlmostEqual(row["expected_payout"], (0.8 + 1.9) / 2)
        self.assertAlmostEqual(row["expected_uncovered"], (0.6 + 0.1) / 2)
        # Uncovered 2 at frequency 0.15, then 1 at 0.05.
        self.assertAlmostEqual(row["cvar_uncovered_b0_8"], (0.15 * 2 + 0.05) / 0.2)
        members = [
            evaluate(cover, sample, split="out_of_sample").iloc[-1]
            for sample in (first, second)
        ]
        for column in ("expected_loss", "expected_payout", "payout_frequency"):
            self.assertAlmostEqual(
                row[column], (members[0][column] + members[1][column]) / 2
            )

    def test_year_sets_of_other_lengths(self) -> None:
        first = _sample(FIRST, _years([[0], [], [0, 1]]))
        second = _sample(SECOND, _years([[1], [0]]))
        mixture = ScenarioSet({"window": [first, second]}).sample("window")
        self.assertIsInstance(mixture.basis, MixtureYears)
        self.assertEqual(mixture.n_years, 5)
        # Occupied years 1/6, 1/6 (first), 1/4, 1/4 (second); the empty one 1/6.
        np.testing.assert_allclose(
            mixture.measure.weights, [1 / 6, 1 / 6, 1 / 4, 1 / 4, 1 / 6]
        )
        cover = _cover(first, [35.0, 55.0], [4.0, 7.0])
        row = evaluate(
            cover,
            mixture,
            split="out_of_sample",
            options=MetricOptions(betas=(0.75,)),
        ).iloc[-1]
        # Years (L, P, U): (2, 0, 2), (8, 4, 4) of the first; (8, 7, 1) and
        # (3, 4, 0) of the second.
        self.assertAlmostEqual(row["expected_loss"], 10 / 6 + 11 / 4)
        self.assertAlmostEqual(row["expected_payout"], 4 / 6 + 11 / 4)
        self.assertAlmostEqual(row["expected_uncovered"], 6 / 6 + 1 / 4)
        self.assertAlmostEqual(row["payout_frequency"], 1 / 6 + 1 / 2)
        self.assertAlmostEqual(row["var_uncovered_b0_75"], 2.0)
        self.assertAlmostEqual(row["cvar_uncovered_b0_75"], (4 / 6 + 2 / 12) / 0.25)
        self.assertAlmostEqual(row["n_years"], 5)

    def test_year_sets_of_equal_lengths_are_plain_year_sets(self) -> None:
        first = _sample(FIRST, _years([[0], [], [0, 1]]))
        second = _sample(SECOND, _years([[1], [0], [2, 2]]))
        mixture = ScenarioSet({"window": [first, second]}).sample("window")
        plain = YearSets.from_sampling_vect(
            mixture.event_id, mixture.basis.sampling_vect
        ).measure(mixture.event_id, mixture.frequency)
        self.assertEqual((mixture.measure.grouping != plain.grouping).nnz, 0)
        np.testing.assert_allclose(mixture.measure.weights, plain.weights, rtol=1e-15)

    def test_catalogue_years_of_other_lengths(self) -> None:
        first = _sample(
            (FIRST[0], FIRST[1], [0.25, 0.25]),
            lambda imp: CatalogueYears.from_labels(imp.event_id, [0, 2], n_years=4),
        )
        second = _sample(
            (SECOND[0], SECOND[1], [0.5] * 3),
            lambda imp: CatalogueYears.from_labels(imp.event_id, [0, 1, 1], n_years=2),
        )
        mixture = ScenarioSet({"window": [first, second]}).sample("window")
        basis = mixture.basis
        self.assertIsInstance(basis, MixtureCatalogue)
        # Places (k + 1/2) / N: 1/8, 3/8, 5/8, 7/8 and 1/4, 3/4, interleaved.
        np.testing.assert_array_equal(basis.year_of(mixture.event_id), [0, 3, 1, 4, 4])
        np.testing.assert_allclose(
            basis.year_weight, [1 / 8, 1 / 4, 1 / 8, 1 / 8, 1 / 4, 1 / 8]
        )
        np.testing.assert_allclose(
            mixture.frequency, [1 / 8, 1 / 8, 1 / 4, 1 / 4, 1 / 4]
        )
        self.assertEqual(list(mixture.block), ["0:0", "0:2", "1:0", "1:1", "1:1"])
        cover = _cover(first, [35.0, 55.0], [4.0, 7.0])
        row = evaluate(
            cover, mixture, split="out_of_sample", options=MetricOptions(betas=(0.7,))
        ).iloc[-1]
        self.assertAlmostEqual(row["expected_loss"], (8 / 4 + 11 / 2) / 2)
        # Years' losses 8 (1/4) and 6 (1/8) lead: the worst 0.3 is 8 and 6.
        self.assertAlmostEqual(row["cvar_loss_b0_7"], (8 / 4 + 6 * 0.05) / 0.3)
        self.assertEqual(row["n_years"], 6)
        train, test = holdout(mixture, train_fraction=0.5)
        np.testing.assert_array_equal(train.years, [0, 1, 2])
        np.testing.assert_allclose(train.measure.weights, [0.25, 0.5, 0.25])
        self.assertAlmostEqual(
            train.measure.expectation(train.loss.toarray()[:, 0]), 2.0
        )
        self.assertEqual(test.n_years, 3)

    def test_the_mixture_classes_check_their_inputs(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "one weight per year"):
            MixtureYears(np.array([1, 2]), [[0], [1]])
        with self.assertRaisesRegex(ParametricInputError, "summing to 1"):
            MixtureYears(np.array([1, 2]), [[0], [1]], year_weight=[0.5, 0.6])
        first = _sample(FIRST, _years([[0], [1]]))
        second = _sample(SECOND, _years([[1], [0]]))
        years = ScenarioSet({"w": [first, second]}).sample("w")
        with self.assertRaisesRegex(YearRouteError, "members"):
            years.basis.measure(years.event_id, years.frequency * 2)
        catalogue = ScenarioSet(
            {
                "w": [
                    _sample(
                        (FIRST[0], FIRST[1], [0.5, 0.5]),
                        lambda imp: CatalogueYears.from_labels(
                            imp.event_id, [0, 1], n_years=2
                        ),
                    )
                ]
                * 2
            }
        ).sample("w")
        with self.assertRaisesRegex(YearRouteError, "year's weight"):
            catalogue.basis.measure(catalogue.event_id, catalogue.frequency * 2)
        with self.assertRaisesRegex(ParametricInputError, "outside the given years"):
            catalogue.basis.measure(
                catalogue.event_id, catalogue.frequency, years=[0, 1]
            )

    def test_draws(self) -> None:
        _, _, units = array_index([[50.0]], [[1.0]])
        index = ParametricIndex([[50.0]], [1], units.unit_id, units_definition())
        first = OutcomeSample(
            index, [[4.0], [8.0]], [0.2], units, EventBasis(), n_draws=2
        )
        second = OutcomeSample(
            index, [[6.0], [2.0]], [0.4], units, EventBasis(), n_draws=2
        )
        mixture = ScenarioSet({"w": [first, second]}).sample("w")
        # Rows by draw, then event: draw 0 of both members, then draw 1.
        np.testing.assert_array_equal(mixture.loss.toarray()[:, 0], [4, 6, 8, 2])
        np.testing.assert_allclose(mixture.measure.weights, [0.05, 0.1, 0.05, 0.1])
        cover = _cover(first, [40.0], [5.0])
        row = evaluate(cover, mixture, split="out_of_sample").iloc[-1]
        self.assertAlmostEqual(row["expected_loss"], (1.2 + 1.6) / 2)
        self.assertAlmostEqual(row["expected_uncovered"], 0.1 * 1 + 0.05 * 3)
        single = OutcomeSample(index, [[4.0]], [0.2], units, EventBasis())
        with self.assertRaisesRegex(ScenarioError, "impact-function draws"):
            ScenarioSet({"w": [first, single]})

    def test_the_members_share_units_and_splits(self) -> None:
        first = _sample(FIRST)
        valued = _with_values(first, [2.0])
        with self.assertRaisesRegex(ScenarioError, "other units"):
            ScenarioSet({"w": [first, valued]})
        # Other exposures are another scenario's.
        self.assertEqual(ScenarioSet({"a": first, "b": valued}).labels, ("a", "b"))
        sample, _ = random_portfolio(3, basis="event")
        train, _ = holdout(sample, rng=np.random.default_rng(1))
        with self.assertRaisesRegex(ScenarioError, "all whole samples"):
            ScenarioSet({"w": [train, sample]})
        folds = year_block_kfold(sample, 2, rng=np.random.default_rng(2))
        with self.assertRaisesRegex(ScenarioError, "one side of one split"):
            ScenarioSet({"w": [folds[0][1], folds[1][1]]})

    def test_nested_mixtures_keep_the_weights(self) -> None:
        first = _sample(FIRST, _years([[0], [], [0, 1]]))
        second = _sample(SECOND, _years([[1], [0]]))
        third = _sample(SECOND, _years([[2], [1, 0]]))
        inner = ScenarioSet({"in": [first, second]}).sample("in")
        outer = ScenarioSet({"out": [inner, third]}).sample("out")
        losses = [s.measure.expectation(s.loss.toarray()[:, 0]) for s in (inner, third)]
        self.assertAlmostEqual(
            outer.measure.expectation(outer.loss.toarray()[:, 0]), sum(losses) / 2
        )
        catalogue = [
            _sample(
                (FIRST[0], FIRST[1], [0.25, 0.25]),
                lambda imp: CatalogueYears.from_labels(imp.event_id, [0, 2], n_years=4),
            ),
            _sample(
                (SECOND[0], SECOND[1], [0.5] * 3),
                lambda imp: CatalogueYears.from_labels(
                    imp.event_id, [0, 1, 1], n_years=2
                ),
            ),
        ]
        inner = ScenarioSet({"in": catalogue}).sample("in")
        outer = ScenarioSet({"out": [inner, catalogue[1]]}).sample("out")
        losses = [
            s.measure.expectation(s.loss.toarray()[:, 0]) for s in (inner, catalogue[1])
        ]
        self.assertAlmostEqual(
            outer.measure.expectation(outer.loss.toarray()[:, 0]), sum(losses) / 2
        )
        self.assertAlmostEqual(float(outer.basis.year_weight.sum()), 1.0)


def units_definition() -> IndexDefinition:
    """Return the definition of :func:`array_index`'s indices."""
    index, _, _ = array_index([[1.0]], [[1.0]])
    return index.definition


class TestScenarioSet(unittest.TestCase):
    """Labels, samples and the checks of a set."""

    def test_a_single_sample_keeps_its_rows_and_takes_the_label(self) -> None:
        sample = _sample(BASE)
        scenarios = ScenarioSet({"future": sample})
        found = scenarios.sample("future")
        self.assertEqual((found.scenario, sample.scenario), ("future", "baseline"))
        np.testing.assert_array_equal(found.event_id, sample.event_id)
        self.assertEqual((found.loss != sample.loss).nnz, 0)
        np.testing.assert_array_equal(found.measure.weights, sample.measure.weights)
        self.assertIs(ScenarioSet({"baseline": sample}).sample("baseline"), sample)
        self.assertIs(ScenarioSet({"baseline": [sample]}).sample("baseline"), sample)

    def test_properties_and_summary(self) -> None:
        first, second = _sample(FIRST), _sample(SECOND)
        scenarios = ScenarioSet({"a": first, "w": (first, second)})
        self.assertEqual(scenarios.labels, ("a", "w"))
        self.assertEqual((scenarios.basis, scenarios.route), ("event", "frequency"))
        self.assertEqual(scenarios.unit_id, ("U0",))
        self.assertEqual(scenarios.definition, first.index.definition)
        self.assertEqual(
            repr(scenarios),
            "ScenarioSet(labels=['a', 'w'], basis='event', route='frequency')",
        )
        self.assertEqual(len(scenarios.members["w"]), 2)
        self.assertEqual(scenarios.fold("w"), 0)
        summary = scenarios.summary()
        self.assertEqual(
            list(summary.columns),
            [
                "scenario",
                "members",
                "basis",
                "route",
                "n_years",
                "n_events",
                "n_draws",
                "split",
                "fold",
            ],
        )
        self.assertEqual(list(summary["members"]), [1, 2])
        self.assertEqual(list(summary["n_events"]), [2, 5])
        self.assertTrue(summary["n_years"].isna().all())
        with self.assertRaisesRegex(ParametricInputError, "no scenario 'b'"):
            scenarios.sample("b")
        with self.assertRaisesRegex(ParametricInputError, "no scenario 'b'"):
            scenarios.fold("b")

    def test_units_and_index_definitions_must_match(self) -> None:
        one = _sample(FIRST)
        two = _sample(([[30.0, 40.0]], [[1.0, 2.0]], [0.1]))
        with self.assertRaisesRegex(ScenarioError, "unit ids"):
            ScenarioSet({"a": one, "b": two})
        index, impact, units = array_index(
            FIRST[0],
            FIRST[1],
            frequency=FIRST[2],
            definition=IndexDefinition("table", unit="m/s", source="other"),
        )
        other = OutcomeSample.from_impact(
            index, impact, units=units, basis=EventBasis()
        )
        with self.assertRaisesRegex(ScenarioError, "index definition"):
            ScenarioSet({"a": one, "b": other})

    def test_invalid_members_raise(self) -> None:
        sample = _sample(FIRST)
        for members in (
            [sample],
            {},
            {1: sample},
            {"": sample},
            {"a": "sample"},
            {"a": []},
            {"a": [sample, "other"]},
            {"a": 3},
        ):
            with self.subTest(members=members), self.assertRaises(ParametricInputError):
                ScenarioSet(members)


class TestFromImpacts(unittest.TestCase):
    """A set from an index and an ``Impact`` per scenario."""

    def setUp(self) -> None:
        self.first = array_index(*FIRST[:2], frequency=FIRST[2])
        self.second = array_index(*SECOND[:2], frequency=SECOND[2])

    def test_pairs_and_mixtures(self) -> None:
        units = self.first[2]
        scenarios = ScenarioSet.from_impacts(
            {
                "a": self.first[:2],
                "w": [self.first[:2], self.second[:2]],
            },
            units=units,
            basis=EventBasis(),
        )
        expected = OutcomeSample.from_impact(
            *self.first[:2], units=units, basis=EventBasis()
        )
        found = scenarios.sample("a")
        self.assertEqual(found.scenario, "a")
        self.assertEqual((found.loss != expected.loss).nnz, 0)
        np.testing.assert_array_equal(found.frequency, expected.frequency)
        self.assertEqual(scenarios.sample("w").n_events, 5)
        rng = np.random.default_rng(4)
        on_years = ScenarioSet.from_impacts(
            {"a": self.first[:2], "b": self.second[:2]},
            units={"a": units, "b": self.second[2]},
            basis=lambda impact: YearSets.from_frequencies(impact, n_years=30, rng=rng),
        )
        self.assertEqual(on_years.route, "yearsets")

    def test_errors_name_the_scenario_and_the_member(self) -> None:
        units = self.first[2]
        with self.assertRaises(ParametricInputError) as found:
            ScenarioSet.from_impacts(
                {"a": self.first[:2]}, units={"b": units}, basis=EventBasis()
            )
        self.assertIn(
            "raised by ScenarioSet.from_impacts at scenario 'a'",
            found.exception.__notes__,
        )
        with self.assertRaisesRegex(ParametricInputError, "tuple") as found:
            ScenarioSet.from_impacts(
                {"w": [self.first[:2], self.second[1]]}, units=units, basis=EventBasis()
            )
        self.assertIn(
            "raised by ScenarioSet.from_impacts at scenario 'w', member 1",
            found.exception.__notes__,
        )


class TestFromHazards(unittest.TestCase):
    """A set from hazard sets, the units' exposures and the impact functions."""

    def setUp(self) -> None:
        self.hazard = grid_hazard()
        self.units = InsuredUnits.from_column(grid_exposures(), "region_id")
        # Region 1 grows at the second centroid: values 1, 3, 2 become 1, 1, 4.
        points = [(10.0, 20.0, 1.0, 1), (10.001, 20.0, 1.0, 1), (10.0, 21.0, 4.0, 1)]
        grown = grid_exposures(points + [(11.0, 20.0, 4.0, 2), (11.0, 21.0, 4.0, 2)])
        self.grown = InsuredUnits.from_column(grown, "region_id")
        self.impfs = ImpactFuncSet(
            [
                ImpactFunc(
                    haz_type="FL",
                    id=1,
                    intensity=np.array([-10.0, 0.0, 5.0]),
                    mdd=np.array([0.0, 0.0, 1.0]),
                    paa=np.ones(3),
                    intensity_unit="m",
                    name="linear",
                )
            ]
        )
        self.definition = ParametricIndex.from_hazard(
            self.hazard, self.units, index_stat="mean", weight="value"
        ).definition

    def _set(self, **kwargs: Any) -> ScenarioSet:
        return ScenarioSet.from_hazards(
            {"baseline": self.hazard, "growth": self.hazard},
            definition=self.definition,
            units={"baseline": self.units, "growth": self.grown},
            impact_funcs=self.impfs,
            basis=EventBasis(),
            **kwargs,
        )

    def test_value_weights_follow_the_exposures_unless_stated(self) -> None:
        # Depths of events 30, 10, 20 at the first two centroids: (0, 2),
        # (1, 3), (0.5, 0); region 1's weights 4/6, 2/6 at inception.
        own = self._set()
        np.testing.assert_allclose(
            own.sample("baseline").index.values[:, 0], [2 / 3, 5 / 3, 1 / 3]
        )
        np.testing.assert_allclose(
            own.sample("growth").index.values[:, 0], [4 / 3, 7 / 3, 1 / 6]
        )
        fixed = self._set(index_units=self.units)
        np.testing.assert_allclose(
            fixed.sample("growth").index.values[:, 0], [2 / 3, 5 / 3, 1 / 3]
        )
        # The loss follows the scenario's exposures either way: 40 % of 4.
        for scenarios in (own, fixed):
            self.assertAlmostEqual(
                scenarios.sample("growth").loss.toarray()[0, 0], 0.4 * 4
            )
        self.assertAlmostEqual(own.sample("baseline").loss.toarray()[0, 0], 0.4 * 2)
        np.testing.assert_allclose(
            own.sample("growth").index.values[:, 1], [0.5, 0.0, 3.0]
        )

    def test_windows_of_other_hazard_sets(self) -> None:
        windows = [
            grid_hazard(event_id=(100, 200, 300)),
            grid_hazard(FLOOD_SHIFTED, event_id=(400, 500, 600)),
        ]
        windows[0].frequency = np.array([0.05, 0.2, 0.1])
        windows[1].frequency = np.array([0.3, 0.1, 0.02])
        rng = np.random.default_rng(42)
        scenarios = ScenarioSet.from_hazards(
            {"present": self.hazard, "rcp45, 2040 to 2060": windows},
            definition=self.definition,
            units=self.units,
            impact_funcs=self.impfs,
            basis=lambda hazard: YearSets.from_frequencies(hazard, n_years=40, rng=rng),
        )
        self.assertEqual(list(scenarios.summary()["n_years"]), [40, 80])
        cover = ParametricCover(
            self.definition,
            StepPayout([0.5, 2.0], [0.2, 0.6]),
            unit_id=self.units.unit_id,
            payout_unit="USD",
        )
        table = evaluate_scenarios(cover, scenarios, split="out_of_sample")
        members = [
            evaluate(cover, member, split="out_of_sample").iloc[-1]["expected_loss"]
            for member in scenarios.members["rcp45, 2040 to 2060"]
        ]
        row = _total(table, "rcp45, 2040 to 2060")
        self.assertAlmostEqual(row["expected_loss"], sum(members) / 2)
        self.assertEqual(row["n_years"], 80)
        text = setup_text(table).splitlines()
        self.assertIn("Scenario: present; rcp45, 2040 to 2060", text)
        self.assertIn(
            "Basis: annual basis on 40, 80 sampled years (route yearsets)", text
        )

    def test_errors_name_the_scenario_and_the_member(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "climada Hazard") as found:
            ScenarioSet.from_hazards(
                {"w": [self.hazard, "hazard"]},
                definition=self.definition,
                units=self.units,
                impact_funcs=self.impfs,
                basis=EventBasis(),
            )
        self.assertIn(
            "raised by ScenarioSet.from_hazards at scenario 'w', member 1",
            found.exception.__notes__,
        )
        with self.assertRaisesRegex(ParametricInputError, "InsuredUnits or None"):
            self._set(index_units="baseline")
        other = InsuredUnits.whole(grid_exposures())
        with self.assertRaisesRegex(ParametricInputError, "same unit ids") as found:
            self._set(index_units=other)
        self.assertIn(
            "raised by ScenarioSet.from_hazards at scenario 'baseline'",
            found.exception.__notes__,
        )
        with self.assertRaisesRegex(ParametricInputError, "impact_funcs has no entry"):
            ScenarioSet.from_hazards(
                {"a": self.hazard},
                definition=self.definition,
                units=self.units,
                impact_funcs={"b": self.impfs},
                basis=EventBasis(),
            )


#: Depths of the second window of :meth:`TestFromHazards.test_windows_of_other_hazard_sets`.
FLOOD_SHIFTED = ((0.5, 3.0, 2.0, 1.0), (2.0, 1.0, 0.0, 0.5), (0.0, 0.5, 1.0, 3.0))


class TestEvaluateScenarios(unittest.TestCase):
    """One cover or design on every scenario."""

    def setUp(self) -> None:
        self.first, self.second = _sample(FIRST), _sample(SECOND)
        self.cover = _cover(self.first, [35.0, 55.0], [4.0, 7.0])

    def test_rows_are_those_of_evaluate(self) -> None:
        scenarios = ScenarioSet({"a": self.first, "b": self.second})
        table = evaluate_scenarios(self.cover, scenarios, split="out_of_sample")
        expected = []
        for label, sample in (("a", self.first), ("b", self.second)):
            rows = evaluate(self.cover, sample, split="out_of_sample")
            rows["scenario"] = label
            expected.append(rows)
        pd.testing.assert_frame_equal(table, pd.concat(expected, ignore_index=True))

    def test_a_design_keeps_its_contract_price(self) -> None:
        result = calibrate(
            template(self.first, StepPayout([35.0, 55.0], [Free(), Free()])),
            self.first,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=0.8),
        )
        scenarios = ScenarioSet({"a": self.first, "b": self.second})
        table = evaluate_scenarios(
            result, scenarios, split={"a": "in_sample", "b": "out_of_sample"}
        )
        self.assertEqual(
            list(table["split"]), ["in_sample"] * 2 + ["out_of_sample"] * 2
        )
        np.testing.assert_allclose(table["premium"], [result.premium] * 4)
        paid = result.cover.payout_matrix(self.second.index)[:, 0]
        self.assertAlmostEqual(
            _total(table, "b")["premium_repriced"], float(self.second.frequency @ paid)
        )

    def test_invalid_arguments_raise(self) -> None:
        scenarios = ScenarioSet({"a": self.first, "b": self.second})
        with self.assertRaisesRegex(ParametricInputError, "lacks \\['b'\\]"):
            evaluate_scenarios(self.cover, scenarios, split={"a": "in_sample"})
        with self.assertRaisesRegex(ParametricInputError, "has \\['c'\\]"):
            evaluate_scenarios(
                self.cover,
                scenarios,
                split={"a": "in_sample", "b": "in_sample", "c": "in_sample"},
            )
        with self.assertRaisesRegex(ParametricInputError, "ScenarioSet"):
            evaluate_scenarios(self.cover, self.first, split="in_sample")
        unresolved = template(self.first, StepPayout([35.0], [Free()]))
        with self.assertRaises(UnresolvedParameterError) as found:
            evaluate_scenarios(unresolved, scenarios, split="in_sample")
        self.assertIn(
            "raised by evaluate_scenarios at scenario 'a'", found.exception.__notes__
        )

    def test_leakage_is_checked_member_by_member(self) -> None:
        first, _ = random_portfolio(5, basis="event")
        second = _shifted(random_portfolio(6, basis="event")[0], 1000)
        train, test = ScenarioSet({"w": [first, second]}).holdout(
            rng=np.random.default_rng(7)
        )
        template_cover = template(
            first, StepPayout.uniform(attachment=25.0, exhaustion=75.0, n_steps=3)
        )
        on_mixture = calibrate(
            template_cover,
            train.sample("w"),
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        on_member = calibrate(
            template_cover,
            train.members["w"][0],
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        for design in (on_mixture, on_member):
            table = evaluate_scenarios(design, test, split="out_of_sample")
            self.assertEqual(list(table["scenario"]), ["w"])
            with self.assertRaises(LeakageError) as found:
                evaluate_scenarios(design, train, split="out_of_sample")
            self.assertIn(
                "raised by evaluate_scenarios at scenario 'w'",
                found.exception.__notes__,
            )
            evaluate_scenarios(design, train, split="in_sample")

    def test_the_fold_is_the_members(self) -> None:
        first, _ = random_portfolio(5, basis="event")
        second = _shifted(random_portfolio(6, basis="event")[0], 1000)
        sides = [
            year_block_kfold(sample, 2, rng=np.random.default_rng(8))[1][1]
            for sample in (first, second)
        ]
        cover = _cover(first, [30.0, 60.0], [10.0, 30.0])
        table = evaluate_scenarios(
            cover, ScenarioSet({"w": sides, "one": sides[0]}), split="out_of_sample"
        )
        self.assertEqual(set(table["fold"]), {1})
        self.assertEqual(table["fold"].dtype, np.int64)


class TestCalibratePerScenario(unittest.TestCase):
    """One design per scenario, and the designs as one table."""

    def setUp(self) -> None:
        self.first, _ = random_portfolio(5, basis="event")
        self.second = _shifted(random_portfolio(6, basis="event")[0], 1000)
        self.scenarios = ScenarioSet({"a": self.first, "w": [self.first, self.second]})
        self.template = template(
            self.first, StepPayout.uniform(attachment=25.0, exhaustion=75.0, n_steps=3)
        )

    def test_designs_equal_separate_calibrations(self) -> None:
        objective = WeightedCVaR(alpha=0.9, beta=0.9)
        designs = calibrate_per_scenario(
            self.template,
            self.scenarios,
            objective,
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        self.assertEqual(list(designs), ["a", "w"])
        for label, design in designs.items():
            alone = calibrate(
                self.template,
                self.scenarios.sample(label),
                objective,
                scope="joint",
                budget=Budget(ratio=1.0),
            )
            pd.testing.assert_frame_equal(
                design.summary().drop(columns="seconds"),
                alone.summary().drop(columns="seconds"),
            )
            pd.testing.assert_frame_equal(
                design.parameters_frame(), alone.parameters_frame()
            )
            self.assertEqual(design.design_sample.scenario, label)

    def test_designs_table(self) -> None:
        designs = calibrate_per_scenario(
            self.template,
            self.scenarios,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=1.0),
        )
        table = designs_table(designs)
        summary = designs["a"].summary()
        self.assertEqual(
            list(table.columns[: 1 + summary.shape[1]]),
            ["scenario", *summary.columns],
        )
        self.assertEqual(list(table["scenario"]), ["a"] * 3 + ["w"] * 3)
        self.assertEqual(list(table["problem_id"]), ["U0", "U1", "U2"] * 2)
        for label, design in designs.items():
            rows = table[table["scenario"] == label].reset_index(drop=True)
            for k, unit in enumerate(design.cover.unit_id):
                for name, value in design.cover.payout[unit].parameters().items():
                    self.assertEqual(rows.loc[k, f"{unit}:{name}"], value)
                    other = rows.loc[(k + 1) % 3, f"{unit}:{name}"]
                    self.assertTrue(np.isnan(other))
        for invalid in ({}, {"a": "design"}, [designs["a"]]):
            with self.subTest(invalid=invalid), self.assertRaises(ParametricInputError):
                designs_table(invalid)

    def test_a_failing_scenario_raises_and_names_it(self) -> None:
        fixed = template(self.first, StepPayout([20.0, 50.0], [10.0, Free()]))
        with self.assertRaises(InfeasibleCalibrationError) as found:
            calibrate_per_scenario(
                fixed,
                self.scenarios,
                ExpectedUncovered(),
                scope="joint",
                budget=Budget(ratio=0.0),
            )
        self.assertIn(
            "raised by calibrate_per_scenario at scenario 'a'",
            found.exception.__notes__,
        )
        with self.assertRaisesRegex(ParametricInputError, "ScenarioSet"):
            calibrate_per_scenario(
                fixed, {"a": self.first}, ExpectedUncovered(), scope="joint"
            )

    def test_out_of_sample_designs_on_year_sets(self) -> None:
        scenarios = ScenarioSet(
            {"a": _year_sets_sample(11, 300), "w": [_year_sets_sample(12, 300)] * 2}
        )
        train, test = scenarios.holdout(rng=np.random.default_rng(13))
        sides = train.members["w"]
        self.assertEqual(train.sample("w").n_years, sum(side.n_years for side in sides))
        expected = [
            side.measure.expectation(side.loss.sum(axis=1).A1) for side in sides
        ]
        mixture = train.sample("w")
        self.assertAlmostEqual(
            mixture.measure.expectation(mixture.loss.sum(axis=1).A1),
            sum(expected) / 2,
        )
        designs = calibrate_per_scenario(
            self.template,
            train,
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        table = evaluate_scenarios(designs["w"], test, split="out_of_sample")
        self.assertEqual(list(table["scenario"]), ["a", "w"])
        self.assertEqual(
            list(table["n_years"]),
            [test.sample("a").n_years, test.sample("w").n_years],
        )
        with self.assertRaises(ParametricInputError) as found:
            scenarios.holdout()
        self.assertIn(
            "raised by ScenarioSet.holdout at scenario 'a'", found.exception.__notes__
        )


if __name__ == "__main__":
    unittest.main()
