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

Integration tests of scenarios and of the lineage of samples in
:mod:`climada_petals.engine.parametric`, on tropical cyclones for Jamaica
from CLIMADA's Data API:

* the present ``random_walk`` set, and its rcp 4.5 sets at 2040, 2060 and
  2080 as one scenario, an equal-weight mixture, on year sets;
* the present set and its frequencies scaled by
  ``TropCyclone.apply_climate_scenario_knu``, one event set, cut alike;
* a design on the 41 observed years, evaluated on STORM's 10,000.

The data, the helpers and the command that runs these tests are those of
:mod:`climada_petals.test.test_parametric`.
"""

from __future__ import annotations

import functools
import unittest

import numpy as np
from climada.hazard import Hazard, TropCyclone
from climada.hazard.tc_tracks import SAFFIR_SIM_CAT

from climada_petals.engine.parametric import (
    Budget,
    CatalogueYears,
    EventBasis,
    ExpectedUncovered,
    InsuredUnits,
    LeakageError,
    Limit,
    OutcomeSample,
    ParametricCover,
    ParametricIndex,
    ScenarioSet,
    StepPayout,
    WeightedCVaR,
    calibrate,
    calibrate_per_scenario,
    check_no_leakage,
    designs_table,
    evaluate,
    evaluate_scenarios,
    labels_from_names,
)
from climada_petals.test.test_parametric import (
    N_YEARS,
    TC,
    TC_OBSERVED,
    TC_RCP45,
    TC_STORM,
    data_api_hazard,
    impact_of,
    jamaica,
    litpop,
    tc_impfset,
    tc_parents,
    total,
    year_sets,
)

#: The label of the rcp 4.5 scenario, a mixture of three reference years.
RCP45 = "rcp45, 2040 to 2080"

#: STORM's event names, ``..._1000_YEARS_<k>.txt-<y>-<n>``: year ``y`` of
#: the ``k``-th thousand simulated years.
STORM_NAME = (
    r"STORM_DATA_IBTRACS_[A-Z]+_1000_YEARS_(?P<k>[0-9])\.txt-(?P<y>[0-9]+)-[0-9]+"
)

#: Knots to metres per second.
KN_TO_MS = 1852.0 / 3600.0


def with_categories(hazard: Hazard) -> TropCyclone:
    """Return a tropical cyclone set with a category and a basin per event.

    The Data API's files hold neither, and
    ``TropCyclone.apply_climate_scenario_knu`` then returns the set
    unchanged. Here an event's category is the Saffir-Simpson category of
    its largest wind over the set's centroids (-1 for no storm), a lower
    bound of its track's, and every basin is the North Atlantic, the basin
    of the storms that reach Jamaica: the others have category -1, which
    the scaling leaves alone.
    """
    wind = hazard.intensity.max(axis=1).toarray().ravel()
    bounds = np.asarray(SAFFIR_SIM_CAT[:-1], dtype=float) * KN_TO_MS
    category = np.searchsorted(bounds, wind, side="right") - 1
    return TropCyclone(
        category=category,
        basin=["NA"] * hazard.size,
        units=hazard.units,
        centroids=hazard.centroids,
        event_id=hazard.event_id,
        event_name=hazard.event_name,
        frequency=hazard.frequency,
        frequency_unit=hazard.frequency_unit,
        date=hazard.date,
        orig=hazard.orig,
        intensity=hazard.intensity,
        fraction=hazard.fraction,
    )


@functools.cache
def jamaica_scenarios() -> ScenarioSet:
    """Return the present set and the rcp 4.5 mixture, on year sets."""
    case = jamaica()
    rng = np.random.default_rng(3913)
    return ScenarioSet.from_hazards(
        {
            "present": case.hazard,
            RCP45: [data_api_hazard(TC, name) for name in TC_RCP45],
        },
        definition=case.index.definition,
        units=case.units,
        impact_funcs=case.impfset,
        basis=lambda hazard: year_sets(hazard, rng, tc_parents(hazard)),
    )


class TestScenarioMixture(unittest.TestCase):
    """Jamaica's present set, and rcp 4.5 at 2040, 2060 and 2080 as one mixture."""

    def test_the_set(self) -> None:
        summary = jamaica_scenarios().summary().set_index("scenario")
        self.assertEqual(summary.loc["present", "members"], 1)
        self.assertEqual(summary.loc[RCP45, "members"], 3)
        self.assertEqual(summary.loc["present", "n_years"], N_YEARS)
        self.assertEqual(summary.loc[RCP45, "n_years"], 3 * N_YEARS)
        self.assertEqual(set(summary["route"]), {"yearsets"})

    def test_a_cover_weighs_the_members_equally(self) -> None:
        case = jamaica()
        cover = ParametricCover(
            case.index.definition,
            StepPayout.uniform(
                attachment=30.0,
                exhaustion=70.0,
                n_steps=5,
                levels=[1e7, 3e7, 1e8, 3e8, 6e8],
            ),
            unit_id=case.units.unit_id,
            payout_unit=case.impact.unit,
            label="given",
        )
        scenarios = jamaica_scenarios()
        table = evaluate_scenarios(cover, scenarios, split="out_of_sample")
        mixture = total(table, scenario=RCP45)
        members = [
            total(evaluate(cover, member, split="out_of_sample"))
            for member in scenarios.members[RCP45]
        ]
        for column in ("expected_loss", "expected_payout", "expected_uncovered"):
            with self.subTest(column=column):
                self.assertAlmostEqual(
                    mixture[column] / np.mean([row[column] for row in members]),
                    1.0,
                    delta=1e-12,
                )
        # On the event basis, the members' expected losses are their
        # Impacts' aai_agg.
        events = ScenarioSet.from_hazards(
            {RCP45: [data_api_hazard(TC, name) for name in TC_RCP45]},
            definition=case.index.definition,
            units=case.units,
            impact_funcs=case.impfset,
            basis=EventBasis(),
        )
        aai = [
            impact_of(case.exposures, case.impfset, data_api_hazard(TC, name)).aai_agg
            for name in TC_RCP45
        ]
        row = total(evaluate_scenarios(cover, events, split="out_of_sample"))
        self.assertAlmostEqual(row["expected_loss"] / np.mean(aai), 1.0, delta=1e-12)

    def test_designs_per_scenario(self) -> None:
        case = jamaica()
        scenarios = jamaica_scenarios()
        train, test = scenarios.holdout(rng=np.random.default_rng(3914))
        for side in (train, test):
            for member in (*side.members[RCP45], *side.members["present"]):
                self.assertGreaterEqual(member.n_years, 10_000)
        designs = calibrate_per_scenario(
            case.template(30.0, 70.0, 5, label="step"),
            train,
            WeightedCVaR(alpha=0.9, beta=0.99),
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        frame = designs_table(designs)
        self.assertEqual(frame["scenario"].tolist(), ["present", RCP45])
        for label, design in designs.items():
            with self.subTest(design=label):
                self.assertEqual(design.design_sample.scenario, label)
                table = evaluate_scenarios(design, test, split="out_of_sample")
                self.assertEqual(
                    table["scenario"].unique().tolist(), ["present", RCP45]
                )
                for scenario in ("present", RCP45):
                    row = total(table, scenario=scenario)
                    self.assertAlmostEqual(row["premium"], design.premium)
                    self.assertEqual(row["n_years"], test.sample(scenario).n_years)
        # The whole set holds the events the mixture's design was calibrated on.
        with self.assertRaises(LeakageError):
            evaluate_scenarios(designs[RCP45], scenarios, split="out_of_sample")


class TestClimateScenarioKnu(unittest.TestCase):
    """Jamaica's present set and its frequencies scaled for rcp 4.5 at 2060."""

    LABEL = "rcp45, 2060, knu"

    @classmethod
    def setUpClass(cls) -> None:
        case = jamaica()
        present = with_categories(case.hazard)
        cls.future = present.apply_climate_scenario_knu(
            percentile="50", scenario="4.5", target_year=2060
        )
        rng = np.random.default_rng(3915)
        cls.scenarios = ScenarioSet.from_hazards(
            {"present": present, cls.LABEL: cls.future},
            definition=case.index.definition,
            units=case.units,
            impact_funcs=case.impfset,
            basis=lambda hazard: year_sets(hazard, rng, tc_parents(hazard)),
        )
        cls.train, cls.test = cls.scenarios.holdout(rng=rng)

    def test_the_scaled_set_keeps_the_events(self) -> None:
        present, future = (
            self.scenarios.sample(label) for label in self.scenarios.labels
        )
        hazard = jamaica().hazard
        self.assertFalse(np.allclose(self.future.frequency, hazard.frequency))
        np.testing.assert_array_equal(future.event_id, present.event_id)
        self.assertEqual(future.source, present.source)
        self.assertFalse(np.allclose(future.frequency, present.frequency))

    def test_the_two_are_cut_alike(self) -> None:
        for side in (self.train, self.test):
            present, future = (side.sample(label) for label in side.labels)
            with self.subTest(side=present.split.side):
                np.testing.assert_array_equal(future.event_id, present.event_id)
        # Each member keeps its own rate of events on both sides.
        for label in self.scenarios.labels:
            whole = self.scenarios.sample(label).frequency.sum()
            for side in (self.train, self.test):
                self.assertAlmostEqual(
                    side.sample(label).frequency.sum() / whole, 1.0, delta=1e-9
                )

    def test_a_baseline_design_on_the_scaled_set(self) -> None:
        case = jamaica()
        design = calibrate(
            case.template(30.0, 70.0, 5, label="present"),
            self.train.sample("present"),
            ExpectedUncovered(),
            scope="joint",
            budget=Budget(ratio=1.0),
        )
        table = evaluate_scenarios(design, self.test, split="out_of_sample")
        row = total(table, scenario=self.LABEL)
        self.assertAlmostEqual(row["premium"], design.premium)
        self.assertEqual(row["premium_basis"], "design")
        with self.assertRaises(LeakageError):
            evaluate(design, self.scenarios.sample(self.LABEL), split="out_of_sample")
        with self.assertRaises(LeakageError):
            evaluate_scenarios(design, self.scenarios, split="out_of_sample")


class TestShortRecordOnStorm(unittest.TestCase):
    """A design on Jamaica's 41 observed years, evaluated on STORM's 10,000."""

    @classmethod
    def setUpClass(cls) -> None:
        exposures, impfset = litpop("JAM", "TC"), tc_impfset()
        units = InsuredUnits.whole(exposures, unit_id="JAM")
        observed = data_api_hazard(TC, TC_OBSERVED)
        index = ParametricIndex.from_hazard(
            observed, units, index_stat="mean", weight="value"
        )
        cls.observed = OutcomeSample.from_impact(
            index,
            impact_of(exposures, impfset, observed),
            units=units,
            basis=CatalogueYears.from_dates(observed),
        )
        storm = data_api_hazard(TC, TC_STORM)
        found = labels_from_names(storm.event_name, STORM_NAME, ("k", "y"))
        year = 1000 * np.asarray(found["k"], dtype=int) + np.asarray(
            found["y"], dtype=int
        )
        cls.storm = OutcomeSample.from_impact(
            ParametricIndex.from_definition(index.definition, storm, units),
            impact_of(exposures, impfset, storm),
            units=units,
            basis=CatalogueYears.from_labels(storm.event_id, year, n_years=10_000),
        )
        template = ParametricCover(
            index.definition,
            StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=5),
            unit_id=units.unit_id,
            payout_unit="USD",
            limit=Limit.largest_loss(),
            label="observed",
        )
        cls.design = calibrate(
            template,
            cls.observed,
            ExpectedUncovered(),
            scope="individual",
            budget=Budget(ratio=1.0),
        )

    def test_the_routes(self) -> None:
        self.assertEqual(
            (self.observed.route, self.observed.n_years), ("catalogue", 41)
        )
        self.assertEqual((self.storm.route, self.storm.n_years), ("catalogue", 10_000))

    def test_another_hazard_set_is_out_of_sample(self) -> None:
        # The two sets share catalogue years, 1980 to 2020, so a comparison
        # by ids alone would refuse them; they are other event sets.
        self.assertNotEqual(self.storm.source, self.observed.source)
        with self.assertRaisesRegex(LeakageError, "blocks are in both"):
            check_no_leakage(self.design.design_sample, self.storm)
        table = evaluate(self.design, self.storm, split="out_of_sample")
        self.assertEqual(total(table)["n_years"], 10_000)

    def test_the_short_record_leaves_amounts_unidentified(self) -> None:
        unidentified = self.design.unidentified[self.observed.unit_id[0]]
        self.assertGreater(len(unidentified), 0)
        inside = total(evaluate(self.design, self.observed, split="in_sample"))
        outside = total(evaluate(self.design, self.storm, split="out_of_sample"))
        self.assertEqual(inside["unidentified_share"], 0.0)
        self.assertGreater(outside["unidentified_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
