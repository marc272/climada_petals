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

Tests of the outcome measures and the bases."""

from __future__ import annotations

import datetime as dt
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp

from climada_petals.engine.parametric import (
    ENSEMBLE_PARENT,
    CatalogueYears,
    EventAlignmentError,
    EventBasis,
    ImpactInputError,
    OutcomeMeasure,
    ParametricInputError,
    YearRouteError,
    YearSets,
    labels_from_names,
    sample_years,
)


def _dated(*dates: dt.date) -> SimpleNamespace:
    """Return events 1, 2, ... with these dates, as a hazard holds them."""
    return SimpleNamespace(
        event_id=np.arange(1, len(dates) + 1),
        date=np.array([day.toordinal() for day in dates]),
    )


def _named(*names: str) -> SimpleNamespace:
    """Return events 1, 2, ... with these names, as a hazard holds them."""
    return SimpleNamespace(event_id=np.arange(1, len(names) + 1), event_name=names)


class TestOutcomeMeasure(unittest.TestCase):
    """The measure's own fields, sums and checks."""

    def test_years_group_events_and_empty_years_are_pooled(self) -> None:
        years = CatalogueYears.from_labels([1, 2, 3], [3, 3, 7], n_years=10)
        measure = years.measure([1, 2, 3], [0.1, 0.1, 0.1])
        np.testing.assert_array_equal(
            measure.grouping.toarray(), [[1, 1, 0], [0, 0, 1], [0, 0, 0]]
        )
        np.testing.assert_allclose(measure.probability, [0.1, 0.1, 0.8])
        np.testing.assert_allclose(measure.aggregate([1.0, 2.0, 5.0]), [3.0, 5.0, 0.0])
        self.assertAlmostEqual(measure.expectation([1.0, 2.0, 5.0]), 0.8)
        np.testing.assert_allclose(measure.event_probability, [0.1, 0.1, 0.1])
        self.assertEqual(measure.grouping_matrix.shape, (3, 3))
        self.assertEqual((measure.basis, measure.route), ("annual", "catalogue"))
        self.assertEqual((measure.n_years, measure.n_events), (10, 3))
        self.assertFalse(measure.weights.flags.writeable)
        self.assertFalse(measure.grouping.data.flags.writeable)

    def test_a_block_of_years_reweights_each_year(self) -> None:
        years = CatalogueYears.from_labels([1, 2], [1, 2], n_years=10)
        measure = years.measure([1, 2], [0.1, 0.1], years=[0, 1, 2, 3])
        np.testing.assert_allclose(measure.probability, [0.25, 0.25, 0.5])
        self.assertEqual(measure.n_years, 4)
        with self.assertRaisesRegex(ParametricInputError, "outside the given years"):
            years.measure([1, 2], [0.1, 0.1], years=[0, 1])

    def test_invalid_fields_raise(self) -> None:
        good: dict[str, Any] = {
            "grouping": sp.identity(2),
            "weights": [0.5, 0.5],
            "basis": "annual",
            "route": "yearsets",
            "n_years": 2,
        }
        self.assertEqual(OutcomeMeasure(**good).n_outcomes, 2)
        cases: list[tuple[dict[str, Any], str]] = [
            ({"basis": "season"}, "basis and route"),
            ({"route": "frequency"}, "basis and route"),
            ({"weights": []}, "non-empty"),
            ({"weights": [1.5, -0.5]}, "non-empty"),
            ({"weights": [0.0, 0.0]}, "positive sum"),
            ({"weights": [0.5, 0.4]}, "must sum to 1"),
            ({"grouping": np.ones((3, 2))}, "3 rows for 2 weights"),
            ({"grouping": [[0.5, 0.0], [0.0, 1.0]]}, "whole counts"),
            ({"grouping": [[-1.0, 0.0], [0.0, 1.0]]}, "whole counts"),
            ({"n_years": None}, "n_years must be an int"),
            ({"basis": "event", "route": "frequency"}, "has no n_years"),
        ]
        for change, match in cases:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                OutcomeMeasure(**{**good, **change})

    def test_aggregate_needs_one_row_per_event(self) -> None:
        measure = OutcomeMeasure(sp.identity(2), [0.5, 0.5], "annual", "yearsets", 2)
        np.testing.assert_allclose(
            measure.aggregate([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 2.0], [3.0, 4.0]]
        )
        with self.assertRaisesRegex(ParametricInputError, "2 rows"):
            measure.aggregate([1.0, 2.0, 3.0])


class TestEventBasis(unittest.TestCase):
    """One outcome per event, weighted by its frequency, whatever the sum."""

    def test_frequencies_are_the_weights(self) -> None:
        basis = EventBasis()
        measure = basis.measure([5, 7], [0.6, 0.9])
        np.testing.assert_allclose(measure.weights, [0.6, 0.9])
        self.assertAlmostEqual(measure.total_weight, 1.5)
        self.assertIsNone(measure.n_years)
        # E[X] is per year: 0.6 x 1 + 0.9 x 2.
        self.assertAlmostEqual(measure.expectation([1.0, 2.0]), 2.4)
        np.testing.assert_array_equal(basis.blocks([7, 5]), [7, 5])
        self.assertEqual((basis.kind, basis.route), ("event", "frequency"))

    def test_draws_share_their_event_frequency(self) -> None:
        measure = EventBasis().measure([1, 2], [0.2, 0.4], n_draws=2)
        np.testing.assert_allclose(measure.weights, [0.1, 0.2, 0.1, 0.2])
        np.testing.assert_allclose(measure.event_weight, [0.1, 0.2, 0.1, 0.2])

    def test_ensemble_members_take_their_parent_as_block(self) -> None:
        names = ["AL01", "AL01_gen1", "AL01_gen10", "EP02", "EP02_gen1"]
        parents = labels_from_names(names, ENSEMBLE_PARENT)["block"]
        self.assertEqual(parents, ["AL01", "AL01", "AL01", "EP02", "EP02"])
        basis = EventBasis(event_id=[1, 2, 3, 4, 5], block=parents)
        np.testing.assert_array_equal(basis.blocks([3, 4]), ["AL01", "EP02"])
        self.assertFalse(basis.block.flags.writeable)
        with self.assertRaisesRegex(EventAlignmentError, "have no block"):
            basis.blocks([6])
        with self.assertRaisesRegex(EventAlignmentError, "have no block"):
            basis.measure([1, 6], [0.1, 0.1])

    def test_invalid_arguments_raise(self) -> None:
        cases: list[tuple[dict[str, Any], str]] = [
            ({"event_id": [1, 2]}, "together"),
            ({"block": ["a", "b"]}, "together"),
            ({"event_id": [1, 2], "block": ["a"]}, "one label per event"),
            ({"event_id": [1, 2], "block": ["a", None]}, "missing labels"),
            ({"event_id": [1, 1], "block": ["a", "b"]}, "unique"),
        ]
        for kwargs, match in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                EventBasis(**kwargs)
        measures: list[tuple[tuple[Any, ...], dict[str, Any], str]] = [
            (([], []), {}, "without events"),
            (([1, 2], [0.1]), {}, "one value >= 0 per event"),
            (([1, 2], [0.1, -0.1]), {}, "one value >= 0 per event"),
            (([1, 2], [0.0, 0.0]), {}, "positive sum"),
            (([1, 2], [0.1, 0.1]), {"n_draws": 0}, "n_draws must be >= 1"),
        ]
        for args, kwargs, match in measures:
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                EventBasis().measure(*args, **kwargs)


class TestCatalogueYears(unittest.TestCase):
    """A year label per event, from the dates, the names or an array."""

    def test_labels(self) -> None:
        years = CatalogueYears.from_labels(
            [10, 20, 30], [1981, 1983, 1983], n_years=5, first_year=1980
        )
        np.testing.assert_array_equal(years.years, np.arange(1980, 1985))
        np.testing.assert_array_equal(years.year_of([30, 10]), [1983, 1981])
        np.testing.assert_array_equal(years.blocks([30, 10]), [1983, 1981])
        self.assertEqual((years.kind, years.route), ("annual", "catalogue"))
        with self.assertRaisesRegex(EventAlignmentError, "have no year"):
            years.year_of([40])
        parents = CatalogueYears.from_labels(
            [10, 20, 30], [0, 1, 2], n_years=3, block=["a", "b", "a"]
        )
        np.testing.assert_array_equal(parents.blocks([30, 20]), ["a", "b"])

    def test_frequencies_must_be_one_over_the_years(self) -> None:
        years = CatalogueYears.from_labels([1, 2], [0, 1], n_years=10)
        with self.assertRaisesRegex(YearRouteError, "1 of the sample's 2 events"):
            years.measure([1, 2], [0.1, 0.01])
        # A relative error of 1e-7 is within the tolerance.
        years.measure([1, 2], [0.1, 0.1 * (1 + 1e-7)])

    def test_draws_are_outcomes_of_their_own(self) -> None:
        years = CatalogueYears.from_labels([1, 2], [0, 0], n_years=4)
        measure = years.measure([1, 2], [0.25, 0.25], n_draws=2)
        np.testing.assert_array_equal(
            measure.grouping.toarray(), [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]
        )
        np.testing.assert_allclose(measure.weights, [0.125, 0.125, 0.75])

    def test_calendar_years_and_seasons_from_the_dates(self) -> None:
        hazard = _dated(dt.date(2001, 1, 5), dt.date(2001, 8, 1), dt.date(2002, 3, 1))
        calendar = CatalogueYears.from_dates(hazard)
        np.testing.assert_array_equal(calendar.year, [2001, 2001, 2002])
        np.testing.assert_array_equal(calendar.years, [2001, 2002])
        self.assertIn("calendar years", calendar.source)
        seasons = CatalogueYears.from_dates(hazard, season_start_month=7)
        np.testing.assert_array_equal(seasons.year, [2000, 2001, 2001])
        np.testing.assert_array_equal(seasons.years, [2000, 2001])
        stated = CatalogueYears.from_dates(hazard, n_years=5, first_year=1999)
        np.testing.assert_array_equal(stated.years, np.arange(1999, 2004))

    def test_dates_that_do_not_hold_the_years_raise(self) -> None:
        hazard = _dated(dt.date(2001, 1, 5), dt.date(2003, 8, 1))
        cases: list[tuple[dict[str, Any], str]] = [
            ({"n_years": 2}, "3 years, more than n_years = 2"),
            ({"n_years": 5}, "pass first_year"),
            ({"n_years": 5, "first_year": 2002}, "outside the 5 simulation years"),
        ]
        for kwargs, match in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(YearRouteError, match),
            ):
                CatalogueYears.from_dates(hazard, **kwargs)
        no_date = SimpleNamespace(event_id=np.array([1, 2]), date=np.array([0, 1]))
        with self.assertRaisesRegex(YearRouteError, "one ordinal date"):
            CatalogueYears.from_dates(no_date)
        for month in (0, 13):
            with self.subTest(month=month), self.assertRaises(ParametricInputError):
                CatalogueYears.from_dates(hazard, season_start_month=month)
        empty = SimpleNamespace(event_id=np.array([], dtype=int), date=np.array([]))
        with self.assertRaisesRegex(ParametricInputError, "has no events"):
            CatalogueYears.from_dates(empty)

    def test_years_and_blocks_from_the_event_names(self) -> None:
        iris = _named("NA_0_0", "NA_7_1", "NA_9_0")
        years = CatalogueYears.from_event_names(iris, r"NA_(?P<year>[0-9]+)_[0-9]+")
        np.testing.assert_array_equal(years.year, [0, 7, 9])
        np.testing.assert_array_equal(years.years, np.arange(10))
        self.assertIsNone(years.block)
        members = _named("1990_A_x", "1990_A_y", "1991_B_x")
        pattern = r"(?P<year>[0-9]+)_(?P<block>[A-Z])_[a-z]"
        parents = CatalogueYears.from_event_names(members, pattern)
        np.testing.assert_array_equal(parents.blocks([1, 3]), ["A", "B"])

    def test_names_that_do_not_fit_raise(self) -> None:
        pattern = r"NA_(?P<year>[0-9]+)_[0-9]+"
        with self.assertRaisesRegex(YearRouteError, "1 of 2 event names"):
            CatalogueYears.from_event_names(_named("NA_1_0", "NA_x_0"), pattern)
        with self.assertRaisesRegex(YearRouteError, "whole number"):
            CatalogueYears.from_event_names(_named("NA_x_0"), r"NA_(?P<year>\w)_0")
        with self.assertRaisesRegex(ParametricInputError, "named group 'year'"):
            CatalogueYears.from_event_names(_named("NA_1_0"), r"NA_[0-9]+_0")
        short = SimpleNamespace(event_id=np.array([1, 2]), event_name=["NA_1_0"])
        with self.assertRaisesRegex(ParametricInputError, "1 event names for 2"):
            CatalogueYears.from_event_names(short, pattern)
        with self.assertRaisesRegex(YearRouteError, "with the groups"):
            labels_from_names(["A"], r"(?P<block>A)|(?P<other>B)", ["block", "other"])
        with self.assertRaisesRegex(ParametricInputError, "no named groups"):
            labels_from_names(["A"], r"A")

    def test_invalid_labels_raise(self) -> None:
        with self.assertRaisesRegex(YearRouteError, "outside the 3 simulation years"):
            CatalogueYears.from_labels([1, 2], [0, 3], n_years=3)
        with self.assertRaisesRegex(ParametricInputError, "1 entries for 2"):
            CatalogueYears.from_labels([1, 2], [0], n_years=3)
        with self.assertRaisesRegex(ParametricInputError, "n_years must be >= 1"):
            CatalogueYears.from_labels([1], [0], n_years=0)
        years = CatalogueYears.from_labels([1], [0], n_years=3)
        for chosen, match in (([], "non-empty"), ([1, 1], "unique"), ([3], "not")):
            with (
                self.subTest(years=chosen),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                years.check_years(chosen)


class TestYearSets(unittest.TestCase):
    """Sampled years: positions per year, counted with repeats."""

    def test_a_sampling_vect_by_hand(self) -> None:
        # Three events; year 0 holds the first twice and the third, year 1
        # nothing, year 2 the second, which the sample leaves out.
        sets = YearSets.from_sampling_vect([10, 20, 30], [[0, 0, 2], [], [1]])
        self.assertEqual(sets.n_years, 3)
        np.testing.assert_array_equal(sets.sampling_vect[0], [0, 0, 2])
        measure = sets.measure([10, 30], [0.5, 0.2])
        np.testing.assert_array_equal(measure.grouping.toarray(), [[2, 1], [0, 0]])
        np.testing.assert_allclose(measure.weights, [1 / 3, 2 / 3])
        self.assertAlmostEqual(measure.expectation([1.0, 5.0]), 7 / 3)
        self.assertEqual((measure.route, measure.n_years), ("yearsets", 3))
        np.testing.assert_array_equal(sets.blocks([30]), [30])
        with self.assertRaisesRegex(EventAlignmentError, "sampled years"):
            sets.measure([40], [0.1])
        parents = YearSets.from_sampling_vect([10, 20], [[1]], block=["p", "q"])
        np.testing.assert_array_equal(parents.blocks([20, 10]), ["q", "p"])

    def test_invalid_sampling_vects_raise(self) -> None:
        with self.assertRaisesRegex(YearRouteError, "not event ids"):
            YearSets.from_sampling_vect([10, 20], [[10], [20]])
        with self.assertRaisesRegex(ParametricInputError, "at least one year"):
            YearSets.from_sampling_vect([10, 20], [])
        with self.assertRaisesRegex(ParametricInputError, "one value >= 0"):
            YearSets([10, 20], [[0]], frequency=[0.1, -0.1])

    def test_years_from_the_frequencies(self) -> None:
        impact = SimpleNamespace(
            event_id=np.array([3, 1, 2]),
            frequency=np.array([0.5, 0.25, 0.25]),
            frequency_unit="1/year",
        )
        state = np.random.get_state()
        sets = YearSets.from_frequencies(
            impact, n_years=2_000, rng=np.random.default_rng(7)
        )
        self.assertEqual(sets.n_years, 2_000)
        again = YearSets.from_frequencies(
            impact, n_years=2_000, rng=np.random.default_rng(7)
        )
        for first, second in zip(sets.sampling_vect, again.sampling_vect):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        measure = sets.measure([3, 1, 2], [0.5, 0.25, 0.25])
        # About lambda = 1 event per year, half of them the first.
        self.assertAlmostEqual(measure.expectation([1.0, 1.0, 1.0]), 1.0, delta=0.05)
        self.assertAlmostEqual(measure.expectation([1.0, 0.0, 0.0]), 0.5, delta=0.05)
        with self.assertRaisesRegex(YearRouteError, "not those the years were"):
            sets.measure([3, 1], [0.5, 0.3])
        impact.frequency_unit = "1/month"
        with self.assertRaisesRegex(ImpactInputError, "frequency_unit"):
            YearSets.from_frequencies(impact, n_years=10, rng=np.random.default_rng(7))


class TestSampleYears(unittest.TestCase):
    """Poisson counts and draws with replacement, from one generator."""

    def test_rare_events_keep_their_rate(self) -> None:
        # With replacement, each event occurs at its own rate, however rare.
        frequency = np.array([0.01, 0.99])
        years = sample_years(frequency, 20_000, np.random.default_rng(311))
        counts = np.bincount(np.concatenate(years), minlength=2) / 20_000
        np.testing.assert_allclose(counts, frequency, rtol=0.1)

    def test_invalid_arguments_raise(self) -> None:
        rng = np.random.default_rng(0)
        cases: list[tuple[tuple[Any, ...], str]] = [
            (([], 10, rng), "non-empty"),
            (([0.0], 10, rng), "positive sum"),
            (([0.1], 0, rng), "n_years must be >= 1"),
            (([0.1], 10, 7), "numpy.random.Generator"),
        ]
        for args, match in cases:
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                sample_years(*args)


if __name__ == "__main__":
    unittest.main()
