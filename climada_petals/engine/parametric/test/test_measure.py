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

Tests of the outcome measures and the risk measures."""

from __future__ import annotations

import unittest

import numpy as np

from climada_petals.engine.parametric import (
    ParametricInputError,
    outcome_measure,
    var_cvar,
)


class TestAnnualMeasure(unittest.TestCase):
    """One outcome per year; the loss-free years pooled into one outcome."""

    def test_years_group_events_and_empty_years_are_pooled(self) -> None:
        measure = outcome_measure(
            [3, 3, 7],
            [0.1, 0.1, 0.1],
            basis="annual",
            years=range(10),
            n_catalogue_years=10,
        )
        np.testing.assert_array_equal(measure.outcome_of_event, [0, 0, 1])
        np.testing.assert_allclose(measure.probability, [0.1, 0.1, 0.8])
        np.testing.assert_allclose(measure.aggregate([1.0, 2.0, 5.0]), [3.0, 5.0, 0.0])
        self.assertAlmostEqual(measure.expectation([1.0, 2.0, 5.0]), 0.8)
        np.testing.assert_allclose(measure.event_probability, [0.1, 0.1, 0.1])
        self.assertEqual(measure.grouping_matrix.shape, (3, 3))

    def test_a_block_of_years_reweights_each_year(self) -> None:
        measure = outcome_measure(
            [1, 2],
            [0.1, 0.1],
            basis="annual",
            years=[0, 1, 2, 3],
            n_catalogue_years=10,
        )
        np.testing.assert_allclose(measure.probability, [0.25, 0.25, 0.5])
        self.assertEqual(measure.n_years, 4)

    def test_frequencies_must_be_one_over_the_years(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "1 / N"):
            outcome_measure(
                [1, 2],
                [0.1, 0.01],
                basis="annual",
                years=range(10),
                n_catalogue_years=10,
            )

    def test_events_outside_the_years_raise(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "outside"):
            outcome_measure(
                [1, 12],
                [0.1, 0.1],
                basis="annual",
                years=range(10),
                n_catalogue_years=10,
            )


class TestEventMeasure(unittest.TestCase):
    """One outcome per event, plus "no event"."""

    def test_no_event_outcome_takes_the_rest(self) -> None:
        measure = outcome_measure(
            [3, 3, 7],
            [0.1, 0.1, 0.1],
            basis="event",
            years=range(10),
            n_catalogue_years=10,
        )
        np.testing.assert_allclose(measure.probability, [0.1, 0.1, 0.1, 0.7])
        np.testing.assert_array_equal(measure.outcome_of_event, [0, 1, 2])

    def test_more_than_one_event_per_year_raises(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "annual basis"):
            outcome_measure(
                [0, 1], [0.6, 0.6], basis="event", years=[0, 1], n_catalogue_years=2
            )


class TestVarCvar(unittest.TestCase):
    """Lower quantile and Rockafellar-Uryasev CVaR."""

    def test_example(self) -> None:
        self.assertEqual(var_cvar([0.0, 1.0, 2.0, 10.0], [0.25] * 4, 0.5), (1.0, 6.0))

    def test_atom_at_the_var_is_split(self) -> None:
        var, cvar = var_cvar([0.0, 4.0], [0.9, 0.1], 0.8)
        self.assertEqual(var, 0.0)
        self.assertAlmostEqual(cvar, 2.0)

    def test_a_level_on_a_step_gives_the_exact_lower_quantile(self) -> None:
        values = np.arange(100.0)
        prob = np.full(100, 0.01)
        for beta in (0.9, 0.95, 0.99):
            var, cvar = var_cvar(values, prob, beta)
            self.assertEqual(var, round(beta * 100) - 1)
            self.assertAlmostEqual(cvar, values[values > var].mean())

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ParametricInputError):
            var_cvar([1.0], [1.0], 1.0)
        with self.assertRaises(ParametricInputError):
            var_cvar([1.0, 2.0], [0.5, 0.6], 0.5)


if __name__ == "__main__":
    unittest.main()
