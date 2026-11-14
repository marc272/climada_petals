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

Tests of the risk measures: VaR, CVaR and tail weights, by hand."""

from __future__ import annotations

import unittest

import numpy as np

from climada_petals.engine.parametric import ParametricInputError, var_cvar
from climada_petals.engine.parametric.riskmeasures import tail_weights, var_cvar_levels


class TestVarCvar(unittest.TestCase):
    """VaR and the Rockafellar-Uryasev CVaR, with probabilities and frequencies."""

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

    def test_frequencies_above_one_per_year(self) -> None:
        # Frequencies 2.0, 0.6, 0.3 and 0.1 per year, 3.0 in all. The worst
        # 0.25 per year: all of 0.1 at 10 and 0.15 of the 0.3 at 5.
        values = [1.0, 2.0, 5.0, 10.0]
        frequency = [2.0, 0.6, 0.3, 0.1]
        var, cvar = var_cvar(values, frequency, 0.75)
        self.assertEqual(var, 5.0)
        self.assertAlmostEqual(cvar, (0.1 * 10.0 + 0.15 * 5.0) / 0.25)
        # Shuffled, the same.
        self.assertEqual(var_cvar(values[::-1], frequency[::-1], 0.75), (var, cvar))
        # 1 - beta exactly on the step at 10: the lower atom, 5.
        var, cvar = var_cvar(values, frequency, 0.9)
        self.assertEqual(var, 5.0)
        self.assertAlmostEqual(cvar, 10.0)

    def test_missing_mass_counts_as_zero(self) -> None:
        # Frequencies sum to 0.05: the rest of the year, 0.95, holds nothing.
        var, cvar = var_cvar([4.0, 8.0], [0.04, 0.01], 0.9)
        self.assertEqual(var, 0.0)
        self.assertAlmostEqual(cvar, (0.04 * 4.0 + 0.01 * 8.0) / 0.1)
        var, cvar = var_cvar([4.0, 8.0], [0.04, 0.01], 0.98)
        self.assertEqual(var, 4.0)
        self.assertAlmostEqual(cvar, (0.01 * 8.0 + 0.01 * 4.0) / 0.02)
        # The same as an outcome of 0 holding the missing mass.
        self.assertEqual(
            var_cvar([4.0, 8.0], [0.04, 0.01], 0.97),
            var_cvar([4.0, 8.0, 0.0], [0.04, 0.01, 0.95], 0.97),
        )

    def test_several_levels_at_once(self) -> None:
        values = np.arange(100.0)
        prob = np.full(100, 0.01)
        betas = (0.5, 0.9, 0.99)
        self.assertEqual(
            var_cvar_levels(values, prob, betas),
            [var_cvar(values, prob, beta) for beta in betas],
        )

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ParametricInputError):
            var_cvar([1.0], [1.0], 1.0)
        with self.assertRaisesRegex(ParametricInputError, "positive sum"):
            var_cvar([1.0, 2.0], [0.0, 0.0], 0.5)
        for values, prob in (([], []), ([1.0, 2.0], [1.0]), ([1.0], [-1.0])):
            with (
                self.subTest(values=values, prob=prob),
                self.assertRaisesRegex(ParametricInputError, "non-empty"),
            ):
                var_cvar(values, prob, 0.5)


class TestTailWeights(unittest.TestCase):
    """The CVaR weights of each outcome."""

    def test_by_hand(self) -> None:
        values = [0.0, 1.0, 2.0, 10.0]
        weights = tail_weights(values, [0.25] * 4, 0.625)
        np.testing.assert_allclose(weights, [0.0, 0.0, 0.125, 0.25])
        _, cvar = var_cvar(values, [0.25] * 4, 0.625)
        self.assertAlmostEqual(float(weights @ values) / 0.375, cvar)

    def test_the_atom_at_the_var_is_shared_pro_rata(self) -> None:
        # Two outcomes at the VaR, 3, with weights 0.1 and 0.3; 0.2 of the
        # tail is left for them after the 0.1 above.
        weights = tail_weights([3.0, 3.0, 9.0, 1.0], [0.1, 0.3, 0.1, 0.5], 0.7)
        np.testing.assert_allclose(weights, [0.05, 0.15, 0.1, 0.0])

    def test_frequencies(self) -> None:
        values = [1.0, 2.0, 5.0, 10.0]
        weights = tail_weights(values, [2.0, 0.6, 0.3, 0.1], 0.75)
        np.testing.assert_allclose(weights, [0.0, 0.0, 0.15, 0.1])
        # Missing mass at a VaR of 0: its share holds no value and is left out.
        weights = tail_weights([0.0, 4.0], [0.01, 0.04], 0.9)
        np.testing.assert_allclose(weights, [0.06 * 0.01 / 0.96, 0.04])
        np.testing.assert_allclose(tail_weights([4.0], [0.04], 0.9), [0.04])


if __name__ == "__main__":
    unittest.main()
