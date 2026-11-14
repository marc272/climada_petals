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

Tests of the objectives, worked out by hand.

Two simulation years of ten hold one event each: year 0 has the loss 4 and
the payout 1, year 1 the loss 10 and the payout 12. The other eight years
have neither.
"""

from __future__ import annotations

import unittest

from climada_petals.engine.parametric import (
    ExpectedUncovered,
    MeanSquaredError,
    ParametricInputError,
    WeightedCVaR,
    outcome_measure,
)

LOSS = [4.0, 10.0]
PAYOUT = [1.0, 12.0]


def _measure():
    return outcome_measure(
        [0, 1], [0.1, 0.1], basis="annual", years=range(10), n_catalogue_years=10
    )


class TestObjectives(unittest.TestCase):
    """Each objective evaluated for given losses and payouts."""

    def test_mean_squared_error_counts_over_payment(self) -> None:
        # Residuals 3 and -2 in two years of probability 0.1 each.
        value = MeanSquaredError().evaluate(LOSS, PAYOUT, _measure())
        self.assertAlmostEqual(value, 0.1 * 9.0 + 0.1 * 4.0)
        self.assertFalse(MeanSquaredError.is_linear)

    def test_expected_uncovered_ignores_over_payment(self) -> None:
        value = ExpectedUncovered().evaluate(LOSS, PAYOUT, _measure())
        self.assertAlmostEqual(value, 0.1 * 3.0)

    def test_weighted_cvar_mixes_the_cvar_and_the_mean(self) -> None:
        measure = _measure()
        # U is 3 in one year of ten and 0 otherwise: CVaR_0.8 = 3 / 2.
        self.assertAlmostEqual(
            WeightedCVaR(alpha=1.0, beta=0.8).evaluate(LOSS, PAYOUT, measure), 1.5
        )
        self.assertAlmostEqual(
            WeightedCVaR(alpha=0.5, beta=0.8).evaluate(LOSS, PAYOUT, measure),
            0.5 * 1.5 + 0.5 * 0.3,
        )
        self.assertAlmostEqual(
            WeightedCVaR(alpha=0.0, beta=0.8).evaluate(LOSS, PAYOUT, measure), 0.3
        )

    def test_invalid_parameters_raise(self) -> None:
        for alpha, beta in ((-0.1, 0.9), (1.1, 0.9), (0.5, 0.0), (0.5, 1.0)):
            with (
                self.subTest(alpha=alpha, beta=beta),
                self.assertRaises(ParametricInputError),
            ):
                WeightedCVaR(alpha=alpha, beta=beta)


if __name__ == "__main__":
    unittest.main()
