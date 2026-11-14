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

Tests of the pricings, worked out by hand.

Two simulation years of ten hold one event each, with the payouts 1 and 12.
The other eight years have none.
"""

from __future__ import annotations

import unittest

from climada_petals.engine.parametric import (
    CatalogueYears,
    CostOfCapitalLoad,
    EventBasis,
    ParametricInputError,
    PurePremium,
)

PAYOUT = [1.0, 12.0]


def _measure():
    years = CatalogueYears.from_labels([1, 2], [0, 1], n_years=10)
    return years.measure([1, 2], [0.1, 0.1])


class TestPricing(unittest.TestCase):
    """The pure premium and the cost-of-capital load."""

    def test_premiums(self) -> None:
        measure = _measure()
        expected = 0.1 * 1.0 + 0.1 * 12.0
        self.assertAlmostEqual(PurePremium().premium(PAYOUT, measure), expected)
        # A rate of 0 is the pure premium.
        self.assertAlmostEqual(
            CostOfCapitalLoad(rate=0.0, q=0.9).premium(PAYOUT, measure), expected
        )
        # CVaR_0.9 of the payout is its worst year, 12.
        self.assertAlmostEqual(
            CostOfCapitalLoad(rate=0.1, q=0.9).premium(PAYOUT, measure),
            0.9 * expected + 0.1 * 12.0,
        )

    def test_premiums_on_the_event_basis_above_one_per_year(self) -> None:
        # Frequencies 2.0, 0.6, 0.3 and 0.1 per year: the worst 0.25 per year of
        # the payout is all of 0.1 at 12 and 0.15 of 0.3 at 5.
        measure = EventBasis().measure([1, 2, 3, 4], [2.0, 0.6, 0.3, 0.1])
        payout = [0.0, 1.0, 5.0, 12.0]
        expected = 0.6 * 1.0 + 0.3 * 5.0 + 0.1 * 12.0
        self.assertAlmostEqual(PurePremium().premium(payout, measure), expected)
        cvar = (0.1 * 12.0 + 0.15 * 5.0) / 0.25
        self.assertAlmostEqual(
            CostOfCapitalLoad(rate=0.1, q=0.75).premium(payout, measure),
            0.9 * expected + 0.1 * cvar,
        )

    def test_documents(self) -> None:
        self.assertEqual(PurePremium().to_dict(), {"kind": "pure"})
        self.assertEqual(
            CostOfCapitalLoad(rate=0.06, q=0.99).to_dict(),
            {"kind": "cost_of_capital", "q": 0.99, "rate": 0.06},
        )

    def test_invalid_parameters_raise(self) -> None:
        for rate, level in ((-0.1, 0.9), (1.1, 0.9), (0.1, 1.0)):
            with (
                self.subTest(rate=rate, level=level),
                self.assertRaises(ParametricInputError),
            ):
                CostOfCapitalLoad(rate=rate, q=level)


if __name__ == "__main__":
    unittest.main()
