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

import copy
import pickle
import unittest

import numpy as np

from climada_petals.engine.parametric import (
    MSE,
    CatalogueYears,
    EventBasis,
    ExpectedUncovered,
    ParametricInputError,
    UserObjective,
    WeightedCVaR,
)
from climada_petals.engine.parametric.objectives import tail_degeneracy

LOSS = [4.0, 10.0]
PAYOUT = [1.0, 12.0]


def _measure():
    years = CatalogueYears.from_labels([1, 2], [0, 1], n_years=10)
    return years.measure([1, 2], [0.1, 0.1])


def shortfall_squared(loss, payout, weight):
    """Return the expected squared shortfall, ``sum_o w_o ((L_o - P_o)^+)^2``."""
    return float(weight @ np.maximum(loss - payout, 0.0) ** 2)


class TestObjectives(unittest.TestCase):
    """Each objective evaluated for given losses and payouts."""

    def test_mean_squared_error_counts_over_payment(self) -> None:
        # Residuals 3 and -2 in two years of probability 0.1 each.
        value = MSE().evaluate(LOSS, PAYOUT, _measure())
        self.assertAlmostEqual(value, 0.1 * 9.0 + 0.1 * 4.0)
        self.assertFalse(MSE.is_linear)

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

    def test_on_the_event_basis_above_one_per_year(self) -> None:
        # U is 3 at frequency 0.2 and 0 at 1.3 per year: the worst 0.5 of
        # frequency holds 0.2 at 3 and 0.3 at 0.
        measure = EventBasis().measure([1, 2], [0.2, 1.3])
        self.assertAlmostEqual(
            ExpectedUncovered().evaluate(LOSS, PAYOUT, measure), 0.2 * 3.0
        )
        self.assertAlmostEqual(
            WeightedCVaR(alpha=1.0, beta=0.5).evaluate(LOSS, PAYOUT, measure),
            0.2 * 3.0 / 0.5,
        )

    def test_documents(self) -> None:
        self.assertEqual(MSE().to_dict(), {"kind": "mse"})
        self.assertEqual(ExpectedUncovered().to_dict(), {"kind": "expected_uncovered"})
        self.assertEqual(
            WeightedCVaR(alpha=0.9, beta=0.99).to_dict(),
            {"kind": "weighted_cvar", "alpha": 0.9, "beta": 0.99},
        )

    def test_invalid_parameters_raise(self) -> None:
        for alpha, beta in ((-0.1, 0.9), (1.1, 0.9), (0.5, 0.0), (0.5, 1.0)):
            with (
                self.subTest(alpha=alpha, beta=beta),
                self.assertRaises(ParametricInputError),
            ):
                WeightedCVaR(alpha=alpha, beta=beta)


class TestUserObjective(unittest.TestCase):
    """A user's function of the outcome arrays."""

    def test_it_sees_the_outcomes(self) -> None:
        # Shortfalls 3 and 0 in two years of ten: 0.1 * 9. The function gets
        # three outcomes: the two years, and the eight without an event pooled.
        seen = []

        def spy(loss, payout, weight):
            seen.append((loss.copy(), payout.copy(), weight.copy()))
            return shortfall_squared(loss, payout, weight)

        value = UserObjective(spy, name="spy").evaluate(LOSS, PAYOUT, _measure())
        self.assertAlmostEqual(value, 0.9)
        loss, payout, weight = seen[0]
        self.assertEqual(loss.tolist(), [4.0, 10.0, 0.0])
        self.assertEqual(payout.tolist(), [1.0, 12.0, 0.0])
        np.testing.assert_allclose(weight, [0.1, 0.1, 0.8])

    def test_document_equality_and_pickling(self) -> None:
        objective = UserObjective(
            shortfall_squared, name="squared", parameters={"power": 2, "note": "a"}
        )
        self.assertEqual(
            objective.to_dict(),
            {
                "kind": "user",
                "name": "squared",
                "parameters": {"note": "a", "power": 2},
            },
        )
        self.assertFalse(UserObjective.is_linear)
        same = UserObjective(
            shortfall_squared, name="squared", parameters={"note": "a", "power": 2}
        )
        self.assertEqual(objective, same)
        self.assertEqual(hash(objective), hash(same))
        for copied in (pickle.loads(pickle.dumps(objective)), copy.deepcopy(objective)):
            self.assertEqual(copied, objective)
        with self.assertRaises(TypeError):
            objective.parameters["power"] = 3

    def test_invalid_objectives_raise(self) -> None:
        cases = [
            ({"function": "f", "name": "f"}, "callable"),
            ({"function": shortfall_squared, "name": ""}, "non-empty string"),
            ({"function": shortfall_squared, "name": "f", "parameters": [1]}, "map"),
            (
                {"function": shortfall_squared, "name": "f", "parameters": {"x": []}},
                "map names",
            ),
            (
                {"function": shortfall_squared, "name": "f", "parameters": {1: 2.0}},
                "map names",
            ),
            (
                {
                    "function": shortfall_squared,
                    "name": "f",
                    "parameters": {"x": np.nan},
                },
                "map names",
            ),
        ]
        for fields, match in cases:
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                function = fields.pop("function")
                UserObjective(function, **fields)

    def test_invalid_values_raise(self) -> None:
        for value, match in (
            (np.nan, "not finite"),
            ("1.0", "real number"),
            (True, "real number"),
        ):
            objective = UserObjective(lambda *_, v=value: v, name="bad")
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                objective.evaluate(LOSS, PAYOUT, _measure())


class TestTailDegeneracy(unittest.TestCase):
    """When the weighted CVaR's tail degenerates, worked out by hand."""

    def test_annual_basis(self) -> None:
        # Two years of ten have a loss: their weight is 0.2.
        measure = _measure()
        cases = {
            0.75: "expected_uncovered",
            0.8: "expected_uncovered",
            0.85: None,
            0.9: "single_outcome",
            0.95: "single_outcome",
        }
        for beta, flag in cases.items():
            with self.subTest(beta=beta):
                self.assertEqual(
                    tail_degeneracy(WeightedCVaR(0.5, beta), LOSS, measure),
                    (0.2, flag),
                )
        self.assertEqual(
            tail_degeneracy(ExpectedUncovered(), LOSS, measure), (0.2, None)
        )
        self.assertEqual(
            tail_degeneracy(WeightedCVaR(0.0, 0.95), LOSS, measure), (0.2, None)
        )

    def test_event_basis(self) -> None:
        # Losses at frequencies 1.5 and 0.5, and an event without loss at 0.01:
        # the smallest weight of an outcome with a loss is 0.5, not 0.01.
        measure = EventBasis().measure([1, 2, 3], [1.5, 0.5, 0.01])
        loss = [4.0, 10.0, 0.0]
        self.assertEqual(
            tail_degeneracy(WeightedCVaR(0.9, 0.6), loss, measure),
            (2.0, "single_outcome"),
        )
        self.assertEqual(
            tail_degeneracy(WeightedCVaR(0.9, 0.4), loss, measure), (2.0, None)
        )


if __name__ == "__main__":
    unittest.main()
