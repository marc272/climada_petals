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

Tests of the step payout."""

from __future__ import annotations

import unittest

import numpy as np

from climada_petals.engine.parametric import ParametricInputError, StepPayout
from climada_petals.engine.parametric.payout import NO_STEP


class TestStepPayout(unittest.TestCase):
    """Steps, the attachment modes and the payout per event."""

    def test_uniform_breakpoints(self) -> None:
        payout = StepPayout(attachment=20.0, limit=70.0, n_steps=5)
        np.testing.assert_array_equal(payout.edges, [20, 30, 40, 50, 60, 70])
        self.assertFalse(payout.edges.flags.writeable)

    def test_steps_are_half_open_and_clip_at_the_limit(self) -> None:
        payout = StepPayout(20.0, 70.0, 5, below_attachment="first_bin")
        values = [0.0, 19.9, 20.0, 29.9, 30.0, 69.9, 70.0, 99.0]
        np.testing.assert_array_equal(payout.step_of(values), [0, 0, 0, 0, 1, 4, 4, 4])

    def test_zero_mode_pays_nothing_below_the_attachment(self) -> None:
        payout = StepPayout(20.0, 70.0, 5)
        self.assertEqual(payout.below_attachment, "zero")
        steps = payout.step_of([0.0, 19.9, 20.0, 30.0, 70.0])
        np.testing.assert_array_equal(steps, [NO_STEP, NO_STEP, 0, 1, 4])
        levels = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        paid = payout.evaluate(np.array([[0.0], [19.9], [20.0], [99.0]]), levels)
        np.testing.assert_array_equal(paid[:, 0], [0.0, 0.0, 1.0, 5.0])

    def test_basis_matrix_matches_evaluate(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.uniform(0.0, 90.0, size=(50, 2))
        levels = np.sort(rng.uniform(0.0, 10.0, size=(2, 4)), axis=1)
        for mode in ("zero", "first_bin"):
            payout = StepPayout(20.0, 70.0, 4, below_attachment=mode)
            paid = payout.evaluate(values, levels)
            for j in range(2):
                basis = payout.basis_matrix(values[:, j])
                self.assertEqual(basis.shape, (50, 4))
                np.testing.assert_allclose(
                    basis @ levels[j], paid[:, j], rtol=0, atol=0
                )
                self.assertTrue(np.all(basis.sum(axis=1) <= 1))

    def test_explicit_breakpoints_for_categories(self) -> None:
        payout = StepPayout(1, 6, 5, breakpoints=(1, 2, 3, 4, 5, 6))
        np.testing.assert_array_equal(
            payout.step_of([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), [-1, 0, 1, 2, 3, 4]
        )

    def test_invalid_parameters_raise(self) -> None:
        bad = [
            {"attachment": 1.0, "limit": 1.0, "n_steps": 2},
            {"attachment": 0.0, "limit": 1.0, "n_steps": 0},
            {"attachment": 0.0, "limit": 1.0, "n_steps": True},
            {"attachment": float("nan"), "limit": 1.0, "n_steps": 2},
            {"attachment": 0.0, "limit": 1.0, "n_steps": 2, "breakpoints": (0, 1)},
            {"attachment": 0.0, "limit": 2.0, "n_steps": 2, "breakpoints": (0, 3, 2)},
            {"attachment": 0.0, "limit": 2.0, "n_steps": 2, "breakpoints": (1, 1.5, 2)},
            {"attachment": 0.0, "limit": 1.0, "n_steps": 2, "below_attachment": "x"},
            {"attachment": 0.0, "limit": 1e-13, "n_steps": 1},
            {"attachment": 0.0, "limit": 2.0, "n_steps": 2, "breakpoints": ("a",) * 3},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ParametricInputError):
                StepPayout(**kwargs)

    def test_non_finite_index_raises(self) -> None:
        payout = StepPayout(0.0, 1.0, 2)
        with self.assertRaises(ParametricInputError):
            payout.step_of([0.5, np.nan])
        with self.assertRaises(ParametricInputError):
            payout.evaluate(np.zeros((2, 1)), np.zeros((1, 3)))


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestStepPayout)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
