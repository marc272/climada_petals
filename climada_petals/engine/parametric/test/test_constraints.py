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

Tests of the constraints: the budget, the limit, the amounts and the fill rule."""

from __future__ import annotations

import copy
import pickle
import unittest
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from climada_petals.engine.parametric import (
    AmountConstraints,
    BinaryPayout,
    Budget,
    CoverDefinitionError,
    Free,
    Limit,
    LinearPayout,
    ParametricInputError,
    PayoutFunction,
    StepPayout,
)
from climada_petals.engine.parametric.constraints import (
    check_bounded,
    check_unpaid_at_zero,
    fill_amounts,
    final_amounts,
    resolve_limit,
    resolved_cover,
    snap_amounts,
    unit_amounts,
    violation,
)
from climada_petals.engine.parametric.test._synthetic import array_sample, template


@dataclass(frozen=True)
class _Square(PayoutFunction):
    """A payout that is not linear in an amount: ``factor x^2``."""

    kind: ClassVar[str] = "test_square"

    factor: float
    tail: str = "upper"

    def evaluate(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return ``factor x^2``."""
        return float(self.factor) * np.square(np.asarray(index, dtype=float))


def _chain(lower: list[float], upper: list[float]) -> AmountConstraints:
    """Bounds with monotone rows."""
    return AmountConstraints(np.array(lower), np.array(upper)).with_monotone_rows()


class TestBudget(unittest.TestCase):
    """A ratio, or an amount: one, or one per unit."""

    def test_a_ratio_or_an_amount(self) -> None:
        self.assertEqual(Budget(ratio=0.8).bound(("U0",), 10.0), 8.0)
        self.assertEqual(Budget(amount=3.0).bound(("U0", "U1"), 10.0), 3.0)
        by_unit = Budget(amount={"U0": 1.0, "U1": 2.0})
        self.assertEqual(by_unit.bound(("U1",), 10.0), 2.0)
        self.assertEqual(Budget(ratio=1).to_dict(), {"ratio": 1.0})
        self.assertEqual(Budget(amount=3).to_dict(), {"amount": 3.0})
        self.assertEqual(
            Budget(amount={"U0": 1.0, 7: 2.0}).to_dict(),
            {"amount": {"U0": 1.0, "7": 2.0}},
        )
        for budget in (Budget(ratio=0.5), by_unit):
            with self.subTest(budget=budget):
                self.assertEqual(hash(budget), hash(copy.deepcopy(budget)))
                self.assertEqual(pickle.loads(pickle.dumps(budget)), budget)

    def test_invalid_budgets_raise(self) -> None:
        bad = [
            {},
            {"ratio": 0.5, "amount": 1.0},
            {"ratio": -0.1},
            {"ratio": float("inf")},
            {"amount": -1.0},
            {"amount": {}},
            {"amount": {"U0": -1.0}},
        ]
        for fields in bad:
            with self.subTest(fields=fields), self.assertRaises(ParametricInputError):
                Budget(**fields)

    def test_the_budget_fits_the_scope(self) -> None:
        units = ("U0", "U1")
        Budget(ratio=0.5).check("individual", units)
        Budget(amount=1.0).check("joint", units)
        Budget(amount=1.0).check("individual", ("U0",))
        Budget(amount={"U0": 1.0, "U1": 2.0}).check("individual", units)
        cases = [
            (Budget(amount=1.0), "individual", "takes a mapping by unit id"),
            (Budget(amount={"U0": 1.0, "U1": 2.0}), "joint", "one budget on the"),
            (Budget(amount={"U0": 1.0}), "individual", "one amount per unit id"),
        ]
        for budget, scope, match in cases:
            with (
                self.subTest(scope=scope, match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                budget.check(scope, units)


class TestLimit(unittest.TestCase):
    """``Limit.largest_loss()`` becomes numbers per unit."""

    def test_the_largest_loss_of_each_unit(self) -> None:
        loss = sp.csr_matrix([[1.0, 0.0], [4.0, 2.0], [3.0, 5.0]])
        resolved = resolve_limit(Limit.largest_loss(), ("U0", "U1"), loss)
        self.assertEqual(resolved, Limit({"U0": 4.0, "U1": 5.0}))
        stated = Limit(7.0)
        self.assertIs(resolve_limit(stated, ("U0", "U1"), loss), stated)
        self.assertIsNone(resolve_limit(None, ("U0", "U1"), loss))


class TestAmounts(unittest.TestCase):
    """Snapping, measuring and filling amounts."""

    def test_snapping_to_bounds_and_order(self) -> None:
        bounds = AmountConstraints(np.zeros(3), np.array([5.0, 5.0, np.inf]))
        np.testing.assert_array_equal(
            snap_amounts(np.array([-1e-16, 5.0 + 1e-12, 7.0]), bounds), [0.0, 5.0, 7.0]
        )
        # Monotone: a level an ulp below the one before is lifted to it, and the
        # bounds are tightened to the order first.
        chain = _chain([0.0, 1.0, 0.0], [9.0, 9.0, 4.0])
        snapped = snap_amounts(np.array([3.0, 3.0 - 1e-15, 4.0 + 1e-12]), chain)
        np.testing.assert_array_equal(snapped, [3.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            snap_amounts(np.array([-1.0, 0.5, 2.0]), chain), [0.0, 1.0, 2.0]
        )
        self.assertTrue(np.all(np.diff(snapped) >= 0))

    def test_violation(self) -> None:
        chain = _chain([0.0, 0.0], [5.0, 5.0])
        self.assertEqual(violation(np.array([1.0, 2.0]), chain), 0.0)
        self.assertAlmostEqual(violation(np.array([-0.5, 2.0]), chain), 0.5)
        self.assertAlmostEqual(violation(np.array([1.0, 6.0]), chain), 1.0)
        self.assertAlmostEqual(violation(np.array([3.0, 1.0]), chain), 2.0)
        bounds = AmountConstraints(np.zeros(1), np.ones(1))
        self.assertEqual(violation(np.array([1.0]), bounds), 0.0)

    def test_the_fill_rule(self) -> None:
        chain = _chain([0.0] * 4, [10.0] * 4)
        values = np.array([1.0, 7.0, 3.0, 4.0])
        unidentified = np.array([False, True, False, True])
        np.testing.assert_allclose(
            fill_amounts(values, chain, unidentified, "lowest"), [1.0, 1.0, 3.0, 3.0]
        )
        np.testing.assert_allclose(
            fill_amounts(values, chain, unidentified, "highest"), [1.0, 3.0, 3.0, 10.0]
        )
        # Without rows, the bounds.
        bounds = AmountConstraints(np.array([0.5, 0.0]), np.array([2.0, 3.0]))
        np.testing.assert_allclose(
            fill_amounts(
                np.array([1.0, 1.0]), bounds, np.array([True, False]), "lowest"
            ),
            [0.5, 1.0],
        )
        np.testing.assert_array_equal(
            fill_amounts(values, chain, np.zeros(4, dtype=bool), "lowest"), values
        )

    def test_a_fill_that_cannot_be_made_raises(self) -> None:
        open_top = _chain([0.0, 0.0], [10.0, np.inf])
        unidentified = np.array([False, True])
        with self.assertRaisesRegex(ParametricInputError, "upper bound"):
            fill_amounts(np.array([1.0, 0.0]), open_top, unidentified, "highest")
        tight = _chain([0.0, 0.0], [10.0, 1.0])
        with self.assertRaisesRegex(CoverDefinitionError, "can't be set"):
            fill_amounts(np.array([2.0, 0.0]), tight, unidentified, "lowest")


class TestUnitAmounts(unittest.TestCase):
    """The amounts of a template's units on a design sample."""

    def setUp(self) -> None:
        # One unit's indices reach the first of two steps only.
        self.sample, _ = array_sample(
            [[30.0, 60.0], [35.0, 25.0]], [[4.0, 9.0], [6.0, 1.0]], [0, 1], 10
        )

    def test_bases_constraints_and_unidentified_amounts(self) -> None:
        cover = template(self.sample, StepPayout.uniform(20.0, 70.0, 2))
        limit = resolve_limit(cover.limit, self.sample.unit_id, self.sample.loss)
        units = unit_amounts(cover, self.sample, scope="individual", limit=limit)
        self.assertEqual([unit.unit_id for unit in units], ["U0", "U1"])
        np.testing.assert_array_equal(units[0].basis.toarray(), [[1, 0], [1, 0]])
        np.testing.assert_array_equal(units[0].constraints.upper, [6.0, 6.0])
        self.assertEqual(units[0].unidentified_names, ("levels[1]",))
        self.assertEqual(units[1].unidentified_names, ())
        check_unpaid_at_zero(self.sample, units)
        amounts = final_amounts(units, np.array([5.0, 9.0, 1.0, 8.0]), "lowest")
        np.testing.assert_array_equal(amounts["U0"], [5.0, 5.0])
        resolved = resolved_cover(cover, units, amounts, limit)
        self.assertTrue(resolved.is_resolved)
        self.assertEqual(resolved.limit, Limit({"U0": 6.0, "U1": 9.0}))
        self.assertEqual(resolved.payout["U1"].levels, (1.0, 8.0))

    def test_fixed_amounts_are_not_calibrated(self) -> None:
        payout = BinaryPayout(trigger=28.0, amount=Free())
        fixed = LinearPayout(attachment=20.0, exhaustion=70.0, limit=2.0)
        cover = template(self.sample, {"U0": payout, "U1": fixed}, limit=None)
        units = unit_amounts(cover, self.sample, scope="joint", limit=None)
        np.testing.assert_array_equal([unit.free[0] for unit in units], [True, False])
        np.testing.assert_array_equal(units[1].constraints.lower, [2.0])
        with self.assertRaisesRegex(ParametricInputError, "no upper bound"):
            check_bounded(units, "expected_uncovered")
        check_bounded(units[1:], "expected_uncovered")

    def test_routes_this_version_does_not_have_raise(self) -> None:
        free_threshold = StepPayout([20.0, Free(30.0, 50.0, grid=3)], [Free(), Free()])
        cases = [
            (_Square(1.0), "individual", "not linear in its amounts"),
            (free_threshold, "individual", "need the grid route"),
            (free_threshold, "joint", "thresholds are fixed in a joint design"),
        ]
        for payout, scope, match in cases:
            cover = template(self.sample, payout, limit=None)
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                unit_amounts(cover, self.sample, scope=scope, limit=None)

    def test_a_fixed_amount_above_the_limit_raises(self) -> None:
        payout = StepPayout.uniform(20.0, 70.0, 2, levels=[Free(), 20.0])
        cover = template(self.sample, payout)
        limit = resolve_limit(cover.limit, self.sample.unit_id, self.sample.loss)
        with self.assertRaisesRegex(CoverDefinitionError, "can't stay within"):
            unit_amounts(cover, self.sample, scope="individual", limit=limit)

    def test_a_payout_that_may_pay_at_zero_on_left_out_events(self) -> None:
        sample, _ = array_sample(
            [[30.0], [60.0], [0.0]], [[4.0], [10.0], [0.0]], [0, 1, 2], 10
        )
        cover = template(sample, StepPayout.uniform(0.0, 70.0, 2), limit=None)
        units = unit_amounts(cover, sample, scope="individual", limit=None)
        with self.assertRaisesRegex(ParametricInputError, "left out 1 events"):
            check_unpaid_at_zero(sample, units)
        capped = template(sample, StepPayout.uniform(0.0, 70.0, 2), limit=Limit(0.0))
        units = unit_amounts(capped, sample, scope="individual", limit=Limit(0.0))
        check_unpaid_at_zero(sample, units)


if __name__ == "__main__":
    unittest.main()
