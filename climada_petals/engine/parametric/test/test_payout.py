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

Tests of the payout families, the payout protocol and the registry."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Any, ClassVar
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp

from climada_petals.engine.cat_bonds.subarea_calculations import calc_payout
from climada_petals.engine.parametric import (
    BinaryPayout,
    CoverDefinitionError,
    Free,
    LinearInAmounts,
    LinearPayout,
    ParametricInputError,
    PayoutFunction,
    PiecewiseLinearPayout,
    StepPayout,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from climada_petals.engine.parametric.payout import payout_from_dict, register_payout

EPS = 1e-9


def _families() -> list[LinearInAmounts]:
    """Return resolved payouts of every family, on both tails."""
    return [
        StepPayout([25.0, 35.0, 50.0], [10.0, 40.0, 100.0]),
        StepPayout([-1.0, -2.0], [5.0, 20.0], tail="lower"),
        LinearPayout(30.0, 50.0, 1000.0),
        LinearPayout(-1.0, -3.0, 100.0, tail="lower"),
        PiecewiseLinearPayout([50.0, 80.0, 120.0], [50.0, 250.0, 700.0]),
        PiecewiseLinearPayout([-1.0, -2.0, -4.0], [0.0, 30.0, 30.0], tail="lower"),
        BinaryPayout(210.0, 1800.0),
        BinaryPayout(-1.5, 60.0, tail="lower"),
    ]


def _first_threshold(payout: LinearInAmounts) -> float:
    """Return the first threshold of a resolved built-in payout, in index units."""
    for name in ("breakpoints[0]", "attachment", "knots[0]", "trigger"):
        if name in payout.parameters():
            return float(payout.parameters()[name])
    raise AssertionError(payout)


class TestStepPayout(unittest.TestCase):
    """Steps at and around their breakpoints, on both tails."""

    def test_payouts_at_and_around_the_breakpoints(self) -> None:
        payout = StepPayout([25.0, 35.0, 50.0], [10.0, 40.0, 100.0])
        values = [-5.0, 0.0, 25 - EPS, 25.0, 25 + EPS, 35 - EPS, 35.0, 50 - EPS]
        values += [50.0, 50 + EPS, 1e9]
        expected = [0, 0, 0, 10, 10, 10, 40, 40, 100, 100, 100]
        np.testing.assert_array_equal(payout.evaluate(values), expected)
        self.assertEqual(payout.n_steps, 3)

    def test_a_lower_tail_pays_as_the_index_falls(self) -> None:
        payout = StepPayout([-1.0, -2.0], [5.0, 20.0], tail="lower")
        values = [0.0, -1 + EPS, -1.0, -1.5, -2 + EPS, -2.0, -30.0]
        np.testing.assert_array_equal(payout.evaluate(values), [0, 0, 5, 5, 5, 20, 20])

    def test_uniform_is_the_prototypes_construction(self) -> None:
        payout = StepPayout.uniform(20.0, 70.0, 4)
        self.assertEqual(payout.breakpoints, (20.0, 32.5, 45.0, 57.5))
        self.assertEqual(payout.levels, (Free(),) * 4)
        self.assertEqual(payout.amount_names(), tuple(f"levels[{k}]" for k in range(4)))
        lower = StepPayout.uniform(70, 20, np.int64(2), levels=[1, 2], tail="lower")
        self.assertEqual(lower.breakpoints, (70.0, 45.0))
        self.assertEqual(lower.levels, (1.0, 2.0))
        np.testing.assert_array_equal(
            lower.evaluate([80, 70, 46, 45, 0]), [0, 1, 1, 2, 2]
        )

    def test_uniform_checks_its_arguments(self) -> None:
        bad = [
            (20.0, 70.0, 0, "upper"),
            (20.0, 70.0, True, "upper"),
            (20.0, 70.0, 2.0, "upper"),
            (70.0, 20.0, 2, "upper"),
            (20.0, 70.0, 2, "lower"),
            (20.0, 20.0, 2, "upper"),
            (math.nan, 70.0, 2, "upper"),
            (20.0, 70.0, 2, "both"),
        ]
        for attachment, exhaustion, n_steps, tail in bad:
            with (
                self.subTest(args=(attachment, exhaustion, n_steps, tail)),
                self.assertRaises(CoverDefinitionError),
            ):
                StepPayout.uniform(attachment, exhaustion, n_steps, tail=tail)

    def test_invalid_parameters_raise(self) -> None:
        bad: list[dict[str, Any]] = [
            {"breakpoints": [], "levels": []},
            {"breakpoints": [2.0, 1.0], "levels": [1.0, 2.0]},
            {"breakpoints": [1.0, 1.0], "levels": [1.0, 2.0]},
            {"breakpoints": [1.0, 2.0], "levels": [1.0]},
            {"breakpoints": [1.0, 2.0], "levels": [1.0, -2.0]},
            {"breakpoints": [1.0, 2.0], "levels": [2.0, 1.0]},
            {"breakpoints": [1.0, math.inf], "levels": [1.0, 2.0]},
            {"breakpoints": [1.0, 2.0], "levels": [1.0, 2.0], "tail": "up"},
            {"breakpoints": [1.0, 2.0], "levels": [1.0, 2.0], "tail": 1},
            {"breakpoints": [1.0, 2.0], "levels": [1.0, 2.0], "monotone": 1},
            {"breakpoints": [1.0, Free(1.0, 3.0)], "levels": [1.0, 2.0]},
            {"breakpoints": [1.0, Free(1.0)], "levels": [1.0, 2.0]},
            {"breakpoints": [1.0, 2.0], "levels": [Free(-1.0), 2.0]},
            {"breakpoints": 1.0, "levels": [1.0]},
            {"breakpoints": "12", "levels": [1.0, 2.0]},
            {"breakpoints": [1.0, 2.0], "levels": [1.0, 2.0], "tail": "lower"},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(CoverDefinitionError):
                StepPayout(**kwargs)

    def test_monotone_false_allows_falling_levels(self) -> None:
        payout = StepPayout([1.0, 2.0], [10.0, 5.0], monotone=False)
        np.testing.assert_array_equal(payout.evaluate([0.0, 1.0, 2.0]), [0, 10, 5])
        constraints = payout.amount_constraints()
        self.assertEqual(constraints.a_ub.shape, (0, 2))

    def test_index_of_any_shape_and_type(self) -> None:
        payout = StepPayout([25.0, 35.0], [10.0, 40.0])
        paid = payout.evaluate(np.array([[20, 25], [35, 30]], dtype=np.int64))
        np.testing.assert_array_equal(paid, [[0, 10], [40, 10]])
        self.assertEqual(payout.evaluate(30).shape, ())
        self.assertEqual(payout.evaluate(np.zeros((0, 3))).shape, (0, 3))
        for index in ([25.0, math.nan], [math.inf], ["a"], [[1.0], [1.0, 2.0]]):
            with self.subTest(index=index), self.assertRaises(ParametricInputError):
                payout.evaluate(index)


class TestLinearPayout(unittest.TestCase):
    """The linear family and its agreement with cat_bonds' calc_payout."""

    def test_payouts_at_and_around_the_attachment_and_exhaustion(self) -> None:
        payout = LinearPayout(attachment=30.0, exhaustion=50.0, limit=1000.0)
        values = [0.0, 30 - EPS, 30.0, 32.0, 40.0, 50 - 1e-6, 50.0, 70.0]
        expected = [0, 0, 0, 100, 500, 1000 - 5e-5, 1000, 1000]
        np.testing.assert_allclose(payout.evaluate(values), expected, rtol=1e-12)

    def test_a_lower_tail(self) -> None:
        payout = LinearPayout(-1.0, -3.0, 100.0, tail="lower")
        values = [0.0, -1.0, -2.0, -3.0, -4.0]
        np.testing.assert_allclose(payout.evaluate(values), [0, 0, 50, 100, 100])

    def test_from_triggers_takes_cat_bonds_words(self) -> None:
        payout = LinearPayout.from_triggers(min_trig=30.0, max_trig=50.0, max_pay=1e3)
        self.assertEqual(payout, LinearPayout(30.0, 50.0, 1000.0))
        lower = LinearPayout.from_triggers(-1.0, -3.0, 5.0, tail="lower")
        self.assertEqual(lower.tail, "lower")

    def test_equals_calc_payout_on_a_float_index(self) -> None:
        rng = np.random.default_rng(15)
        values = np.concatenate([rng.uniform(0.0, 90.0, 200), [30.0, 50.0, 29.0]])
        for min_trig, max_trig, max_pay in ((30.0, 50.0, 1e6), (0.5, 80.25, 3.0)):
            expected = calc_payout(min_trig, max_trig, pd.DataFrame(values), max_pay)
            payout = LinearPayout.from_triggers(min_trig, max_trig, max_pay)
            with self.subTest(triggers=(min_trig, max_trig)):
                np.testing.assert_allclose(
                    payout.evaluate(values), expected, rtol=1e-12, atol=0.0
                )

    def test_an_integer_index_is_not_truncated(self) -> None:
        payout = LinearPayout(30.0, 50.0, 1.0)
        integers = np.array([35, 40, 60], dtype=np.int64)
        np.testing.assert_allclose(payout.evaluate(integers), [0.25, 0.5, 1.0])
        truncated = calc_payout(30.0, 50.0, pd.DataFrame(integers), 1.0)
        np.testing.assert_array_equal(truncated, [0, 0, 1])

    def test_invalid_parameters_raise(self) -> None:
        bad = [
            (50.0, 30.0, 1.0, "upper"),
            (30.0, 30.0, 1.0, "upper"),
            (30.0, 50.0, 1.0, "lower"),
            (30.0, 50.0, -1.0, "upper"),
            (30.0, Free(40.0, 60.0), 1.0, "upper"),
            (30.0, 50.0, Free(-1.0), "upper"),
            (30.0, 50.0, math.nan, "upper"),
        ]
        for attachment, exhaustion, limit, tail in bad:
            with (
                self.subTest(args=(attachment, exhaustion, limit, tail)),
                self.assertRaises(CoverDefinitionError),
            ):
                LinearPayout(attachment, exhaustion, limit, tail=tail)


class TestPiecewiseLinearPayout(unittest.TestCase):
    """Interpolation between the knots, on both tails."""

    def test_payouts_at_and_around_the_knots(self) -> None:
        payout = PiecewiseLinearPayout([50.0, 80.0, 120.0], [50.0, 250.0, 700.0])
        values = [0.0, 50 - EPS, 50.0, 65.0, 80 - EPS, 80.0, 100.0, 120.0, 200.0]
        expected = [0, 0, 50, 150, 250, 250, 475, 700, 700]
        np.testing.assert_allclose(payout.evaluate(values), expected, rtol=1e-7)

    def test_a_lower_tail(self) -> None:
        payout = PiecewiseLinearPayout([-1.0, -2.0], [0.0, 100.0], tail="lower")
        values = [0.0, -1.0, -1.5, -2.0, -5.0]
        np.testing.assert_allclose(payout.evaluate(values), [0, 0, 50, 100, 100])

    def test_monotone_false_and_the_number_of_knots(self) -> None:
        payout = PiecewiseLinearPayout([1.0, 2.0], [10.0, 0.0], monotone=False)
        np.testing.assert_allclose(payout.evaluate([1.0, 1.5, 3.0]), [10, 5, 0])
        bad: list[dict[str, Any]] = [
            {"knots": [1.0], "levels": [1.0]},
            {"knots": [1.0, 2.0], "levels": [2.0, 1.0]},
            {"knots": [1.0, 2.0], "levels": [1.0, 2.0, 3.0]},
            {"knots": [2.0, 1.0], "levels": [1.0, 2.0]},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(CoverDefinitionError):
                PiecewiseLinearPayout(**kwargs)


class TestBinaryPayout(unittest.TestCase):
    """The binary family at and around its trigger."""

    def test_payouts_at_and_around_the_trigger(self) -> None:
        payout = BinaryPayout(trigger=210.0, amount=1800.0)
        values = [0.0, 210 - EPS, 210.0, 210 + EPS, 1e6]
        np.testing.assert_array_equal(payout.evaluate(values), [0, 0, 1800, 1800, 1800])
        lower = BinaryPayout(-1.5, 60.0, tail="lower")
        np.testing.assert_array_equal(lower.evaluate([0.0, -1.5, -9.0]), [0, 60, 60])

    def test_invalid_parameters_raise(self) -> None:
        for trigger, amount in ((math.nan, 1.0), (1.0, -1.0), (Free(1.0), 1.0)):
            with (
                self.subTest(trigger=trigger, amount=amount),
                self.assertRaises(CoverDefinitionError),
            ):
                BinaryPayout(trigger, amount)


class TestProtocol(unittest.TestCase):
    """What every family promises: basis, monotonicity, parameters, templates."""

    def test_basis_matches_evaluate(self) -> None:
        rng = np.random.default_rng(314)
        for payout in _families():
            first = _first_threshold(payout)
            values = first + rng.uniform(-100.0, 100.0, 400)
            basis = payout.basis(values)
            with self.subTest(payout=payout):
                self.assertIsInstance(basis, sp.csr_matrix)
                self.assertEqual(basis.shape, (400, len(payout.amount_names())))
                self.assertGreaterEqual(basis.min(), 0.0)
                self.assertTrue(np.all(basis.sum(axis=1) <= 1.0 + 1e-12))
                np.testing.assert_allclose(
                    basis @ payout.amounts(), payout.evaluate(values), rtol=0, atol=0
                )

    def test_monotone_families_never_pay_less_as_the_severity_rises(self) -> None:
        for payout in _families():
            values = _first_threshold(payout) + np.linspace(-300.0, 300.0, 20_001)
            severity = payout.severity(values)
            paid = payout.evaluate(values)
            in_order = paid[np.argsort(severity)]
            with self.subTest(payout=payout):
                self.assertTrue(np.all(np.diff(in_order) >= -1e-9 * paid.max()))
                below = severity < payout.severity(_first_threshold(payout))
                self.assertTrue(below.any())
                np.testing.assert_array_equal(paid[below], 0.0)

    def test_parameters_and_free_parameters(self) -> None:
        template = StepPayout([25.0, Free(30.0, 40.0, grid=3)], [Free(), 5.0])
        self.assertEqual(
            list(template.parameters()),
            ["breakpoints[0]", "breakpoints[1]", "levels[0]", "levels[1]"],
        )
        self.assertEqual(template.free_parameters(), ("breakpoints[1]", "levels[0]"))
        self.assertFalse(template.is_resolved)
        self.assertEqual(template.index_names, ("index",))
        linear = LinearPayout(30.0, 50.0, Free())
        self.assertEqual(
            list(linear.parameters()), ["attachment", "exhaustion", "limit"]
        )
        self.assertEqual(linear.free_parameters(), ("limit",))

    def test_a_template_does_not_pay(self) -> None:
        free_amounts = StepPayout([25.0, 35.0], [Free(), Free()])
        basis = free_amounts.basis([20.0, 30.0, 40.0])
        np.testing.assert_array_equal(basis.toarray(), [[0, 0], [1, 0], [0, 1]])
        free_shape = BinaryPayout(Free(0.0, 10.0, grid=3), 1.0)
        for call in (
            lambda: free_amounts.evaluate([30.0]),
            free_amounts.amounts,
            free_amounts.check_resolved,
            lambda: free_shape.basis([1.0]),
            lambda: free_shape.evaluate([1.0]),
        ):
            with self.subTest(call=call), self.assertRaises(UnresolvedParameterError):
                call()

    def test_a_basis_takes_one_unit(self) -> None:
        with self.assertRaisesRegex(ParametricInputError, "1-D"):
            StepPayout([1.0], [1.0]).basis(np.zeros((2, 2)))

    def test_with_values_resolves_and_checks_again(self) -> None:
        template = StepPayout([25.0, Free(30.0, 40.0, grid=3)], [Free(), Free(0, 50)])
        partial = template.with_values({"breakpoints[1]": 32.5})
        self.assertEqual(partial.breakpoints, (25.0, 32.5))
        self.assertEqual(template.breakpoints[1], Free(30.0, 40.0, grid=3))
        resolved = partial.with_values({"levels[0]": 10, "levels[1]": 50.0})
        self.assertTrue(resolved.is_resolved)
        np.testing.assert_array_equal(resolved.evaluate([30.0, 33.0]), [10, 50])
        linear = LinearPayout(30.0, 50.0, Free()).with_values({"limit": 7.0})
        self.assertEqual(linear.limit, 7.0)
        bad: list[Any] = [
            {"levels[2]": 1.0},
            {"breakpoints[0]": 26.0},
            {"levels[1]": 60.0},
            {"levels[0]": math.nan},
            {"levels[0]": 20.0, "levels[1]": 10.0},
            [("levels[0]", 1.0)],
        ]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(CoverDefinitionError):
                template.with_values(values)

    def test_amount_constraints(self) -> None:
        payout = StepPayout([1.0, 2.0, 3.0], [Free(0.0, 50.0), Free(5.0), 40.0])
        constraints = payout.amount_constraints(limit=30.0)
        np.testing.assert_array_equal(constraints.lower, [0.0, 5.0, 40.0])
        np.testing.assert_array_equal(constraints.upper, [30.0, 30.0, 30.0])
        self.assertFalse(constraints.bounds_feasible)
        np.testing.assert_array_equal(constraints.a_ub, [[1, -1, 0], [0, 1, -1]])
        unbounded = payout.amount_constraints()
        np.testing.assert_array_equal(unbounded.upper, [50.0, math.inf, 40.0])
        self.assertTrue(unbounded.bounds_feasible)
        linear = LinearPayout(1.0, 2.0, Free()).amount_constraints(limit=5)
        self.assertEqual((linear.lower.tolist(), linear.upper.tolist()), ([0.0], [5.0]))
        self.assertEqual(linear.a_ub.shape, (0, 1))
        curve = PiecewiseLinearPayout([1.0, 2.0], [Free(), Free()])
        self.assertEqual(curve.amount_constraints().a_ub.shape, (1, 2))
        for limit in (-1.0, math.nan):
            with self.subTest(limit=limit), self.assertRaises(CoverDefinitionError):
                payout.amount_constraints(limit=limit)


class TestDocuments(unittest.TestCase):
    """Payout documents and the registry."""

    def test_round_trip_of_every_family(self) -> None:
        templates = [
            StepPayout.uniform(25.0, 51.0, 5),
            LinearPayout(Free(20.0, 40.0, grid=5), 60.0, Free(0.0, 1e6)),
            PiecewiseLinearPayout([50.0, 80.0], [Free(), Free()], monotone=False),
            BinaryPayout(Free(1.0, 3.0, grid=[1.0, 2.0, 3.0]), 9.0, tail="lower"),
        ]
        for payout in _families() + templates:
            document = payout.to_dict()
            with self.subTest(payout=payout):
                self.assertEqual(document["kind"], payout.kind)
                self.assertEqual(payout_from_dict(document), payout)
                self.assertEqual(type(payout).from_dict(document), payout)

    def test_the_parity_format(self) -> None:
        free = {"free": {"grid": None, "lower": 0.0, "upper": None}}
        document = {
            "breakpoints": [25.0, 30.0, 36.0, 43.0, 51.0],
            "kind": "step",
            "levels": [free] * 5,
            "monotone": True,
            "tail": "upper",
        }
        payout = payout_from_dict(document)
        self.assertEqual(payout, StepPayout([25, 30, 36, 43, 51], [Free()] * 5))
        self.assertEqual(payout.to_dict(), document)
        linear = {"kind": "linear", "attachment": 50.0, "exhaustion": 150.0}
        self.assertEqual(
            payout_from_dict({**linear, "limit": 1000.0, "tail": "upper"}).to_dict(),
            {**linear, "limit": 1000.0, "tail": "upper"},
        )
        self.assertEqual(payout_from_dict({**linear, "limit": 1.0}).tail, "upper")

    def test_invalid_documents_raise(self) -> None:
        step = StepPayout([1.0], [1.0]).to_dict()
        bad: list[Any] = [
            None,
            {"breakpoints": [1.0], "levels": [1.0]},
            {**step, "kind": 3},
            {**step, "extra": 1},
            {"kind": "step", "levels": [1.0]},
            {**step, "levels": [{"free": {"low": 0.0}}]},
            {**step, "levels": [-1.0]},
        ]
        for document in bad:
            with (
                self.subTest(document=document),
                self.assertRaises(CoverDefinitionError),
            ):
                payout_from_dict(document)
        with self.assertRaises(CoverDefinitionError):
            StepPayout.from_dict({**step, "kind": "linear"})
        with self.assertRaisesRegex(UnknownPayoutKindError, "register_payout"):
            payout_from_dict({**step, "kind": "quadratic"})
        self.assertTrue(issubclass(UnknownPayoutKindError, CoverDefinitionError))


@dataclass(frozen=True)
class _QuadraticPayout(PayoutFunction):
    """A user family: the share of the way to the exhaustion, squared."""

    kind: ClassVar[str] = "test_quadratic"

    attachment: float
    exhaustion: float
    amount: Any
    tail: str = "upper"

    def evaluate(self, index: Any) -> np.ndarray:
        self.check_resolved()
        share = (self.severity(index) - self.attachment) / (
            self.exhaustion - self.attachment
        )
        return self.amount * np.clip(share, 0.0, 1.0) ** 2


@dataclass(frozen=True)
class _RampPayout(LinearInAmounts):
    """A user family linear in two amounts, with the default evaluate."""

    kind: ClassVar[str] = "test_ramp"

    low: Any
    high: Any
    tail: str = "upper"

    def amount_names(self) -> tuple[str, ...]:
        return ("low", "high")

    def basis(self, index: Any) -> sp.csr_matrix:
        severity = self.severity(index)
        return sp.csr_matrix(
            np.column_stack([severity >= 1.0, severity >= 2.0]).astype(float) / 2.0
        )


class _NotADataclass(PayoutFunction):
    """A user family that is not a dataclass and overrides nothing."""

    kind: ClassVar[str] = "test_plain"
    tail = "upper"

    def evaluate(self, index: Any) -> np.ndarray:
        return np.zeros(np.shape(index))


class TestUserDefined(unittest.TestCase):
    """User families through the protocol and the registry."""

    def setUp(self) -> None:
        patcher = mock.patch.dict("climada_petals.engine.parametric.payout._REGISTRY")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_registered_family_round_trips(self) -> None:
        self.assertIs(register_payout(_QuadraticPayout), _QuadraticPayout)
        self.assertIs(register_payout(_QuadraticPayout), _QuadraticPayout)
        template = _QuadraticPayout(10.0, 20.0, Free())
        self.assertEqual(template.free_parameters(), ("amount",))
        payout = template.with_values({"amount": 100.0})
        np.testing.assert_allclose(payout.evaluate([5.0, 15.0, 25.0]), [0, 25, 100])
        self.assertEqual(payout_from_dict(payout.to_dict()), payout)
        self.assertEqual(payout_from_dict(template.to_dict()), template)

    def test_a_linear_user_family_gets_evaluate_and_constraints(self) -> None:
        register_payout(_RampPayout)
        payout = _RampPayout(2.0, 6.0)
        np.testing.assert_allclose(payout.evaluate([0.5, 1.0, 2.0]), [0, 1, 4])
        constraints = _RampPayout(Free(), 6.0).amount_constraints(limit=5.0)
        self.assertEqual(constraints.upper.tolist(), [5.0, 5.0])
        self.assertFalse(constraints.bounds_feasible)

    def test_a_family_that_is_not_a_dataclass_must_override(self) -> None:
        plain = _NotADataclass()
        np.testing.assert_array_equal(plain.evaluate([1.0]), [0.0])
        for call in (plain.parameters, plain.to_dict):
            with (
                self.subTest(call=call),
                self.assertRaisesRegex(CoverDefinitionError, "not a dataclass"),
            ):
                call()

    def test_registration_errors(self) -> None:
        class Unnamed(_QuadraticPayout):
            kind: ClassVar[str] = ""

        class Clash(_QuadraticPayout):
            kind: ClassVar[str] = "step"

        for cls in (int, "step", LinearInAmounts, Unnamed, Clash):
            with self.subTest(cls=cls), self.assertRaises(CoverDefinitionError):
                register_payout(cls)


if __name__ == "__main__":
    unittest.main()
