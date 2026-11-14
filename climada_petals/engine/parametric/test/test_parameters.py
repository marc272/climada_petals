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

Tests of the parameters: free parameters, their documents and the checks."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from climada_petals.engine.parametric import (
    AmountConstraints,
    CoverDefinitionError,
    Free,
)
from climada_petals.engine.parametric.parameters import (
    amount_parameter,
    check_flag,
    check_lengths,
    check_monotone,
    check_order,
    constructor_fields,
    field_from_document,
    field_to_document,
    is_parameter,
    non_negative,
    parameter_tuple,
    real_number,
    required_field,
    shape_parameter,
)


class TestFree(unittest.TestCase):
    """Bounds, grids and the document of a free parameter."""

    def test_defaults_are_a_free_amount(self) -> None:
        free = Free()
        self.assertEqual((free.lower, free.upper, free.grid), (0.0, None, None))
        self.assertEqual(free, Free(0, None, grid=None))
        self.assertEqual(hash(free), hash(Free()))

    def test_grids_are_stored_as_explicit_values(self) -> None:
        self.assertEqual(Free(25.0, 35.0, grid=3).grid, (25.0, 30.0, 35.0))
        self.assertEqual(Free(25.0, 35.0, grid=np.int64(2)).grid, (25.0, 35.0))
        self.assertEqual(Free(25, 35, grid=[25, 31.5]).grid, (25.0, 31.5))
        self.assertEqual(Free(0.0, grid=np.array([1.0, 1e6])).grid, (1.0, 1e6))
        self.assertIsInstance(Free(np.float32(1.0), 2).lower, float)
        self.assertEqual(Free(1.0, 1.0).upper, 1.0)

    def test_invalid_bounds_and_grids_raise(self) -> None:
        bad = [
            {"lower": math.nan},
            {"lower": True},
            {"lower": "0"},
            {"lower": 0.0, "upper": math.inf},
            {"lower": 2.0, "upper": 1.0},
            {"lower": 0.0, "upper": None, "grid": 3},
            {"lower": 0.0, "upper": 1.0, "grid": 1},
            {"lower": 0.0, "upper": 1.0, "grid": True},
            {"lower": 0.0, "upper": 1.0, "grid": []},
            {"lower": 0.0, "upper": 1.0, "grid": [0.5, 0.5]},
            {"lower": 0.0, "upper": 1.0, "grid": [0.5, 0.2]},
            {"lower": 0.0, "upper": 1.0, "grid": [-0.5, 0.5]},
            {"lower": 0.0, "upper": 1.0, "grid": [0.5, 1.5]},
            {"lower": 0.0, "upper": 1.0, "grid": [[0.5]]},
            {"lower": 0.0, "upper": 1.0, "grid": [0.5, math.nan]},
            {"lower": 0.0, "upper": 1.0, "grid": ["a"]},
        ]
        for kwargs in bad:
            grid = kwargs.pop("grid", None)
            with (
                self.subTest(kwargs=kwargs, grid=grid),
                self.assertRaises(CoverDefinitionError),
            ):
                Free(**kwargs, grid=grid)

    def test_check_takes_values_within_the_bounds(self) -> None:
        free = Free(10.0, 20.0, grid=[10.0, 20.0])
        self.assertEqual(free.check("x", 15), 15.0)  # need not be on the grid
        self.assertEqual(free.check("x", 20.0), 20.0)
        self.assertEqual(Free().check("x", 1e12), 1e12)
        for value in (9.9, 20.1, math.nan, "15", None):
            with self.subTest(value=value), self.assertRaises(CoverDefinitionError):
                free.check("x", value)

    def test_document_round_trip(self) -> None:
        for free in (Free(), Free(5.0), Free(25.0, 35.0, grid=3)):
            with self.subTest(free=free):
                document = free.to_dict()
                self.assertEqual(set(document), {"free"})
                self.assertEqual(Free.from_dict(document), free)
        self.assertEqual(
            Free(25.0, 35.0, grid=3).to_dict(),
            {"free": {"grid": [25.0, 30.0, 35.0], "lower": 25.0, "upper": 35.0}},
        )
        self.assertEqual(Free.from_dict({"free": {}}), Free())

    def test_invalid_documents_raise(self) -> None:
        bad = [
            None,
            {"free": None},
            {"free": {"low": 0.0}},
            {"free": {}, "other": 1},
            {"fixed": 1.0},
            {"free": {"lower": 1.0, "upper": 0.0}},
        ]
        for document in bad:
            with (
                self.subTest(document=document),
                self.assertRaises(CoverDefinitionError),
            ):
                Free.from_dict(document)


class TestChecks(unittest.TestCase):
    """The checks the payout families share."""

    def test_numbers(self) -> None:
        self.assertEqual(real_number("x", np.int64(3)), 3.0)
        self.assertEqual(non_negative("x", 0), 0.0)
        for value in (True, np.bool_(True), "1", None, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(CoverDefinitionError):
                real_number("x", value)
        with self.assertRaisesRegex(CoverDefinitionError, ">= 0"):
            non_negative("x", -1e-12)
        self.assertTrue(check_flag("flag", np.bool_(False)) is False)
        with self.assertRaisesRegex(CoverDefinitionError, "bool"):
            check_flag("flag", 1)

    def test_shape_parameters_and_amounts(self) -> None:
        self.assertEqual(shape_parameter("t", 5), 5.0)
        grid = Free(1.0, 2.0, grid=2)
        self.assertIs(shape_parameter("t", grid), grid)
        for free in (Free(1.0), Free(1.0, 2.0)):
            with (
                self.subTest(free=free),
                self.assertRaisesRegex(CoverDefinitionError, "grid"),
            ):
                shape_parameter("t", free)
        self.assertEqual(amount_parameter("a", 2), 2.0)
        free_amount = Free()
        self.assertIs(amount_parameter("a", free_amount), free_amount)
        with self.assertRaisesRegex(CoverDefinitionError, "lower >= 0"):
            amount_parameter("a", Free(-1.0))
        with self.assertRaisesRegex(CoverDefinitionError, ">= 0"):
            amount_parameter("a", -1.0)

    def test_sequences_lengths_order_and_monotonicity(self) -> None:
        self.assertEqual(
            parameter_tuple("b", np.array([1, 2]), shape_parameter, 1), (1.0, 2.0)
        )
        for values in ("12", {"a": 1}, 3, [], [[1.0]]):
            with self.subTest(values=values), self.assertRaises(CoverDefinitionError):
                parameter_tuple("b", values, shape_parameter, 1)
        with self.assertRaisesRegex(CoverDefinitionError, "one length"):
            check_lengths(("b", (1.0,)), ("l", (1.0, 2.0)))
        free = Free(0.0, 10.0, grid=2)
        check_order("b", (1.0, free, 3.0), "upper")
        check_order("b", (3.0, free, 1.0), "lower")
        for values, tail in (((1.0, 1.0), "upper"), ((1.0, 2.0), "lower")):
            with (
                self.subTest(values=values, tail=tail),
                self.assertRaisesRegex(CoverDefinitionError, "strictly"),
            ):
                check_order("b", values, tail)
        check_monotone("l", (1.0, Free(), 1.0, 2.0))
        with self.assertRaisesRegex(CoverDefinitionError, "non-decreasing"):
            check_monotone("l", (2.0, Free(), 1.0))


class TestDocuments(unittest.TestCase):
    """The document helpers of dataclass payouts."""

    def test_fields_and_values(self) -> None:
        @dataclass(frozen=True)
        class Example:
            first: float
            second: tuple[float, ...] = ()

        names = [item.name for item in constructor_fields(Example)]
        self.assertEqual(names, ["first", "second"])
        self.assertEqual(
            [required_field(item) for item in constructor_fields(Example(1.0))],
            [True, False],
        )
        with self.assertRaisesRegex(CoverDefinitionError, "not a dataclass"):
            constructor_fields(object())
        self.assertTrue(is_parameter(np.float32(1.0)))
        self.assertTrue(is_parameter(Free()))
        self.assertFalse(is_parameter(True))
        self.assertFalse(is_parameter("1"))

    def test_field_documents(self) -> None:
        value = (1, np.int64(2), np.float64(2.5), Free(), True, np.bool_(False))
        document = field_to_document(value, "x")
        self.assertEqual(document, [1, 2, 2.5, Free().to_dict(), True, False])
        self.assertIsInstance(document[1], int)
        self.assertEqual(field_to_document(None, "x"), None)
        self.assertEqual(field_to_document("upper", "x"), "upper")
        with self.assertRaisesRegex(CoverDefinitionError, "override to_dict"):
            field_to_document({1, 2}, "x")
        self.assertEqual(field_from_document([1.0, Free().to_dict()]), (1.0, Free()))
        self.assertEqual(field_from_document("upper"), "upper")


class TestAmountConstraints(unittest.TestCase):
    """Bounds and rows on the amounts."""

    def test_rows_default_to_none_and_are_read_only(self) -> None:
        constraints = AmountConstraints([0.0, 1.0], [math.inf, 1.0])
        self.assertEqual(constraints.a_ub.shape, (0, 2))
        self.assertEqual(constraints.b_ub.shape, (0,))
        self.assertTrue(constraints.bounds_feasible)
        self.assertFalse(constraints.upper.flags.writeable)
        self.assertFalse(AmountConstraints([2.0], [1.0]).bounds_feasible)

    def test_monotone_rows(self) -> None:
        rows = AmountConstraints([0.0] * 3, [5.0] * 3).with_monotone_rows()
        np.testing.assert_array_equal(rows.a_ub, [[1, -1, 0], [0, 1, -1]])
        np.testing.assert_array_equal(rows.b_ub, [0.0, 0.0])
        single = AmountConstraints([0.0], [5.0]).with_monotone_rows()
        self.assertEqual(single.a_ub.shape, (0, 1))

    def test_invalid_constraints_raise(self) -> None:
        bad = [
            {"lower": [0.0, 0.0], "upper": [1.0]},
            {"lower": [[0.0]], "upper": [[1.0]]},
            {"lower": [0.0], "upper": [1.0], "a_ub": [[1.0, 1.0]], "b_ub": [0.0]},
            {"lower": [0.0], "upper": [1.0], "a_ub": [[1.0]], "b_ub": [[0.0]]},
            {"lower": [-math.inf], "upper": [1.0]},
            {"lower": [0.0], "upper": [math.nan]},
            {"lower": [0.0], "upper": [1.0], "a_ub": [[math.nan]], "b_ub": [0.0]},
            {"lower": [0.0], "upper": [1.0], "a_ub": [[1.0]], "b_ub": [math.inf]},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(CoverDefinitionError):
                AmountConstraints(**kwargs)


if __name__ == "__main__":
    unittest.main()
