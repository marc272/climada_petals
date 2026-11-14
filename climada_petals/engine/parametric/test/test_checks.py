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

Tests of the shared argument checks.
"""

from __future__ import annotations

import unittest

import numpy as np

from climada_petals.engine.parametric import ParametricInputError
from climada_petals.engine.parametric._checks import (
    check_int,
    check_level,
    check_real,
    finite_array,
    integer_array,
    read_only,
)


class TestScalarChecks(unittest.TestCase):
    """Real numbers, integers and levels."""

    def test_check_real(self) -> None:
        self.assertEqual(check_real("x", np.float32(0.5), low=0.0, high=1.0), 0.5)
        for value in ("1", True, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ParametricInputError, "must be a real number"),
            ):
                check_real("x", value)
        with self.assertRaisesRegex(ParametricInputError, r"in \(0.0, 1.0\]"):
            check_real("x", 0.0, low=0.0, high=1.0, include_low=False)
        with self.assertRaisesRegex(ParametricInputError, "finite"):
            check_real("x", float("inf"))

    def test_check_int_and_level(self) -> None:
        self.assertEqual(check_int("n", np.int64(3), low=1), 3)
        with self.assertRaisesRegex(ParametricInputError, "must be >= 1"):
            check_int("n", 0, low=1)
        self.assertEqual(check_level("beta", 0.9), 0.9)
        for value in (0.0, 1.0):
            with self.subTest(value=value), self.assertRaises(ParametricInputError):
                check_level("beta", value)


class TestArrayChecks(unittest.TestCase):
    """Finite float arrays, whole-number arrays and read-only copies."""

    def test_finite_array(self) -> None:
        arr = finite_array([[1, 2]], "values", 2)
        self.assertEqual(arr.dtype, np.float64)
        with self.assertRaisesRegex(ParametricInputError, "must be numeric"):
            finite_array(["a"], "values", 1)
        with self.assertRaisesRegex(ParametricInputError, "must be 1-D"):
            finite_array([[1.0]], "values", 1)
        with self.assertRaisesRegex(ParametricInputError, "must be finite"):
            finite_array([np.nan], "values", 1)

    def test_integer_array(self) -> None:
        np.testing.assert_array_equal(integer_array([1.0, 2.0], "years"), [1, 2])
        self.assertEqual(integer_array([1.0], "years").dtype, np.int64)
        self.assertEqual(integer_array([], "years").size, 0)
        with self.assertRaisesRegex(ParametricInputError, "must be 1-D"):
            integer_array([[1]], "years")
        for values in ([1.5], [np.nan], ["a"]):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ParametricInputError, "whole numbers"),
            ):
                integer_array(values, "years")

    def test_read_only_copies(self) -> None:
        source = np.arange(3.0)
        copy = read_only(source)
        self.assertFalse(copy.flags.writeable)
        source[0] = 9.0
        self.assertEqual(copy[0], 0.0)


if __name__ == "__main__":
    unittest.main()
