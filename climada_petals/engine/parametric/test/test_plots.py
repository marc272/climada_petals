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

Tests of the plots: each returns its axes and never shows them."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import ClassVar
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from climada_petals.engine.parametric import (
    BinaryPayout,
    EventBasis,
    IndexDefinition,
    LinearPayout,
    OutcomeSample,
    ParametricCover,
    ParametricInputError,
    PayoutFunction,
    StepPayout,
    UnresolvedParameterError,
    evaluate,
    payout_impact,
    plot_exceedance,
    plot_payout,
    plot_payout_vs_loss,
    uncovered_impact,
)
from climada_petals.engine.parametric.test._synthetic import (
    TABLE_WIND,
    random_portfolio,
)

LOWER = IndexDefinition("table", unit="mm", tail="lower", source="synthetic")


@dataclass(frozen=True)
class _Constant(PayoutFunction):
    """A payout with no parameter: 1 at every index value."""

    kind: ClassVar[str] = "constant_for_tests"

    tail: str = "upper"

    def evaluate(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return 1 at every index value."""
        return np.ones_like(np.asarray(index, dtype=np.float64))


class _PlotTest(unittest.TestCase):
    """Closes the figures, and fails if a plot calls ``plt.show()``."""

    def setUp(self) -> None:
        plt.switch_backend("Agg")
        patcher = mock.patch.object(plt, "show", side_effect=AssertionError("shown"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(plt.close, "all")


class TestPlotPayout(_PlotTest):
    """The payout function against the index."""

    def setUp(self) -> None:
        super().setUp()
        self.cover = ParametricCover(
            TABLE_WIND,
            {
                "a": StepPayout([25.0, 35.0, 50.0], [10.0, 40.0, 100.0]),
                "b": LinearPayout(30.0, 60.0, 80.0),
            },
            unit_id=["a", "b"],
            payout_unit="USD",
        )

    def test_every_unit_or_one(self) -> None:
        ax = plot_payout(self.cover)
        self.assertEqual(len(ax.get_lines()), 2)
        self.assertEqual(ax.get_xlabel(), "Index (m/s)")
        self.assertEqual(ax.get_ylabel(), "Payout (USD)")
        # The range runs a quarter of the thresholds' span beyond them.
        x_values = ax.get_lines()[0].get_xdata()
        self.assertAlmostEqual(float(x_values.min()), 25.0 - 0.25 * 35.0)
        self.assertAlmostEqual(float(x_values.max()), 60.0 + 0.25 * 35.0)
        # A step is drawn as a jump at its breakpoint.
        y_values = ax.get_lines()[0].get_ydata()
        at = np.flatnonzero(x_values == 35.0)[0]
        self.assertEqual((y_values[at - 1], y_values[at]), (10.0, 40.0))
        _, given = plt.subplots()
        self.assertIs(plot_payout(self.cover, "b", given, index_range=(0, 100)), given)
        self.assertEqual(len(given.get_lines()), 1)

    def test_a_lower_tail_and_one_threshold(self) -> None:
        cover = ParametricCover(
            LOWER, BinaryPayout(-2.0, 5.0, tail="lower"), unit_id=[1], payout_unit="USD"
        )
        ax = plot_payout(cover)
        line = ax.get_lines()[0]
        self.assertAlmostEqual(float(line.get_xdata().min()), -2.5)
        self.assertEqual(float(line.get_ydata()[0]), 5.0)
        self.assertEqual(float(line.get_ydata()[-1]), 0.0)

    def test_invalid_arguments_raise(self) -> None:
        template = ParametricCover(
            TABLE_WIND,
            StepPayout.uniform(20.0, 70.0, 4),
            unit_id=["a"],
            payout_unit="USD",
        )
        constant = ParametricCover(
            TABLE_WIND, _Constant(), unit_id=["a"], payout_unit="USD"
        )
        with self.assertRaises(UnresolvedParameterError):
            plot_payout(template)
        cases = [
            ((self.cover.payout["a"],), {}, "ParametricCover"),
            ((self.cover, "c"), {}, "no unit"),
            ((self.cover,), {"index_range": (5.0,)}, "two numbers"),
            ((self.cover,), {"index_range": (5.0, 5.0)}, "low < high"),
            ((constant,), {}, "index_range"),
        ]
        for args, kwargs, match in cases:
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                plot_payout(*args, **kwargs)
        ax = plot_payout(constant, index_range=(0.0, 1.0))
        self.assertTrue(np.all(ax.get_lines()[0].get_ydata() == 1.0))


class TestPlotPayoutVsLoss(_PlotTest):
    """Each event's payout against its loss."""

    def setUp(self) -> None:
        super().setUp()
        self.sample, _ = random_portfolio(4)
        self.cover = ParametricCover(
            TABLE_WIND,
            StepPayout([30.0, 50.0], [100.0, 900.0]),
            unit_id=self.sample.unit_id,
            payout_unit="USD",
        )

    def test_the_total_or_a_unit(self) -> None:
        ax = plot_payout_vs_loss(self.cover, self.sample)
        points = ax.collections[0].get_offsets()
        loss = self.sample.loss.toarray()
        paid = self.cover.payout_matrix(self.sample.index)
        np.testing.assert_allclose(points[:, 0], loss.sum(axis=1))
        np.testing.assert_allclose(points[:, 1], paid.sum(axis=1))
        self.assertEqual(ax.get_xlabel(), "Loss (USD)")
        ax = plot_payout_vs_loss(self.cover, self.sample, unit_id="U1")
        np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], paid[:, 1])
        with self.assertRaisesRegex(ParametricInputError, "no unit"):
            plot_payout_vs_loss(self.cover, self.sample, unit_id="U9")


class TestPlotExceedance(_PlotTest):
    """Exceedance curves from Impacts and from tables."""

    def setUp(self) -> None:
        super().setUp()
        self.sample, self.impact = random_portfolio(4)
        self.cover = ParametricCover(
            TABLE_WIND,
            StepPayout([30.0, 50.0], [100.0, 900.0]),
            unit_id=self.sample.unit_id,
            payout_unit="USD",
        )

    def test_impacts(self) -> None:
        paid = payout_impact(self.cover, self.sample)
        ax = plot_exceedance(paid)
        curve = paid.calc_freq_curve()
        np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), curve.return_per)
        self.assertEqual(ax.get_xscale(), "log")
        ax = plot_exceedance(
            {
                "loss": self.impact,
                "payout": paid,
                "uncovered": uncovered_impact(self.cover, self.sample),
            }
        )
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertEqual(labels, ["loss", "payout", "uncovered"])

    def test_a_table(self) -> None:
        table = evaluate(self.cover, self.sample, split="in_sample")
        ax = plot_exceedance(table, quantities=("loss", "net_loss"))
        # Four rows, two quantities each.
        self.assertEqual(len(ax.get_lines()), 8)
        first = ax.get_lines()[0]
        np.testing.assert_allclose(first.get_xdata(), [2, 5, 10, 25, 50, 100, 250])
        self.assertEqual(
            first.get_label(), f"{table['cover_label'].iloc[0]} unit U0 loss"
        )
        # On the event basis the net loss is null, and left out.
        on_events = OutcomeSample(
            self.sample.index,
            self.sample.loss,
            self.sample.frequency,
            self.sample.units,
            EventBasis(),
        )
        table = evaluate(self.cover, on_events, split="in_sample")
        ax = plot_exceedance(table.iloc[[-1]], quantities=("loss", "net_loss"))
        self.assertEqual(len(ax.get_lines()), 1)

    def test_invalid_sources_raise(self) -> None:
        table = evaluate(self.cover, self.sample, split="in_sample")
        with self.assertRaisesRegex(ParametricInputError, "no return-period column"):
            plot_exceedance(table[["cover_label", "expected_loss"]])
        for source in ({"a": 1.0}, [self.impact], "loss"):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ParametricInputError, "source must be"),
            ):
                plot_exceedance(source)


if __name__ == "__main__":
    unittest.main()
