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

Tests of the budget frontier: each row is the calibration at its budget.
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from climada_petals.engine.parametric import (
    MSE,
    Budget,
    CalibrationResult,
    CostOfCapitalLoad,
    ExpectedUncovered,
    Free,
    InfeasibleCalibrationError,
    ParametricInputError,
    StepPayout,
    WeightedCVaR,
    calibrate,
    calibrate_frontier,
)
from climada_petals.engine.parametric.test._synthetic import random_portfolio, template

PAYOUT = StepPayout.uniform(attachment=20.0, exhaustion=70.0, n_steps=3)


def _expected_rows(result: CalibrationResult, budget: Budget) -> pd.DataFrame:
    """Return the rows a frontier holds for one calibration, built by hand."""
    rows = []
    for problem in result.problems:
        row: dict[str, Any] = {}
        for unit in problem.unit_id:
            for name, value in result.cover.payout[unit].parameters().items():
                row[f"{unit}:{name}"] = value
        rows.append(row)
    budgets = pd.DataFrame(
        {
            "budget_ratio": [np.nan if budget.ratio is None else budget.ratio]
            * len(rows),
            "budget_amount": [
                (
                    budget.amount[problem.unit_id[0]]
                    if isinstance(budget.amount, Mapping)
                    else np.nan if budget.amount is None else budget.amount
                )
                for problem in result.problems
            ],
        }
    )
    return pd.concat([budgets, result.summary(), pd.DataFrame(rows)], axis=1)


class TestFrontier(unittest.TestCase):
    """Rows, their order and their columns."""

    def setUp(self) -> None:
        self.sample, _ = random_portfolio(7)
        self.template = template(self.sample, PAYOUT)

    def _compare(self, frontier: pd.DataFrame, expected: list[pd.DataFrame]) -> None:
        """Compare a frontier with hand-built rows, but for the solve times."""
        pd.testing.assert_frame_equal(
            frontier.drop(columns="seconds"),
            pd.concat(expected, ignore_index=True).drop(columns="seconds"),
            check_dtype=False,
        )

    def test_rows_equal_separate_calibrations(self) -> None:
        cases: list[tuple[Any, str, Any]] = [
            (ExpectedUncovered(), "individual", None),
            (WeightedCVaR(alpha=0.9, beta=0.9), "joint", None),
            (MSE(), "joint", CostOfCapitalLoad(rate=0.1, q=0.9)),
        ]
        budgets = [Budget(ratio=ratio) for ratio in (0.3, 0.6, 1.0)]
        for objective, scope, pricing in cases:
            extra = {} if pricing is None else {"pricing": pricing}
            frontier = calibrate_frontier(
                self.template,
                self.sample,
                objective,
                budgets=budgets,
                scope=scope,
                **extra,
            )
            expected = [
                _expected_rows(
                    calibrate(
                        self.template,
                        self.sample,
                        objective,
                        scope=scope,
                        budget=budget,
                        **extra,
                    ),
                    budget,
                )
                for budget in budgets
            ]
            with self.subTest(objective=objective, scope=scope):
                self._compare(frontier, expected)
                self.assertEqual(len(frontier), 3 * (3 if scope == "individual" else 1))

    def test_rows_follow_the_budgets_and_dont_depend_on_their_order(self) -> None:
        budgets = [Budget(ratio=1.0), Budget(ratio=0.2), Budget(ratio=0.6)]
        common = {"budgets": budgets, "scope": "joint"}
        frontier = calibrate_frontier(
            self.template, self.sample, ExpectedUncovered(), **common
        )
        self.assertEqual(frontier["budget_ratio"].tolist(), [1.0, 0.2, 0.6])
        common["budgets"] = budgets[::-1]
        reverse = calibrate_frontier(
            self.template, self.sample, ExpectedUncovered(), **common
        )
        pd.testing.assert_frame_equal(
            frontier.drop(columns="seconds"),
            reverse.iloc[::-1].reset_index(drop=True).drop(columns="seconds"),
        )
        # A measurement: no column ranks the budgets or names a best one.
        for word in ("best", "rank", "optimal", "knee", "marginal"):
            self.assertFalse(any(word in column for column in frontier.columns))
        # On the design sample, more premium never leaves more uncovered.
        values = frontier.sort_values("budget_ratio")["objective_value"]
        self.assertTrue(np.all(np.diff(values.to_numpy()) <= 1e-9))

    def test_budgets_by_unit_and_a_grid(self) -> None:
        units = self.sample.unit_id
        budgets = [
            Budget(amount={unit: 50.0 * (k + 1) for k, unit in enumerate(units)}),
            Budget(amount={unit: 100.0 * (k + 1) for k, unit in enumerate(units)}),
        ]
        payout = StepPayout([Free(20.0, 30.0, grid=2), 50.0], [Free(), Free()])
        cover = template(self.sample, payout)
        frontier = calibrate_frontier(
            cover, self.sample, ExpectedUncovered(), budgets=budgets, scope="individual"
        )
        self.assertEqual(
            frontier["budget_amount"].tolist(), [50, 100, 150, 100, 200, 300]
        )
        self.assertTrue(frontier["budget_ratio"].isna().all())
        self.assertEqual(frontier["n_candidates"].tolist(), [2] * 6)
        first = frontier[frontier["problem_id"] == "U0"]
        self.assertTrue(first["U1:levels[0]"].isna().all())
        self.assertTrue(first["U0:breakpoints[0]"].isin([20.0, 30.0]).all())
        expected = [
            _expected_rows(
                calibrate(
                    cover,
                    self.sample,
                    ExpectedUncovered(),
                    scope="individual",
                    budget=budget,
                ),
                budget,
            )
            for budget in budgets
        ]
        self._compare(frontier, expected)

    def test_invalid_budgets_raise(self) -> None:
        cases = {
            "a sequence of Budget": Budget(ratio=0.5),
            "sequence of Budget,": "0.5",
            "non-empty sequence": [],
            "of Budget, got": [Budget(ratio=0.5), 0.5],
            "distinct": [Budget(ratio=0.5), Budget(ratio=0.2), Budget(ratio=0.5)],
        }
        for match, budgets in cases.items():
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                calibrate_frontier(
                    self.template,
                    self.sample,
                    ExpectedUncovered(),
                    budgets=budgets,
                    scope="joint",
                )

    def test_a_budget_the_template_cant_meet_raises_and_names_it(self) -> None:
        fixed = template(self.sample, StepPayout([20.0, 50.0], [10.0, Free()]))
        with self.assertRaises(InfeasibleCalibrationError) as found:
            calibrate_frontier(
                fixed,
                self.sample,
                ExpectedUncovered(),
                budgets=[Budget(ratio=0.5), Budget(ratio=0.0)],
                scope="joint",
            )
        self.assertIn("Budget(ratio=0.0", " ".join(found.exception.__notes__))


if __name__ == "__main__":
    unittest.main()
