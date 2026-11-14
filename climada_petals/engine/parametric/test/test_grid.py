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

Tests of the grid route: designs worked out by hand, ties, the profile, and
what the route refuses.

Most designs use one unit and ten simulation years, three of them with one
event each: year 0 has the index 30 and no loss (a payout there is a false
positive), year 1 the index 45 and the loss 6, year 2 the index 60 and the
loss 10. So ``E[L] = 1.6``.
"""

from __future__ import annotations

import dataclasses
import pickle
import unittest
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from climada_petals.engine.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CalibrationLimitError,
    CalibrationOptions,
    CoverDefinitionError,
    ExpectedUncovered,
    Free,
    InfeasibleCalibrationError,
    Limit,
    ParametricInputError,
    PayoutFunction,
    StepPayout,
    UserObjective,
    WeightedCVaR,
    calibrate,
    evaluate,
)
from climada_petals.engine.parametric.test._synthetic import (
    array_sample,
    random_portfolio,
    template,
)

BUDGET = Budget(ratio=1.0)
INDIVIDUAL = "individual"


def _sample(index: ArrayLike = (30.0, 45.0, 60.0), loss: ArrayLike = (0.0, 6.0, 10.0)):
    """Return the module docstring's sample: one unit, one event per year."""
    values = np.asarray(index, dtype=float)[:, None]
    losses = np.asarray(loss, dtype=float)[:, None]
    return array_sample(values, losses, list(range(values.shape[0])), 10)[0]


def _attachment(grid: list[float], levels: Any = None) -> StepPayout:
    """Return a step payout with a free attachment on ``grid`` and a step at 55."""
    chosen = [Free(), Free()] if levels is None else levels
    return StepPayout([Free(min(grid), max(grid), grid=grid), 55.0], chosen)


def shortfall_squared(
    loss: NDArray[np.float64], payout: NDArray[np.float64], weight: NDArray[np.float64]
) -> float:
    """Return the expected squared shortfall, ``sum_o w_o ((L_o - P_o)^+)^2``."""
    return float(weight @ np.maximum(loss - payout, 0.0) ** 2)


@dataclass(frozen=True)
class _Quadratic(PayoutFunction):
    """A payout that is not linear in its amounts: ``rate (s - trigger)^2`` above it."""

    kind: ClassVar[str] = "test_quadratic"

    trigger: Any
    rate: Any
    tail: str = "upper"

    def evaluate(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return the payout at each index value."""
        self.check_resolved()
        severity = self.severity(index)
        return self.rate * np.maximum(severity - self.trigger, 0.0) ** 2


@dataclass(frozen=True)
class _Negative(PayoutFunction):
    """A payout that returns a negative value: the grid refuses it."""

    kind: ClassVar[str] = "test_negative"

    amount: Any
    tail: str = "upper"

    def evaluate(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return ``-amount`` everywhere."""
        return np.full(np.shape(index), -self.amount)


class TestShapeGrid(unittest.TestCase):
    """Free shape parameters of a payout linear in its amounts."""

    def test_a_free_attachment_with_the_lp_at_each_candidate(self) -> None:
        # Budget 1.6. At 25 the first step also pays the false positive, so a
        # unit of it costs twice what it covers: the top level takes 10 of
        # the budget and the first 3, E[U] = 0.1 * 3. At 40 the levels (6, 10)
        # cover every loss. At 50 the first step is reached by no event, and
        # the loss 6 stays uncovered: E[U] = 0.6.
        sample = _sample()
        result = calibrate(
            template(sample, _attachment([25.0, 40.0, 50.0])),
            sample,
            ExpectedUncovered(),
            scope=INDIVIDUAL,
            budget=BUDGET,
        )
        profile = result.profile
        np.testing.assert_allclose(
            profile["objective_value"], [0.3, 0.0, 0.6], atol=1e-9
        )
        np.testing.assert_allclose(profile["premium"], [1.6, 1.6, 1.0], atol=1e-9)
        self.assertEqual(profile["breakpoints[0]"].tolist(), [25.0, 40.0, 50.0])
        self.assertEqual(profile["chosen"].tolist(), [False, True, False])
        self.assertEqual(profile["status"].tolist(), ["feasible"] * 3)
        payout = result.cover.payout["U0"]
        self.assertEqual(payout.breakpoints, (40.0, 55.0))
        np.testing.assert_allclose(payout.levels, [6.0, 10.0], atol=1e-9)
        self.assertAlmostEqual(result.objective_value, 0.0, places=9)
        self.assertEqual((result.ties, result.problems[0].n_candidates), (0, 3))
        summary = result.summary().iloc[0]
        self.assertEqual((summary["route"], summary["ties"]), ("lp", 0))
        self.assertIsNone(result.unique)  # the LP route without ranges
        self.assertEqual(
            dict(result.calibrated)["U0"], ("breakpoints[0]", "levels[0]", "levels[1]")
        )
        frame = result.parameters_frame()
        self.assertEqual(frame["calibrated"].tolist(), [True, False, True, True])

    def test_a_free_attachment_with_the_qp_at_each_candidate(self) -> None:
        # The least squares at 25 put the first level at the mean of 0 and 6:
        # MSE = 0.1 (9 + 9). At 40: 0. At 50: 0.1 * 36.
        sample = _sample()
        result = calibrate(
            template(sample, _attachment([25.0, 40.0, 50.0])),
            sample,
            MSE(),
            scope=INDIVIDUAL,
        )
        np.testing.assert_allclose(
            result.profile["objective_value"], [1.8, 0.0, 3.6], atol=1e-8
        )
        np.testing.assert_allclose(
            result.cover.payout["U0"].levels, [6.0, 10.0], atol=1e-6
        )
        self.assertEqual(result.problems[0].solver.route, "qp")
        self.assertIs(result.unique, True)  # the Hessian at 40, and no ties

    def test_ties_go_to_the_first_candidate_in_lexicographic_order(self) -> None:
        # 32 and 40 both lie between the indices 30 and 45: the same payouts.
        sample = _sample()
        result = calibrate(
            template(sample, _attachment([32.0, 40.0, 50.0])),
            sample,
            ExpectedUncovered(),
            scope=INDIVIDUAL,
            budget=BUDGET,
            options=CalibrationOptions(ranges=True),
        )
        profile = result.profile
        self.assertEqual(profile["tied"].tolist(), [True, True, False])
        self.assertEqual(profile["chosen"].tolist(), [True, False, False])
        self.assertEqual(result.cover.payout["U0"].breakpoints[0], 32.0)
        self.assertEqual(result.ties, 1)
        self.assertIs(result.unique, False)  # unique levels, but a tie
        ranges = result.ranges["U0"]
        self.assertAlmostEqual(ranges["levels[0]"][0], 6.0, delta=1e-6)
        self.assertAlmostEqual(ranges["levels[1]"][1], 10.0, delta=1e-6)
        table = evaluate(result, sample, split="in_sample")
        self.assertEqual(table["ties"].tolist(), [1, 1])
        self.assertEqual(table["unique"].tolist(), [False, False])

    def test_candidates_are_ordered_and_pruned(self) -> None:
        # Two free breakpoints: the product has 9 points, and (50, 40) is out
        # of order. The candidates follow the first breakpoint, then the
        # second.
        sample = _sample()
        payout = StepPayout(
            [Free(20.0, 50.0, grid=[20.0, 35.0, 50.0]), Free(40.0, 60.0, grid=3)],
            [Free(), Free()],
        )
        result = calibrate(
            template(sample, payout),
            sample,
            WeightedCVaR(alpha=0.5, beta=0.8),
            scope=INDIVIDUAL,
            budget=BUDGET,
        )
        pairs = result.profile[["breakpoints[0]", "breakpoints[1]"]]
        self.assertEqual(
            [tuple(row) for row in pairs.itertuples(index=False)],
            [
                (20.0, 40.0),
                (20.0, 50.0),
                (20.0, 60.0),
                (35.0, 40.0),
                (35.0, 50.0),
                (35.0, 60.0),
                (50.0, 60.0),
            ],
        )
        self.assertEqual(result.problems[0].n_candidates, 7)
        self.assertEqual(result.profile["candidate"].tolist(), list(range(7)))

    def test_infeasible_candidates_are_recorded(self) -> None:
        # Fixed levels 5 and 10: at 25 the premium is 2.0 and at 40 it is 1.5,
        # against a budget of 1.6.
        sample = _sample()
        cover = template(sample, _attachment([25.0, 40.0], levels=[5.0, 10.0]))
        for objective in (ExpectedUncovered(), MSE()):
            result = calibrate(
                cover, sample, objective, scope=INDIVIDUAL, budget=BUDGET
            )
            with self.subTest(objective=objective):
                self.assertEqual(
                    result.profile["status"].tolist(), ["infeasible", "feasible"]
                )
                self.assertTrue(np.isnan(result.profile["objective_value"].iloc[0]))
                self.assertEqual(result.cover.payout["U0"].breakpoints[0], 40.0)
                with self.assertRaisesRegex(
                    InfeasibleCalibrationError, "no candidate of the grid is feasible"
                ):
                    calibrate(
                        cover,
                        sample,
                        objective,
                        scope=INDIVIDUAL,
                        budget=Budget(ratio=0.1),
                    )

    def test_a_unit_on_a_grid_next_to_one_on_the_exact_route(self) -> None:
        sample, _ = random_portfolio(2, n_units=2)
        payouts = {
            "U0": _attachment([25.0, 35.0]),
            "U1": StepPayout.uniform(20.0, 70.0, 3),
        }
        result = calibrate(
            template(sample, payouts),
            sample,
            ExpectedUncovered(),
            scope=INDIVIDUAL,
            budget=Budget(ratio=0.5),
        )
        grid, exact = result.problems
        self.assertEqual((grid.n_candidates, exact.n_candidates), (2, None))
        self.assertIsNone(exact.ties)
        self.assertEqual(result.ties, grid.ties)
        self.assertEqual(set(result.profile["problem_id"]), {"U0"})
        # U1's design is the one it has next to a unit on the exact route.
        exact_only = calibrate(
            template(sample, {**payouts, "U0": StepPayout.uniform(20.0, 70.0, 3)}),
            sample,
            ExpectedUncovered(),
            scope=INDIVIDUAL,
            budget=Budget(ratio=0.5),
        )
        np.testing.assert_array_equal(
            result.cover.payout["U1"].levels, exact_only.cover.payout["U1"].levels
        )
        self.assertEqual(
            exact, dataclasses.replace(exact_only.problems[1], solver=exact.solver)
        )

    def test_the_amounts_need_a_budget_or_bounds(self) -> None:
        sample = _sample()
        with self.assertRaisesRegex(ParametricInputError, "needs a budget or bounded"):
            calibrate(
                template(sample, _attachment([25.0, 40.0]), limit=None),
                sample,
                ExpectedUncovered(),
                scope=INDIVIDUAL,
            )


class TestNumpyGrid(unittest.TestCase):
    """A UserObjective, or a payout that is not linear in its amounts."""

    def test_a_user_objective_worked_out_by_hand(self) -> None:
        # Binary payouts: (trigger, amount) in lexicographic order. At 35 the
        # payout reaches the losses 6 and 10, at 50 only the loss 10. The
        # amount 10 at 35 costs 2.0, above the budget of 1.6.
        sample = _sample()
        payout = BinaryPayout(
            Free(35.0, 50.0, grid=[35.0, 50.0]), Free(0.0, 10.0, grid=3)
        )
        objective = UserObjective(shortfall_squared, name="shortfall_squared")
        result = calibrate(
            template(sample, payout), sample, objective, scope=INDIVIDUAL, budget=BUDGET
        )
        profile = result.profile
        np.testing.assert_allclose(
            profile["objective_value"], [13.6, 2.6, 0.0, 13.6, 6.1, 3.6], atol=1e-12
        )
        np.testing.assert_allclose(
            profile["premium"], [0.0, 1.0, 2.0, 0.0, 0.5, 1.0], atol=1e-12
        )
        self.assertEqual(
            profile["status"].tolist(),
            ["feasible", "feasible", "above_budget"] + ["feasible"] * 3,
        )
        self.assertEqual(profile["chosen"].tolist(), [False, True] + [False] * 4)
        self.assertEqual(
            result.cover.payout["U0"].parameters(),
            {
                "trigger": 35.0,
                "amount": 5.0,
            },
        )
        self.assertAlmostEqual(result.objective_value, 2.6, places=12)
        self.assertAlmostEqual(result.premium, 1.0, places=12)
        summary = result.summary().iloc[0]
        self.assertEqual((summary["route"], summary["solver"]), ("grid", "numpy"))
        self.assertEqual(
            summary["objective"], "user(name=shortfall_squared, parameters={})"
        )
        self.assertEqual((result.ties, result.unique), (0, True))
        # A limit of 8 rules the amount 10 out at both triggers.
        capped = calibrate(
            template(sample, payout, limit=Limit(8.0)),
            sample,
            objective,
            scope=INDIVIDUAL,
            budget=BUDGET,
        )
        self.assertEqual(
            capped.profile["status"].tolist(),
            ["feasible", "feasible", "above_limit"] * 2,
        )
        copied = pickle.loads(pickle.dumps(result))
        self.assertEqual(copied.objective, objective)
        self.assertTrue(copied.profile.equals(result.profile))

    def test_a_payout_not_linear_in_its_amounts(self) -> None:
        # rate (s - trigger)^2. With the trigger at 25 any positive rate pays
        # above the limit of 10 at the index 60; at 35 with the rate 0.01 the
        # payouts are 1 and 6.25, E[U] = 0.1 (5 + 3.75).
        sample = _sample()
        payout = _Quadratic(
            Free(25.0, 45.0, grid=3), Free(0.0, 0.02, grid=[0.0, 0.01, 0.02])
        )
        result = calibrate(
            template(sample, payout),
            sample,
            ExpectedUncovered(),
            scope=INDIVIDUAL,
            budget=BUDGET,
        )
        profile = result.profile
        self.assertEqual(
            profile["status"].tolist(),
            ["feasible", "above_limit", "above_limit"]
            + ["feasible", "feasible", "above_limit"]
            + ["feasible"] * 3,
        )
        np.testing.assert_allclose(
            profile["objective_value"].iloc[[0, 4, 7, 8]],
            [1.6, 0.875, 1.375, 1.15],
            atol=1e-12,
        )
        self.assertEqual(int(np.flatnonzero(profile["chosen"])[0]), 4)
        self.assertEqual(dict(result.unidentified), {"U0": ()})
        self.assertAlmostEqual(result.premium, 0.725, places=12)

    def test_the_tie_rule_sets_an_unidentified_amount(self) -> None:
        # No index reaches the second step at 70: its level ties across its
        # grid, and the first candidate in lexicographic order sets it.
        sample = _sample()
        payout = StepPayout(
            [40.0, 70.0], [Free(0.0, 5.0, grid=2), Free(5.0, 10.0, grid=2)]
        )
        result = calibrate(
            template(sample, payout),
            sample,
            UserObjective(shortfall_squared, name="shortfall_squared"),
            scope=INDIVIDUAL,
        )
        self.assertEqual(result.cover.payout["U0"].levels, (5.0, 5.0))
        self.assertEqual(dict(result.unidentified), {"U0": ("levels[1]",)})
        self.assertEqual((result.ties, result.unique), (1, False))
        self.assertAlmostEqual(result.objective_value, 2.6, places=12)

    def test_invalid_payouts_and_grids_raise(self) -> None:
        sample = _sample()
        objective = UserObjective(shortfall_squared, name="shortfall_squared")
        cases = [
            (BinaryPayout(40.0, Free(0.0, 10.0)), ParametricInputError, "none for"),
            (_Negative(Free(1.0, 2.0, grid=2)), CoverDefinitionError, "finite payouts"),
        ]
        for payout, error, match in cases:
            with self.subTest(match=match), self.assertRaisesRegex(error, match):
                calibrate(
                    template(sample, payout, limit=None),
                    sample,
                    objective,
                    scope=INDIVIDUAL,
                )


class TestRefusals(unittest.TestCase):
    """What the grid route refuses, before anything is solved."""

    def setUp(self) -> None:
        self.sample = _sample()
        self.objective = UserObjective(shortfall_squared, name="shortfall_squared")

    def test_the_joint_scope_raises(self) -> None:
        cases = [
            (_attachment([25.0, 40.0]), ExpectedUncovered(), "thresholds are fixed"),
            (StepPayout([40.0], [Free(0.0, 5.0, grid=2)]), self.objective, "is a Us"),
            (_Quadratic(30.0, Free(0.0, 1.0, grid=2)), MSE(), "not linear in its"),
        ]
        for payout, objective, match in cases:
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(ParametricInputError, match),
            ):
                calibrate(
                    template(self.sample, payout),
                    self.sample,
                    objective,
                    scope="joint",
                    budget=BUDGET,
                )

    def test_more_candidates_than_max_grid_points_raise(self) -> None:
        options = CalibrationOptions(max_grid_points=2)
        with self.assertRaisesRegex(
            CalibrationLimitError, "max_grid_points=2"
        ) as found:
            calibrate(
                template(self.sample, _attachment([25.0, 40.0, 50.0])),
                self.sample,
                ExpectedUncovered(),
                scope=INDIVIDUAL,
                budget=BUDGET,
                options=options,
            )
        self.assertEqual(found.exception.status, "max_grid_points")
        # At the limit, the design goes ahead.
        calibrate(
            template(self.sample, _attachment([25.0, 40.0])),
            self.sample,
            ExpectedUncovered(),
            scope=INDIVIDUAL,
            budget=BUDGET,
            options=options,
        )

    def test_a_grid_with_no_valid_payout_raises(self) -> None:
        payout = StepPayout([Free(50.0, 60.0, grid=2), 45.0], [Free(), Free()])
        with self.assertRaisesRegex(ParametricInputError, "accepts no point"):
            calibrate(
                template(self.sample, payout),
                self.sample,
                ExpectedUncovered(),
                scope=INDIVIDUAL,
                budget=BUDGET,
            )

    def test_a_grid_that_may_pay_at_zero_on_left_out_events_raises(self) -> None:
        # The fourth event has no loss and an index of 0: the sample leaves it out.
        sample = _sample((30.0, 45.0, 60.0, 0.0), (0.0, 6.0, 10.0, 0.0))
        self.assertEqual(sample.n_left_out, 1)
        cases = [
            (_attachment([0.0, 40.0]), ExpectedUncovered()),
            (
                BinaryPayout(Free(0.0, 40.0, grid=2), Free(0.0, 5.0, grid=2)),
                self.objective,
            ),
        ]
        for payout, objective in cases:
            with (
                self.subTest(payout=payout.kind),
                self.assertRaisesRegex(ParametricInputError, "left out 1 events"),
            ):
                calibrate(
                    template(sample, payout),
                    sample,
                    objective,
                    scope=INDIVIDUAL,
                    budget=BUDGET,
                )
        # Grids above an index of 0 go ahead.
        result = calibrate(
            template(sample, _attachment([25.0, 40.0])),
            sample,
            ExpectedUncovered(),
            scope=INDIVIDUAL,
            budget=BUDGET,
        )
        self.assertEqual(result.cover.payout["U0"].breakpoints[0], 40.0)


if __name__ == "__main__":
    unittest.main()
