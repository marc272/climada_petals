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

Tests of parent storms: the rule that reads them, the parts and mixtures
that keep them, the checks that compare them across hazard sets, and
``ScenarioSet.holdout``, which cuts the members that share them alike.

The hazard sets are ensembles named as CLIMADA names perturbed tracks: the
parent ``sid`` and its members ``sid_gen1``, ``sid_gen2``, ..., with the
parent as block. A "future" set holds the same parents with other index
values and losses, so its source token differs from the "present" one's.
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Any

import numpy as np

from climada_petals.engine.parametric import (
    ENSEMBLE_PARENT,
    Budget,
    CatalogueYears,
    EventBasis,
    ExpectedUncovered,
    LeakageError,
    OutcomeSample,
    ParametricIndex,
    ParametricInputError,
    ScenarioError,
    ScenarioSet,
    StepPayout,
    YearSets,
    calibrate,
    evaluate,
    evaluate_scenarios,
    holdout,
    labels_from_names,
)
from climada_petals.engine.parametric._parts import parent_labels, parents_in
from climada_petals.engine.parametric.sample import (
    check_no_leakage,
    check_out_of_sample,
)
from climada_petals.engine.parametric.test._synthetic import array_index, template

#: Ten parent storms, named as IBTrACS names them.
SIDS = tuple(f"{1980 + p}{150 + p}N{10 + p}{300 + p}" for p in range(10))

#: Members per parent: the parent and three perturbed tracks.
MEMBERS = 4


def _names(sids: Sequence[str], members: int = MEMBERS) -> list[str]:
    """Return the event names of an ensemble: ``sid``, then ``sid_gen<m>``."""
    return [sid if m == 0 else f"{sid}_gen{m}" for sid in sids for m in range(members)]


def _storms(
    seed: int,
    *,
    sids: Sequence[str] = SIDS,
    basis: str = "event",
    offset: int = 0,
    block: Any = "parents",
) -> OutcomeSample:
    """Return a sample of an ensemble on two units, its blocks the parents.

    ``seed`` draws the index and the losses, so two seeds are two hazard
    sets with other tokens. ``basis`` is ``"event"``, ``"yearsets"`` (400
    years sampled from the frequencies) or ``"catalogue"`` (one year per
    event, the members of a parent next to each other). ``offset`` moves
    the event ids; ``block`` is ``"parents"``, the labels to take, or None
    for the basis' own blocks.
    """
    rng = np.random.default_rng(seed)
    names = _names(sids)
    values = rng.uniform(30.0, 70.0, size=(len(names), 2))
    loss = (values - 25.0) ** 2 * rng.lognormal(0.0, 0.3, size=values.shape)
    index, impact, units = array_index(values, loss, frequency=1.0 / len(names))
    impact.event_id = impact.event_id + offset
    impact.event_name = names
    index = ParametricIndex(
        index.values,
        impact.event_id,
        index.unit_id,
        index.definition,
        event_name=names,
    )
    labels = (
        labels_from_names(names, ENSEMBLE_PARENT)["block"]
        if isinstance(block, str)
        else block
    )
    outcomes: Any
    if basis == "event":
        outcomes = (
            EventBasis()
            if labels is None
            else EventBasis(event_id=impact.event_id, block=labels)
        )
    elif basis == "yearsets":
        outcomes = YearSets.from_frequencies(impact, n_years=400, rng=rng, block=labels)
    else:
        outcomes = CatalogueYears.from_labels(
            impact.event_id,
            np.arange(len(names)),
            n_years=len(names),
            block=labels,
        )
    return OutcomeSample.from_impact(index, impact, units=units, basis=outcomes)


def _design(sample: OutcomeSample) -> Any:
    """Return an E[U] design of a three-step cover on ``sample``."""
    return calibrate(
        template(sample, StepPayout.uniform(30.0, 70.0, 3)),
        sample,
        ExpectedUncovered(),
        scope="joint",
        budget=Budget(ratio=1.0),
    )


class TestTheRule(unittest.TestCase):
    """Which blocks are parent storms, and which are local labels."""

    def test_the_parents_of_perturbed_tracks(self) -> None:
        sample = _storms(1)
        expected = dict(
            zip(sample.event_id.tolist(), [sid for sid in SIDS for _ in range(4)])
        )
        self.assertEqual(dict(sample.parents), expected)
        self.assertEqual(parents_in(sample), set(SIDS))
        with self.assertRaises(TypeError):
            sample.parents[1] = "other"  # type: ignore[index]

    def test_other_blocks_are_local(self) -> None:
        names = _names(SIDS)
        ids = np.arange(1, len(names) + 1)
        cases = {
            # Windstorm's members: event_id // 100, an integer of one set.
            "integer labels": ("event", ids // MEMBERS),
            # The parents' names in other words.
            "other labels": ("event", [f"storm {name[:4]}" for name in names]),
            "the event ids": ("event", None),
            "catalogue years": ("catalogue", None),
        }
        for case, (basis, block) in cases.items():
            with self.subTest(case=case):
                sample = _storms(1, basis=basis, block=block)
                self.assertEqual(dict(sample.parents), {})
                self.assertEqual(parents_in(sample), set())

    def test_names_without_members_or_without_a_name(self) -> None:
        sample = _storms(1)
        # Names without "_gen<m>": every event's block is its own name.
        index = ParametricIndex(
            sample.index.values,
            sample.event_id,
            sample.unit_id,
            sample.index.definition,
            event_name=[f"{sid}x{k}" for k, sid in enumerate(sample.block)],
        )
        self.assertEqual(parent_labels(index, index.event_name), {})
        unnamed = ParametricIndex(
            sample.index.values,
            sample.event_id,
            sample.unit_id,
            sample.index.definition,
        )
        self.assertEqual(parent_labels(unnamed, sample.block), {})
        self.assertEqual(len(parent_labels(sample.index, sample.block)), 40)
        # One block that is not its event's parent makes all of them local.
        block = list(sample.block)
        block[5] = "elsewhere"
        self.assertEqual(parent_labels(sample.index, block), {})

    def test_parents_can_be_stated(self) -> None:
        local = _storms(1, block=[f"b{k // 4}" for k in range(40)])
        stated = OutcomeSample(
            local.index,
            local.loss,
            local.frequency,
            local.units,
            local.basis,
            parents={1: "A", np.int64(2): "A", 3: "B"},
        )
        self.assertEqual(dict(stated.parents), {1: "A", 2: "A", 3: "B"})
        self.assertEqual(parents_in(stated), {"A", "B"})
        for parents in ([("A", 1)], {"1": "A"}, {1: ""}, {1: 2}):
            with self.subTest(parents=parents):
                with self.assertRaisesRegex(ParametricInputError, "parents must map"):
                    OutcomeSample(
                        local.index,
                        local.loss,
                        local.frequency,
                        local.units,
                        local.basis,
                        parents=parents,
                    )


class TestPartsKeepTheParents(unittest.TestCase):
    """Sides, selections, units, designs and mixtures keep the parent storms."""

    def test_parts_and_records(self) -> None:
        sample = _storms(1, basis="yearsets")
        train, test = holdout(sample, rng=np.random.default_rng(2))
        self.assertTrue(parents_in(train).isdisjoint(parents_in(test)))
        self.assertEqual(parents_in(train) | parents_in(test), set(SIDS))
        selected = sample.select(SIDS[:2], rng=np.random.default_rng(3))
        for part in (train, test, sample.unit("U1"), selected):
            with self.subTest(part=part.split):
                self.assertEqual(
                    parents_in(part),
                    {sample.parents[event] for event in part.event_id.tolist()},
                )
        record = _design(train).design_sample
        self.assertEqual(
            dict(record.parents),
            {event: train.parents[event] for event in train.event_id.tolist()},
        )

    def test_a_mixture_keeps_its_members_parents(self) -> None:
        present, future = _storms(1), _storms(2, sids=SIDS[:6])
        mixture = ScenarioSet({"both": [present, future]}).sample("both")
        self.assertEqual(parents_in(mixture), set(SIDS))
        for position, member in enumerate((present, future)):
            for event in member.event_id.tolist():
                self.assertEqual(
                    mixture.parents[-(2 * event + position + 1)],
                    member.parents[event],
                )
        # A split of the mixture keeps every member of a storm on one side,
        # in both of its members.
        train, test = holdout(mixture, rng=np.random.default_rng(3))
        self.assertTrue(parents_in(train).isdisjoint(parents_in(test)))
        check_no_leakage(train, test)

    def test_a_mixture_with_a_local_member(self) -> None:
        local = _storms(3, block=np.arange(1, 41) // MEMBERS)
        mixture = ScenarioSet({"both": [_storms(1), local]}).sample("both")
        self.assertEqual(len(mixture.parents), 40)
        self.assertEqual(parents_in(mixture), set(SIDS))
        train, test = holdout(mixture, rng=np.random.default_rng(4))
        self.assertTrue(parents_in(train).isdisjoint(parents_in(test)))


class TestOutOfSample(unittest.TestCase):
    """Parent storms are compared across hazard sets; local blocks are not."""

    def test_a_shared_parent_raises_across_hazard_sets(self) -> None:
        present, future = _storms(1), _storms(2, offset=1000)
        self.assertNotEqual(present.source, future.source)
        train, _ = holdout(present, rng=np.random.default_rng(5))
        design = _design(train)
        for other in (future, holdout(future, rng=np.random.default_rng(6))[1]):
            with self.assertRaisesRegex(LeakageError, "parent storms are in both"):
                evaluate(design, other, split="out_of_sample")
        # The blocks of a plain sample are its parents; a mixture's are
        # "<member>:<parent>", but its parents are compared as well.
        with self.assertRaisesRegex(LeakageError, "blocks are in both"):
            check_no_leakage(train, future)
        mixture = ScenarioSet({"future": [future, _storms(3)]}).sample("future")
        with self.assertRaisesRegex(LeakageError, "parent storms are in both"):
            check_no_leakage(train, mixture)
        # Another ensemble, of other storms, is out of sample.
        others = _storms(2, sids=[f"{sid}X" for sid in SIDS])
        evaluate(design, others, split="out_of_sample")
        check_out_of_sample(design.design_sample, others)
        # Covers given as they are, and in-sample rows, are never compared.
        evaluate(design.cover, future, split="out_of_sample")
        evaluate(design, future, split="in_sample")

    def test_local_blocks_raise_no_false_alarm(self) -> None:
        # Two hazard sets with the event ids 1 to 40 and integer blocks, as
        # windstorm's event_id // 100: the labels coincide by chance.
        first = _storms(1, block=np.arange(1, 41) // MEMBERS)
        second = _storms(2, block=np.arange(1, 41) // MEMBERS)
        train, _ = holdout(first, rng=np.random.default_rng(7))
        design = _design(train)
        check_out_of_sample(design.design_sample, second)
        evaluate(design, second, split="out_of_sample")
        # Within one hazard set the blocks are compared, as before.
        with self.assertRaisesRegex(LeakageError, "events are in both"):
            evaluate(design, first, split="out_of_sample")

    def test_a_record_without_parents_compares_none(self) -> None:
        present, future = _storms(1), _storms(2)
        record = _design(present).design_sample
        bare = type(record)(**{**record.__dict__, "parents": {}})
        check_out_of_sample(bare, future)


class TestScenarioHoldout(unittest.TestCase):
    """``ScenarioSet.holdout`` cuts members that share parents with their sides."""

    def _check_sides(self, train: ScenarioSet, test: ScenarioSet) -> None:
        """Check that no parent storm is on both sides of any member."""
        on_train = set().union(
            *(parents_in(s) for g in train.members.values() for s in g)
        )
        on_test = set().union(
            *(parents_in(s) for g in test.members.values() for s in g)
        )
        self.assertTrue(on_train.isdisjoint(on_test))
        self.assertEqual(on_train | on_test, set(SIDS))

    def test_the_future_takes_the_presents_sides(self) -> None:
        for basis in ("event", "yearsets"):
            with self.subTest(basis=basis):
                present = _storms(1, basis=basis)
                futures = [_storms(2, basis=basis), _storms(3, basis=basis)]
                scenarios = ScenarioSet({"present": present, "future": futures})
                train, test = scenarios.holdout(rng=np.random.default_rng(8))
                self._check_sides(train, test)
                for member, sides in zip(
                    futures, zip(train.members["future"], test.members["future"])
                ):
                    # Each side keeps the member's lambda.
                    for side in sides:
                        self.assertAlmostEqual(
                            side.frequency.sum() / member.frequency.sum(), 1.0
                        )
                design = _design(train.sample("present"))
                table = evaluate_scenarios(design, test, split="out_of_sample")
                self.assertEqual(set(table["split"]), {"out_of_sample"})
                evaluate(design, test.sample("future"), split="out_of_sample")
                # The future's whole members and training sides hold storms
                # the design was calibrated on.
                for unseen in (
                    scenarios.sample("future"),
                    train.sample("future"),
                    *futures,
                ):
                    with self.assertRaisesRegex(LeakageError, "parent storms"):
                        evaluate(design, unseen, split="out_of_sample")
                with self.assertRaisesRegex(LeakageError, "parent storms") as found:
                    evaluate_scenarios(
                        design,
                        ScenarioSet({"future": futures}),
                        split="out_of_sample",
                    )
                self.assertIn(
                    "raised by evaluate_scenarios at scenario 'future'",
                    found.exception.__notes__,
                )

    def test_new_parents_are_pooled(self) -> None:
        present = _storms(1, sids=SIDS[:6])
        future = _storms(2, sids=SIDS[3:])
        scenarios = ScenarioSet({"present": present, "future": future})
        train, test = scenarios.holdout(rng=np.random.default_rng(9))
        self._check_sides(train, test)
        for parent in SIDS[3:6]:
            self.assertEqual(
                parent in parents_in(train.sample("future")),
                parent in parents_in(train.sample("present")),
            )
        # New parents whose frequency is 0 go to the training side.
        silent = _storms(2)
        silent = OutcomeSample(
            silent.index,
            silent.loss,
            np.where(np.isin(silent.block, SIDS[6:]), 0.0, silent.frequency),
            silent.units,
            silent.basis,
        )
        train, _ = ScenarioSet({"present": present, "future": silent}).holdout(
            rng=np.random.default_rng(9)
        )
        self.assertTrue(set(SIDS[6:]) <= parents_in(train.sample("future")))

    def test_members_without_shared_parents_split_as_before(self) -> None:
        present = _storms(1, basis="yearsets")
        others = _storms(2, basis="yearsets", sids=[f"{sid}X" for sid in SIDS])
        rng = np.random.default_rng(10)
        expected = [holdout(present, rng=rng), holdout(others, rng=rng)]
        train, test = ScenarioSet({"a": present, "b": others}).holdout(
            rng=np.random.default_rng(10)
        )
        for label, (first, second) in zip(("a", "b"), expected):
            for side, part in ((train, first), (test, second)):
                with self.subTest(label=label, side=part.split.side):
                    np.testing.assert_array_equal(
                        side.sample(label).event_id, part.event_id
                    )
                    self.assertEqual(
                        [list(year) for year in side.sample(label).basis.sampling_vect],
                        [list(year) for year in part.basis.sampling_vect],
                    )

    def test_catalogue_years(self) -> None:
        present = _storms(1, basis="catalogue")
        future = _storms(2, basis="catalogue")
        train, test = ScenarioSet({"present": present, "future": future}).holdout()
        self._check_sides(train, test)
        np.testing.assert_array_equal(
            train.sample("future").years, train.sample("present").years
        )
        # Other years, or a parent's members on other years, can't be cut alike.
        names = _names(SIDS)
        longer = _storms(2, basis="event")
        longer = OutcomeSample(
            longer.index,
            longer.loss,
            np.full(40, 1 / 50),
            longer.units,
            CatalogueYears.from_labels(
                longer.event_id, np.arange(40), n_years=50, block=present.block
            ),
        )
        moved = _storms(2, basis="event")
        moved = OutcomeSample(
            moved.index,
            moved.loss,
            moved.frequency,
            moved.units,
            CatalogueYears.from_labels(
                moved.event_id,
                np.arange(40)[::-1],
                n_years=40,
                block=labels_from_names(names, ENSEMBLE_PARENT)["block"],
            ),
        )
        for other, match in (
            (longer, "shares parent storms .* 50 simulation years"),
            (moved, "puts 3 parent storms"),
        ):
            with (
                self.subTest(match=match),
                self.assertRaisesRegex(ScenarioError, match) as found,
            ):
                ScenarioSet({"present": present, "future": other}).holdout()
            self.assertIn(
                "raised by ScenarioSet.holdout at scenario 'future'",
                found.exception.__notes__,
            )


if __name__ == "__main__":
    unittest.main()
