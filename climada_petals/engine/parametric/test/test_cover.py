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

Tests of covers: the index definition, the limit, the terms and their JSON."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import re
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from climada_petals.engine.parametric import (
    BinaryPayout,
    CoverDefinitionError,
    Free,
    IndexDefinition,
    Limit,
    LinearPayout,
    ParametricCover,
    ParametricInputError,
    PayoutFunction,
    PiecewiseLinearPayout,
    StepPayout,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from climada_petals.engine.parametric.test._synthetic import array_index

WIND = IndexDefinition("table", unit="m/s", source="synthetic")
DROUGHT = IndexDefinition("table", unit="1", tail="lower")
FREE = {"free": {"grid": None, "lower": 0.0, "upper": None}}

#: A template in the format of the parity fixtures.
TEMPLATE_DOCUMENT: dict[str, Any] = {
    "index": {
        "index_stat": None,
        "kind": "table",
        "parameters": None,
        "source": "synthetic",
        "tail": "upper",
        "unit": "m/s",
        "weight": None,
    },
    "label": None,
    "limit": {"rule": "largest_loss"},
    "payout": {
        "breakpoints": [25.0, 30.0, 36.0, 43.0, 51.0],
        "kind": "step",
        "levels": [FREE] * 5,
        "monotone": True,
        "tail": "upper",
    },
    "payout_unit": "USD",
    "unit_id": ["u1", "u2"],
}


def _cover(**changes: Any) -> ParametricCover:
    """Return a resolved two-unit cover, with fields changed."""
    fields: dict[str, Any] = {
        "index": WIND,
        "payout": {
            "u1": StepPayout([30.0, 45.0, 60.0], [100.0, 800.0, 2500.0]),
            "u2": StepPayout([25.0, 36.0, 50.0], [50.0, 300.0, 900.0]),
        },
        "unit_id": ["u1", "u2"],
        "payout_unit": "USD",
    }
    fields.update(changes)
    index = fields.pop("index")
    payout = fields.pop("payout")
    return ParametricCover(index, payout, **fields)


@dataclass(frozen=True)
class _Doubling(PayoutFunction):
    """A user payout that pays twice the severity, or what it is told to."""

    kind: ClassVar[str] = "test_doubling"

    factor: float = 2.0
    tail: str = "upper"
    result: Any = None

    def evaluate(self, index: Any) -> np.ndarray:
        if self.result is not None:
            return np.asarray(self.result, dtype=float)
        return self.factor * np.maximum(self.severity(index), 0.0)


class _TwoIndices(_Doubling):
    """A user payout on two index components."""

    index_names: ClassVar[tuple[str, ...]] = ("wind", "rain")


class TestIndexDefinition(unittest.TestCase):
    """Fields, checks and the document of an index definition."""

    def test_fields_and_document(self) -> None:
        index = IndexDefinition(
            "hazard",
            unit="m/s",
            index_stat="mean",
            weight="value",
            source="TC Jamaica",
            parameters={
                "radius_km": np.float64(50.0),
                "ids": [np.int64(1), 2],
                "options": {"absent": None, "dense": np.bool_(True), "name": "km"},
            },
        )
        document = index.to_dict()
        self.assertEqual(
            document,
            {
                "index_stat": "mean",
                "kind": "hazard",
                "parameters": {
                    "ids": [1, 2],
                    "options": {"absent": None, "dense": True, "name": "km"},
                    "radius_km": 50.0,
                },
                "source": "TC Jamaica",
                "tail": "upper",
                "unit": "m/s",
                "weight": "value",
            },
        )
        self.assertEqual(IndexDefinition.from_dict(document), index)
        self.assertEqual(hash(IndexDefinition.from_dict(document)), hash(index))
        self.assertIsInstance(index.parameters, MappingProxyType)
        with self.assertRaises(TypeError):
            index.parameters["radius_km"] = 1.0
        self.assertEqual(index.parameters["ids"], (1, 2))
        self.assertEqual(
            IndexDefinition.from_dict({"kind": "table", "unit": "mm"}),
            IndexDefinition("table", unit="mm"),
        )

    def test_index_statistics(self) -> None:
        for value, stored in (
            ("max", "max"),
            ("at_centroid", "at_centroid"),
            (90, 90.0),
        ):
            with self.subTest(value=value):
                index = IndexDefinition("hazard", unit="m/s", index_stat=value)
                self.assertEqual(index.index_stat, stored)
        self.assertEqual(IndexDefinition("h", unit="m", index_stat=0).index_stat, 0.0)

    def test_invalid_fields_raise(self) -> None:
        bad: list[dict[str, Any]] = [
            {"kind": ""},
            {"kind": 3},
            {"unit": ""},
            {"tail": "up"},
            {"index_stat": "median"},
            {"index_stat": 100.5},
            {"index_stat": -1},
            {"index_stat": True},
            {"index_stat": math.nan},
            {"weight": "area", "index_stat": "mean"},
            {"weight": "value", "index_stat": "max"},
            {"weight": "value"},
            {"source": ""},
            {"source": 3},
            {"parameters": [1, 2]},
            {"parameters": {1: 2}},
            {"parameters": {"a": math.nan}},
            {"parameters": {"a": {"b": object()}}},
            {"parameters": {"a": [{"b": math.inf}]}},
        ]
        for change in bad:
            fields = {"kind": "hazard", "unit": "m/s", **change}
            kind = fields.pop("kind")
            with self.subTest(change=change), self.assertRaises(CoverDefinitionError):
                IndexDefinition(kind, **fields)

    def test_invalid_documents_raise(self) -> None:
        for document in (None, {"kind": "table"}, {"kind": "t", "unit": "m", "x": 1}):
            with (
                self.subTest(document=document),
                self.assertRaises(CoverDefinitionError),
            ):
                IndexDefinition.from_dict(document)


class TestLimit(unittest.TestCase):
    """Stated limits and the largest-loss rule."""

    def test_forms_and_documents(self) -> None:
        one = Limit(1000)
        self.assertEqual(one.amount_of("anything"), 1000.0)
        self.assertEqual(one.to_dict(), {"amount": 1000.0})
        per_unit = Limit({1: 5, np.int64(2): np.float64(6.5)})
        self.assertIsInstance(per_unit.amount, MappingProxyType)
        self.assertEqual(per_unit.amount_of(2), 6.5)
        self.assertEqual(per_unit.to_dict(), {"amount": {"1": 5.0, "2": 6.5}})
        self.assertEqual(Limit.from_dict(per_unit.to_dict(), unit_id=[1, 2]), per_unit)
        rule = Limit.largest_loss()
        self.assertIsNone(rule.amount_of("u1"))
        self.assertEqual(rule.to_dict(), {"rule": "largest_loss"})
        for limit in (one, rule):
            with self.subTest(limit=limit):
                self.assertEqual(Limit.from_dict(limit.to_dict()), limit)
        self.assertEqual(len({one, Limit(1000.0), rule, per_unit}), 3)
        with self.assertRaises(KeyError):
            per_unit.amount_of(3)

    def test_invalid_limits_raise(self) -> None:
        bad: list[dict[str, Any]] = [
            {},
            {"amount": 1.0, "rule": "largest_loss"},
            {"rule": "largest"},
            {"amount": -1.0},
            {"amount": math.nan},
            {"amount": {}},
            {"amount": {"u1": -1.0}},
            {"amount": {True: 1.0}},
            {"amount": "1000"},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(CoverDefinitionError):
                Limit(**kwargs)
        documents: list[Any] = [
            None,
            {"rule": "largest_loss", "amount": 1.0},
            {"amounts": 1.0},
            {"amount": {"u3": 1.0}},
        ]
        for document in documents:
            with (
                self.subTest(document=document),
                self.assertRaises(CoverDefinitionError),
            ):
                Limit.from_dict(document, unit_id=["u1"])


class TestParametricCover(unittest.TestCase):
    """The terms: units, payouts, parameters, the limit and resolution."""

    def test_one_payout_for_every_unit(self) -> None:
        payout = StepPayout.uniform(25.0, 51.0, 5)
        cover = ParametricCover(
            WIND, payout, unit_id=("u1", np.int64(7)), payout_unit="USD"
        )
        self.assertEqual(cover.unit_id, ("u1", 7))
        self.assertIsInstance(cover.unit_id[1], int)
        self.assertEqual(cover.n_units, 2)
        self.assertIsInstance(cover.payout, MappingProxyType)
        self.assertIs(cover.payout["u1"], payout)
        self.assertIs(cover.payout[7], payout)
        self.assertEqual(len(cover.free_parameters()), 10)
        self.assertEqual(
            cover.free_parameters()[:2], (("u1", "levels[0]"), ("u1", "levels[1]"))
        )
        self.assertFalse(cover.is_resolved)

    def test_with_values_resolves_unit_by_unit(self) -> None:
        template = ParametricCover(
            WIND,
            LinearPayout(30.0, 50.0, Free(0.0, 1e6)),
            unit_id=["u1", "u2"],
            payout_unit="USD",
            limit=Limit(5e5),
            label="draft",
        )
        partial = template.with_values({"u1": {"limit": 4e5}})
        self.assertEqual(partial.free_parameters(), (("u2", "limit"),))
        cover = partial.with_values({"u2": {"limit": 2e5}})
        self.assertTrue(cover.is_resolved)
        cover.check_resolved()
        self.assertEqual((cover.label, cover.limit), ("draft", Limit(5e5)))
        self.assertEqual(cover.payout["u1"].limit, 4e5)
        self.assertEqual(template.free_parameters(), (("u1", "limit"), ("u2", "limit")))
        bad: list[Any] = [
            {"u3": {"limit": 1.0}},
            {"u1": {"attachment": 31.0}},
            {"u1": {"limit": 6e5}},
            [("u1", {"limit": 1.0})],
        ]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(CoverDefinitionError):
                template.with_values(values)

    def test_a_template_does_not_pay(self) -> None:
        index = SimpleNamespace(values=np.zeros((2, 2)), unit_id=["u1", "u2"])
        template = ParametricCover.from_dict(TEMPLATE_DOCUMENT)
        with self.assertRaisesRegex(UnresolvedParameterError, "free parameters"):
            template.payout_matrix(index)
        fixed = replace(_cover(), limit=Limit.largest_loss())
        self.assertFalse(fixed.is_resolved)
        with self.assertRaisesRegex(UnresolvedParameterError, "largest_loss"):
            fixed.payout_matrix(index)

    def test_payout_matrix_matches_units_by_id(self) -> None:
        values = [[20.0, 30.0, 61.0], [45.0, 25.0, 36.0], [0.0, 99.0, 50.0]]
        index, _ = array_index(values, np.zeros((3, 3)), [0, 1, 2], 10)
        cover = _cover(
            unit_id=["U2", "U0"],
            payout={
                "U2": StepPayout([30.0, 45.0, 60.0], [100.0, 800.0, 2500.0]),
                "U0": StepPayout([25.0, 36.0, 50.0], [50.0, 300.0, 900.0]),
            },
        )
        paid = cover.payout_matrix(index)
        np.testing.assert_array_equal(paid, [[2500, 0], [100, 300], [800, 0]])
        drought = ParametricCover(
            DROUGHT,
            BinaryPayout(-1.5, 60.0, tail="lower"),
            unit_id=["U1"],
            payout_unit="EUR",
        )
        stand_in = SimpleNamespace(
            values=[[0.0], [-1.5], [-3.0]], unit_id=("U1",), definition=DROUGHT
        )
        np.testing.assert_array_equal(
            drought.payout_matrix(stand_in), [[0], [60], [60]]
        )

    def test_payout_matrix_checks_the_index(self) -> None:
        cover = _cover()
        bad: list[Any] = [
            SimpleNamespace(values=np.zeros((2, 1)), unit_id=["u1"], definition=WIND),
            SimpleNamespace(values=np.zeros((2, 2)), unit_id=["u1"], definition=WIND),
            SimpleNamespace(values=np.zeros(2), unit_id=["u1", "u2"], definition=WIND),
            SimpleNamespace(values=[["a", "b"]], unit_id=["u1", "u2"], definition=WIND),
            SimpleNamespace(values=np.zeros((2, 2)), unit_id=["u1", "u2"]),
            np.zeros((2, 2)),
        ]
        for index in bad:
            with self.subTest(index=index), self.assertRaises(ParametricInputError):
                cover.payout_matrix(index)

    def test_payout_matrix_needs_the_covers_definition(self) -> None:
        values = [[20.0, 30.0], [45.0, 25.0]]
        on_wind, _ = array_index(values, np.zeros((2, 2)), [0, 1], 10)
        cover = _cover(unit_id=["U0", "U1"], payout=StepPayout([30.0], [100.0]))
        np.testing.assert_array_equal(
            cover.payout_matrix(on_wind), [[0, 100], [100, 0]]
        )
        for other in (
            IndexDefinition("table", unit="km/h", source="synthetic"),
            IndexDefinition("table", unit="m/s"),
            IndexDefinition("hazard", unit="m/s", index_stat="max", source="synthetic"),
        ):
            index, _ = array_index(
                values, np.zeros((2, 2)), [0, 1], 10, definition=other
            )
            with (
                self.subTest(other=other),
                self.assertRaisesRegex(ParametricInputError, "cover's definition"),
            ):
                cover.payout_matrix(index)

    def test_user_payouts_are_checked_where_they_pay(self) -> None:
        index = SimpleNamespace(values=[[1.0], [3.0]], unit_id=["u1"], definition=WIND)
        within = ParametricCover(
            WIND, _Doubling(), unit_id=["u1"], payout_unit="USD", limit=Limit(6.0)
        )
        np.testing.assert_array_equal(within.payout_matrix(index), [[2.0], [6.0]])
        above = replace(within, limit=Limit(5.0))
        with self.assertRaisesRegex(CoverDefinitionError, "above its limit"):
            above.payout_matrix(index)
        for result in ([1.0], [1.0, -1.0], [1.0, math.nan]):
            cover = replace(within, payout=_Doubling(result=result))
            with (
                self.subTest(result=result),
                self.assertRaisesRegex(CoverDefinitionError, "must return"),
            ):
                cover.payout_matrix(index)

    def test_a_stated_limit_bounds_the_amounts(self) -> None:
        cover = _cover(limit=Limit({"u1": 2500.0, "u2": 900.0}))
        self.assertEqual(cover.limit.amount_of("u2"), 900.0)
        bad: list[Any] = [
            Limit(2000.0),
            Limit({"u1": 2500.0}),
            Limit({"u1": 2500.0, "u2": 900.0, "u3": 1.0}),
            1000.0,
        ]
        for limit in bad:
            with self.subTest(limit=limit), self.assertRaises(CoverDefinitionError):
                _cover(limit=limit)
        for payout in (
            StepPayout([1.0, 2.0], [Free(10.0), Free()]),
            LinearPayout(1.0, 2.0, 1e6),
            PiecewiseLinearPayout([1.0, 2.0], [Free(), 7.0]),
        ):
            with (
                self.subTest(payout=payout),
                self.assertRaisesRegex(CoverDefinitionError, "within its limit"),
            ):
                ParametricCover(
                    WIND, payout, unit_id=["u1"], payout_unit="USD", limit=Limit(5.0)
                )

    def test_invalid_terms_raise(self) -> None:
        step = StepPayout([1.0], [1.0])
        bad: list[dict[str, Any]] = [
            {"index": {"kind": "table", "unit": "m/s"}},
            {"unit_id": []},
            {"unit_id": ["u1", "u1"]},
            {"unit_id": ["1", 1], "payout": step},
            {"unit_id": [True], "payout": step},
            {"unit_id": [1.5], "payout": step},
            {"unit_id": [""], "payout": step},
            {"unit_id": "u1", "payout": step},
            {"unit_id": {"u1": 1}, "payout": step},
            {"payout": {"u1": step}},
            {"payout": {"u1": step, "u2": step, "u3": step}},
            {"payout": {"u1": step, "u2": "step"}},
            {"payout": (1.0, 2.0)},
            {"payout": StepPayout([-1.0], [1.0], tail="lower")},
            {"payout": _TwoIndices()},
            {"payout_unit": ""},
            {"payout_unit": 3},
            {"label": 3},
        ]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(CoverDefinitionError):
                _cover(**change)


class TestDocuments(unittest.TestCase):
    """The JSON round trip and the definition hash."""

    def test_the_parity_format_round_trips(self) -> None:
        cover = ParametricCover.from_dict(TEMPLATE_DOCUMENT)
        self.assertEqual(cover.to_dict(), TEMPLATE_DOCUMENT)
        self.assertEqual(cover.limit, Limit.largest_loss())
        self.assertEqual(
            cover.payout["u1"], StepPayout([25, 30, 36, 43, 51], [Free()] * 5)
        )
        per_unit = _cover().to_dict()
        self.assertEqual(set(per_unit["payout"]), {"u1", "u2"})
        self.assertEqual(ParametricCover.from_dict(per_unit), _cover())

    def test_equal_payouts_are_written_once(self) -> None:
        step = StepPayout([1.0], [1.0])
        cover = _cover(payout={"u1": step, "u2": StepPayout([1.0], [1.0])})
        self.assertEqual(cover.to_dict()["payout"], step.to_dict())
        self.assertEqual(cover, _cover(payout=step))

    def test_json_files_and_text(self) -> None:
        cover = _cover(
            unit_id=[3, 1],
            payout={
                3: LinearPayout(30.0, 50.0, 1e3),
                1: PiecewiseLinearPayout([50.0, 80.0], [50.0, 250.0]),
            },
            limit=Limit({3: 1e3, 1: 300.0}),
            label="two units",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cover.json"
            cover.write_json(path)
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("}\n"))
            self.assertEqual(json.loads(text), cover.to_dict())
            for source in (path, str(path), text, "  " + text):
                with self.subTest(source=type(source)):
                    again = ParametricCover.from_json(source)
                    self.assertEqual(again, cover)
                    self.assertEqual(again.unit_id, (3, 1))
                    self.assertEqual(hash(again), hash(cover))
        self.assertNotEqual(cover, cover.to_dict())

    def test_covers_pickle_and_copy(self) -> None:
        covers = [
            _cover(limit=Limit({"u1": 3e3, "u2": 1e3}), label="stated"),
            ParametricCover.from_dict(TEMPLATE_DOCUMENT),
            replace(
                _cover(limit=Limit(5e3)),
                index=IndexDefinition("hazard", unit="m/s", parameters={"r": [1]}),
            ),
        ]
        for cover in covers:
            with self.subTest(cover=cover.label):
                for again in (pickle.loads(pickle.dumps(cover)), copy.deepcopy(cover)):
                    self.assertEqual(again, cover)
                    self.assertEqual(again.limit, cover.limit)
                    self.assertEqual(again.index, cover.index)

    def test_invalid_documents_raise(self) -> None:
        document = copy.deepcopy(TEMPLATE_DOCUMENT)
        bad: list[Any] = [
            None,
            {**document, "extra": 1},
            {key: value for key, value in document.items() if key != "payout"},
            {**document, "unit_id": "u1"},
            {**document, "payout": [document["payout"]]},
            {**document, "payout": {"u1": document["payout"]}},
            {**document, "limit": {"rule": "largest"}},
            {**document, "index": {"kind": "table"}},
        ]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(CoverDefinitionError):
                ParametricCover.from_dict(item)
        unknown = {**document, "payout": {**document["payout"], "kind": "spline"}}
        with self.assertRaises(UnknownPayoutKindError):
            ParametricCover.from_dict(unknown)
        with self.assertRaisesRegex(CoverDefinitionError, "not a JSON document"):
            ParametricCover.from_json("{not json")
        with self.assertRaises(ParametricInputError):
            ParametricCover.from_json(3)
        optional = {
            key: value
            for key, value in document.items()
            if key not in ("label", "limit")
        }
        self.assertIsNone(ParametricCover.from_dict(optional).limit)

    def test_definition_hash(self) -> None:
        cover = _cover()
        canonical = (
            '{"index":{"index_stat":null,"kind":"table","parameters":null,'
            '"source":"synthetic","tail":"upper","unit":"m/s","weight":null},'
            '"limit":null,"payout":{"u1":{"breakpoints":[30.0,45.0,60.0],'
            '"kind":"step","levels":[100.0,800.0,2500.0],"monotone":true,'
            '"tail":"upper"},"u2":{"breakpoints":[25.0,36.0,50.0],"kind":"step",'
            '"levels":[50.0,300.0,900.0],"monotone":true,"tail":"upper"}},'
            '"payout_unit":"USD","unit_id":["u1","u2"]}'
        )
        digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(cover.definition_hash, digest)
        self.assertEqual(
            digest,
            "sha256:f3b2a593d5b1165f570f28fca0644643a49330fc582c4df47a62edec86549e6b",
        )
        self.assertEqual(replace(cover, label="another name").definition_hash, digest)
        again = ParametricCover.from_json(json.dumps(cover.to_dict()))
        self.assertEqual(again.definition_hash, digest)
        payouts = dict(cover.payout)
        changed = [
            replace(
                cover,
                unit_id=["u1", "u3"],
                payout={"u1": payouts["u1"], "u3": payouts["u2"]},
            ),
            replace(cover, payout_unit="EUR"),
            replace(cover, limit=Limit(1e4)),
            replace(cover, index=IndexDefinition("table", unit="km/h")),
            replace(
                cover,
                payout={
                    **payouts,
                    "u1": StepPayout([30.0, 45.0, 60.0], [100.0, 800.0, 2500.5]),
                },
            ),
        ]
        for other in changed:
            with self.subTest(other=other):
                self.assertNotEqual(other.definition_hash, digest)
                self.assertNotEqual(other, cover)


if __name__ == "__main__":
    unittest.main()
