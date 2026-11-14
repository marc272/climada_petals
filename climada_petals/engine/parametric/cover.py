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

A parametric cover: the terms of the contract.

A :class:`ParametricCover` holds the terms and nothing else: the index it
pays on (:class:`IndexDefinition`), one payout function per insured unit
with its parameters, the payouts' unit, and the most one event pays at one
unit (:class:`Limit`). It holds no events, years, frequencies or solver
output, so one cover can be stored, compared, and applied to another hazard
set.

* A cover with a free parameter, or with a limit stated as a rule, is a
  **template**. The calibration resolves it, and
  :meth:`ParametricCover.with_values` resolves one by hand. Only a resolved
  cover pays (:meth:`ParametricCover.payout_matrix`).
* A cover round-trips through JSON: :meth:`~ParametricCover.to_dict`,
  :meth:`~ParametricCover.write_json`, :meth:`~ParametricCover.from_dict`
  and :meth:`~ParametricCover.from_json`. Its document holds ``index``,
  ``payout`` (one payout document, or one per unit id when the units'
  payouts differ), ``unit_id``, ``payout_unit``, ``limit`` and ``label``.
* :attr:`~ParametricCover.definition_hash` identifies the terms in tables:
  the SHA-256 of the canonical JSON of the document without its label.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from .errors import CoverDefinitionError, ParametricInputError, UnresolvedParameterError
from .parameters import non_negative, real_number
from .payout import LinearInAmounts, PayoutFunction, check_tail, payout_from_dict

if TYPE_CHECKING:
    from .index import ParametricIndex

__all__ = [
    "INDEX_STATS",
    "LARGEST_LOSS",
    "IndexDefinition",
    "Limit",
    "ParametricCover",
]

INDEX_STATS: Final[tuple[str, ...]] = ("at_centroid", "mean", "max")
"""The named statistics of :attr:`IndexDefinition.index_stat`."""

LARGEST_LOSS: Final[str] = "largest_loss"
"""The rule of :meth:`Limit.largest_loss`."""

_LIMIT_RTOL: Final[float] = 1e-9
_COVER_KEYS: Final[frozenset[str]] = frozenset(
    {"index", "label", "limit", "payout", "payout_unit", "unit_id"}
)
_INDEX_KEYS: Final[frozenset[str]] = frozenset(
    {"index_stat", "kind", "parameters", "source", "tail", "unit", "weight"}
)


@dataclass(frozen=True)
class IndexDefinition:
    """What a cover pays on: how its index is computed, and in what unit.

    It travels with the cover, so the cover can be applied to another hazard
    set, and every table can state the index behind a number.

    Parameters
    ----------
    kind : str
        How the index is built, e.g. ``"hazard"`` (a statistic of a hazard
        footprint), ``"cat_in_circle"`` or ``"table"`` (a user's table, such
        as station readings).
    unit : str
        Unit of the index values, e.g. ``"m/s"``. Keyword-only.
    tail : {"upper", "lower"}, default "upper"
        Which side of the index is severe. Every payout of a cover on this
        index has the same tail.
    index_stat : str or float, optional
        The statistic over a unit's centroids: ``"at_centroid"``,
        ``"mean"``, ``"max"``, or a percentile in ``[0, 100]``, as in
        ``cat_bonds``. None where none applies.
    weight : {"value"}, optional
        Weights of the ``"mean"``: the exposure values. None: equal weights.
    source : str, optional
        What the index is computed from, e.g. the name of a hazard set.
    parameters : mapping, optional
        Further settings of the index, e.g. a radius, as JSON values with
        string keys. Stored read-only.

    Raises
    ------
    CoverDefinitionError
        If a field is invalid, or ``weight`` is given for a statistic other
        than ``"mean"``.
    """

    kind: str
    unit: str = field(kw_only=True)
    tail: str = field(default="upper", kw_only=True)
    index_stat: str | float | None = field(default=None, kw_only=True)
    weight: str | None = field(default=None, kw_only=True)
    source: str | None = field(default=None, kw_only=True)
    parameters: Mapping[str, Any] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the fields; store the parameters read-only."""
        index_stat = _index_stat(self.index_stat)
        if self.weight is not None and (self.weight != "value" or index_stat != "mean"):
            raise CoverDefinitionError(
                "weight must be None, or 'value' with index_stat='mean'; got "
                f"weight={self.weight!r} and index_stat={index_stat!r}"
            )
        if self.parameters is not None and not isinstance(self.parameters, Mapping):
            raise CoverDefinitionError(
                f"parameters must be a mapping, got {self.parameters!r}"
            )
        values = {
            "kind": _text("kind", self.kind),
            "unit": _text("unit", self.unit),
            "tail": check_tail(self.tail),
            "index_stat": index_stat,
            "source": None if self.source is None else _text("source", self.source),
            "parameters": (
                None
                if self.parameters is None
                else _frozen(self.parameters, "parameters")
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __hash__(self) -> int:
        """Hash the document, since ``parameters`` is a read-only mapping."""
        return hash(_canonical(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        """Return the index definition's document, every field included."""
        return {
            "index_stat": self.index_stat,
            "kind": self.kind,
            "parameters": _thawed(self.parameters),
            "source": self.source,
            "tail": self.tail,
            "unit": self.unit,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> IndexDefinition:
        """Read an index definition from its document (:meth:`to_dict`).

        Parameters
        ----------
        document : mapping
            The fields. ``kind`` and ``unit`` are required; the others
            default as in the constructor.

        Returns
        -------
        IndexDefinition

        Raises
        ------
        CoverDefinitionError
            If a key is unknown or missing, or a value is invalid.
        """
        _check_keys("an index definition", document, _INDEX_KEYS, {"kind", "unit"})
        fields = {key: value for key, value in document.items() if key != "kind"}
        return cls(document["kind"], **fields)


@dataclass(frozen=True)
class Limit:
    """The most one event pays at one unit: a term of the cover.

    Exactly one of ``amount`` and ``rule`` is given.

    Parameters
    ----------
    amount : float or mapping of unit id to float, optional
        The limit in payout units, >= 0: one for every unit, or one per unit
        id, for every unit of the cover.
    rule : {"largest_loss"}, optional, keyword-only
        A rule the calibration resolves into amounts; see
        :meth:`largest_loss`.

    Raises
    ------
    CoverDefinitionError
        If both or neither are given, an amount is negative or not finite,
        or the rule is unknown.

    Notes
    -----
    A cover checks a stated limit against its payouts: every amount of a
    payout that is linear in its amounts must be able to stay within it
    (:meth:`~climada_petals.engine.parametric.payout.LinearInAmounts.amount_constraints`),
    and another payout must not pay more where it is evaluated.
    """

    amount: float | Mapping[Hashable, float] | None = None
    rule: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate and store the amount as a float or a read-only mapping."""
        if (self.amount is None) == (self.rule is None):
            raise CoverDefinitionError(
                "a Limit takes an amount, or is Limit.largest_loss(); got "
                f"amount={self.amount!r} and rule={self.rule!r}"
            )
        if self.rule is not None:
            if self.rule != LARGEST_LOSS:
                raise CoverDefinitionError(
                    f"Limit.rule must be {LARGEST_LOSS!r}, got {self.rule!r}"
                )
        elif isinstance(self.amount, Mapping):
            if not self.amount:
                raise CoverDefinitionError("Limit.amount must not be an empty mapping")
            amounts = {
                _unit_id(unit): non_negative(f"Limit.amount[{unit!r}]", value)
                for unit, value in self.amount.items()
            }
            object.__setattr__(self, "amount", MappingProxyType(amounts))
        else:
            object.__setattr__(
                self, "amount", non_negative("Limit.amount", self.amount)
            )

    def __hash__(self) -> int:
        """Hash the rule or the amounts."""
        if isinstance(self.amount, Mapping):
            return hash(frozenset(self.amount.items()))
        return hash((self.amount, self.rule))

    @classmethod
    def largest_loss(cls) -> Limit:
        """Return the rule: each unit's largest event loss on the design sample.

        The calibration writes the numbers into the resolved cover. A cover
        with this rule is a template.
        """
        return cls(rule=LARGEST_LOSS)

    def amount_of(self, unit_id: Hashable) -> float | None:
        """Return the stated limit of a unit, or None under a rule.

        Raises
        ------
        KeyError
            If the limit is stated per unit and not for ``unit_id``.
        """
        if self.rule is not None:
            return None
        if isinstance(self.amount, Mapping):
            return float(self.amount[unit_id])
        return float(self.amount)

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"rule": ...}`` or ``{"amount": ...}``, keyed by ``str(unit)``."""
        if self.rule is not None:
            return {"rule": self.rule}
        if isinstance(self.amount, Mapping):
            return {"amount": {str(unit): value for unit, value in self.amount.items()}}
        return {"amount": self.amount}

    @classmethod
    def from_dict(
        cls, document: Mapping[str, Any], *, unit_id: Sequence[Hashable] = ()
    ) -> Limit:
        """Read a limit from its document (:meth:`to_dict`).

        Parameters
        ----------
        document : mapping
            ``{"rule": "largest_loss"}``, ``{"amount": x}`` or
            ``{"amount": {str(unit): x, ...}}``.
        unit_id : sequence of str or int
            The cover's units, to read the keys of amounts per unit.

        Returns
        -------
        Limit

        Raises
        ------
        CoverDefinitionError
            If the document has another form, or names a unit not in
            ``unit_id``.
        """
        if not isinstance(document, Mapping) or set(document) not in (
            {"rule"},
            {"amount"},
        ):
            raise CoverDefinitionError(
                f"a limit document is {{'rule': ...}} or {{'amount': ...}}, got "
                f"{document!r}"
            )
        if "rule" in document:
            return cls(rule=document["rule"])
        amount = document["amount"]
        if isinstance(amount, Mapping):
            amount = _by_unit(amount, unit_id, "the limit's amounts")
        return cls(amount)


@dataclass(frozen=True, eq=False)
class ParametricCover:
    """A parametric cover: the terms of the contract (module docstring).

    Parameters
    ----------
    index : IndexDefinition
        What the cover pays on.
    payout : PayoutFunction or mapping of unit id to PayoutFunction
        One payout for every unit, or one per unit id, for every unit.
        Stored as a read-only mapping by unit id. Each reads one index
        component and has the index's ``tail``.
    unit_id : sequence of str or int
        The insured units, unique, also as strings. Keyword-only.
    payout_unit : str
        Unit of the payouts and amounts, e.g. ``"USD"``: the unit of the
        loss the cover is designed against. Keyword-only.
    limit : Limit, optional
        The most one event pays at one unit. None: the amounts are bounded
        by their own upper bounds, or by a premium budget, alone.
    label : str, optional
        A name for tables. Not part of the terms: it changes neither
        :attr:`definition_hash` nor how the cover pays.

    Raises
    ------
    CoverDefinitionError
        If a field is invalid, the payouts and the units don't match, a
        payout's ``tail`` is not the index's, or a payout's amounts can't
        stay within a stated limit.

    Examples
    --------
    A template with five free levels on a wind index, and the same cover
    resolved by hand:

    >>> wind = IndexDefinition("hazard", unit="m/s", index_stat="max")
    >>> template = ParametricCover(
    ...     wind,
    ...     StepPayout.uniform(attachment=33.0, exhaustion=80.0, n_steps=5),
    ...     unit_id=["parish_1"],
    ...     payout_unit="USD",
    ...     limit=Limit(1e6),
    ... )
    >>> template.is_resolved
    False
    >>> levels = {f"levels[{k}]": 2e5 * (k + 1) for k in range(5)}
    >>> cover = template.with_values({"parish_1": levels})
    >>> cover.is_resolved
    True
    """

    index: IndexDefinition
    payout: PayoutFunction | Mapping[Hashable, PayoutFunction]
    unit_id: Sequence[Hashable] = field(kw_only=True)
    payout_unit: str = field(kw_only=True)
    limit: Limit | None = field(default=None, kw_only=True)
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the terms; store the units as a tuple and the payouts by unit."""
        if not isinstance(self.index, IndexDefinition):
            raise CoverDefinitionError(
                f"index must be an IndexDefinition, got {self.index!r}"
            )
        units = _unit_ids(self.unit_id)
        payouts = _payouts(self.payout, units, self.index.tail)
        if self.limit is not None:
            _check_limit(self.limit, payouts)
        if self.label is not None and not isinstance(self.label, str):
            raise CoverDefinitionError(f"label must be a string, got {self.label!r}")
        object.__setattr__(self, "unit_id", units)
        object.__setattr__(self, "payout", MappingProxyType(payouts))
        object.__setattr__(self, "payout_unit", _text("payout_unit", self.payout_unit))

    def __eq__(self, other: object) -> bool:
        """Two covers are equal if their documents are, label included."""
        if not isinstance(other, ParametricCover):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        """Hash the document, label included."""
        return hash(_canonical(self.to_dict()))

    @property
    def n_units(self) -> int:
        """Number of insured units."""
        return len(self.unit_id)

    def free_parameters(self) -> tuple[tuple[Hashable, str], ...]:
        """Return the free parameters as ``(unit_id, name)`` pairs.

        Returns
        -------
        tuple of (unit id, str)
            In the order of the units, then of each payout's parameters.
        """
        return tuple(
            (unit, name)
            for unit in self.unit_id
            for name in self.payout[unit].free_parameters()
        )

    @property
    def is_resolved(self) -> bool:
        """Whether no parameter is free and the limit, if any, is stated."""
        return not self.free_parameters() and (
            self.limit is None or self.limit.rule is None
        )

    def check_resolved(self) -> None:
        """Raise :class:`UnresolvedParameterError` if the cover is a template."""
        free = self.free_parameters()
        if free:
            raise UnresolvedParameterError(
                f"the cover has {len(free)} free parameters, e.g. {list(free[:3])}: "
                "resolve them with with_values, or calibrate the cover"
            )
        if self.limit is not None and self.limit.rule is not None:
            raise UnresolvedParameterError(
                f"the cover's limit is the rule {self.limit.rule!r}, which the "
                "calibration resolves on a design sample; state Limit(amount) to "
                "use the cover as it is"
            )

    def with_values(
        self, values: Mapping[Hashable, Mapping[str, float]]
    ) -> ParametricCover:
        """Return the cover with values for some or all free parameters.

        Parameters
        ----------
        values : mapping of unit id to mapping of str to float
            Per unit, a value per free parameter, by name, as
            :meth:`~climada_petals.engine.parametric.payout.PayoutFunction.with_values`
            takes them. Units and parameters not named stay as they are.

        Returns
        -------
        ParametricCover
            A new cover, checked again. The limit and the label are kept.

        Raises
        ------
        CoverDefinitionError
            If a unit is not the cover's, a name is not a free parameter of
            its payout, a value is out of bounds, or the new cover is
            invalid.
        """
        if not isinstance(values, Mapping):
            raise CoverDefinitionError(
                f"values must map unit ids to parameter values, got {values!r}"
            )
        unknown = [unit for unit in values if unit not in self.payout]
        if unknown:
            raise CoverDefinitionError(f"the cover has no units {unknown[:5]}")
        payouts = dict(self.payout)
        for unit, unit_values in values.items():
            payouts[unit] = self.payout[unit].with_values(unit_values)
        return replace(self, payout=payouts)

    def payout_matrix(self, index: ParametricIndex) -> NDArray[np.float64]:
        """Return the payout of every event and unit of ``index``.

        Parameters
        ----------
        index : ParametricIndex
            An index that has a column for every unit of the cover, matched
            by unit id; other columns are not read.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_events, n_units)``, in the cover's unit order and in
            payout units.

        Raises
        ------
        UnresolvedParameterError
            If the cover is a template.
        ParametricInputError
            If ``index`` lacks a unit of the cover.
        CoverDefinitionError
            If a user-defined payout returns payouts of another shape,
            negative or not finite ones, or ones above the unit's limit.
        """
        self.check_resolved()
        values, columns = _index_columns(index, self.unit_id)
        out = np.empty((values.shape[0], self.n_units))
        for position, unit in enumerate(self.unit_id):
            out[:, position] = self._unit_payout(unit, values[:, columns[position]])
        return out

    def _unit_payout(
        self, unit: Hashable, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return and check the payout of one unit."""
        payout = self.payout[unit]
        paid = np.asarray(payout.evaluate(values), dtype=np.float64)
        if paid.shape != values.shape or not np.all(np.isfinite(paid) & (paid >= 0)):
            raise CoverDefinitionError(
                f"the {payout.kind} payout of unit {unit!r} must return finite "
                f"payouts >= 0 of shape {values.shape}"
            )
        cap = None if self.limit is None else self.limit.amount_of(unit)
        if (
            cap is not None
            and not isinstance(payout, LinearInAmounts)
            and np.any(paid > cap * (1.0 + _LIMIT_RTOL))
        ):
            raise CoverDefinitionError(
                f"the {payout.kind} payout of unit {unit!r} pays {paid.max()}, above "
                f"its limit {cap}"
            )
        return paid

    def to_dict(self) -> dict[str, Any]:
        """Return the cover's document (module docstring).

        Returns
        -------
        dict
            JSON-ready. ``payout`` is one payout document if every unit's is
            the same, otherwise a mapping from ``str(unit_id)`` to each.
        """
        documents = [self.payout[unit].to_dict() for unit in self.unit_id]
        if all(document == documents[0] for document in documents):
            payout: Any = documents[0]
        else:
            payout = {
                str(unit): document
                for unit, document in zip(self.unit_id, documents, strict=True)
            }
        return {
            "index": self.index.to_dict(),
            "label": self.label,
            "limit": None if self.limit is None else self.limit.to_dict(),
            "payout": payout,
            "payout_unit": self.payout_unit,
            "unit_id": list(self.unit_id),
        }

    def write_json(self, path: str | os.PathLike[str]) -> None:
        """Write the cover's document to a JSON file, keys sorted.

        Parameters
        ----------
        path : str or path-like
            The file to write; it is replaced if it exists.
        """
        text = json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        Path(path).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ParametricCover:
        """Read a cover from its document (:meth:`to_dict`).

        Parameters
        ----------
        document : mapping
            ``index``, ``payout``, ``unit_id`` and ``payout_unit``, and
            optionally ``limit`` and ``label`` (default None). Payout kinds
            are looked up among the registered families
            (:func:`~climada_petals.engine.parametric.payout.register_payout`).

        Returns
        -------
        ParametricCover

        Raises
        ------
        CoverDefinitionError
            If a key is unknown or missing, or a value is invalid.
        UnknownPayoutKindError
            If a payout's ``kind`` is not registered.
        """
        _check_keys(
            "a cover",
            document,
            _COVER_KEYS,
            {"index", "payout", "unit_id", "payout_unit"},
        )
        if not isinstance(document["unit_id"], list):
            raise CoverDefinitionError(
                f"a cover's unit_id is a list, got {document['unit_id']!r}"
            )
        units = _unit_ids(document["unit_id"])
        payout = document["payout"]
        if isinstance(payout, Mapping) and isinstance(payout.get("kind"), str):
            payouts: Any = payout_from_dict(payout)
        elif isinstance(payout, Mapping):
            payouts = {
                unit: payout_from_dict(item)
                for unit, item in _by_unit(payout, units, "the payouts").items()
            }
        else:
            raise CoverDefinitionError(
                f"a cover's payout is a payout document or a mapping by unit id, got "
                f"{payout!r}"
            )
        limit = document.get("limit")
        return cls(
            IndexDefinition.from_dict(document["index"]),
            payouts,
            unit_id=units,
            payout_unit=document["payout_unit"],
            limit=None if limit is None else Limit.from_dict(limit, unit_id=units),
            label=document.get("label"),
        )

    @classmethod
    def from_json(cls, path_or_text: str | os.PathLike[str]) -> ParametricCover:
        """Read a cover from a JSON file, or from JSON text.

        Parameters
        ----------
        path_or_text : str or path-like
            A path, or the JSON text itself: a string that starts with
            ``{``.

        Returns
        -------
        ParametricCover

        Raises
        ------
        CoverDefinitionError
            If the text is not JSON, or not a valid cover document.
        ParametricInputError
            If ``path_or_text`` is neither a string nor a path.
        """
        if isinstance(path_or_text, str) and path_or_text.lstrip().startswith("{"):
            text = path_or_text
        elif isinstance(path_or_text, str | os.PathLike):
            text = Path(path_or_text).read_text(encoding="utf-8")
        else:
            raise ParametricInputError(
                f"path_or_text must be a path or JSON text, got {path_or_text!r}"
            )
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CoverDefinitionError(f"not a JSON document: {exc}") from exc
        return cls.from_dict(document)

    @property
    def definition_hash(self) -> str:
        """``"sha256:<hex>"`` of the canonical JSON of the terms.

        The canonical JSON is the document without ``label``, with sorted
        keys and no whitespace. The hash changes with the index, the
        payouts and their parameters, the units, the payout unit and the
        limit, and not with the label.
        """
        document = self.to_dict()
        del document["label"]
        digest = hashlib.sha256(_canonical(document).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


def _canonical(document: object) -> str:
    """Return the canonical JSON of a document: sorted keys, no whitespace."""
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _text(name: str, value: object) -> str:
    """Return ``value`` if it is a non-empty string."""
    if not isinstance(value, str) or not value:
        raise CoverDefinitionError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _index_stat(value: object) -> str | float | None:
    """Return a valid ``index_stat``: None, a named statistic or a percentile."""
    if value is None or (isinstance(value, str) and value in INDEX_STATS):
        return value
    if isinstance(value, str):
        raise CoverDefinitionError(
            f"index_stat must be one of {list(INDEX_STATS)} or a percentile in "
            f"[0, 100], got {value!r}"
        )
    number = real_number("index_stat", value)
    if not 0.0 <= number <= 100.0:
        raise CoverDefinitionError(
            f"index_stat as a percentile must be in [0, 100], got {number}"
        )
    return number


def _frozen(value: object, path: str) -> Any:
    """Return a JSON value with mappings read-only and lists as tuples."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.bool_ | np.integer | np.floating):
        return _frozen(value.item(), path)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return MappingProxyType(
            {key: _frozen(item, f"{path}[{key!r}]") for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_frozen(item, f"{path}[{k}]") for k, item in enumerate(value))
    raise CoverDefinitionError(
        f"{path} must hold JSON values (finite numbers, strings, lists and "
        f"mappings with string keys), got {value!r}"
    )


def _thawed(value: object) -> Any:
    """Return a frozen JSON value with dicts and lists again."""
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return value


def _check_keys(
    what: str, document: object, allowed: frozenset[str], required: set[str]
) -> None:
    """Raise unless ``document`` is a mapping with known keys and the required ones."""
    if not isinstance(document, Mapping):
        raise CoverDefinitionError(f"{what} document is a mapping, got {document!r}")
    unknown = sorted(set(document) - allowed)
    missing = sorted(required - set(document))
    if unknown or missing:
        raise CoverDefinitionError(
            f"{what} document has unknown keys {unknown} and misses {missing}"
        )


def _unit_id(value: object) -> str | int:
    """Return a unit id if it is a string or an integer."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_):
        return int(value)
    raise CoverDefinitionError(
        f"a unit id is a non-empty string or an integer, got {value!r}"
    )


def _unit_ids(values: object) -> tuple[str | int, ...]:
    """Return the cover's unit ids: non-empty, unique, and unique as strings."""
    if isinstance(values, str | Mapping) or not isinstance(
        values, Sequence | np.ndarray
    ):
        raise CoverDefinitionError(f"unit_id must be a sequence, got {values!r}")
    units = tuple(_unit_id(value) for value in values)
    if not units or len({str(unit) for unit in units}) != len(units):
        raise CoverDefinitionError(
            "unit_id must hold at least one unit, each unique, also as a string; "
            f"got {list(units)[:10]}"
        )
    return units


def _by_unit(
    document: Mapping[str, Any], units: Sequence[Hashable], what: str
) -> dict[Hashable, Any]:
    """Return a document keyed by ``str(unit)`` keyed by the units themselves."""
    by_text = {str(unit): unit for unit in units}
    if set(document) != set(by_text):
        raise CoverDefinitionError(
            f"{what} must have one entry per unit id; got keys {sorted(document)} "
            f"for units {list(by_text)[:10]}"
        )
    return {by_text[key]: value for key, value in document.items()}


def _payouts(
    payout: object, units: tuple[str | int, ...], tail: str
) -> dict[Hashable, PayoutFunction]:
    """Return one payout per unit, checked against the units and the tail."""
    if isinstance(payout, Mapping):
        if set(payout) != set(units):
            raise CoverDefinitionError(
                "a mapping of payouts must have one entry per unit id; got keys "
                f"{list(payout)[:10]} for units {list(units)[:10]}"
            )
        found = {unit: payout[unit] for unit in units}
    else:
        found = dict.fromkeys(units, payout)
    for unit, item in found.items():
        if not isinstance(item, PayoutFunction):
            raise CoverDefinitionError(
                f"the payout of unit {unit!r} must be a PayoutFunction, got {item!r}"
            )
        if len(item.index_names) != 1:
            raise CoverDefinitionError(
                f"the payout of unit {unit!r} reads {len(item.index_names)} index "
                "components; a cover's payouts read one"
            )
        if item.tail != tail:
            raise CoverDefinitionError(
                f"the payout of unit {unit!r} has tail={item.tail!r}, and the index "
                f"has tail={tail!r}"
            )
    return found


def _check_limit(limit: object, payouts: Mapping[Hashable, PayoutFunction]) -> None:
    """Raise unless a stated limit covers every unit and its amounts can meet it."""
    if not isinstance(limit, Limit):
        raise CoverDefinitionError(f"limit must be a Limit, got {limit!r}")
    if limit.rule is not None:
        return
    if isinstance(limit.amount, Mapping) and set(limit.amount) != set(payouts):
        raise CoverDefinitionError(
            "a limit per unit must have one amount per unit id; got "
            f"{list(limit.amount)[:10]} for units {list(payouts)[:10]}"
        )
    for unit, payout in payouts.items():
        if not isinstance(payout, LinearInAmounts):
            continue
        cap = limit.amount_of(unit)
        if not payout.amount_constraints(cap).bounds_feasible:
            raise CoverDefinitionError(
                f"the {payout.kind} payout of unit {unit!r} can't stay within its "
                f"limit {cap}: a fixed amount, or the lower bound of a free one, is "
                "above it"
            )


def _index_columns(
    index: Any, units: Sequence[Hashable]
) -> tuple[NDArray[np.float64], list[int]]:
    """Return an index's values and the column of each of the cover's units."""
    try:
        values = np.asarray(index.values, dtype=np.float64)
        index_units = list(index.unit_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ParametricInputError(
            f"index must be a ParametricIndex, with values and unit_id: {exc}"
        ) from exc
    if values.ndim != 2 or values.shape[1] != len(index_units):
        raise ParametricInputError(
            "the index's values must have one column per unit id, got shape "
            f"{values.shape} for {len(index_units)} units"
        )
    column = {unit: position for position, unit in enumerate(index_units)}
    missing = [unit for unit in units if unit not in column]
    if missing:
        raise ParametricInputError(f"the index has no column for units {missing[:5]}")
    return values, [column[unit] for unit in units]
