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

The fields of a payout that is a frozen dataclass: checks, parameters,
new values and documents.

:class:`~climada_petals.engine.parametric.payout.PayoutFunction` has
defaults for a frozen dataclass whose fields are its constructor's
arguments, as the built-in families are; they are implemented here:

* :func:`field_parameters`: its parameters are the fields that hold a
  number or :class:`~climada_petals.engine.parametric.parameters.Free`, or
  a tuple of them, named ``"field"`` or ``"field[k]"``;
* :func:`with_field_values` replaces free parameters and builds the payout
  again, so its own checks run;
* :func:`field_document` and :func:`payout_from_fields`: its document
  holds ``kind`` and every field.

:func:`store_levels` checks and stores the fields of the step and the
piecewise-linear families, thresholds and levels; :func:`store_fields` sets
the fields of a frozen payout in its ``__post_init__``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from .errors import CoverDefinitionError
from .parameters import (
    Free,
    Parameter,
    amount_parameter,
    check_flag,
    check_lengths,
    check_monotone,
    check_order,
    constructor_fields,
    field_from_document,
    field_to_document,
    is_parameter,
    parameter_tuple,
    required_field,
    shape_parameter,
)

if TYPE_CHECKING:
    from .payout import PayoutFunction

__all__ = [
    "field_document",
    "field_parameters",
    "payout_from_fields",
    "store_fields",
    "store_levels",
    "with_field_values",
]

_P = TypeVar("_P", bound="PayoutFunction")


def store_fields(payout: PayoutFunction, **values: Any) -> None:
    """Set the fields of a frozen payout in its ``__post_init__``.

    Parameters
    ----------
    payout : PayoutFunction
        The payout being built.
    **values
        The checked value of each field, by name.
    """
    for name, value in values.items():
        object.__setattr__(payout, name, value)


def store_levels(
    payout: PayoutFunction, tail: str, thresholds: str, minimum: int
) -> None:
    """Check and store the thresholds, levels and ``monotone`` of a payout.

    The fields of the step family (``breakpoints``) and of the
    piecewise-linear family (``knots``): the thresholds strictly increase in
    severity, there are as many levels, at least ``minimum``, and with
    ``monotone=True`` the fixed levels don't fall.

    Parameters
    ----------
    payout : PayoutFunction
        The payout being built, with the fields ``thresholds``, ``levels``
        and ``monotone`` as given.
    tail : {"upper", "lower"}
        The payout's tail, checked.
    thresholds : str
        The name of the thresholds' field, ``"breakpoints"`` or ``"knots"``.
    minimum : int
        The fewest thresholds.

    Raises
    ------
    CoverDefinitionError
        If a field is invalid.
    """
    points = parameter_tuple(
        thresholds, getattr(payout, thresholds), shape_parameter, minimum
    )
    check_order(thresholds, points, tail)
    levels = parameter_tuple(
        "levels", getattr(payout, "levels"), amount_parameter, minimum
    )
    check_lengths((thresholds, points), ("levels", levels))
    monotone = check_flag("monotone", getattr(payout, "monotone"))
    if monotone:
        check_monotone("levels", levels)
    store_fields(
        payout, tail=tail, **{thresholds: points}, levels=levels, monotone=monotone
    )


def field_parameters(payout: PayoutFunction) -> dict[str, Parameter]:
    """Return every parameter of a dataclass payout by name, in field order.

    Parameters
    ----------
    payout : PayoutFunction
        A frozen dataclass.

    Returns
    -------
    dict of str to float or Free
        As :meth:`~climada_petals.engine.parametric.payout.PayoutFunction.parameters`.
    """
    out: dict[str, Parameter] = {}
    for field_def in constructor_fields(payout):
        value = getattr(payout, field_def.name)
        if is_parameter(value):
            out[field_def.name] = value
        elif isinstance(value, tuple) and all(is_parameter(v) for v in value):
            for position, entry in enumerate(value):
                out[f"{field_def.name}[{position}]"] = entry
    return out


def with_field_values(payout: _P, values: Mapping[str, float]) -> _P:
    """Return a dataclass payout with values for some of its free parameters.

    Parameters
    ----------
    payout : PayoutFunction
        A frozen dataclass.
    values : mapping of str to float
        As in
        :meth:`~climada_petals.engine.parametric.payout.PayoutFunction.with_values`.

    Returns
    -------
    PayoutFunction
        A new payout of the same family, checked again.

    Raises
    ------
    CoverDefinitionError
        As
        :meth:`~climada_petals.engine.parametric.payout.PayoutFunction.with_values`.
    """
    if not isinstance(values, Mapping):
        raise CoverDefinitionError(
            f"values must map parameter names to numbers, got {values!r}"
        )
    current = payout.parameters()
    changes: dict[str, Any] = {}
    for name, value in values.items():
        parameter = current.get(name)
        if not isinstance(parameter, Free):
            raise CoverDefinitionError(
                f"{name!r} is not a free parameter of the {payout.kind} payout; its "
                f"free parameters are {list(payout.free_parameters())}"
            )
        number = parameter.check(name, value)
        field_name, _, position = name.partition("[")
        if position:
            entries = changes.setdefault(field_name, list(getattr(payout, field_name)))
            entries[int(position[:-1])] = number
        else:
            changes[field_name] = number
    for field_name, change in changes.items():
        if isinstance(change, list):
            changes[field_name] = tuple(change)
    return dataclasses.replace(payout, **changes)


def field_document(payout: PayoutFunction) -> dict[str, Any]:
    """Return a dataclass payout's document: ``kind`` and every field.

    Parameters
    ----------
    payout : PayoutFunction
        A frozen dataclass.

    Returns
    -------
    dict
        As :meth:`~climada_petals.engine.parametric.payout.PayoutFunction.to_dict`.
    """
    document: dict[str, Any] = {"kind": payout.kind}
    for field_def in constructor_fields(payout):
        document[field_def.name] = field_to_document(
            getattr(payout, field_def.name), field_def.name
        )
    return document


def payout_from_fields(cls: type[_P], document: Mapping[str, Any]) -> _P:
    """Read a dataclass payout of the family ``cls`` from its document.

    Parameters
    ----------
    cls : type
        A frozen dataclass family.
    document : mapping
        As in
        :meth:`~climada_petals.engine.parametric.payout.PayoutFunction.from_dict`.

    Returns
    -------
    PayoutFunction
        A payout of the family ``cls``.

    Raises
    ------
    CoverDefinitionError
        As
        :meth:`~climada_petals.engine.parametric.payout.PayoutFunction.from_dict`.
    """
    if not isinstance(document, Mapping) or document.get("kind") != cls.kind:
        raise CoverDefinitionError(
            f"a {cls.__name__} document has kind {cls.kind!r}, got {document!r}"
        )
    fields = {field_def.name: field_def for field_def in constructor_fields(cls)}
    unknown = set(document) - set(fields) - {"kind"}
    missing = {name for name, field_def in fields.items() if required_field(field_def)}
    missing -= set(document)
    if unknown or missing:
        raise CoverDefinitionError(
            f"a {cls.kind} payout document has unknown keys {sorted(unknown)} "
            f"and misses {sorted(missing)}"
        )
    kwargs = {
        name: field_from_document(document[name]) for name in fields if name in document
    }
    return cls(**kwargs)
