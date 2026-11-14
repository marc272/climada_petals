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

The samples of a scenario set, built from an index and an impact, or from a
hazard set, per member.

:meth:`~climada_petals.engine.parametric.scenarios.ScenarioSet.from_impacts`
and :meth:`~climada_petals.engine.parametric.scenarios.ScenarioSet.from_hazards`
build one
:class:`~climada_petals.engine.parametric.sample.OutcomeSample` per member
of each scenario: :func:`impact_sample` from a pair ``(index, impact)``,
:func:`hazard_sample` from a ``Hazard``, the units' exposures and the
impact functions. :func:`built_samples` runs either over the members of
every label, and notes on an error which scenario and member raised it.
Arguments given per label, a mapping by label, are read with
:func:`for_label`, and a basis given as a function of the member with
:func:`basis_for`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from climada.engine import ImpactCalc
from climada.hazard import Hazard

from ._mixture import as_sequence, labelled_items, where
from .cover import IndexDefinition
from .errors import ParametricError, ParametricInputError
from .index import ParametricIndex
from .sample import Basis, OutcomeSample
from .units import InsuredUnits

__all__ = [
    "BasisSource",
    "HazardSetup",
    "basis_for",
    "built_samples",
    "for_label",
    "hazard_sample",
    "impact_sample",
    "is_pair",
]

BasisSource = Basis | Callable[[Any], Basis]
"""A basis for every member, or a function of a member's hazard or impact."""


def for_label(value: Any, label: str, what: str) -> Any:
    """Return ``value``, or its entry for ``label`` if it is a mapping by label.

    Parameters
    ----------
    value : object
        One value for every scenario, or a mapping of labels to values.
    label : str
        The scenario.
    what : str
        The argument's name, for the error.

    Returns
    -------
    object
        ``value``, or ``value[label]``.

    Raises
    ------
    ParametricInputError
        If a mapping has no entry for ``label``.
    """
    if not isinstance(value, Mapping):
        return value
    if label not in value:
        raise ParametricInputError(f"{what} has no entry for scenario {label!r}")
    return value[label]


def basis_for(basis: BasisSource, source: Any) -> Any:
    """Return the basis, or the basis a function gives for a hazard or an impact.

    Parameters
    ----------
    basis : EventBasis, CatalogueYears, YearSets, or a function
        The basis of every member, or a function of a member's source.
    source : climada.hazard.Hazard or climada.engine.Impact
        The member's hazard set or impact.

    Returns
    -------
    EventBasis, CatalogueYears or YearSets
        The member's basis.
    """
    if callable(basis):
        return basis(source)
    return basis


def is_pair(value: object) -> bool:
    """Whether ``value`` is an index and an impact, as a tuple.

    Parameters
    ----------
    value : object
        A member, or a label's sequence of members.

    Returns
    -------
    bool
        True for a tuple of two whose first entry is a
        :class:`~climada_petals.engine.parametric.index.ParametricIndex`.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], ParametricIndex)
    )


@dataclass(frozen=True)
class HazardSetup:
    """The arguments of ``ScenarioSet.from_hazards`` but the hazards.

    Parameters
    ----------
    definition : IndexDefinition
        The index a cover pays on.
    units : InsuredUnits or mapping of str to InsuredUnits
        The insured units, or each label's.
    impact_funcs : climada.entity.ImpactFuncSet or mapping of str to it
        The impact functions, or each label's.
    basis : EventBasis, CatalogueYears, YearSets, or a function
        The basis of every sample, or a function of a hazard set.
    index_units : InsuredUnits or None
        The units the index is computed on, or None for each scenario's
        own.
    """

    definition: IndexDefinition
    units: Any
    impact_funcs: Any
    basis: BasisSource
    index_units: InsuredUnits | None


def hazard_sample(hazard: Any, label: str, *, setup: HazardSetup) -> OutcomeSample:
    """Return the sample of one hazard set: its index, its impact and its basis.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        The member's hazard set.
    label : str
        The scenario.
    setup : HazardSetup
        The other arguments of ``ScenarioSet.from_hazards``.

    Returns
    -------
    OutcomeSample
        With ``scenario`` the label.

    Raises
    ------
    ParametricInputError
        If ``hazard`` is not a ``Hazard``, or ``index_units`` has other unit
        ids than the scenario's units.
    """
    if not isinstance(hazard, Hazard):
        raise ParametricInputError(
            f"a member is a climada Hazard, got {type(hazard).__name__}"
        )
    units = for_label(setup.units, label, "units")
    index_units = units if setup.index_units is None else setup.index_units
    if tuple(index_units.unit_id) != tuple(units.unit_id):
        raise ParametricInputError(
            f"index_units has the units {list(index_units.unit_id)[:5]} and the "
            f"scenario {list(units.unit_id)[:5]}: they must be the same unit ids, in "
            "order"
        )
    index = ParametricIndex.from_definition(setup.definition, hazard, index_units)
    impact = ImpactCalc(
        units.exposures.copy(),
        for_label(setup.impact_funcs, label, "impact_funcs"),
        hazard,
    ).impact(save_mat=True)
    return OutcomeSample.from_impact(
        index,
        impact,
        units=units,
        basis=basis_for(setup.basis, hazard),
        scenario=label,
    )


def impact_sample(
    pair: Any, label: str, *, units: Any, basis: BasisSource
) -> OutcomeSample:
    """Return the sample of one index and impact.

    Parameters
    ----------
    pair : tuple of ParametricIndex and climada.engine.Impact
        The member.
    label : str
        The scenario.
    units : InsuredUnits or mapping of str to InsuredUnits
        The insured units, or each label's.
    basis : EventBasis, CatalogueYears, YearSets, or a function
        The basis of every sample, or a function of an impact.

    Returns
    -------
    OutcomeSample
        With ``scenario`` the label.

    Raises
    ------
    ParametricInputError
        If ``pair`` is not a tuple of an index and an impact.
    """
    if not is_pair(pair):
        raise ParametricInputError(
            "a member is a tuple (index, impact), with a ParametricIndex first; "
            f"got {type(pair).__name__}"
        )
    return OutcomeSample.from_impact(
        pair[0],
        pair[1],
        units=for_label(units, label, "units"),
        basis=basis_for(basis, pair[1]),
        scenario=label,
    )


def built_samples(
    members: object,
    is_one: Callable[[object], bool],
    build: Callable[[Any, str], OutcomeSample],
    method: str,
) -> dict[str, list[OutcomeSample]]:
    """Return each label's samples, ``build(member, label)`` for each of its members.

    A label maps to one member (``is_one``) or a non-empty sequence of them.
    An error has a note naming the method, the scenario and the member.

    Parameters
    ----------
    members : mapping of str to object
        Each label to its member or members.
    is_one : callable
        Whether a label's value is one member.
    build : callable
        Returns the sample of a member and its label.
    method : str
        The ``ScenarioSet`` method, for the note.

    Returns
    -------
    dict of str to list of OutcomeSample
        Each label's samples, in order.

    Raises
    ------
    ParametricInputError
        If ``members`` is not a non-empty mapping of labels to members or
        non-empty sequences of them.
    ParametricError, ValueError
        As ``build``, with the note.
    """
    samples: dict[str, list[OutcomeSample]] = {}
    for label, value in labelled_items(members):
        group = (value,) if is_one(value) else as_sequence(value, f"scenario {label!r}")
        for position, member in enumerate(group):
            try:
                samples.setdefault(label, []).append(build(member, label))
            except (ParametricError, ValueError) as error:
                error.add_note(
                    f"raised by ScenarioSet.{method} at "
                    + where(label, position, len(group))
                )
                raise
    return samples
