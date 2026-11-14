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

Mixtures: several outcome samples that form one scenario, as one sample;
and the checks that the samples of a scenario set fit together.

A scenario of a
:class:`~climada_petals.engine.parametric.scenarios.ScenarioSet` can hold
several samples, its **members**: the windows of a pathway (one hazard set
per reference year) or the members of an ensemble (one per climate model).
Together they are one scenario, an **equal-weight mixture**, in which each
of the ``m`` members has the weight ``1 / m``. :func:`mixture_sample`
returns it as one
:class:`~climada_petals.engine.parametric.sample.OutcomeSample`, so that the
evaluation and the calibration read it as they read any sample:

==================  =========================================================
Route               The mixture
==================  =========================================================
``EventBasis``      The members' events, each with its frequency divided by
                    ``m``
``CatalogueYears``  The members' years, a year of member ``i`` of weight
                    ``1 / (m N_i)``, with ``N_i`` the member's years
                    (:class:`MixtureCatalogue`); the years are interleaved
                    in the order of their position in their member, so the
                    first years of the mixture are the first of every
                    member. An event's frequency is its year's weight
``YearSets``        The members' sampled years, a year of member ``i`` of
                    weight ``1 / (m N_i)`` (:class:`MixtureYears`); an
                    event's frequency is divided by ``m``
==================  =========================================================

So an expectation is the mean of the members' expectations, and a VaR or a
CVaR is that of the mixed distribution: on the annual basis a year of the
mixture is a year of one member.

The members are different hazard sets. As
:func:`~climada_petals.engine.parametric.sample.check_no_leakage` treats
two such sets, their event ids and blocks say nothing about each other, so
a mixture keeps them apart:

* **Events** are numbered ``-(m |e| + i + 1)`` for the event ``e`` of
  member ``i`` (from 0): negative, so that they don't coincide with the
  positive ids that CLIMADA's hazard sets use, and the same for the same
  event of the same member in two mixtures of ``m`` members, such as the
  training and the test set of
  :meth:`~climada_petals.engine.parametric.scenarios.ScenarioSet.holdout`.
  An event's name is ``"<i>:<name>"``, with the event's name in its member
  (its id where the member has no names).
* **Blocks** are the members' own, labelled ``"<i>:<block>"``: the given
  labels (an ensemble's parent storm, say), else the year on catalogue
  years and the event id on the other routes. So the same storm in two
  members, such as two windows of a pathway, counts as two blocks.
* **Draws**: the members have the same number of draws, and the rows are
  ordered by draw, then event, as in any sample.
* **Source**: the mixture's source token is that of its members' tokens,
  in order (:func:`~climada_petals.engine.parametric._parts.mixture_token`).
  The mixture of the whole members and the mixtures of their sides number
  their events alike, and have one token, so that an evaluation compares
  them (:func:`~climada_petals.engine.parametric.sample.check_out_of_sample`).

**Splits.** On an annual route a split of a mixture would not mix its
members' sides: on year sets each side would sample its years from the
pooled frequencies of every member, and on catalogue years each member
would weigh by its number of years on the side. So the mixture's bases
(:attr:`MixtureYears.split_refusal`, :attr:`MixtureCatalogue.split_refusal`)
make ``holdout``, ``year_block_kfold`` and ``select`` raise
:class:`~climada_petals.engine.parametric.errors.ScenarioError`;
:meth:`~climada_petals.engine.parametric.scenarios.ScenarioSet.holdout`
splits the members instead. On the event basis a mixture is one set of
events with frequencies, and splits it as any other: each side keeps its
events, with their frequencies divided by the side's share.

:func:`scenario_members` checks a set's samples before any mixture is
built: every sample of a set has the same basis and route, unit ids and
index definition, and the members of one scenario have the same units,
exposures included, the same number of draws, and come from the same side
of one split, or none.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from ._checks import finite_array, read_only
from ._parts import mixture_token
from .annual import CatalogueYears, YearSets
from .errors import ParametricInputError, ScenarioError, YearRouteError
from .index import ParametricIndex
from .measure import (
    ANNUAL,
    CATALOGUE,
    EVENT,
    FREQUENCY_ATOL,
    PROBABILITY_ATOL,
    YEARSETS,
    EventBasis,
    OutcomeMeasure,
    _events,
    _positions_of,
)
from .sample import OutcomeSample
from .units import InsuredUnits

__all__ = [
    "SPLIT_REFUSAL",
    "MixtureCatalogue",
    "MixtureYears",
    "as_sequence",
    "labelled_items",
    "mixture_sample",
    "scenario_members",
    "weighted_annual",
    "where",
]

_FREQUENCY_RTOL = 1e-9
"""Relative tolerance of a sample's frequencies against the year sets'."""

SPLIT_REFUSAL: Final[str] = (
    "this sample mixes the members of a scenario on an annual route, and a split "
    "of it would not mix the members' sides: on year sets each side would sample "
    "its years from the pooled frequencies of every member, so a year would hold "
    "the events of several members, and on catalogue years each member would "
    "weigh by its number of years on the side. Split the members instead: "
    "ScenarioSet.holdout splits each member and mixes their sides"
)
"""Why ``holdout``, ``year_block_kfold`` and ``select`` refuse a mixture on an annual route."""


def where(label: str, position: int, size: int) -> str:
    """Return the scenario and, in a mixture, the member, for a message."""
    return f"scenario {label!r}" + (f", member {position}" if size > 1 else "")


def as_sequence(value: object, what: str) -> tuple[Any, ...]:
    """Return a non-empty sequence as a tuple.

    Raises
    ------
    ParametricInputError
        If ``value`` is a string, not a sequence, or empty.
    """
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise ParametricInputError(
            f"{what} must be one item or a non-empty sequence of them, got "
            f"{type(value).__name__}"
        )
    return tuple(value)


def labelled_items(members: object, what: str = "members") -> list[tuple[str, Any]]:
    """Return the items of a non-empty mapping with non-empty string labels.

    Raises
    ------
    ParametricInputError
        If ``members`` (named ``what`` in the message) is not such a mapping.
    """
    if not isinstance(members, Mapping) or not members:
        raise ParametricInputError(
            f"{what} must be a non-empty mapping with scenario labels as keys, "
            f"got {type(members).__name__}"
        )
    bad = [label for label in members if not isinstance(label, str) or not label]
    if bad:
        raise ParametricInputError(f"labels must be non-empty strings, got {bad[:5]}")
    return list(members.items())


def _labelled(sample: OutcomeSample, label: str) -> OutcomeSample:
    """Return the sample with ``label`` as its ``scenario``."""
    if sample.scenario == label:
        return sample
    return dataclasses.replace(sample, scenario=label)


def _same_units(first: InsuredUnits, other: InsuredUnits) -> bool:
    """Whether two units hold the same points, values and grouping."""
    if first is other:
        return True
    return (
        tuple(first.unit_id) == tuple(other.unit_id)
        and first.n_points == other.n_points
        and np.allclose(first.coords, other.coords, rtol=0.0, atol=1e-9)
        and np.array_equal(first.value, other.value)
        and (first.membership != other.membership).nnz == 0
    )


def _check_alike(found: Mapping[str, tuple[OutcomeSample, ...]]) -> None:
    """Raise unless every sample has the first's route, units and index definition.

    Raises
    ------
    ScenarioError
        If a sample differs.
    """
    first_label = next(iter(found))
    first = found[first_label][0]
    for label, group in found.items():
        for position, sample in enumerate(group):
            place = where(label, position, len(group))
            if (sample.kind, sample.route) != (first.kind, first.route):
                raise ScenarioError(
                    f"{place} is on the {sample.kind} basis (route {sample.route}) "
                    f"and scenario {first_label!r} on the {first.kind} basis (route "
                    f"{first.route}): the scenarios of a comparison take one route, "
                    "since the route changes tail numbers. A frequency-scaled set "
                    "takes the event basis or YearSets, as its baseline then must"
                )
            if sample.unit_id != first.unit_id:
                raise ScenarioError(
                    f"{place} has the units {list(sample.unit_id)[:5]} and scenario "
                    f"{first_label!r} {list(first.unit_id)[:5]}: one cover pays on "
                    "every scenario, so they share the unit ids, in order"
                )
            if sample.index.definition != first.index.definition:
                raise ScenarioError(
                    f"{place} has the index {sample.index.definition!r} and scenario "
                    f"{first_label!r} {first.index.definition!r}: one cover pays on "
                    "every scenario, so they share the index definition. Rebuild a "
                    "cover's index on a hazard set with "
                    "ParametricIndex.from_definition"
                )


def _split_key(sample: OutcomeSample) -> tuple[str, str, int, int] | None:
    """Return what identifies a sample's split, or None for a whole sample."""
    if sample.split is None:
        return None
    record = sample.split
    return (record.method, record.side, record.fold, record.n_folds)


def _check_members(label: str, group: tuple[OutcomeSample, ...]) -> None:
    """Raise unless the members of a mixture fit together (module docstring).

    Raises
    ------
    ScenarioError
        If two members have other exposures, other numbers of draws, or
        come from other splits.
    """
    first = group[0]
    for position, member in enumerate(group[1:], start=1):
        place = f"member {position} of scenario {label!r}"
        if not _same_units(first.units, member.units):
            raise ScenarioError(
                f"{place} has other units than member 0: the members of one "
                "scenario share the exposure points, their values and the units"
            )
        if member.n_draws != first.n_draws:
            raise ScenarioError(
                f"{place} has {member.n_draws} impact-function draws and member 0 "
                f"{first.n_draws}: the members of one scenario have the same draws"
            )
        if _split_key(member) != _split_key(first):
            raise ScenarioError(
                f"{place} comes from {_split_key(member)} and member 0 from "
                f"{_split_key(first)} (split method, side, fold, folds): the members "
                "of one scenario are all whole samples, or all one side of one split"
            )


def scenario_members(members: object) -> dict[str, tuple[OutcomeSample, ...]]:
    """Return a set's samples by label, each labelled, after checking that they fit.

    Parameters
    ----------
    members : mapping of str to OutcomeSample or sequence of OutcomeSample
        As :class:`~climada_petals.engine.parametric.scenarios.ScenarioSet`
        takes it.

    Returns
    -------
    dict of str to tuple of OutcomeSample
        Each sample with its label as ``scenario``.

    Raises
    ------
    ScenarioError
        If two samples are on other routes, units or index definitions, or
        two members of one scenario have other exposures, draws or splits.
    ParametricInputError
        If ``members`` is not a non-empty mapping of non-empty string
        labels to samples or non-empty sequences of them.
    """
    found: dict[str, tuple[OutcomeSample, ...]] = {}
    for label, value in labelled_items(members):
        group = (
            (value,)
            if isinstance(value, OutcomeSample)
            else as_sequence(value, f"scenario {label!r}")
        )
        if not all(isinstance(sample, OutcomeSample) for sample in group):
            raise ParametricInputError(
                f"scenario {label!r} must be an OutcomeSample or a sequence of them"
            )
        found[label] = tuple(_labelled(sample, label) for sample in group)
    _check_alike(found)
    for label, group in found.items():
        _check_members(label, group)
    return found


def weighted_annual(
    counts: sp.csr_matrix, weights: NDArray[np.float64], n_draws: int, route: str
) -> OutcomeMeasure:
    """Return the annual measure of ``counts`` (years x events) with each year's weight.

    Each year and draw has the weight ``weights[year] / n_draws``; the years
    without events are pooled into one last outcome with their summed
    weight. With equal weights this is the measure of the annual routes.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        The count of each event in each year.
    weights : numpy.ndarray of float
        One weight per year, summing to 1.
    n_draws : int
        Draws per event; rows are ordered by draw, then event.
    route : {"catalogue", "yearsets"}
        The route the measure records.

    Returns
    -------
    OutcomeMeasure
    """
    occupied = np.diff(counts.indptr) > 0
    grouping = sp.block_diag([counts[occupied]] * n_draws, format="csr")
    outcome_weight = np.tile(weights[occupied], n_draws) / n_draws
    if not occupied.all():
        grouping = sp.vstack(
            [grouping, sp.csr_matrix((1, grouping.shape[1]))], format="csr"
        )
        outcome_weight = np.append(outcome_weight, float(weights[~occupied].sum()))
    return OutcomeMeasure(grouping, outcome_weight, ANNUAL, route, counts.shape[0])


def _year_weights(weights: ArrayLike | None, n_years: int) -> NDArray[np.float64]:
    """Return one weight per year, >= 0 and summing to 1, read-only.

    Raises
    ------
    ParametricInputError
        If there is not one such weight per year.
    """
    if weights is None:
        raise ParametricInputError("year_weight must hold one weight per year")
    found = finite_array(weights, "year_weight", 1)
    if (
        found.shape != (n_years,)
        or np.any(found < 0)
        or abs(float(found.sum()) - 1.0) > PROBABILITY_ATOL
    ):
        raise ParametricInputError(
            f"year_weight must hold one weight >= 0 per year ({n_years}), summing "
            f"to 1; got shape {found.shape} and sum {float(found.sum())!r}"
        )
    return read_only(found)


@dataclass(frozen=True, eq=False)
class MixtureYears(YearSets):
    """Year sets of a mixture: each sampled year with its own weight.

    Parameters
    ----------
    event_id, sampling_vect, block, frequency
        As :class:`~climada_petals.engine.parametric.annual.YearSets`'.
    year_weight : array_like of float
        One weight per sampled year, >= 0, summing to 1: ``1 / (m N_i)``
        for a year of member ``i`` (module docstring).

    Raises
    ------
    ParametricInputError
        If an argument is invalid.
    """

    year_weight: NDArray[np.float64] | None = field(default=None, kw_only=True)
    split_refusal: ClassVar[str] = SPLIT_REFUSAL

    def __post_init__(self) -> None:
        """Validate the years as year sets do, and the weights."""
        super().__post_init__()
        object.__setattr__(
            self, "year_weight", _year_weights(self.year_weight, self.n_years)
        )

    def measure(
        self, event_id: ArrayLike, frequency: ArrayLike, *, n_draws: int = 1
    ) -> OutcomeMeasure:
        """Return the measure of the given events over the years, with their weights.

        As :meth:`YearSets.measure`, with ``year_weight`` in place of
        ``1 / N``.

        Raises
        ------
        YearRouteError
            If the frequencies are not the year sets'.
        EventAlignmentError
            If an event is not one of the year sets'.
        ParametricInputError
            If there is no event or a frequency is invalid.
        """
        ids, freq, draws = _events(event_id, frequency, n_draws)
        rows = _positions_of(self.event_id, ids, "sampled years")
        given = self.frequency
        if given is not None and not np.allclose(
            freq, given[rows], rtol=_FREQUENCY_RTOL, atol=0.0
        ):
            raise YearRouteError(
                "the sample's frequencies are not those of the mixture's members"
            )
        return weighted_annual(
            self._counts(rows), np.asarray(self.year_weight), draws, YEARSETS
        )

    def _counts(self, rows: NDArray[np.intp]) -> sp.csr_matrix:
        """Return the count of each given event (its position, ``rows``) in each year.

        The positions of other events are left out of each year.
        """
        place = np.full(self.event_id.size, -1, dtype=np.int64)
        place[rows] = np.arange(rows.size)
        found = place[self._positions]
        sampled = found >= 0
        of_year = np.repeat(np.arange(self.n_years), np.diff(self._offsets))
        return sp.coo_matrix(
            (np.ones(int(sampled.sum())), (of_year[sampled], found[sampled])),
            shape=(self.n_years, rows.size),
        ).tocsr()


def _year_counts(chosen: NDArray[np.int64], year: NDArray[np.int64]) -> sp.csr_matrix:
    """Return the count of each event (its year, ``year``) in each of ``chosen``.

    Raises
    ------
    ParametricInputError
        If an event's year is not among ``chosen``.
    """
    column = pd.Index(chosen).get_indexer(year)
    if np.any(column < 0):
        raise ParametricInputError(
            f"{int((column < 0).sum())} events lie outside the given years"
        )
    return sp.csr_matrix(
        (np.ones(year.size), (column, np.arange(year.size))),
        shape=(chosen.size, year.size),
    )


@dataclass(frozen=True, eq=False)
class MixtureCatalogue(CatalogueYears):
    """Catalogue years of a mixture: each simulation year with its own weight.

    Parameters
    ----------
    event_id, year, n_years, first_year, block, source
        As :class:`~climada_petals.engine.parametric.annual.CatalogueYears`'.
    year_weight : array_like of float
        One weight per simulation year, >= 0, summing to 1: ``1 / (m N_i)``
        for a year of member ``i`` (module docstring). An event's frequency
        must be its year's weight.

    Raises
    ------
    YearRouteError
        If a year is outside the simulation years.
    ParametricInputError
        If an argument is invalid.
    """

    year_weight: NDArray[np.float64] | None = field(default=None, kw_only=True)
    split_refusal: ClassVar[str] = SPLIT_REFUSAL

    def __post_init__(self) -> None:
        """Validate the labels as catalogue years do, and the weights."""
        super().__post_init__()
        object.__setattr__(
            self, "year_weight", _year_weights(self.year_weight, self.n_years)
        )

    def measure(
        self,
        event_id: ArrayLike,
        frequency: ArrayLike,
        *,
        n_draws: int = 1,
        years: ArrayLike | None = None,
    ) -> OutcomeMeasure:
        """Return the measure of the given events over the years, with their weights.

        As :meth:`CatalogueYears.measure`, with ``year_weight`` in place of
        ``1 / N``; on a part of the years (a side of a split), their weights
        divided by their sum.

        Raises
        ------
        YearRouteError
            If an event's frequency is not its year's weight.
        EventAlignmentError
            If an event has no year.
        ParametricInputError
            If there is no event, ``years`` is invalid, or an event's year is
            not among ``years``.
        """
        ids, freq, draws = _events(event_id, frequency, n_draws)
        year = self.year_of(ids)
        weight = np.asarray(self.year_weight)
        wrong = np.abs(freq / weight[year - self.first_year] - 1.0) > FREQUENCY_ATOL
        if wrong.any():
            raise YearRouteError(
                f"{int(wrong.sum())} of the sample's {ids.size} events have a "
                "frequency that is not their year's weight in the mixture"
            )
        chosen = self.years if years is None else self.check_years(years)
        part = weight[chosen - self.first_year]
        return weighted_annual(
            _year_counts(chosen, year), part / part.sum(), draws, CATALOGUE
        )


def _blocks(members: Sequence[OutcomeSample]) -> NDArray[Any]:
    """Return the members' blocks as ``"<i>:<block>"``, one per event, in order."""
    return np.array(
        [
            f"{position}:{block}"
            for position, member in enumerate(members)
            for block in np.asarray(member.block).tolist()
        ],
        dtype=object,
    )


def _event_ids(members: Sequence[OutcomeSample]) -> NDArray[np.int64]:
    """Return the mixture's event ids, ``-(m |e| + i + 1)`` (module docstring)."""
    count = len(members)
    return np.concatenate(
        [
            -(count * np.abs(member.event_id) + position + 1)
            for position, member in enumerate(members)
        ]
    ).astype(np.int64)


def _index(
    members: Sequence[OutcomeSample], event_id: NDArray[np.int64]
) -> ParametricIndex:
    """Return the mixture's index: the members' rows, renumbered and named."""
    first = members[0]
    names = [
        f"{position}:{name}"
        for position, member in enumerate(members)
        for name in (
            member.index.event_name
            if member.index.event_name is not None
            else member.event_id.tolist()
        )
    ]
    return ParametricIndex(
        np.vstack([member.index.values for member in members]),
        event_id,
        first.unit_id,
        first.index.definition,
        event_name=names,
    )


def _loss(members: Sequence[OutcomeSample]) -> sp.csr_matrix:
    """Return the mixture's loss: per draw, the members' events in order."""
    return sp.vstack(
        [
            member.loss[draw * member.n_events : (draw + 1) * member.n_events]
            for draw in range(members[0].n_draws)
            for member in members
        ],
        format="csr",
    )


def _own_weights(member: OutcomeSample) -> NDArray[np.float64]:
    """Return the weight of each year of an annual member, in its order.

    ``1 / N`` per year, or the year weights of a member that is a mixture
    itself, on the member's years and divided by their sum.
    """
    basis = member.basis
    if isinstance(basis, MixtureYears):
        return np.asarray(basis.year_weight)
    if isinstance(basis, MixtureCatalogue):
        own = np.asarray(basis.year_weight)[np.asarray(member.years) - basis.first_year]
        return own / own.sum()
    size = (
        basis.n_years if isinstance(basis, YearSets) else np.asarray(member.years).size
    )
    return np.full(size, 1.0 / size)


def _member_years(member: OutcomeSample, offset: int) -> list[NDArray[np.int64]]:
    """Return a year-sets member's years, as positions of its events in the mixture."""
    basis = member.basis
    column = np.full(basis.event_id.size, -1, dtype=np.int64)
    column[pd.Index(basis.event_id).get_indexer(member.event_id)] = offset + np.arange(
        member.n_events
    )
    years = []
    for year in basis.sampling_vect:
        drawn = column[np.asarray(year)]
        years.append(drawn[drawn >= 0])
    return years


def _year_sets(
    members: Sequence[OutcomeSample],
    event_id: NDArray[np.int64],
    block: NDArray[Any],
) -> tuple[MixtureYears, NDArray[np.float64]]:
    """Return the year sets of a mixture and its events' frequencies."""
    count = len(members)
    offsets = np.cumsum([0, *(member.n_events for member in members)])
    years: list[NDArray[np.int64]] = []
    for position, member in enumerate(members):
        years.extend(_member_years(member, int(offsets[position])))
    frequency = np.concatenate([member.frequency for member in members]) / count
    basis = MixtureYears(
        event_id,
        years,
        block=block,
        frequency=frequency,
        year_weight=np.concatenate([_own_weights(member) for member in members])
        / count,
    )
    return basis, frequency


def _catalogue(
    members: Sequence[OutcomeSample],
    event_id: NDArray[np.int64],
    block: NDArray[Any],
) -> tuple[MixtureCatalogue, NDArray[np.float64]]:
    """Return the catalogue years of a mixture and its events' frequencies.

    The ``k``-th of a member's ``N_i`` years goes to the place of ``(k + 1/2)
    / N_i`` among all years, ties in the members' order. An event's frequency
    is its year's weight.
    """
    sizes = np.array([np.asarray(member.years).size for member in members])
    member_of = np.repeat(np.arange(len(members)), sizes)
    position = np.concatenate([np.arange(size) for size in sizes])
    order = np.lexsort((member_of, (position + 0.5) / sizes[member_of]))
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    weight = np.empty(order.size)
    weight[rank] = np.concatenate([_own_weights(member) for member in members])
    weight /= len(members)
    starts = np.cumsum(sizes) - sizes
    year = np.concatenate(
        [
            rank[
                starts[k] + np.searchsorted(np.asarray(member.years), member.event_year)
            ]
            for k, member in enumerate(members)
        ]
    )
    basis = MixtureCatalogue(
        event_id,
        year,
        n_years=int(order.size),
        block=block,
        source="a mixture of catalogue years",
        year_weight=weight,
    )
    return basis, weight[year]


def mixture_sample(label: str, members: Sequence[OutcomeSample]) -> OutcomeSample:
    """Return the equal-weight mixture of several samples as one sample.

    Parameters
    ----------
    label : str
        The scenario's label, the sample's ``scenario``.
    members : sequence of OutcomeSample
        Two or more samples with the same units, index definition, basis,
        route and number of draws.

    Returns
    -------
    OutcomeSample
        A whole sample (``split`` None) of the members' events (module
        docstring), with the sum of their ``n_left_out`` and the token of
        their source tokens.
    """
    first = members[0]
    event_id = _event_ids(members)
    block = _blocks(members)
    basis: EventBasis | MixtureYears | MixtureCatalogue
    if first.kind == EVENT:
        frequency = np.concatenate([member.frequency for member in members])
        frequency = frequency / len(members)
        basis = EventBasis(event_id=event_id, block=block)
    elif first.route == YEARSETS:
        basis, frequency = _year_sets(members, event_id, block)
    else:
        basis, frequency = _catalogue(members, event_id, block)
    return OutcomeSample(
        _index(members, event_id),
        _loss(members),
        frequency,
        first.units,
        basis,
        n_draws=first.n_draws,
        scenario=label,
        n_left_out=sum(member.n_left_out for member in members),
        source=mixture_token([str(member.source) for member in members]),
    )
