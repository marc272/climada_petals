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

The annual basis: catalogue years and year sets.

One outcome per simulated year, each of weight ``1 / N``, on a route the
caller states (:mod:`~climada_petals.engine.parametric.measure` says what the
measure is and which route suits which hazard):

* :class:`CatalogueYears`: a year label on each event, from the dates
  (:meth:`~CatalogueYears.from_dates`), the event names
  (:meth:`~CatalogueYears.from_event_names`) or an array
  (:meth:`~CatalogueYears.from_labels`). Every event a sample keeps must
  have the frequency ``1 / N``; otherwise
  :class:`~climada_petals.engine.parametric.errors.YearRouteError`.
* :class:`YearSets`: years of Poisson draws from the frequencies, with
  replacement and the caller's ``numpy.random.Generator``
  (:meth:`~YearSets.from_frequencies`, :func:`sample_years`), or a
  ``sampling_vect`` of ``climada.util.yearsets``
  (:meth:`~YearSets.from_sampling_vect`). Any frequencies work, so a
  climate-scaled set does.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.util.constants import DEF_FREQ_UNIT
from climada.util.yearsets import sample_events
from numpy.typing import ArrayLike, NDArray

from ._checks import check_int, finite_array, integer_array, read_only
from .errors import ImpactInputError, ParametricInputError, YearRouteError
from .measure import (
    ANNUAL,
    CATALOGUE,
    FREQUENCY_ATOL,
    YEARSETS,
    OutcomeMeasure,
    _annual,
    _event_ids,
    _events,
    _labels,
    _positions_of,
    labels_from_names,
)

__all__ = ["CatalogueYears", "YearSets", "sample_years"]

_LAST_ORDINAL: Final[int] = dt.date.max.toordinal()
"""Ordinal of 31 December 9999, the last date that ``datetime`` holds."""

_FREQUENCY_RTOL: Final[float] = 1e-9


@dataclass(frozen=True, eq=False)
class CatalogueYears:
    """Catalogue years: the annual route on a label per event (module docstring).

    Use :meth:`from_dates`, :meth:`from_event_names` or :meth:`from_labels`;
    the constructor is :meth:`from_labels`'s.

    Parameters
    ----------
    event_id : array_like of int
        The labelled events, unique.
    year : array_like of int
        Each event's simulation year.
    n_years : int
        ``N``, the number of simulation years, including years without
        events.
    first_year : int, default 0
        The first simulation year: the years are ``first_year`` to
        ``first_year + N - 1``.
    block : array_like, optional
        One label per event, the unit a split keeps together, e.g. a
        member's parent storm. Default: the year.
    source : str, default "labels"
        Where the years were read, for the messages.

    Raises
    ------
    YearRouteError
        If a year is outside the simulation years.
    ParametricInputError
        If an argument is invalid.
    """

    event_id: NDArray[np.int64]
    year: NDArray[np.int64]
    n_years: int = field(kw_only=True)
    first_year: int = field(default=0, kw_only=True)
    block: NDArray[Any] | None = field(default=None, kw_only=True)
    source: str = field(default="labels", kw_only=True)

    def __post_init__(self) -> None:
        """Validate the labels and store read-only copies."""
        ids = _event_ids(self.event_id)
        year = integer_array(self.year, "year")
        if year.size != ids.size:
            raise ParametricInputError(
                f"year has {year.size} entries for {ids.size} events"
            )
        count = check_int("n_years", self.n_years, low=1)
        first = check_int("first_year", self.first_year)
        outside = np.unique(year[(year < first) | (year >= first + count)])
        if outside.size:
            raise YearRouteError(
                f"event years outside the {count} simulation years {first} to "
                f"{first + count - 1} ({self.source}), e.g. {outside[:5].tolist()}"
            )
        object.__setattr__(self, "event_id", ids)
        object.__setattr__(self, "year", read_only(year))
        object.__setattr__(self, "n_years", count)
        object.__setattr__(self, "first_year", first)
        if self.block is not None:
            object.__setattr__(self, "block", _labels(self.block, ids.size, "block"))

    @classmethod
    def from_labels(
        cls,
        event_id: ArrayLike,
        year: ArrayLike,
        *,
        n_years: int,
        first_year: int = 0,
        block: ArrayLike | None = None,
    ) -> CatalogueYears:
        """Catalogue years from one year per event, given as an array.

        This is the route for every labelling the dates and names don't
        give: STORM's ``k x 1000 + y`` from its names, or the members of an
        ensemble. Number the member-years of one parent next to each other,
        e.g. ``n_members x (calendar_year - first) + member``, with the
        parent as ``block``: a split then keeps every member with its parent
        and still cuts the years into contiguous blocks.

        Parameters
        ----------
        event_id, year, n_years, first_year, block
            As the class's.

        Returns
        -------
        CatalogueYears
        """
        return cls(
            np.asarray(event_id),
            np.asarray(year),
            n_years=n_years,
            first_year=first_year,
            block=None if block is None else np.asarray(block),
        )

    @classmethod
    def from_dates(
        cls,
        hazard: Any,
        *,
        season_start_month: int = 1,
        n_years: int | None = None,
        first_year: int | None = None,
    ) -> CatalogueYears:
        """Catalogue years from the events' dates: the calendar year or season.

        Every event of ``hazard`` needs a date. Without ``n_years``, ``N`` is
        the span of the dates, so a set whose dates don't carry its years
        (all tracks on one date, or members on their parent's date) raises
        when its sample is built: its frequencies imply another ``N``.

        Parameters
        ----------
        hazard : climada.hazard.Hazard or climada.engine.Impact
            Its ``event_id`` and ``date`` (proleptic ordinals).
        season_start_month : int, default 1
            The month a season starts in, 1 to 12. A date before it belongs
            to the season that started the year before, e.g. ``7`` for
            windstorm seasons from July to June. Default: calendar years.
        n_years : int, optional
            ``N``, if the record is known to be longer than the span of its
            dates. Default: the span.
        first_year : int, optional
            The first simulation year, needed when ``n_years`` is longer
            than the span: where the years without events lie isn't in the
            dates. Default: the first dated year.

        Returns
        -------
        CatalogueYears

        Raises
        ------
        YearRouteError
            If a date is missing or invalid (a date holds the years 1 to
            9,999), or ``n_years`` and ``first_year`` don't hold the dates'
            span.
        ParametricInputError
            If the hazard has no events, or an argument is invalid.
        """
        ids = _event_ids(hazard.event_id)
        month = check_int("season_start_month", season_start_month, low=1)
        if month > 12:
            raise ParametricInputError(
                f"season_start_month must be 1 to 12, got {month}"
            )
        season = _seasons(hazard.date, ids.size, month)
        what = "calendar years" if month == 1 else f"seasons from month {month}"
        return cls._spanned(ids, season, (n_years, first_year), f"the dates' {what}")

    @classmethod
    def from_event_names(
        cls,
        hazard: Any,
        pattern: str,
        *,
        n_years: int | None = None,
        first_year: int | None = None,
    ) -> CatalogueYears:
        """Catalogue years read from the event names by a regular expression.

        Each name must match ``pattern`` as a whole. Its group ``year``
        gives the event's year, and an optional group ``block`` its block,
        e.g. ``r"NA_(?P<year>[0-9]+)_[0-9]+"`` for IRIS's ``NA_9999_0``.

        Parameters
        ----------
        hazard : climada.hazard.Hazard or climada.engine.Impact
            Its ``event_id`` and ``event_name``.
        pattern : str
            The regular expression, with a named group ``year``.
        n_years, first_year : int, optional
            As in :meth:`from_dates`. Default: the span of the years read.

        Returns
        -------
        CatalogueYears

        Raises
        ------
        YearRouteError
            If a name doesn't match, its year isn't a whole number, or
            ``n_years`` and ``first_year`` don't hold the span.
        ParametricInputError
            If the pattern has no group ``year``, or the hazard has no
            events.
        """
        regex = re.compile(pattern)
        if "year" not in regex.groupindex:
            raise ParametricInputError(
                f"pattern must have a named group 'year', got {pattern!r}"
            )
        ids = _event_ids(hazard.event_id)
        names = [str(name) for name in hazard.event_name]
        groups = ["year", *(["block"] if "block" in regex.groupindex else [])]
        found = labels_from_names(names, pattern, groups)
        try:
            year = np.array([int(text) for text in found["year"]], dtype=np.int64)
        except ValueError as exc:
            raise YearRouteError(
                f"the group 'year' of the event names is not a whole number: {exc}"
            ) from exc
        if year.size != ids.size:
            raise ParametricInputError(f"{year.size} event names for {ids.size} events")
        return cls._spanned(
            ids,
            year,
            (n_years, first_year),
            "the event names",
            block=found.get("block"),
        )

    @classmethod
    def _spanned(
        cls,
        ids: NDArray[np.int64],
        year: NDArray[np.int64],
        stated: tuple[int | None, int | None],
        source: str,
        *,
        block: Sequence[str] | None = None,
    ) -> CatalogueYears:
        """Return the years of ``year``'s span, or of the stated ``N`` and first year.

        Raises
        ------
        YearRouteError
            If ``N`` is longer than the span and no first year is stated, or
            the stated years don't hold the span.
        ParametricInputError
            If there are no events.
        """
        if ids.size == 0:
            raise ParametricInputError("the hazard has no events")
        low, high = int(year.min()), int(year.max())
        span = high - low + 1
        n_years, first_year = stated
        count = span if n_years is None else check_int("n_years", n_years, low=1)
        if count < span:
            raise YearRouteError(
                f"{source} run from {low} to {high}, {span} years, more than "
                f"n_years = {count}"
            )
        if first_year is None and count > span:
            raise YearRouteError(
                f"{source} run from {low} to {high}, {span} years, but n_years is "
                f"{count}: where the other years lie is not in the data; pass "
                "first_year, the first simulation year"
            )
        first = low if first_year is None else check_int("first_year", first_year)
        return cls(
            ids,
            year,
            n_years=count,
            first_year=first,
            block=None if block is None else np.asarray(block),
            source=f"{source} from {first}",
        )

    @property
    def kind(self) -> str:
        """``"annual"``."""
        return ANNUAL

    @property
    def route(self) -> str:
        """``"catalogue"``."""
        return CATALOGUE

    @property
    def years(self) -> NDArray[np.int64]:
        """Every simulation year, ``first_year`` to ``first_year + N - 1``."""
        return np.arange(self.first_year, self.first_year + self.n_years)

    def year_of(self, event_id: ArrayLike) -> NDArray[np.int64]:
        """Return the year of each given event.

        Raises
        ------
        EventAlignmentError
            If an event has no year.
        """
        ids = integer_array(event_id, "event_id")
        return read_only(self.year[_positions_of(self.event_id, ids, "year")])

    def blocks(self, event_id: ArrayLike) -> NDArray[Any]:
        """Return the block of each given event: its label, or its year.

        Raises
        ------
        EventAlignmentError
            If an event has no year.
        """
        ids = integer_array(event_id, "event_id")
        rows = _positions_of(self.event_id, ids, "year")
        return read_only(self.year[rows] if self.block is None else self.block[rows])

    def measure(
        self,
        event_id: ArrayLike,
        frequency: ArrayLike,
        *,
        n_draws: int = 1,
        years: ArrayLike | None = None,
    ) -> OutcomeMeasure:
        """Return the measure of the given events over the years.

        Parameters
        ----------
        event_id : array_like of int
            The events a sample keeps (module docstring).
        frequency : array_like of float
            Each event's frequency per year: ``1 / N`` within
            :data:`~climada_petals.engine.parametric.measure.FREQUENCY_ATOL` relative to
            ``1``.
        n_draws : int, default 1
            Draws per event; rows are ordered by draw, then event.
        years : array_like of int, optional
            The years the outcomes are, a subset of :attr:`years`, each of
            weight ``1 / len(years)``: a block of a split. Every event's year
            must be among them. Default: all ``N`` years.

        Returns
        -------
        OutcomeMeasure

        Raises
        ------
        YearRouteError
            If a frequency is not ``1 / N`` (module docstring).
        EventAlignmentError
            If an event has no year.
        ParametricInputError
            If there is no event, ``years`` is invalid, or an event's year is
            not among ``years``.
        """
        ids, freq, draws = _events(event_id, frequency, n_draws)
        implied = freq * self.n_years
        wrong = np.abs(implied - 1.0) > FREQUENCY_ATOL
        if wrong.any():
            raise YearRouteError(
                f"{int(wrong.sum())} of the sample's {ids.size} events have "
                f"frequency x N between {implied[wrong].min():.6g} and "
                f"{implied[wrong].max():.6g}, not 1, with N = {self.n_years} "
                f"({self.source}). On catalogue years every event the sample keeps "
                "has frequency 1 / N. A frequency-scaled or perturbed set takes "
                "YearSets or the event basis; dates that don't carry the years "
                "take from_event_names or from_labels"
            )
        chosen = self.years if years is None else self.check_years(years)
        column = pd.Index(chosen).get_indexer(self.year_of(ids))
        if np.any(column < 0):
            raise ParametricInputError(
                f"{int((column < 0).sum())} events lie outside the given years"
            )
        return _annual(
            sp.csr_matrix(
                (np.ones(ids.size), (column, np.arange(ids.size))),
                shape=(chosen.size, ids.size),
            ),
            draws,
            CATALOGUE,
        )

    def check_years(self, years: ArrayLike) -> NDArray[np.int64]:
        """Return ``years`` sorted, checked against :attr:`years`.

        Raises
        ------
        ParametricInputError
            If the years are empty, repeat or are not simulation years.
        """
        chosen = integer_array(years, "years")
        if chosen.size == 0 or np.unique(chosen).size != chosen.size:
            raise ParametricInputError("years must be non-empty and unique")
        unknown = chosen[
            (chosen < self.first_year) | (chosen >= self.first_year + self.n_years)
        ]
        if unknown.size:
            raise ParametricInputError(
                f"years that are not simulation years, e.g. {unknown[:5].tolist()}"
            )
        return np.sort(chosen)


@dataclass(frozen=True, eq=False)
class YearSets:
    """Year sets: the annual route on sampled years (module docstring).

    Use :meth:`from_frequencies` or :meth:`from_sampling_vect`.

    Parameters
    ----------
    event_id : array_like of int
        The events the sampled positions refer to, unique, in order.
    sampling_vect : sequence of array_like of int
        One array per sampled year with the **positions** in ``event_id`` of
        its events, as ``climada.util.yearsets`` returns them. An event can
        occur twice in a year.
    block : array_like, optional
        One label per event, the unit a split keeps together. Default:
        every event is its own block.
    frequency : array_like of float, optional
        The frequencies per year the years were sampled from, one per event.
        A sample then checks that its events have them.

    Raises
    ------
    YearRouteError
        If a position is outside ``event_id``.
    ParametricInputError
        If an argument is invalid.
    """

    event_id: NDArray[np.int64]
    sampling_vect: Sequence[ArrayLike]
    block: NDArray[Any] | None = field(default=None, kw_only=True)
    frequency: NDArray[np.float64] | None = field(default=None, kw_only=True)
    _positions: NDArray[np.int64] = field(init=False, repr=False)
    _offsets: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the years and store them as positions and offsets."""
        ids = _event_ids(self.event_id)
        years = [
            integer_array(np.asarray(year).reshape(-1), "a year of sampling_vect")
            for year in self.sampling_vect
        ]
        if not years:
            raise ParametricInputError("sampling_vect must hold at least one year")
        positions = read_only(np.concatenate(years))
        if positions.size and (positions.min() < 0 or positions.max() >= ids.size):
            raise YearRouteError(
                f"sampling_vect holds positions outside the {ids.size} events: it "
                "holds positions in the event arrays, as climada.util.yearsets "
                "returns them, not event ids"
            )
        offsets = read_only(np.cumsum([0, *(year.size for year in years)]))
        object.__setattr__(self, "event_id", ids)
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_offsets", offsets)
        object.__setattr__(
            self,
            "sampling_vect",
            tuple(positions[start:end] for start, end in pairwise(offsets)),
        )
        if self.block is not None:
            object.__setattr__(self, "block", _labels(self.block, ids.size, "block"))
        if self.frequency is not None:
            freq = finite_array(self.frequency, "frequency", 1)
            if freq.shape != ids.shape or np.any(freq < 0):
                raise ParametricInputError(
                    "frequency must hold one value >= 0 per event"
                )
            object.__setattr__(self, "frequency", read_only(freq))

    @classmethod
    def from_sampling_vect(
        cls,
        event_id: ArrayLike,
        sampling_vect: Sequence[ArrayLike],
        *,
        block: ArrayLike | None = None,
    ) -> YearSets:
        """Year sets from a ``sampling_vect`` of ``climada.util.yearsets``.

        Parameters
        ----------
        event_id : array_like of int
            The events of the impact the years were sampled from, in its
            order, e.g. ``impact.event_id``.
        sampling_vect : sequence of array_like of int
            Positions in ``event_id``, one array per year, e.g. the second
            value of ``impact_yearset``.
        block : array_like, optional
            As the class's.

        Returns
        -------
        YearSets
        """
        return cls(
            np.asarray(event_id),
            sampling_vect,
            block=None if block is None else np.asarray(block),
        )

    @classmethod
    def from_frequencies(
        cls,
        impact_or_hazard: Any,
        *,
        n_years: int,
        rng: np.random.Generator,
        block: ArrayLike | None = None,
    ) -> YearSets:
        """Sample ``n_years`` years from the frequencies, with replacement.

        Parameters
        ----------
        impact_or_hazard : climada.engine.Impact or climada.hazard.Hazard
            Its ``event_id``, ``frequency`` and ``frequency_unit``.
        n_years : int
            The number of years to sample, >= 1. The sampling error of a
            tail measure falls with it: ask for at least 10,000.
        rng : numpy.random.Generator
            Draws the counts and the events (:func:`sample_years`).
        block : array_like, optional
            As the class's, one label per event of ``impact_or_hazard``.

        Returns
        -------
        YearSets

        Raises
        ------
        ImpactInputError
            If the ``frequency_unit`` is not ``1/year``.
        ParametricInputError
            If an argument is invalid.
        """
        if impact_or_hazard.frequency_unit != DEF_FREQ_UNIT:
            raise ImpactInputError(
                f"the frequency_unit is {impact_or_hazard.frequency_unit!r}; year "
                f"sets read frequencies per year ({DEF_FREQ_UNIT!r})"
            )
        freq = finite_array(impact_or_hazard.frequency, "frequency", 1)
        return cls(
            np.asarray(impact_or_hazard.event_id),
            sample_years(freq, n_years, rng),
            block=None if block is None else np.asarray(block),
            frequency=freq,
        )

    @property
    def kind(self) -> str:
        """``"annual"``."""
        return ANNUAL

    @property
    def route(self) -> str:
        """``"yearsets"``."""
        return YEARSETS

    @property
    def n_years(self) -> int:
        """``N``, the number of sampled years."""
        return int(self._offsets.size - 1)

    def blocks(self, event_id: ArrayLike) -> NDArray[Any]:
        """Return the block of each given event: its label, or the event.

        Raises
        ------
        EventAlignmentError
            If an event is not one of the year sets'.
        """
        ids = integer_array(event_id, "event_id")
        rows = _positions_of(self.event_id, ids, "sampled years")
        return read_only(ids if self.block is None else self.block[rows])

    def measure(
        self, event_id: ArrayLike, frequency: ArrayLike, *, n_draws: int = 1
    ) -> OutcomeMeasure:
        """Return the measure of the given events over the sampled years.

        The positions of other events are left out of each year, which is
        the same as sampling the given events alone at their own rate.

        Parameters
        ----------
        event_id : array_like of int
            The events a sample keeps.
        frequency : array_like of float
            Their frequencies per year: those the years were sampled from,
            if the year sets know them.
        n_draws : int, default 1
            Draws per event; rows are ordered by draw, then event.

        Returns
        -------
        OutcomeMeasure

        Raises
        ------
        YearRouteError
            If the years were sampled from other frequencies.
        EventAlignmentError
            If an event is not one of the year sets'.
        ParametricInputError
            If there is no event or a frequency is invalid.
        """
        ids, freq, draws = _events(event_id, frequency, n_draws)
        rows = _positions_of(self.event_id, ids, "sampled years")
        if self.frequency is not None and not np.allclose(
            freq, self.frequency[rows], rtol=_FREQUENCY_RTOL, atol=0.0
        ):
            raise YearRouteError(
                "the sample's frequencies are not those the years were sampled "
                "from; sample the years from this impact"
            )
        column = np.full(self.event_id.size, -1, dtype=np.int64)
        column[rows] = np.arange(ids.size)
        drawn = column[self._positions]
        year = np.repeat(np.arange(self.n_years), np.diff(self._offsets))
        kept = drawn >= 0
        counts = sp.csr_matrix(
            (np.ones(int(kept.sum())), (year[kept], drawn[kept])),
            shape=(self.n_years, ids.size),
        )
        return _annual(counts, draws, YEARSETS)


def sample_years(
    frequency: ArrayLike, n_years: int, rng: np.random.Generator
) -> list[NDArray[np.int64]]:
    """Sample years of events from their frequencies, with replacement.

    The number of events of each year is Poisson with mean ``sum f``, and
    each event is drawn with probability ``f_e / sum f``: a compound Poisson
    year with each event at its own rate. Both draws use ``rng``, through
    ``numpy.random.Generator.poisson`` and
    ``climada.util.yearsets.sample_events``, so numpy's global random state
    is not touched.

    Parameters
    ----------
    frequency : array_like of float
        Each event's frequency per year, >= 0, with a positive sum.
    n_years : int
        The number of years, >= 1.
    rng : numpy.random.Generator
        The random generator.

    Returns
    -------
    list of numpy.ndarray of int
        A ``sampling_vect``: the positions of each year's events.

    Raises
    ------
    ParametricInputError
        If an argument is invalid.
    """
    freq = finite_array(frequency, "frequency", 1)
    if freq.size == 0 or np.any(freq < 0) or float(freq.sum()) <= 0:
        raise ParametricInputError(
            "frequency must be non-empty and >= 0, with a positive sum"
        )
    count = check_int("n_years", n_years, low=1)
    if not isinstance(rng, np.random.Generator):
        raise ParametricInputError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    per_year = rng.poisson(float(freq.sum()), size=count)
    return [
        np.asarray(year, dtype=np.int64)
        for year in sample_events(per_year, freq, with_replacement=True, seed=rng)
    ]


def _seasons(dates: ArrayLike, n_events: int, month: int) -> NDArray[np.int64]:
    """Return the season (calendar year from ``month``) of each date.

    Raises
    ------
    YearRouteError
        If there is not one valid ordinal date per event.
    """
    days = np.asarray(dates, dtype=np.float64)
    if (
        days.shape != (n_events,)
        or not np.all(np.isfinite(days))
        or np.any(days < 1)
        or np.any(days > _LAST_ORDINAL)
    ):
        raise YearRouteError(
            f"the dates must hold one ordinal date (1 to {_LAST_ORDINAL}) per "
            f"event, and a date holds the calendar years 1 to {dt.MAXYEAR} only; "
            "read the years from the event names (from_event_names) or give them "
            "(from_labels)"
        )
    unique, position = np.unique(days.astype(np.int64), return_inverse=True)
    dates_of = [dt.date.fromordinal(int(day)) for day in unique]
    season = np.array(
        [day.year - (day.month < month) for day in dates_of], dtype=np.int64
    )
    return season[np.asarray(position).reshape(-1)]
