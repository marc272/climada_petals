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

Parametric index: index values per event and insured unit.

:class:`ParametricIndex` holds the index of every event of a CLIMADA
``Hazard`` at every insured unit (one exposure point each), aligned with the
hazard's events, plus each event's simulation year, which the annual basis
needs. Two builders:

* :meth:`ParametricIndex.from_hazard`: the hazard intensity at each exposure
  point's assigned centroid (e.g. the maximum sustained wind, in m/s, for a
  ``TropCyclone``).
* :meth:`ParametricIndex.from_cat_in_circle`: the cat-in-a-circle index
  from ``TCTracks``: the highest Saffir-Simpson category that a storm's
  track reaches within ``radius_km`` of the exposure point. It reads the
  tracks, not the wind field that drives the loss, so index and loss are
  decoupled as in a real contract. The method and its track geometry are in
  :mod:`~climada_petals.engine.parametric.index_tracks`.

Simulation years
----------------
The annual basis groups events into simulation years, and the years of the
hazard set include those without events. Both builders take them in three
parts:

* **Each event's year**, ``event_year``:

  - ``None`` or ``"date"`` (the default): the calendar year of
    ``hazard.date``, as ``TropCyclone.from_tracks`` sets it for historical
    tracks;
  - ``"event_name"`` (opt-in): the year in the event names, which must
    have the form ``<prefix>_<year>_<number>``: the year is the
    second-to-last field between underscores, e.g. ``NA_9999_0`` (basin,
    year, storm) for an IRIS storm. A name of any other form raises;
  - an array of int, one year per event.

* **The number of years** ``N``, ``n_years``: by default ``1 / frequency``,
  which must then be the same for every event.
* **The first year**, ``first_year``: the years are ``first_year`` to
  ``first_year + N - 1``. Without it, the years start at the first event's
  year, which is certain only if the events' years run over exactly ``N``
  years. If they run over fewer, the years without events could lie before
  the first event or after the last one: then the ``"event_name"`` route
  raises, and the date and array routes keep their earlier behaviour,
  starting at the first event's year, and log a warning.

``years``, every simulation year, can replace ``n_years`` and
``first_year``; giving it with either of them raises.

The dates cannot carry the years of every hazard. A CLIMADA date holds the
calendar years 1 to 9999 (:data:`datetime.MAXYEAR`), so not every year of a
10,000-year synthetic catalogue, and the tracks of a synthetic catalogue
often all start on one date. For such a hazard, read the years from the
event names and state the first year: ``event_year="event_name",
first_year=0`` for IRIS, which numbers its years 0 to 9,999. The members of
an ensemble of perturbed tracks (``calc_perturbed_trajectories``) share
their parent's date, and their names (``<parent>_gen<m>``) give no year:
pass ``event_year`` as an array, giving member ``m`` of year ``y`` the year
``y + m N``, where ``N`` is the number of years of the parent catalogue.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from climada_petals.util.config import LOGGER

from ._checks import check_int, check_real, finite_array, integer_array, read_only
from .errors import ParametricInputError
from .index_tracks import (
    DEFAULT_TRACK_STEP_HOURS,
    SAFFIR_SIMPSON_KN,
    Metric,
    cat_in_circle,
)

__all__ = ["ParametricIndex", "YearSource", "split_years"]

YearSource = Literal["date", "event_name"]
"""Where the builders read each event's simulation year (module docstring)."""

_FREQUENCY_RTOL: Final[float] = 1e-6

_EVENT_NAME_YEAR: Final[re.Pattern[str]] = re.compile(
    r"(?P<prefix>.+)_(?P<year>[0-9]+)_(?P<number>[0-9]+)"
)
"""``<prefix>_<year>_<number>``, matched against the whole event name."""

_LAST_ORDINAL: Final[int] = dt.date.max.toordinal()
"""Ordinal of 31 December 9999, the last date that ``datetime`` holds."""


class _YearArgs(NamedTuple):
    """The simulation-year arguments of the builders (module docstring)."""

    event_year: ArrayLike | YearSource | None
    years: ArrayLike | None
    n_years: int | None
    first_year: int | None


@dataclass(frozen=True, eq=False)
class ParametricIndex:
    """Index values per event and insured unit, aligned with a hazard's events.

    Use :meth:`from_hazard` or :meth:`from_cat_in_circle` to build one from
    CLIMADA objects; the constructor takes the arrays directly.

    Parameters
    ----------
    values : array_like of float
        Index values, shape ``(n_events, n_units)``. Finite.
    event_id : array_like of int
        The hazard's event ids, shape ``(n_events,)``, unique.
    event_name : sequence of str
        The hazard's event names, one per event.
    frequency : array_like of float
        Annual frequency of each event, > 0.
    event_year : array_like of int
        Simulation year of each event.
    years : array_like of int
        Every simulation year the hazard set represents, including years
        without events. Unique; every event's year must be among them.
    unit_id : sequence of hashable
        One id per unit (column), unique: the exposures' index labels.
    coords : array_like of float
        Latitude and longitude of each unit, shape ``(n_units, 2)``, as in
        ``Impact.coord_exp``.
    unit : str, optional
        Unit of the index values, e.g. ``"m/s"``.
    definition : str, optional
        What the index is, in words.

    Raises
    ------
    ParametricInputError
        If an argument has the wrong shape or type, or the arrays disagree.
    """

    values: NDArray[np.float64]
    event_id: NDArray[np.int64]
    event_name: Sequence[str]
    frequency: NDArray[np.float64]
    event_year: NDArray[np.int64]
    years: NDArray[np.int64]
    unit_id: Sequence[Hashable]
    coords: NDArray[np.float64]
    unit: str = ""
    definition: str = ""

    def __post_init__(self) -> None:
        """Validate the fields and store read-only copies."""
        values = finite_array(self.values, "values", 2)
        n_events, n_units = values.shape
        event_id = integer_array(self.event_id, "event_id")
        frequency = finite_array(self.frequency, "frequency", 1)
        event_year = integer_array(self.event_year, "event_year")
        years = integer_array(self.years, "years")
        names = tuple(str(name) for name in self.event_name)
        for name, arr in (
            ("event_id", event_id),
            ("event_name", names),
            ("frequency", frequency),
            ("event_year", event_year),
        ):
            if len(arr) != n_events:
                raise ParametricInputError(
                    f"{name} has {len(arr)} entries for {n_events} events"
                )
        if np.unique(event_id).size != n_events:
            raise ParametricInputError("event_id must be unique")
        if np.any(frequency <= 0):
            raise ParametricInputError("frequency must be > 0")
        if years.size == 0 or np.unique(years).size != years.size:
            raise ParametricInputError("years must be non-empty and unique")
        stray = np.setdiff1d(event_year, years)
        if stray.size:
            raise ParametricInputError(
                f"event years outside years, e.g. {stray[:5].tolist()}"
            )
        units = tuple(self.unit_id)
        if len(units) != n_units or len(set(units)) != n_units:
            raise ParametricInputError(
                f"unit_id must hold {n_units} unique ids, got {len(units)}"
            )
        coords = finite_array(self.coords, "coords", 2)
        if coords.shape != (n_units, 2):
            raise ParametricInputError(
                f"coords must have shape ({n_units}, 2), got {coords.shape}"
            )
        object.__setattr__(self, "values", read_only(values))
        object.__setattr__(self, "event_id", read_only(event_id))
        object.__setattr__(self, "event_name", names)
        object.__setattr__(self, "frequency", read_only(frequency))
        object.__setattr__(self, "event_year", read_only(event_year))
        object.__setattr__(self, "years", read_only(np.sort(years)))
        object.__setattr__(self, "unit_id", units)
        object.__setattr__(self, "coords", read_only(coords))

    @property
    def n_events(self) -> int:
        """Number of events (rows)."""
        return int(self.values.shape[0])

    @property
    def n_units(self) -> int:
        """Number of insured units (columns)."""
        return int(self.values.shape[1])

    def event_mask(self, years: ArrayLike | None = None) -> NDArray[np.bool_]:
        """Return the events whose year is in ``years``.

        Parameters
        ----------
        years : array_like of int, optional
            Simulation years, a subset of :attr:`years`. Default: all.

        Returns
        -------
        numpy.ndarray of bool
            Shape ``(n_events,)``.

        Raises
        ------
        ParametricInputError
            If a year is not one of the index's years, or repeats.
        """
        chosen = self.check_years(years)
        return np.isin(self.event_year, chosen)

    def check_years(self, years: ArrayLike | None) -> NDArray[np.int64]:
        """Return ``years`` sorted, checked against the index's years.

        Parameters
        ----------
        years : array_like of int or None
            Simulation years; None means all of :attr:`years`.

        Returns
        -------
        numpy.ndarray of int
            The years, sorted.

        Raises
        ------
        ParametricInputError
            If the years are empty, repeat or are not among :attr:`years`.
        """
        if years is None:
            return self.years
        chosen = integer_array(years, "years")
        if chosen.size == 0 or np.unique(chosen).size != chosen.size:
            raise ParametricInputError("years must be non-empty and unique")
        unknown = np.setdiff1d(chosen, self.years)
        if unknown.size:
            raise ParametricInputError(
                f"years not in the index's years, e.g. {unknown[:5].tolist()}"
            )
        return np.sort(chosen)

    @classmethod
    def from_hazard(
        cls,
        hazard: Any,
        exposures: Any,
        *,
        event_year: ArrayLike | YearSource | None = None,
        years: ArrayLike | None = None,
        n_years: int | None = None,
        first_year: int | None = None,
    ) -> ParametricIndex:
        """Index = hazard intensity at each exposure point's assigned centroid.

        Parameters
        ----------
        hazard : climada.hazard.Hazard
            The hazard, e.g. a ``TropCyclone``. Its events are the index's
            events, in order.
        exposures : climada.entity.Exposures
            One point per insured unit. Uses the exposures' centroid column
            for the hazard type (``centr_<haz_type>``) if it is set, and
            otherwise assigns centroids on a copy (``assign_centroids``),
            leaving ``exposures`` unchanged.
        event_year : {"date", "event_name"} or array_like of int, optional
            Each event's simulation year: from ``hazard.date`` (``None``,
            the default, or ``"date"``), from ``hazard.event_name``
            (``"event_name"``, names ``<prefix>_<year>_<number>``), or one
            year per event.
        years : array_like of int, optional
            Every simulation year, including years without events. Not
            together with ``n_years`` or ``first_year``.
        n_years : int, optional
            Number of simulation years, >= 1. Default: ``1 / frequency``.
        first_year : int, optional
            The first simulation year, in the numbering of the event years
            (calendar years on the date route). Default: the first event's
            year; if the events' years run over fewer than ``n_years``
            years, the ``"event_name"`` route raises and the other routes
            log a warning.

        Returns
        -------
        ParametricIndex
            Index in ``hazard.units``.

        Raises
        ------
        ParametricInputError
            If an exposure point has no centroid, or the years cannot be
            derived or are ambiguous (module docstring, "Simulation years").
        """
        centroids = _assigned_centroids(hazard, exposures)
        values = np.asarray(hazard.intensity[:, centroids].toarray(), dtype=float)
        column = f"centr_{hazard.haz_type}"
        return cls._from_climada(
            values,
            hazard,
            exposures,
            _YearArgs(event_year, years, n_years, first_year),
            unit=str(hazard.units),
            definition=(
                f"intensity of the {hazard.haz_type} hazard at each exposure "
                f"point's assigned centroid ({column})"
            ),
        )

    @classmethod
    def from_cat_in_circle(  # pylint: disable=too-many-arguments
        cls,
        tracks: Any,
        exposures: Any,
        *,
        hazard: Any,
        radius_km: float,
        metric: Metric = "category",
        breakpoints_kn: Sequence[float] = SAFFIR_SIMPSON_KN,
        track_step_hours: float | None = DEFAULT_TRACK_STEP_HOURS,
        event_year: ArrayLike | YearSource | None = None,
        years: ArrayLike | None = None,
        n_years: int | None = None,
        first_year: int | None = None,
    ) -> ParametricIndex:
        """Cat-in-a-circle index from tracks (module docstring).

        Parameters
        ----------
        tracks : climada.hazard.TCTracks
            The storms' tracks, wind in knots. Every hazard event needs
            exactly one track whose ``sid`` is the event's name (as
            ``TropCyclone.from_tracks`` names its events).
        exposures : climada.entity.Exposures
            One point per insured unit.
        hazard : climada.hazard.Hazard
            Sets the events (order, ids, frequencies, dates), so that the
            index is aligned with the hazard and with impacts computed from
            it.
        radius_km : float
            Circle radius in km, > 0.
        metric : {"category", "max_wind"}, default "category"
            The category (0 to ``len(breakpoints_kn)``), or the wind in m/s.
        breakpoints_kn : sequence of float, default SAFFIR_SIMPSON_KN
            Strictly increasing, positive lower bounds of categories 1, 2,
            ..., in knots. Used by ``"category"`` only.
        track_step_hours : float or None, default 0.5
            Interpolate the tracks to this time step before testing the
            distance; None uses the track points as they are.
        event_year, years, n_years, first_year : optional
            Simulation years, as in :meth:`from_hazard` (module docstring,
            "Simulation years").

        Returns
        -------
        ParametricIndex
            The index, in categories or m/s.

        Raises
        ------
        ParametricInputError
            If a parameter is out of range, a hazard event has no track or
            two, a track's wind is not in knots, or the years cannot be
            derived or are ambiguous.
        """
        values, unit, definition = cat_in_circle(
            tracks,
            exposures,
            hazard,
            radius_km=radius_km,
            metric=metric,
            breakpoints_kn=breakpoints_kn,
            track_step_hours=track_step_hours,
        )
        return cls._from_climada(
            values,
            hazard,
            exposures,
            _YearArgs(event_year, years, n_years, first_year),
            unit=unit,
            definition=definition,
        )

    @classmethod
    def _from_climada(
        cls,
        values: NDArray[np.float64],
        hazard: Any,
        exposures: Any,
        year_args: _YearArgs,
        *,
        unit: str,
        definition: str,
    ) -> ParametricIndex:
        """Attach the hazard's events, the years and the exposure points."""
        frequency = np.asarray(hazard.frequency, dtype=float)
        rows, all_years = _simulation_years(
            hazard,
            frequency,
            event_year=year_args.event_year,
            years=year_args.years,
            n_years=year_args.n_years,
            first_year=year_args.first_year,
        )
        return cls(
            values=values,
            event_id=np.asarray(hazard.event_id),
            event_name=tuple(str(name) for name in hazard.event_name),
            frequency=frequency,
            event_year=rows,
            years=all_years,
            unit_id=tuple(exposures.gdf.index),
            coords=np.column_stack([exposures.latitude, exposures.longitude]),
            unit=unit,
            definition=definition,
        )


def split_years(
    index: ParametricIndex, *, train_fraction: float = 0.7
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split the simulation years into a training block and a test block.

    The blocks are contiguous: training is the first
    ``floor(train_fraction * N)`` years of :attr:`ParametricIndex.years`,
    testing the rest. All the events of a year stay in the same block.
    The split reads only :attr:`ParametricIndex.years`, the simulation span
    the index was built with (module docstring, "Simulation years"),
    including the years without events; it never reads dates.

    Parameters
    ----------
    index : ParametricIndex
        The index whose years are split.
    train_fraction : float, default 0.7
        Share of the years in the training block, in ``(0, 1)``.

    Returns
    -------
    train, test : numpy.ndarray of int
        The two blocks of years. Pass them as ``years`` to
        :func:`~climada_petals.engine.parametric.calibrate.calibrate` and
        :func:`~climada_petals.engine.parametric.metrics.basis_risk`.

    Raises
    ------
    ParametricInputError
        If ``train_fraction`` is not in ``(0, 1)`` or a block would be empty.
    """
    fraction = check_real(
        "train_fraction",
        train_fraction,
        low=0.0,
        high=1.0,
        include_low=False,
        include_high=False,
    )
    n_years = index.years.size
    n_train = math.floor(fraction * n_years + 1e-9)
    if not 1 <= n_train <= n_years - 1:
        raise ParametricInputError(
            f"train_fraction {fraction} of {n_years} years gives {n_train} "
            "training years; both blocks need at least one year"
        )
    return index.years[:n_train].copy(), index.years[n_train:].copy()


def _assigned_centroids(hazard: Any, exposures: Any) -> NDArray[np.intp]:
    """Return the centroid of each exposure point, assigning on a copy if needed.

    Raises
    ------
    ParametricInputError
        If an exposure point has no centroid within CLIMADA's threshold.
    """
    try:
        centroids = np.asarray(exposures.hazard_centroids(hazard.haz_type))
    except ValueError:
        assigned = exposures.copy()
        assigned.assign_centroids(hazard)
        centroids = np.asarray(assigned.hazard_centroids(hazard.haz_type))
    centroids = centroids.astype(np.intp)
    unassigned = np.flatnonzero(centroids < 0)
    if unassigned.size:
        labels = [exposures.gdf.index[i] for i in unassigned[:5]]
        raise ParametricInputError(
            f"{unassigned.size} exposure points have no hazard centroid, e.g. {labels}"
        )
    return centroids


def _simulation_years(
    hazard: Any,
    frequency: NDArray[np.float64],
    *,
    event_year: ArrayLike | YearSource | None,
    years: ArrayLike | None,
    n_years: int | None,
    first_year: int | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return each event's year and all years (module docstring).

    Raises
    ------
    ParametricInputError
        If the event years cannot be read, ``years`` is given with
        ``n_years`` or ``first_year``, the number of years is not given and
        the frequencies differ, the event years do not fit the span, or, on
        the ``"event_name"`` route, the first year is not given and the
        event years do not determine it.
    """
    if years is not None and (n_years is not None or first_year is not None):
        raise ParametricInputError(
            "pass either years or n_years and first_year, not both"
        )
    rows, source = _event_years(hazard, event_year, frequency.size)
    if years is not None:
        return rows, integer_array(years, "years")
    if rows.size == 0:
        raise ParametricInputError("the hazard has no events")
    count, count_source = _year_count(frequency, n_years)
    # The event-name route is new and never guesses; the date and array
    # routes keep their earlier result, with a warning (module docstring).
    strict = isinstance(event_year, str) and event_year == "event_name"
    first = _first_year(rows, count, first_year, source, count_source, strict=strict)
    LOGGER.info(
        "simulation years: %d years (%s) from %d, event years from %s",
        count,
        count_source,
        first,
        source,
    )
    return rows, np.arange(first, first + count, dtype=np.int64)


def _event_years(
    hazard: Any,
    event_year: ArrayLike | YearSource | None,
    n_events: int,
) -> tuple[NDArray[np.int64], str]:
    """Return each event's year and where it was read (module docstring).

    Raises
    ------
    ParametricInputError
        If ``event_year`` is an unknown source, or the dates or the event
        names do not give one year per event.
    """
    if event_year is None or (isinstance(event_year, str) and event_year == "date"):
        return _years_of_dates(hazard.date, n_events), "the dates"
    if isinstance(event_year, str):
        if event_year == "event_name":
            return _years_of_names(hazard.event_name), "the event names"
        raise ParametricInputError(
            "event_year must be 'date', 'event_name' or one year per event, "
            f"got {event_year!r}"
        )
    return integer_array(event_year, "event_year"), "event_year"


def _years_of_dates(dates: ArrayLike, n_events: int) -> NDArray[np.int64]:
    """Return the calendar year of each event's date (proleptic ordinals).

    Raises
    ------
    ParametricInputError
        If there is not one valid ordinal per event: a date holds the
        calendar years 1 to 9999 only.
    """
    days = np.asarray(dates, dtype=np.float64)
    if (
        days.shape != (n_events,)
        or not np.all(np.isfinite(days))
        or np.any(days < 1)
        or np.any(days > _LAST_ORDINAL)
    ):
        raise ParametricInputError(
            "hazard.date must hold one ordinal date (1 to "
            f"{_LAST_ORDINAL}) per event to give the event years, and a date "
            f"holds the calendar years 1 to {dt.MAXYEAR} only; pass event_year: "
            "'event_name' to read the years from the event names, or one year "
            "per event"
        )
    unique, position = np.unique(days.astype(np.int64), return_inverse=True)
    calendar = np.array(
        [dt.date.fromordinal(int(day)).year for day in unique], dtype=np.int64
    )
    return calendar[np.asarray(position).reshape(-1)]


def _years_of_names(names: Sequence[object]) -> NDArray[np.int64]:
    """Return the year in each event name ``<prefix>_<year>_<number>``.

    Raises
    ------
    ParametricInputError
        If a name has another form.
    """
    texts = [str(name) for name in names]
    found = [_EVENT_NAME_YEAR.fullmatch(text) for text in texts]
    bad = [text for text, match in zip(texts, found, strict=True) if match is None]
    if bad:
        raise ParametricInputError(
            f"{len(bad)} event names do not have the form "
            "<prefix>_<year>_<number> (e.g. NA_9999_0), so they give no year, "
            f"e.g. {bad[:5]}; pass event_year as one year per event"
        )
    return np.array(
        [int(match["year"]) for match in found if match is not None], dtype=np.int64
    )


def _year_count(frequency: NDArray[np.float64], n_years: int | None) -> tuple[int, str]:
    """Return the number of simulation years and where it comes from.

    Raises
    ------
    ParametricInputError
        If ``n_years`` is not an int >= 1, or it is not given and the
        frequencies are not all ``1 / N`` for one ``N``.
    """
    if n_years is not None:
        return check_int("n_years", n_years, low=1), "n_years"
    implied = 1.0 / frequency
    count = round(float(np.median(implied)))
    if count < 1 or not np.allclose(implied, count, rtol=_FREQUENCY_RTOL, atol=0):
        raise ParametricInputError(
            "cannot derive the number of simulation years from frequencies "
            f"between {frequency.min():.6g} and {frequency.max():.6g}; pass "
            "n_years or years"
        )
    return count, "1 / frequency"


def _first_year(
    rows: NDArray[np.int64],
    count: int,
    first_year: int | None,
    source: str,
    count_source: str,
    *,
    strict: bool,
) -> int:
    """Return the first simulation year (module docstring).

    Without ``first_year``, the first event's year if the event years run
    over exactly ``count`` years. If they run over fewer, the first event's
    year is a guess: with ``strict`` it raises, otherwise it logs a warning
    and returns the first event's year.

    Raises
    ------
    ParametricInputError
        If an event year lies outside the stated span, the event years run
        over more than ``count`` years, or ``strict`` and they run over
        fewer without ``first_year``.
    """
    if first_year is not None:
        first = check_int("first_year", first_year)
        outside = np.unique(rows[(rows < first) | (rows >= first + count)])
        if outside.size:
            raise ParametricInputError(
                f"event years ({source}) outside the {count} simulation years "
                f"{first} to {first + count - 1} (first_year, {count_source}), "
                f"e.g. {outside[:5].tolist()}"
            )
        return first
    low, high = int(rows.min()), int(rows.max())
    span = high - low + 1
    if span == count:
        return low
    if span < count:
        gap = (
            f"the event years ({source}) run from {low} to {high}, {span} "
            f"years, but there are {count} simulation years ({count_source}): "
            f"the {count - span} years without events could lie before {low} "
            f"or after {high}"
        )
        if strict:
            raise ParametricInputError(
                f"{gap}. Pass first_year, the first simulation year, or years"
            )
        LOGGER.warning(
            "%s. Taking the first event's year, %d, as the first simulation "
            "year; pass first_year to state it",
            gap,
            low,
        )
        return low
    raise ParametricInputError(
        f"the event years ({source}) run from {low} to {high}, {span} years, "
        f"more than the {count} simulation years ({count_source}); pass "
        "n_years and first_year, or years"
    )
