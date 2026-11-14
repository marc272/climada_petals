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

Simulation years of an index's events (private and transitional).

:meth:`~climada_petals.engine.parametric.index.ParametricIndex.with_years`
reads each event's simulation year, every simulation year and the events'
frequencies here, from a ``Hazard`` or an ``Impact``, until the outcomes
hold the years. The routes are described there.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from typing import Any, Final, Literal, NamedTuple

import numpy as np
from climada.util.constants import DEF_FREQ_UNIT
from numpy.typing import ArrayLike, NDArray

from climada_petals.util.config import LOGGER

from ._checks import check_int, finite_array, integer_array, read_only
from .errors import ImpactInputError, ParametricInputError

__all__ = ["SimulationYears", "YearSource", "simulation_years"]

YearSource = Literal["date", "event_name"]
"""Where :meth:`ParametricIndex.with_years` reads each event's year."""

_FREQUENCY_RTOL: Final[float] = 1e-6

_EVENT_NAME_YEAR: Final[re.Pattern[str]] = re.compile(
    r"(?P<prefix>.+)_(?P<year>[0-9]+)_(?P<number>[0-9]+)"
)
"""``<prefix>_<year>_<number>``, matched against the whole event name."""

_LAST_ORDINAL: Final[int] = dt.date.max.toordinal()
"""Ordinal of 31 December 9999, the last date that ``datetime`` holds."""


class SimulationYears(NamedTuple):
    """The years an index carries, by index row (``with_years``)."""

    event_year: NDArray[np.int64]
    years: NDArray[np.int64]
    frequency: NDArray[np.float64]

    @classmethod
    def checked(
        cls,
        event_year: NDArray[np.int64],
        years: ArrayLike,
        frequency: NDArray[np.float64],
    ) -> SimulationYears:
        """Return the years, sorted and read-only, after checking them.

        Raises
        ------
        ParametricInputError
            If the years are empty or repeat, an event's year is not among
            them, or a frequency is not positive.
        """
        all_years = integer_array(years, "years")
        if all_years.size == 0 or np.unique(all_years).size != all_years.size:
            raise ParametricInputError("years must be non-empty and unique")
        stray = np.setdiff1d(event_year, all_years)
        if stray.size:
            raise ParametricInputError(
                f"event years outside years, e.g. {stray[:5].tolist()}"
            )
        if np.any(frequency <= 0):
            raise ParametricInputError("frequency must be > 0")
        return cls(
            read_only(event_year), read_only(np.sort(all_years)), read_only(frequency)
        )


def simulation_years(
    source: Any,
    *,
    event_year: ArrayLike | YearSource | None,
    years: ArrayLike | None,
    n_years: int | None,
    first_year: int | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Return each event's year, all years and the frequencies of ``source``.

    Raises
    ------
    ImpactInputError
        If ``source.frequency_unit`` is not ``1/year``.
    ParametricInputError
        As :func:`_simulation_years`.
    """
    if source.frequency_unit != DEF_FREQ_UNIT:
        raise ImpactInputError(
            f"the frequency_unit is {source.frequency_unit!r}; the years read "
            f"frequencies per year ({DEF_FREQ_UNIT!r})"
        )
    frequency = finite_array(source.frequency, "frequency", 1)
    rows, all_years = _simulation_years(
        source,
        frequency,
        event_year=event_year,
        years=years,
        n_years=n_years,
        first_year=first_year,
    )
    return rows, all_years, frequency


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
    rows = integer_array(event_year, "event_year")
    if rows.size != n_events:
        raise ParametricInputError(
            f"event_year has {rows.size} entries for {n_events} events"
        )
    return rows, "event_year"


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
