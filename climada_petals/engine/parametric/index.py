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

Parametric index: the value a cover pays on, per event and insured unit.

A :class:`ParametricIndex` holds one value per event and insured unit
(:class:`~climada_petals.engine.parametric.units.InsuredUnits`), each
event's ``event_id``, and the
:class:`~climada_petals.engine.parametric.cover.IndexDefinition` of how the
values were computed. Nothing in it is specific to a hazard type. Builders:

* :meth:`ParametricIndex.from_hazard`: a statistic of a ``Hazard``'s
  intensity over each unit's centroids, ``index_stat`` as in ``cat_bonds``:
  ``"at_centroid"`` (one-point units), ``"mean"`` (equal weights, or by
  value), ``"max"``, or a percentile in ``[0, 100]``;
* :meth:`ParametricIndex.from_cat_in_circle`: the cat-in-a-circle category
  from tracks (:mod:`~climada_petals.engine.parametric.index_tracks`);
* :meth:`ParametricIndex.from_table`: a user's table keyed on ``event_id``
  and unit, such as station readings; the constructor takes arrays;
* :meth:`ParametricIndex.from_definition`: a cover's index on another
  hazard set.

CLIMADA specifics, each handled by a typed error or an explicit argument:

* **Events are matched on** ``event_id``, never by position
  (:meth:`ParametricIndex.rows_in`). An index from another hazard set than
  the loss's is matched to its ``Impact`` with
  :meth:`ParametricIndex.matched_to`, on ``event_id`` or ``event_name``.
  Other events raise
  :class:`~climada_petals.engine.parametric.errors.EventAlignmentError`.
* **Zero means no recorded intensity** in the sparse ``Hazard.intensity``.
  With ``absent_value``, every zero at a unit's centroids reads as that
  value. A lower-tail index (``tail="lower"``, a drought index, say)
  requires it, so it states what an absent reading means.
* An ``Impact`` without ``imp_mat``, a ``frequency_unit`` other than
  ``1/year``, and exposures with ``cover`` or ``deductible`` raise
  :class:`~climada_petals.engine.parametric.errors.ImpactInputError`; two
  points at one location, with centroids built from the points, make
  ``TropCyclone.from_tracks`` return column indices of -1, which raises
  :class:`~climada_petals.engine.parametric.errors.UnitMappingError`.
  Build the centroids from unique coordinates.

Simulation years (transitional)
-------------------------------
Until the outcomes hold them, an index carries the simulation years of the
annual basis and its events' frequencies, once
:meth:`ParametricIndex.with_years` has attached them from the hazard or the
impact. :func:`split_years` splits them into training and test years.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from climada_petals.util.config import LOGGER

from ._checks import check_real, finite_array, integer_array, read_only
from ._years import SimulationYears, YearSource, simulation_years
from .cover import IndexDefinition
from .errors import EventAlignmentError, ParametricInputError, UnitMappingError
from .index_tracks import (
    CAT_IN_CIRCLE,
    DEFAULT_TRACK_STEP_HOURS,
    SAFFIR_SIMPSON_KN,
    Metric,
    cat_in_circle,
    cat_in_circle_arguments,
)
from .units import InsuredUnits

__all__ = ["HAZARD", "TABLE", "ParametricIndex", "YearSource", "split_years"]

HAZARD: Final[str] = "hazard"
"""The ``kind`` of an :class:`IndexDefinition` built by ``from_hazard``."""

TABLE: Final[str] = "table"
"""The ``kind`` the docs suggest for a user's table (``from_table``)."""

_HAZARD_PARAMETERS: Final[frozenset[str]] = frozenset({"absent_value", "haz_type"})

_CHUNK: Final[int] = 1 << 22
"""Most intensity values densified at once by :meth:`ParametricIndex.from_hazard`."""


@dataclass(frozen=True, eq=False)
class ParametricIndex:
    """Index values per event and insured unit (module docstring).

    Use a builder (:meth:`from_hazard`, :meth:`from_cat_in_circle`,
    :meth:`from_table`, :meth:`from_definition`); the constructor takes the
    arrays directly.

    Parameters
    ----------
    values : array_like of float
        Index values, shape ``(n_events, n_units)``. Finite.
    event_id : array_like of int
        One id per event (row), unique.
    unit_id : sequence of hashable
        One id per unit (column), unique.
    definition : IndexDefinition
        How the values were computed. A cover pays on an index only if it
        is the cover's definition.
    event_name : sequence of str, optional
        One name per event, for :meth:`matched_to` with ``on="event_name"``.
    units : InsuredUnits, optional
        The units the index was built on, with ``unit_id`` as their ids. An
        ``Impact``'s loss is summed into them where the index meets it.

    Raises
    ------
    ParametricInputError
        If an argument has the wrong shape or type, or two disagree.
    """

    values: NDArray[np.float64]
    event_id: NDArray[np.int64]
    unit_id: Sequence[Hashable]
    definition: IndexDefinition
    event_name: Sequence[str] | None = field(default=None, kw_only=True)
    units: InsuredUnits | None = field(default=None, kw_only=True)
    _years: SimulationYears | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the fields and store read-only copies."""
        values = finite_array(self.values, "values", 2)
        n_events, n_units = values.shape
        event_id = integer_array(self.event_id, "event_id")
        if event_id.size != n_events or np.unique(event_id).size != n_events:
            raise ParametricInputError(
                f"event_id must hold {n_events} unique ids, got {event_id.size}"
            )
        units = tuple(self.unit_id)
        if len(units) != n_units or len(set(units)) != n_units:
            raise ParametricInputError(
                f"unit_id must hold {n_units} unique ids, got {len(units)}"
            )
        if not isinstance(self.definition, IndexDefinition):
            raise ParametricInputError(
                f"definition must be an IndexDefinition, got {self.definition!r}"
            )
        names = None
        if self.event_name is not None:
            names = tuple(str(name) for name in self.event_name)
            if len(names) != n_events:
                raise ParametricInputError(
                    f"event_name has {len(names)} entries for {n_events} events"
                )
        if self.units is not None and (
            not isinstance(self.units, InsuredUnits) or self.units.unit_id != units
        ):
            raise ParametricInputError(
                "units must be the InsuredUnits of the index's unit_id, in order"
            )
        object.__setattr__(self, "values", read_only(values))
        object.__setattr__(self, "event_id", read_only(event_id))
        object.__setattr__(self, "unit_id", units)
        object.__setattr__(self, "event_name", names)

    @property
    def n_events(self) -> int:
        """Number of events (rows)."""
        return int(self.values.shape[0])

    @property
    def n_units(self) -> int:
        """Number of insured units (columns)."""
        return int(self.values.shape[1])

    @property
    def years(self) -> NDArray[np.int64]:
        """Every simulation year, sorted (:meth:`with_years`)."""
        return self._simulation().years

    @property
    def event_year(self) -> NDArray[np.int64]:
        """Simulation year of each event (:meth:`with_years`)."""
        return self._simulation().event_year

    @property
    def frequency(self) -> NDArray[np.float64]:
        """Annual frequency of each event (:meth:`with_years`)."""
        return self._simulation().frequency

    def _simulation(self) -> SimulationYears:
        """Return the attached years, or raise if there are none."""
        if self._years is None:
            raise ParametricInputError(
                "the index has no simulation years; attach them with "
                "index.with_years(hazard_or_impact, ...)"
            )
        return self._years

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
            If the index has no years, or a year is not one of them, or
            repeats.
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
            If the index has no years, or the years are empty, repeat or
            are not among :attr:`years`.
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

    def rows_in(self, impact: Any) -> NDArray[np.intp]:
        """Return the row of ``impact`` of each of the index's events, on ``event_id``.

        Parameters
        ----------
        impact : climada.engine.Impact
            An impact with the index's events, in any order.

        Returns
        -------
        numpy.ndarray of int
            Shape ``(n_events,)``: ``impact``'s rows in the index's order.

        Raises
        ------
        EventAlignmentError
            If the impact's events are not exactly the index's.
        """
        other = integer_array(np.asarray(impact.event_id), "impact.event_id")
        position = _positions(other, "the impact's event_id")
        rows = [position.get(int(key), -1) for key in self.event_id]
        missing = rows.count(-1)
        if missing or other.size != self.n_events:
            raise EventAlignmentError(
                f"the impact's event_id are not the index's events: the impact has "
                f"{other.size} events and the index {self.n_events}, of which "
                f"{missing} are not the impact's; match an index from another "
                "hazard set with matched_to(impact)"
            )
        return np.asarray(rows, dtype=np.intp)

    def matched_to(self, impact: Any, *, on: str = "event_id") -> ParametricIndex:
        """Return the index on the impact's events, in the impact's order.

        This is the stated event matching of an index from another hazard
        set than the loss's (module docstring). Index events that the
        impact lacks are left out and counted in a logged warning. The
        result has the impact's ``event_id`` and ``event_name``, and no
        simulation years: attach them after matching.

        Parameters
        ----------
        impact : climada.engine.Impact
            The loss, or anything with its ``event_id`` and ``event_name``.
        on : {"event_id", "event_name"}, default "event_id"
            The key that identifies an event in both. Tracks of one storm
            set give two hazard sets the same names but, often, other ids.

        Returns
        -------
        ParametricIndex

        Raises
        ------
        ParametricInputError
            If ``on`` is unknown, or the index has no event names.
        EventAlignmentError
            If the key repeats on either side, or an event of the impact
            has no index value.
        """
        if on not in ("event_id", "event_name"):
            raise ParametricInputError(
                f"on must be 'event_id' or 'event_name', got {on!r}"
            )
        if on == "event_id":
            own: list[Any] = self.event_id.tolist()
            other: list[Any] = [int(key) for key in impact.event_id]
        elif self.event_name is None:
            raise ParametricInputError("the index has no event names to match on")
        else:
            own = list(self.event_name)
            other = [str(key) for key in impact.event_name]
        position = _positions(own, f"the index's {on}")
        _positions(other, f"the impact's {on}")
        missing = [key for key in other if key not in position]
        if missing:
            raise EventAlignmentError(
                f"{len(missing)} of the impact's {len(other)} events have no index "
                f"value ({on}), e.g. {missing[:5]}"
            )
        if len(own) > len(other):
            LOGGER.warning(
                "matched_to: %d of the index's %d events are not the impact's (%s) "
                "and are left out",
                len(own) - len(other),
                len(own),
                on,
            )
        rows = [position[key] for key in other]
        names = [str(name) for name in impact.event_name]
        return ParametricIndex(
            self.values[rows],
            np.asarray(impact.event_id),
            self.unit_id,
            self.definition,
            event_name=names if len(names) == len(rows) else None,
            units=self.units,
        )

    def with_years(
        self,
        source: Any,
        *,
        event_year: ArrayLike | YearSource | None = None,
        years: ArrayLike | None = None,
        n_years: int | None = None,
        first_year: int | None = None,
    ) -> ParametricIndex:
        """Return the index with simulation years and frequencies attached.

        The years include those without events, in three parts:

        * **each event's year**, ``event_year``: ``None`` or ``"date"``, the
          calendar year of ``source.date``; ``"event_name"``, the year in
          names ``<prefix>_<year>_<number>`` such as IRIS's ``NA_9999_0``
          (another form raises); or one year per event of ``source``;
        * **the number of years** ``N``, ``n_years``: by default
          ``1 / frequency``, which must then be one value;
        * **the first year**, ``first_year``. Without it, the years start
          at the first event's year; if the events' years run over fewer
          than ``N`` years, the ``"event_name"`` route raises, and the
          others take the first event's year with a logged warning.

        ``years``, every simulation year, can replace ``n_years`` and
        ``first_year``. A date holds the years 1 to 9999 only, and a
        synthetic catalogue's tracks often start on one date: use
        ``event_year="event_name", first_year=0`` for IRIS. Members of an
        ensemble share their parent's date: pass ``event_year`` as an
        array, member ``m`` of year ``y`` in year ``y + m N``.

        Parameters
        ----------
        source : climada.hazard.Hazard or climada.engine.Impact
            Holds every event of the index (matched on ``event_id``) with
            its ``date``, ``event_name`` and ``frequency`` per year. Its
            other events count for the span of the years.
        event_year, years, n_years, first_year : optional
            As above.

        Returns
        -------
        ParametricIndex
            The same index, with :attr:`years`, :attr:`event_year` and
            :attr:`frequency`.

        Raises
        ------
        ImpactInputError
            If ``source.frequency_unit`` is not ``1/year``.
        EventAlignmentError
            If an event of the index is not in ``source``.
        ParametricInputError
            If the years cannot be derived or are ambiguous, or a frequency
            is not positive.
        """
        rows, all_years, frequency = simulation_years(
            source,
            event_year=event_year,
            years=years,
            n_years=n_years,
            first_year=first_year,
        )
        position = _positions(
            integer_array(np.asarray(source.event_id), "event_id"), "event_id"
        )
        missing = [key for key in self.event_id.tolist() if key not in position]
        if missing:
            raise EventAlignmentError(
                f"{len(missing)} of the index's events are not in the source of "
                f"the years, e.g. {missing[:5]}"
            )
        index_rows = [position[key] for key in self.event_id.tolist()]
        attached = SimulationYears.checked(
            rows[index_rows], all_years, frequency[index_rows]
        )
        out = copy.copy(self)
        object.__setattr__(out, "_years", attached)
        return out

    @classmethod
    def from_hazard(
        cls,
        hazard: Any,
        units: InsuredUnits,
        *,
        index_stat: str | float = "at_centroid",
        weight: str | None = None,
        tail: str = "upper",
        absent_value: float | None = None,
        source: str | None = None,
    ) -> ParametricIndex:
        """Index = a statistic of the hazard's intensity over each unit's centroids.

        A unit's centroids are the distinct centroids its points are
        assigned to (:meth:`InsuredUnits.centroids`). The hazard need not
        be the one behind the loss: match another with :meth:`matched_to`.

        Parameters
        ----------
        hazard : climada.hazard.Hazard
            Any hazard type, with ``units`` set. Its events are the index's
            events, in order.
        units : InsuredUnits
            The insured units.
        index_stat : {"at_centroid", "mean", "max"} or float, default "at_centroid"
            ``"at_centroid"``: the intensity at the assigned centroid, for
            one-point units only. ``"mean"``: the mean over the unit's
            centroids. ``"max"``: their maximum. A number in ``[0, 100]``:
            that percentile, linearly interpolated (``numpy.percentile``).
        weight : {"value"}, optional
            ``"value"`` weights the ``"mean"`` by the total value of the
            unit's points at each centroid. Default: equal weights per
            centroid.
        tail : {"upper", "lower"}, default "upper"
            Which side of the index is severe.
        absent_value : float, optional
            What a zero (no recorded intensity) reads as. Required with
            ``tail="lower"``. Default: zero stays zero.
        source : str, optional
            What the hazard set is, for the definition, e.g. its name.

        Returns
        -------
        ParametricIndex
            In ``hazard.units``, with the definition ``IndexDefinition("hazard",
            unit=hazard.units, tail=tail, index_stat=index_stat,
            weight=weight, source=source, parameters={"haz_type":
            hazard.haz_type})``, and ``absent_value`` in the parameters if
            given.

        Raises
        ------
        ParametricInputError
            If ``hazard.units`` is empty, the hazard has no events,
            ``index_stat="at_centroid"`` meets a unit of several points, or
            ``tail="lower"`` has no ``absent_value``.
        CoverDefinitionError
            If ``index_stat``, ``weight`` or ``tail`` is invalid.
        UnitMappingError
            If a point of a unit has no centroid in the hazard
            (:meth:`InsuredUnits.centroids`).
        """
        if not isinstance(hazard.units, str) or not hazard.units:
            raise ParametricInputError(
                "hazard.units must name the unit of the intensity, e.g. 'm/s'; got "
                f"{hazard.units!r}"
            )
        parameters: dict[str, Any] = {"haz_type": str(hazard.haz_type)}
        if absent_value is not None:
            parameters["absent_value"] = check_real("absent_value", absent_value)
        definition = IndexDefinition(
            HAZARD,
            unit=hazard.units,
            tail=tail,
            index_stat=index_stat,
            weight=weight,
            source=source,
            parameters=parameters,
        )
        return cls._on_hazard(definition, hazard, units)

    @classmethod
    def from_cat_in_circle(
        cls,
        tracks: Any,
        units: InsuredUnits,
        *,
        hazard: Any,
        radius_km: float,
        metric: Metric = "category",
        breakpoints_kn: Sequence[float] = SAFFIR_SIMPSON_KN,
        track_step_hours: float | None = DEFAULT_TRACK_STEP_HOURS,
    ) -> ParametricIndex:
        """Cat-in-a-circle index from tracks (:mod:`.index_tracks`).

        Parameters
        ----------
        tracks : climada.hazard.TCTracks
            The storms' tracks, one per hazard event, matched on the event
            name (as ``TropCyclone.from_tracks`` names events).
        units : InsuredUnits
            The insured units. A unit of several points reads the track
            within ``radius_km`` of any of them.
        hazard : climada.hazard.Hazard
            Sets the events: their order, ids and names.
        radius_km, metric, breakpoints_kn, track_step_hours
            As in
            :func:`~climada_petals.engine.parametric.index_tracks.cat_in_circle`.

        Returns
        -------
        ParametricIndex
            The index, in categories or m/s, with the definition of
            :func:`~climada_petals.engine.parametric.index_tracks.cat_in_circle`.

        Raises
        ------
        ParametricInputError
            If a parameter is out of range, a hazard event has no track or
            two, or a track's wind is not in knots.
        """
        _check_units(units)
        values, definition = cat_in_circle(
            tracks,
            units,
            hazard,
            radius_km=radius_km,
            metric=metric,
            breakpoints_kn=breakpoints_kn,
            track_step_hours=track_step_hours,
        )
        return cls._of_hazard_events(values, hazard, units, definition)

    @classmethod
    def from_table(
        cls,
        table: pd.DataFrame,
        units: InsuredUnits,
        *,
        definition: IndexDefinition,
        event_id: str = "event_id",
        unit_id: str = "unit_id",
        value: str = "value",
    ) -> ParametricIndex:
        """Index from a user's table in long form: one row per event and unit.

        Parameters
        ----------
        table : pandas.DataFrame
            One row per event and unit, in any order, with every unit for
            every event. Units are matched on their ids as strings.
        units : InsuredUnits
            The insured units; the table names each of them.
        definition : IndexDefinition
            What the values are, e.g. ``IndexDefinition("table", unit="mm",
            source="station readings")``.
        event_id, unit_id, value : str
            The columns of the event ids (integers), the unit ids and the
            index values (finite).

        Returns
        -------
        ParametricIndex
            The events in ascending ``event_id``.

        Raises
        ------
        ParametricInputError
            If a column is missing, the table is empty, an event id is not
            an integer, a value is not finite, or an event and unit appear
            twice or not at all.
        UnitMappingError
            If the table names a unit that ``units`` doesn't hold.
        """
        _check_units(units)
        columns = [event_id, unit_id, value]
        if not isinstance(table, pd.DataFrame) or not set(columns) <= set(table):
            raise ParametricInputError(
                f"table must be a pandas DataFrame with the columns {columns}"
            )
        if table.empty:
            raise ParametricInputError("the table has no rows")
        events, cell = _table_cells(table[event_id], table[unit_id], units)
        values = np.empty(events.size * units.n_units)
        values[cell] = finite_array(table[value].to_numpy(), value, 1)
        return cls(
            values.reshape(events.size, units.n_units),
            events,
            units.unit_id,
            definition,
            units=units,
        )

    @classmethod
    def from_definition(
        cls,
        definition: IndexDefinition,
        hazard: Any,
        units: InsuredUnits,
        *,
        tracks: Any = None,
    ) -> ParametricIndex:
        """Rebuild a cover's index on another hazard set (a scenario, say).

        Parameters
        ----------
        definition : IndexDefinition
            A definition that :meth:`from_hazard` (kind ``"hazard"``) or
            :meth:`from_cat_in_circle` (kind ``"cat_in_circle"``) built, e.g.
            ``cover.index``.
        hazard : climada.hazard.Hazard
            The hazard set: of the definition's ``haz_type`` and unit for a
            hazard index; it sets the events of a cat-in-a-circle index.
        units : InsuredUnits
            The insured units.
        tracks : climada.hazard.TCTracks, optional
            The tracks of a cat-in-a-circle index; none for a hazard index.

        Returns
        -------
        ParametricIndex
            With ``definition`` as its definition.

        Raises
        ------
        ParametricInputError
            If the definition is of another kind (a table can't be rebuilt
            from a hazard: use :meth:`from_table`), its parameters are not
            the builder's, the hazard's type or unit differs from it, or
            ``tracks`` is missing or not wanted.
        """
        if not isinstance(definition, IndexDefinition):
            raise ParametricInputError(
                f"definition must be an IndexDefinition, got {definition!r}"
            )
        if definition.kind == HAZARD and tracks is None:
            return cls._on_hazard(definition, hazard, units)
        if definition.kind == CAT_IN_CIRCLE and tracks is not None:
            index = cls.from_cat_in_circle(
                tracks, units, hazard=hazard, **cat_in_circle_arguments(definition)
            )
            if index.definition != definition:
                raise ParametricInputError(
                    "the definition is not the one from_cat_in_circle builds for "
                    f"its parameters: {definition.to_dict()} against "
                    f"{index.definition.to_dict()}"
                )
            return index
        raise ParametricInputError(
            f"an index of kind {definition.kind!r} can't be rebuilt from a hazard "
            f"{'without' if tracks is None else 'with'} tracks; kind {HAZARD!r} takes "
            f"no tracks, {CAT_IN_CIRCLE!r} needs them, and a table is read with "
            "from_table"
        )

    @classmethod
    def _on_hazard(
        cls, definition: IndexDefinition, hazard: Any, units: InsuredUnits
    ) -> ParametricIndex:
        """Compute a hazard index from its definition (:meth:`from_hazard`)."""
        _check_units(units)
        parameters: Mapping[str, Any] = definition.parameters or {}
        if (
            set(parameters) - _HAZARD_PARAMETERS
            or "haz_type" not in parameters
            or definition.index_stat is None
        ):
            raise ParametricInputError(
                "a hazard index definition has an index_stat and the parameters "
                f"haz_type and, optionally, absent_value; got {definition.to_dict()}"
            )
        if (parameters["haz_type"], definition.unit) != (hazard.haz_type, hazard.units):
            raise ParametricInputError(
                f"the definition is on a {parameters['haz_type']!r} hazard in "
                f"{definition.unit!r}; got {hazard.haz_type!r} in {hazard.units!r}"
            )
        absent = parameters.get("absent_value")
        if definition.tail == "lower" and absent is None:
            raise ParametricInputError(
                "a lower-tail index needs absent_value: the reading that a zero "
                "(no recorded intensity) stands for"
            )
        if not np.asarray(hazard.event_id).size:
            raise ParametricInputError("the hazard has no events")
        values = _footprint(
            hazard,
            units,
            definition,
            None if absent is None else check_real("absent_value", absent),
        )
        return cls._of_hazard_events(values, hazard, units, definition)

    @classmethod
    def _of_hazard_events(
        cls,
        values: NDArray[np.float64],
        hazard: Any,
        units: InsuredUnits,
        definition: IndexDefinition,
    ) -> ParametricIndex:
        """Return an index on the hazard's events, with their names."""
        names = [str(name) for name in hazard.event_name]
        return cls(
            values,
            np.asarray(hazard.event_id),
            units.unit_id,
            definition,
            event_name=names if len(names) == values.shape[0] else None,
            units=units,
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


def _check_units(units: object) -> None:
    """Raise unless ``units`` is an :class:`InsuredUnits`."""
    if not isinstance(units, InsuredUnits):
        raise ParametricInputError(
            "units must be InsuredUnits, e.g. InsuredUnits.points(exposures); got "
            f"{type(units).__name__}"
        )


def _table_cells(
    event_ids: pd.Series, unit_ids: pd.Series, units: InsuredUnits
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return a table's events, sorted, and the flat cell of each row.

    Raises
    ------
    UnitMappingError
        If a row names a unit that ``units`` doesn't hold.
    ParametricInputError
        If an event id is not an integer, or an event and unit appear twice
        or not at all.
    """
    by_text = {str(unit): k for k, unit in enumerate(units.unit_id)}
    keys = [str(unit) for unit in unit_ids]
    unknown = sorted({key for key in keys if key not in by_text})
    if unknown:
        raise UnitMappingError(f"the table names units not in units: {unknown[:5]}")
    events, row = np.unique(
        integer_array(event_ids.to_numpy(), str(event_ids.name)), return_inverse=True
    )
    cell = np.asarray(row).reshape(-1) * units.n_units + [by_text[k] for k in keys]
    counts = np.bincount(cell, minlength=events.size * units.n_units)
    for wrong, what in ((counts > 1, "twice"), (counts == 0, "not at all")):
        if wrong.any():
            first = int(np.flatnonzero(wrong)[0])
            raise ParametricInputError(
                f"{int(wrong.sum())} (event, unit) pairs appear {what} in the "
                f"table, e.g. ({int(events[first // units.n_units])}, "
                f"{units.unit_id[first % units.n_units]!r}); give every unit one "
                "value per event"
            )
    return events, cell


def _positions(keys: Sequence[Any] | NDArray[Any], what: str) -> dict[Any, int]:
    """Return the position of each key; raise if a key repeats."""
    listed = keys.tolist() if isinstance(keys, np.ndarray) else list(keys)
    position = {key: k for k, key in enumerate(listed)}
    if len(position) != len(listed):
        raise EventAlignmentError(
            f"{what} repeat: {len(listed) - len(position)} of {len(listed)} are "
            "duplicates, so they don't identify the events"
        )
    return position


def _footprint(
    hazard: Any,
    units: InsuredUnits,
    definition: IndexDefinition,
    absent: float | None,
) -> NDArray[np.float64]:
    """Return the statistic of the intensity over each unit's centroids.

    Units with the same number of distinct centroids are reduced together,
    on blocks of events of at most :data:`_CHUNK` densified values.

    Raises
    ------
    ParametricInputError
        If ``"at_centroid"`` meets a unit of several points.
    """
    columns, weight, groups = _unit_centroids(
        units, units.centroids(hazard), int(hazard.centroids.size), definition
    )
    intensity = hazard.intensity
    out = np.empty((intensity.shape[0], units.n_units))
    step = max(1, _CHUNK // columns.size)
    for first in range(0, intensity.shape[0], step):
        rows = slice(first, first + step)
        block = intensity[rows][:, columns].toarray()
        if absent is not None:
            block[block == 0.0] = absent
        for members, pairs in groups:
            out[rows, members] = _statistic(
                block[:, pairs], definition.index_stat, weight[pairs]
            )
    return out


def _statistic(
    values: NDArray[np.float64], stat: str | float | None, weight: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reduce ``values`` over its last axis by ``stat``, a unit's centroids."""
    if stat == "mean":
        return (values * weight).sum(axis=2)
    if isinstance(stat, str):
        return values.max(axis=2)
    return np.percentile(values, stat, axis=2)


def _unit_centroids(
    units: InsuredUnits,
    centroid: NDArray[np.intp],
    n_centroids: int,
    definition: IndexDefinition,
) -> tuple[
    NDArray[np.intp],
    NDArray[np.float64],
    list[tuple[NDArray[np.intp], NDArray[np.intp]]],
]:
    """Return the pairs of a unit and one of its distinct centroids.

    Returns
    -------
    columns : numpy.ndarray of int
        The centroid of each pair, the pairs sorted by unit.
    weight : numpy.ndarray of float
        Each pair's weight in its unit's mean: equal, or the value of the
        unit's points at the centroid; they sum to 1 per unit.
    groups : list of (members, pairs)
        Per number ``m`` of centroids, the units that have ``m`` and their
        pairs, shape ``(len(members), m)``.

    Raises
    ------
    ParametricInputError
        If ``"at_centroid"`` meets a unit of several points.
    """
    unit_of = units.point_unit
    inside = unit_of >= 0
    if definition.index_stat == "at_centroid" and np.any(
        np.bincount(unit_of[inside]) != 1
    ):
        raise ParametricInputError(
            "index_stat='at_centroid' takes units of one point each; for groups "
            "use 'mean', 'max' or a percentile"
        )
    pairs, inverse = np.unique(
        unit_of[inside].astype(np.int64) * n_centroids + centroid[inside],
        return_inverse=True,
    )
    pair_unit = pairs // n_centroids
    share = np.ones(pairs.size)
    if definition.weight == "value":
        share = np.bincount(
            np.asarray(inverse).reshape(-1), weights=units.value[inside]
        )
    share = share / np.bincount(pair_unit, weights=share)[pair_unit]
    sizes = np.bincount(pair_unit)
    starts = np.cumsum(sizes) - sizes
    groups = [
        (members, starts[members][:, None] + np.arange(size))
        for size in np.unique(sizes)
        for members in [np.flatnonzero(sizes == size)]
    ]
    return pairs % n_centroids, share, groups
