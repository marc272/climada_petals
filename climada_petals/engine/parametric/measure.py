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

Outcomes and their weights: the event basis and the annual basis.

A CLIMADA hazard is an event set: each event has a frequency per year, and
years exist only if the catalogue happens to carry them. A **basis** says
what one **outcome** is, the thing a tail measure ranks: an event, or a
simulated year. It is always stated; there is no default.

The measure
-----------
An :class:`OutcomeMeasure` holds a sparse matrix ``G`` of shape
``(n_outcomes, n_rows)``, with the count of row ``r`` (an event, or an event
and a draw) in outcome ``o``, and the outcome weights ``w``. A per-row
quantity ``x`` gives the outcome values ``G @ x``, and ``E[X] = w @ G @ x``.
Inside an outcome the uncovered loss is summed over its events,
``U_o = sum_e G_oe (L_e - P_e)^+``: each event pays against its own loss,
so an over-payment in one event doesn't offset a shortfall in another of
the same year. The residual ``R_o = L_o - P_o`` is summed before it is
squared.

The event basis, :class:`EventBasis`
------------------------------------
One outcome per event, with weight ``f_e``, the event's frequency per year.
It is valid for every hazard: nothing requires the frequencies to sum to at
most one event per year, which many event sets exceed.

* ``E[X] = sum_e f_e X_e`` is an expected value **per year**: ``E[L]`` is
  ``Impact.aai_agg`` restricted to the units.
* ``1 - beta`` is read as a **frequency per year**. ``VaR_beta(X)`` is the
  smallest ``x`` with ``sum_{X_e > x} f_e <= 1 - beta``: CLIMADA's event
  exceedance curve at the return period ``T = 1 / (1 - beta)``, up to that
  curve's interpolation. ``CVaR_beta(X)`` is the mean of the worst
  ``1 - beta`` of frequency. If the frequencies sum to less than
  ``1 - beta``, the missing mass counts as zero.
* Frequencies are not divided by their sum: ``beta`` would then depend on
  which loss-free events a file holds, and the tropical-cyclone files hold
  every storm of the global set. With frequencies as weights, loss-free
  events below the attachment change nothing.

The calibration, the pricings and the metrics read these weights as they
are, whatever they sum to
(:mod:`~climada_petals.engine.parametric.riskmeasures`).

The annual basis: ``CatalogueYears`` or ``YearSets``
----------------------------------------------------
One outcome per simulated year, each of weight ``1 / N``: ``CVaR_beta`` is
the mean of the worst ``1 - beta`` of years, so ``beta = 0.99`` is the
100-year level and beyond. No single route fits every hazard, and the route
changes tail numbers, so it is part of the setup of every number. Both
routes are in :mod:`~climada_petals.engine.parametric.annual`.

==================  ==========================  =============================
                    ``CatalogueYears``          ``YearSets``
==================  ==========================  =============================
A year is           a label on each event       a draw of Poisson(lambda)
                                                events from the set
Constructors        ``from_dates``,             ``from_sampling_vect``,
                    ``from_event_names``,       ``from_frequencies``
                    ``from_labels``
Frequencies         each kept event has         any, so climate-scaled
                    ``1 / N``                   frequencies work
Empty years         as the catalogue has them   sampled
Noise               none beyond the             sampling error, stated by
                    catalogue's                 ``n_years`` and the seed
==================  ==========================  =============================

Years without events are outcomes without loss. They are pooled into one
outcome with their summed weight, which changes no expectation, quantile or
CVaR.

* **Catalogue years** raise
  :class:`~climada_petals.engine.parametric.errors.YearRouteError` when
  ``frequency x N`` differs from 1 by more than :data:`FREQUENCY_ATOL` on an
  event the sample keeps: a set scaled by ``apply_climate_scenario_knu``
  does, and so do dates that don't carry the years (a synthetic set whose
  tracks all start on one date spans one year of dates, not ``N``).
* **Year sets** follow ``climada.util.yearsets``, with three differences:
  events are always drawn **with replacement** (without, the rarest events
  are drawn several times too often from a few thousand years on); the
  counts and the draws come from the caller's ``numpy.random.Generator``,
  so numpy's global random state is never touched
  (:func:`~climada_petals.engine.parametric.annual.sample_years`);
  and ``correction_fac`` is not applied, because one factor for every year
  repairs the mean and not the tail. A ``sampling_vect`` holds positions in
  the event arrays, not event ids.

Which route suits which hazard is guidance, not code: catalogue years for a
set whose years are the data (STORM, from the names; flood and wildfire,
which hold one event per year or member-year; observed sets, from the
dates); year sets for ensembles whose members aren't years of their own,
and for every frequency-scaled set.

Draws
-----
A sample can hold several rows per event, one per draw (an impact-function
draw, say), each with frequency ``f_e / n_draws``. Rows are ordered by draw,
then event. On the annual basis an outcome is a year and a draw, of weight
``1 / (N n_draws)``.

Blocks
------
Every basis gives each event a **block**, the unit a split keeps together
(:mod:`~climada_petals.engine.parametric.sample`): the given label, such as
an ensemble member's parent storm (:data:`ENSEMBLE_PARENT`), or else the
event's year on catalogue years and the event itself on the other routes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from ._checks import check_int, finite_array, integer_array, read_only
from .errors import EventAlignmentError, ParametricInputError, YearRouteError

__all__ = [
    "ANNUAL",
    "CATALOGUE",
    "ENSEMBLE_PARENT",
    "EVENT",
    "FREQUENCY",
    "FREQUENCY_ATOL",
    "PROBABILITY_ATOL",
    "YEARSETS",
    "EventBasis",
    "OutcomeMeasure",
    "labels_from_names",
]

EVENT: Final[str] = "event"
"""The event basis, with one outcome per event, weighted by its frequency."""

ANNUAL: Final[str] = "annual"
"""The annual basis, with one outcome per simulated year."""

FREQUENCY: Final[str] = "frequency"
"""The route of the event basis, on which the events' frequencies are the weights."""

CATALOGUE: Final[str] = "catalogue"
"""The annual route of :class:`~climada_petals.engine.parametric.annual.CatalogueYears`."""

YEARSETS: Final[str] = "yearsets"
"""The annual route of :class:`~climada_petals.engine.parametric.annual.YearSets`."""

_ROUTES: Final[frozenset[tuple[str, str]]] = frozenset(
    {(EVENT, FREQUENCY), (ANNUAL, CATALOGUE), (ANNUAL, YEARSETS)}
)

PROBABILITY_ATOL: Final[float] = 1e-9
"""Absolute tolerance for weights summing to one."""

FREQUENCY_ATOL: Final[float] = 1e-6
"""The largest ``|frequency x N - 1|`` on an event kept on catalogue years."""

ENSEMBLE_PARENT: Final[str] = r"(?P<block>.+?)(?:_gen[0-9]+)?"
"""Pattern of :func:`labels_from_names` for the members of perturbed tracks.

``TCTracks.calc_perturbed_trajectories`` names member ``m`` of storm ``sid``
``<sid>_gen<m>``; the ``block`` group is the parent's name, ``sid``.
"""


@dataclass(frozen=True, eq=False)
class OutcomeMeasure:
    """Outcome weights and the rows of each outcome (module docstring).

    Parameters
    ----------
    grouping : array_like or scipy.sparse matrix
        ``G``, shape ``(n_outcomes, n_rows)``: the count of row ``r`` in
        outcome ``o``, a whole number >= 0. Stored as CSR. An outcome may
        hold no row (the pooled years without events).
    weights : array_like of float
        ``w``, shape ``(n_outcomes,)``: finite, >= 0, with a positive sum.
        On the annual basis they sum to one within :data:`PROBABILITY_ATOL`.
    basis : {"event", "annual"}
        What an outcome is.
    route : {"frequency", "catalogue", "yearsets"}
        How the outcomes were built: ``"frequency"`` on the event basis.
    n_years : int, optional
        The number of years the outcomes stand for, on the annual basis;
        None on the event basis.

    Raises
    ------
    ParametricInputError
        If a field is invalid or two disagree.
    """

    grouping: Any
    weights: NDArray[np.float64]
    basis: str
    route: str
    n_years: int | None = None

    def __post_init__(self) -> None:
        """Validate the fields and store read-only copies."""
        if (self.basis, self.route) not in _ROUTES:
            raise ParametricInputError(
                f"basis and route must be one of {sorted(_ROUTES)}, got "
                f"{(self.basis, self.route)!r}"
            )
        weights = finite_array(self.weights, "weights", 1)
        if weights.size == 0 or np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ParametricInputError(
                "weights must be non-empty and >= 0, with a positive sum"
            )
        grouping = sp.csr_matrix(self.grouping, dtype=np.float64, copy=True)
        grouping.sum_duplicates()
        if grouping.shape[0] != weights.size:
            raise ParametricInputError(
                f"grouping has {grouping.shape[0]} rows for {weights.size} weights"
            )
        if np.any(grouping.data < 0) or np.any(
            grouping.data != np.round(grouping.data)
        ):
            raise ParametricInputError("grouping must hold whole counts >= 0")
        if self.basis == ANNUAL:
            if abs(float(weights.sum()) - 1.0) > PROBABILITY_ATOL:
                raise ParametricInputError(
                    f"annual weights must sum to 1, got {float(weights.sum())!r}"
                )
            object.__setattr__(
                self, "n_years", check_int("n_years", self.n_years, low=1)
            )
        elif self.n_years is not None:
            raise ParametricInputError("the event basis has no n_years")
        for part in (grouping.data, grouping.indices, grouping.indptr):
            part.setflags(write=False)
        object.__setattr__(self, "grouping", grouping)
        object.__setattr__(self, "weights", read_only(weights))

    @property
    def n_events(self) -> int:
        """Number of rows the measure maps (events, or events and draws)."""
        return int(self.grouping.shape[1])

    @property
    def n_outcomes(self) -> int:
        """Number of outcomes."""
        return int(self.grouping.shape[0])

    @property
    def total_weight(self) -> float:
        """The sum of the weights, 1 on the annual basis and ``sum f`` on the event basis."""
        return float(self.weights.sum())

    @cached_property
    def event_weight(self) -> NDArray[np.float64]:
        """Weight of each row in an expectation, ``G.T @ w``, shape ``(n_rows,)``.

        ``expectation(values) == event_weight @ values`` for per-row values.
        On the event basis it is each row's frequency.
        """
        out = np.asarray(self.grouping.T @ self.weights, dtype=np.float64).reshape(-1)
        out.setflags(write=False)
        return out

    @property
    def grouping_matrix(self) -> sp.csr_matrix:
        """``G``, the name the solvers read."""
        return self.grouping

    @property
    def probability(self) -> NDArray[np.float64]:
        """The weights, the name the objectives, pricings and metrics read.

        A probability vector on the annual basis; frequencies per year on the
        event basis, which may sum to more or less than 1.
        """
        return self.weights

    @property
    def event_probability(self) -> NDArray[np.float64]:
        """:attr:`event_weight`, the name the solvers read."""
        return self.event_weight

    def aggregate(self, values: ArrayLike) -> NDArray[np.float64]:
        """Sum per-row values into outcomes, ``G @ values``.

        Parameters
        ----------
        values : array_like of float
            Shape ``(n_rows,)`` or ``(n_rows, k)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_outcomes,)`` or ``(n_outcomes, k)``; 0 for outcomes
            without rows.

        Raises
        ------
        ParametricInputError
            If the first dimension is not ``n_rows``.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim not in (1, 2) or arr.shape[0] != self.n_events:
            raise ParametricInputError(
                f"values must have {self.n_events} rows, got shape {arr.shape}"
            )
        return np.asarray(self.grouping @ arr, dtype=np.float64)

    def expectation(self, values: ArrayLike) -> float:
        """Return the expected outcome sum of per-row values, ``w @ G @ values``.

        Parameters
        ----------
        values : array_like of float
            Per-row values, shape ``(n_rows,)``.

        Returns
        -------
        float
            Per year on both bases.
        """
        return float(self.weights @ self.aggregate(values))


@dataclass(frozen=True, eq=False, kw_only=True)
class EventBasis:
    """The event basis: one outcome per event, weighted by its frequency.

    The module docstring says what an expectation, a VaR and a CVaR mean on
    it. Valid for every hazard.

    Parameters
    ----------
    event_id : array_like of int, optional
        The events that ``block`` labels, unique.
    block : array_like, optional
        One label per event of ``event_id``, the unit a split keeps
        together, e.g. ``labels_from_names(hazard.event_name,
        ENSEMBLE_PARENT)`` so that the members of a perturbed storm stay
        together. Default: every event is its own block.

    Raises
    ------
    ParametricInputError
        If only one of ``event_id`` and ``block`` is given, or they don't
        have one label per unique event.
    """

    event_id: NDArray[np.int64] | None = None
    block: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        """Validate the labels and store read-only copies."""
        if (self.event_id is None) != (self.block is None):
            raise ParametricInputError("pass event_id and block together, or neither")
        if self.event_id is not None:
            ids = _event_ids(self.event_id)
            object.__setattr__(self, "event_id", ids)
            object.__setattr__(self, "block", _labels(self.block, ids.size, "block"))

    @property
    def kind(self) -> str:
        """``"event"``."""
        return EVENT

    @property
    def route(self) -> str:
        """``"frequency"``."""
        return FREQUENCY

    def blocks(self, event_id: ArrayLike) -> NDArray[Any]:
        """Return the block of each given event.

        Parameters
        ----------
        event_id : array_like of int
            The events.

        Returns
        -------
        numpy.ndarray
            Each event's block, read-only: its ``block`` label, or the event
            id if the basis has no labels.

        Raises
        ------
        EventAlignmentError
            If a block is given and an event has none.
        """
        ids = integer_array(event_id, "event_id")
        if self.event_id is None or self.block is None:
            return read_only(ids)
        return read_only(self.block[_positions_of(self.event_id, ids, "block")])

    def measure(
        self, event_id: ArrayLike, frequency: ArrayLike, *, n_draws: int = 1
    ) -> OutcomeMeasure:
        """Return the measure of the given events: weight ``f_e / n_draws`` per row.

        Parameters
        ----------
        event_id : array_like of int
            The events, in the order of the rows of each draw.
        frequency : array_like of float
            Each event's frequency per year, >= 0.
        n_draws : int, default 1
            Draws per event; rows are ordered by draw, then event.

        Returns
        -------
        OutcomeMeasure
            On the event basis: one outcome per row, of weight
            ``f_e / n_draws``.

        Raises
        ------
        EventAlignmentError
            If a block is given and an event has none.
        ParametricInputError
            If there is no event, or the frequencies are invalid or sum to 0.
        """
        ids, freq, draws = _events(event_id, frequency, n_draws)
        self.blocks(ids)
        return OutcomeMeasure(
            sp.identity(ids.size * draws, format="csr"),
            np.tile(freq, draws) / draws,
            EVENT,
            FREQUENCY,
        )


def labels_from_names(
    names: Sequence[object], pattern: str, groups: Sequence[str] = ("block",)
) -> dict[str, list[str]]:
    """Read labels from event names with a regular expression.

    Parameters
    ----------
    names : sequence
        The event names, e.g. ``hazard.event_name``.
    pattern : str
        A regular expression that each name must match as a whole, with the
        named ``groups``, e.g. :data:`ENSEMBLE_PARENT`.
    groups : sequence of str, default ("block",)
        The groups to read.

    Returns
    -------
    dict of str to list of str
        The text of each group, per name.

    Raises
    ------
    YearRouteError
        If a name doesn't match, or a group doesn't take part in a match.
    ParametricInputError
        If the pattern lacks a group.
    """
    regex = re.compile(pattern)
    missing = [group for group in groups if group not in regex.groupindex]
    if missing:
        raise ParametricInputError(f"pattern {pattern!r} has no named groups {missing}")
    texts = [str(name) for name in names]
    found = [regex.fullmatch(text) for text in texts]
    bad = [
        text
        for text, match in zip(texts, found, strict=True)
        if match is None or any(match[group] is None for group in groups)
    ]
    if bad:
        raise YearRouteError(
            f"{len(bad)} of {len(texts)} event names don't match {pattern!r} with "
            f"the groups {list(groups)}, e.g. {bad[:5]}"
        )
    return {
        group: [str(match[group]) for match in found if match is not None]
        for group in groups
    }


def _events(
    event_id: ArrayLike, frequency: ArrayLike, n_draws: int
) -> tuple[NDArray[np.int64], NDArray[np.float64], int]:
    """Return a measure's events, frequencies and draws, checked.

    Raises
    ------
    ParametricInputError
        If there is no event, the ids repeat, or a frequency is invalid.
    """
    ids = _event_ids(event_id)
    if ids.size == 0:
        raise ParametricInputError("cannot build an outcome measure without events")
    freq = finite_array(frequency, "frequency", 1)
    if freq.shape != ids.shape or np.any(freq < 0):
        raise ParametricInputError(
            f"frequency must hold one value >= 0 per event, got shape {freq.shape} "
            f"for {ids.size} events"
        )
    return ids, freq, check_int("n_draws", n_draws, low=1)


def _annual(counts: sp.csr_matrix, n_draws: int, route: str) -> OutcomeMeasure:
    """Return the annual measure of ``counts`` (years x events), over the draws.

    Each year and draw has weight ``1 / (n_years n_draws)``; the years
    without events are pooled into one last outcome.
    """
    n_years = counts.shape[0]
    occupied = np.diff(counts.indptr) > 0
    grouping = sp.block_diag([counts[occupied]] * n_draws, format="csr")
    weights = np.full(grouping.shape[0], 1.0 / (n_years * n_draws))
    n_empty = int(n_years - occupied.sum())
    if n_empty:
        grouping = sp.vstack(
            [grouping, sp.csr_matrix((1, grouping.shape[1]))], format="csr"
        )
        weights = np.append(weights, n_empty / n_years)
    return OutcomeMeasure(grouping, weights, ANNUAL, route, n_years)


def _event_ids(event_id: ArrayLike) -> NDArray[np.int64]:
    """Return unique event ids, read-only.

    Raises
    ------
    ParametricInputError
        If the ids are not whole numbers or repeat.
    """
    ids = integer_array(np.asarray(event_id), "event_id")
    if np.unique(ids).size != ids.size:
        raise ParametricInputError("event_id must be unique")
    return read_only(ids)


def _labels(labels: ArrayLike, n_events: int, name: str) -> NDArray[Any]:
    """Return one label per event, read-only.

    Raises
    ------
    ParametricInputError
        If there is not one label per event, or a label is missing.
    """
    arr = np.asarray(labels)
    if arr.shape != (n_events,):
        raise ParametricInputError(
            f"{name} must hold one label per event, got shape {arr.shape} for "
            f"{n_events} events"
        )
    if np.any(pd.isna(arr)):
        raise ParametricInputError(f"{name} holds missing labels")
    return read_only(arr)


def _positions_of(
    own: NDArray[np.int64], wanted: NDArray[np.int64], what: str
) -> NDArray[np.intp]:
    """Return the position in ``own`` (unique ids) of each event of ``wanted``.

    Raises
    ------
    EventAlignmentError
        If an event of ``wanted`` is not in ``own``.
    """
    rows = pd.Index(own).get_indexer(wanted)
    missing = rows < 0
    if missing.any():
        raise EventAlignmentError(
            f"{int(missing.sum())} events have no {what}, e.g. "
            f"{wanted[missing][:5].tolist()}"
        )
    return np.asarray(rows, dtype=np.intp)
