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

Outcomes and their probabilities: the annual and the event basis.

The objectives, the premium, the budget and the basis-risk metrics never use
event frequencies directly. They use an :class:`OutcomeMeasure`, which says
which **outcome** each event belongs to and the **probability** of each
outcome. Per-event quantities (for example the uncovered loss
``(L_e - payout_e)^+``) are summed into outcomes and then weighted with the
probabilities.

``N`` is the number of simulation years of the whole hazard set (the
index's ``years``) and ``S`` the years a calculation uses: all of them, or a
training or test block from
:func:`~climada_petals.engine.parametric.index.split_years`.

``"annual"`` (recommended)
    One outcome per simulation year: the events of a year are summed, so
    the uncovered loss of a year is ``U_y = sum_{e in y} (L_e - payout_e)^+``.
    Each year has probability ``1 / |S|``. Every event must have the
    frequency ``1 / N`` (one simulated history of ``N`` years); otherwise
    the year structure and the frequencies disagree, and the measure
    raises. Years without events are outcomes without loss; they are pooled
    into one last outcome with probability ``(number of such years) / |S|``.
    Pooling changes no expectation, quantile or CVaR, because all those
    years have the value 0 for every per-event quantity. ``CVaR_beta`` is
    then the mean of the worst ``1 - beta`` of years: ``beta = 0.99`` is
    the 100-year level and beyond.
``"event"``
    One outcome per event, with probability ``frequency * N / |S|`` (the
    event's annual frequency, re-expressed over the years used), plus one
    "no event" outcome with the remaining probability. The frequencies must
    sum to at most one event per year.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final, Literal, get_args

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from ._checks import finite_array, integer_array, read_only
from .errors import ParametricInputError

__all__ = [
    "BASES",
    "FREQUENCY_RTOL",
    "PROBABILITY_ATOL",
    "Basis",
    "OutcomeMeasure",
    "outcome_measure",
]

Basis = Literal["annual", "event"]
"""What an outcome is: a simulation year, or an event."""

BASES: Final[tuple[str, ...]] = get_args(Basis)
"""The values of :data:`Basis`: ``("annual", "event")``."""

PROBABILITY_ATOL: Final[float] = 1e-9
"""Absolute tolerance for outcome probabilities summing to one."""

FREQUENCY_RTOL: Final[float] = 1e-6
"""Relative tolerance of the annual basis's check ``frequency = 1 / N``."""


@dataclass(frozen=True, eq=False)
class OutcomeMeasure:
    """Probability measure over outcomes, with the event-to-outcome mapping.

    Parameters
    ----------
    basis : {"annual", "event"}
        The basis the measure was built for.
    outcome_of_event : array_like of int
        Outcome of each event, shape ``(n_events,)``, values in
        ``[0, n_outcomes)``. An outcome may hold no event (the pooled
        loss-free years, or "no event").
    probability : array_like of float
        Probability of each outcome, shape ``(n_outcomes,)``: finite, >= 0
        and summing to one within :data:`PROBABILITY_ATOL`.
    n_years : int
        Number of simulation years the measure covers (``|S|``).

    Raises
    ------
    ParametricInputError
        If an array has the wrong shape, an outcome is out of range or the
        probabilities are not a probability vector.
    """

    basis: str
    outcome_of_event: NDArray[np.int64]
    probability: NDArray[np.float64]
    n_years: int

    def __post_init__(self) -> None:
        """Validate the fields and store read-only copies."""
        if self.basis not in BASES:
            raise ParametricInputError(
                f"basis must be one of {list(BASES)}, got {self.basis!r}"
            )
        outcomes = integer_array(self.outcome_of_event, "outcome_of_event")
        prob = finite_array(self.probability, "probability", 1)
        if prob.size == 0 or np.any(prob < 0):
            raise ParametricInputError("probability must be non-empty and >= 0")
        if abs(float(prob.sum()) - 1.0) > PROBABILITY_ATOL:
            raise ParametricInputError(
                f"probability must sum to 1, got {float(prob.sum())!r}"
            )
        if outcomes.size and (outcomes.min() < 0 or outcomes.max() >= prob.size):
            raise ParametricInputError(f"outcome_of_event must lie in [0, {prob.size})")
        object.__setattr__(self, "outcome_of_event", read_only(outcomes))
        object.__setattr__(self, "probability", read_only(prob))
        object.__setattr__(self, "n_years", int(self.n_years))

    @property
    def n_events(self) -> int:
        """Number of events the measure maps."""
        return int(self.outcome_of_event.shape[0])

    @property
    def n_outcomes(self) -> int:
        """Number of outcomes."""
        return int(self.probability.shape[0])

    @property
    def event_probability(self) -> NDArray[np.float64]:
        """Probability of each event's outcome, shape ``(n_events,)``.

        ``expectation(values) == event_probability @ values`` for per-event
        values, because an expectation of outcome sums is linear.
        """
        return self.probability[self.outcome_of_event]

    @cached_property
    def grouping_matrix(self) -> sp.csr_matrix:
        """Sparse 0/1 matrix that sums events into outcomes.

        Shape ``(n_outcomes, n_events)``: entry ``(o, e)`` is 1 if event
        ``e`` belongs to outcome ``o``. Outcomes without events have an
        empty row.
        """
        n_events = self.n_events
        return sp.csr_matrix(
            (np.ones(n_events), (self.outcome_of_event, np.arange(n_events))),
            shape=(self.n_outcomes, n_events),
        )

    def aggregate(self, values: ArrayLike) -> NDArray[np.float64]:
        """Sum per-event values into outcomes.

        Parameters
        ----------
        values : array_like of float
            Shape ``(n_events,)`` or ``(n_events, k)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_outcomes,)`` or ``(n_outcomes, k)``; 0 for outcomes
            without events.

        Raises
        ------
        ParametricInputError
            If the first dimension is not ``n_events``.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim not in (1, 2) or arr.shape[0] != self.n_events:
            raise ParametricInputError(
                f"values must have {self.n_events} rows, got shape {arr.shape}"
            )
        out = np.zeros((self.n_outcomes, *arr.shape[1:]), dtype=np.float64)
        np.add.at(out, self.outcome_of_event, arr)
        return out

    def expectation(self, values: ArrayLike) -> float:
        """Return the expected outcome sum of per-event values.

        Parameters
        ----------
        values : array_like of float
            Per-event values, shape ``(n_events,)``.

        Returns
        -------
        float
            ``probability @ aggregate(values)``.
        """
        return float(self.probability @ self.aggregate(values))


def outcome_measure(
    event_year: ArrayLike,
    frequency: ArrayLike,
    *,
    basis: str,
    years: ArrayLike,
    n_catalogue_years: int,
) -> OutcomeMeasure:
    """Build the outcome measure of a set of events (module docstring).

    Parameters
    ----------
    event_year : array_like of int
        Simulation year of each event, shape ``(n_events,)``.
    frequency : array_like of float
        Annual frequency of each event, shape ``(n_events,)``.
    basis : {"annual", "event"}
        The basis.
    years : array_like of int
        The simulation years the events represent (``S``), including years
        without events. Unique; every event's year must be among them.
    n_catalogue_years : int
        Number of simulation years of the whole hazard set (``N``).

    Returns
    -------
    OutcomeMeasure
        The measure, aligned with the events.

    Raises
    ------
    ParametricInputError
        If there are no events, a year is missing from ``years``, or the
        frequencies do not fit the basis: on ``"annual"`` a frequency other
        than ``1 / N``, on ``"event"`` frequencies that sum to more than one
        event per year.
    """
    if basis not in BASES:
        raise ParametricInputError(f"basis must be one of {list(BASES)}, got {basis!r}")
    rows = integer_array(event_year, "event_year")
    freq = finite_array(frequency, "frequency", 1)
    sample = np.unique(integer_array(years, "years"))
    if rows.size == 0:
        raise ParametricInputError("cannot build an outcome measure without events")
    if freq.shape != rows.shape or np.any(freq <= 0):
        raise ParametricInputError(
            "frequency must hold one value > 0 per event, "
            f"got shape {freq.shape} for {rows.size} events"
        )
    stray = np.setdiff1d(rows, sample)
    if stray.size:
        raise ParametricInputError(
            f"events lie in years outside the given years, e.g. {stray[:5].tolist()}"
        )
    n_total = int(n_catalogue_years)
    if basis == "event":
        return _event_measure(freq, n_total, sample.size)
    return _annual_measure(rows, freq, n_total, sample.size)


def _event_measure(
    freq: NDArray[np.float64], n_total: int, n_sample: int
) -> OutcomeMeasure:
    """Return the event basis's measure (module docstring).

    Raises
    ------
    ParametricInputError
        If the frequencies sum to more than one event per year.
    """
    prob = freq * (n_total / n_sample)
    none = 1.0 - float(prob.sum())
    if none < -PROBABILITY_ATOL:
        raise ParametricInputError(
            "the event basis needs the frequencies to sum to at most one event "
            f"per year, got {1.0 - none:.6g}; use the annual basis"
        )
    if none <= PROBABILITY_ATOL:
        return OutcomeMeasure("event", np.arange(freq.size), prob, n_sample)
    return OutcomeMeasure(
        "event", np.arange(freq.size), np.append(prob, none), n_sample
    )


def _annual_measure(
    rows: NDArray[np.int64], freq: NDArray[np.float64], n_total: int, n_sample: int
) -> OutcomeMeasure:
    """Return the annual basis's measure (module docstring).

    Raises
    ------
    ParametricInputError
        If a frequency is not ``1 / N``.
    """
    implied = freq * n_total
    if not np.allclose(implied, 1.0, rtol=0.0, atol=FREQUENCY_RTOL):
        raise ParametricInputError(
            "the annual basis needs every event's frequency to be 1 / N with "
            f"N = {n_total} simulation years; got frequency x N between "
            f"{implied.min():.6g} and {implied.max():.6g}. For an ensemble of "
            "perturbed tracks, give each member its own simulation years "
            "(event_year and years of the index)"
        )
    occupied, outcome = np.unique(rows, return_inverse=True)
    prob = np.full(occupied.size, 1.0 / n_sample)
    n_empty = n_sample - occupied.size
    if n_empty:
        prob = np.append(prob, n_empty / n_sample)
    return OutcomeMeasure("annual", outcome.reshape(-1), prob, n_sample)
