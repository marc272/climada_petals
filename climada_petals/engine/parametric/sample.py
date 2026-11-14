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

Outcome samples: what a cover is designed or evaluated on, and their splits.

An :class:`OutcomeSample` joins an index
(:class:`~climada_petals.engine.parametric.index.ParametricIndex`), the loss
per event and insured unit
(:class:`~climada_petals.engine.parametric.units.InsuredUnits`), each event's
frequency per year, and a basis
(:mod:`~climada_petals.engine.parametric.measure`), which says what an
outcome is. The basis is a required argument.

:meth:`OutcomeSample.from_impact` builds one from a CLIMADA ``Impact``. It
keeps the events with a loss or a non-zero index at some unit; the others
add nothing to any outcome of a cover that pays nothing at an index of 0,
and they are counted (``n_left_out``) and logged, not dropped silently. On
a lower-tail index every event is kept, because an index of 0 can be severe
there. The catalogue route's frequency check runs on the kept events: the
tropical-cyclone files hold every storm of the globe, with other
frequencies only on storms that never reach a given country.
:meth:`OutcomeSample.from_impact_draws` takes one ``Impact`` per draw of
the impact functions, with the same rule.

Splits with no leakage
----------------------
:func:`holdout` and :func:`year_block_kfold` split by the **block**, the
unit that must stay together (:mod:`~climada_petals.engine.parametric.measure`),
and record what they did (:class:`SplitRecord`).

=========================  ===============================================
Basis and route            Rule
=========================  ===============================================
``CatalogueYears``         Contiguous blocks of years: the first
                           ``train_fraction`` of the years calibrate, as
                           in the core. With block labels, such as the
                           parent of each ensemble member, a cut moves to
                           the nearest year boundary that no block
                           crosses, the earlier one on a tie;
                           :class:`~climada_petals.engine.parametric.errors.LeakageError`
                           if there is none. Takes no ``rng``.
``YearSets``               Pools of blocks: the blocks, in an order
                           shuffled by ``rng``, fill the pools in turn by
                           frequency, each block in the pool that holds
                           the middle of its frequency. A pool's
                           frequencies are divided by its share of the
                           total, so both sides keep ``lambda``, and each
                           side samples ``round(share N)`` years from its
                           own events: events are split before years are
                           sampled, so no event is on both sides.
``EventBasis``             Pools of blocks, as above, with the
                           frequencies rescaled the same way.
=========================  ===============================================

:func:`check_no_leakage` raises if two samples share an event or a block;
both splits run it on their sides.

Lineage
-------
Every sample has a **source token**, :attr:`OutcomeSample.source`. A whole
sample's token is the fingerprint of its events: their ids, frequencies,
index values and losses, with the unit ids, in the order of the event ids,
whatever the basis. Every part of it keeps that token: the sides of
:func:`holdout` and :func:`year_block_kfold`, a :meth:`~OutcomeSample.select`
and a :meth:`~OutcomeSample.unit`. So two samples have one token when they
hold the same events with the same data, such as two samples of the same
index and ``Impact`` on two bases, or when one was cut from the other, or
both from one sample. A mixture of a scenario's members has the token of
its members' tokens, in order
(:mod:`~climada_petals.engine.parametric._mixture`).

:func:`check_out_of_sample` compares a design sample with an evaluation
sample only when they have one token, and then raises on a shared event or
block, as :func:`check_no_leakage` does. Event ids and years alone can't
decide it: another hazard set's can coincide with the design sample's by
chance. A sample of the same events with other frequencies, exposures or
impact functions (a frequency-scaled set, impact-function draws) has
another token, as another scenario: it is not compared.
:func:`~climada_petals.engine.parametric.evaluate.evaluate` runs the check
on every out-of-sample evaluation of a design, so that label is always
checked.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from ._checks import check_int, check_real, finite_array, read_only
from ._draws import ImpactLosses, impact_losses
from ._parts import (
    SPLITS,
    Part,
    SplitRecord,
    check_rng,
    kfold_pairs,
    no_rng,
    pool_part,
    source_token,
    train_test,
    unit_part,
    year_part,
)
from .annual import CatalogueYears, YearSets
from .errors import LeakageError, ParametricInputError, ScenarioError
from .index import ParametricIndex
from .measure import YEARSETS, EventBasis, OutcomeMeasure
from .units import InsuredUnits

__all__ = [
    "SPLITS",
    "Basis",
    "OutcomeSample",
    "SplitRecord",
    "check_left_out_unpaid",
    "check_no_leakage",
    "check_out_of_sample",
    "holdout",
    "year_block_kfold",
]

Basis = EventBasis | CatalogueYears | YearSets
"""What an outcome is: an ``EventBasis``, ``CatalogueYears`` or ``YearSets``."""


@dataclass(frozen=True, eq=False)
class OutcomeSample:
    """Index, loss and outcomes of a set of events (module docstring).

    Use :meth:`from_impact`; the constructor takes the arrays directly.

    Parameters
    ----------
    index : ParametricIndex
        The index of the sample's events (rows) and units (columns).
    loss : array_like or scipy.sparse matrix
        The loss per row and unit, shape ``(n_draws n_events, n_units)``,
        finite and >= 0: rows are ordered by draw, then event. Stored as CSR.
    frequency : array_like of float
        Each event's frequency per year, shape ``(n_events,)``, >= 0. A draw
        of an event has ``frequency / n_draws``.
    units : InsuredUnits
        The insured units, with the index's ``unit_id`` in order.
    basis : EventBasis, CatalogueYears or YearSets
        What an outcome is (:mod:`~climada_petals.engine.parametric.measure`).
    years : array_like of int, optional
        On catalogue years only: the simulation years the sample covers, a
        block of a split. Default: all of them.
    n_draws : int, default 1
        Draws per event: impact-function draws (:meth:`from_impact_draws`).
    scenario : str, default "baseline"
        A label for the tables.
    split : SplitRecord, optional
        How the sample was cut from another; None for a whole sample.
    n_left_out : int, default 0
        Events of the source left out, because they have no loss and an
        index of 0 at every unit (module docstring).
    source : str, optional
        The source token: that of the whole sample this one was cut from,
        which every part keeps. Default: the fingerprint of the sample's
        own events (module docstring, "Lineage").

    Attributes
    ----------
    measure : OutcomeMeasure
        The outcomes of the rows and their weights, built from the basis.

    Raises
    ------
    YearRouteError
        If the basis doesn't fit the events
        (:mod:`~climada_petals.engine.parametric.measure`).
    EventAlignmentError
        If the basis lacks an event of the index.
    ParametricInputError
        If an argument is invalid, or the sample holds no event.
    """

    index: ParametricIndex
    loss: Any
    frequency: NDArray[np.float64]
    units: InsuredUnits
    basis: Basis
    years: NDArray[np.int64] | None = field(default=None, kw_only=True)
    n_draws: int = field(default=1, kw_only=True)
    scenario: str = field(default="baseline", kw_only=True)
    split: SplitRecord | None = field(default=None, kw_only=True)
    n_left_out: int = field(default=0, kw_only=True)
    source: str | None = field(default=None, kw_only=True, repr=False)
    measure: OutcomeMeasure = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the fields, store read-only copies and build the measure."""
        _check_parts(self.index, self.units, self.basis)
        draws = check_int("n_draws", self.n_draws, low=1)
        shape = (self.index.n_events * draws, self.index.n_units)
        frequency = finite_array(self.frequency, "frequency", 1)
        if frequency.shape != (self.index.n_events,) or np.any(frequency < 0):
            raise ParametricInputError(
                f"frequency must hold one value >= 0 per event, got shape "
                f"{frequency.shape} for {self.index.n_events} events"
            )
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ParametricInputError(
                f"scenario must be a non-empty str, got {self.scenario!r}"
            )
        if self.split is not None and not isinstance(self.split, SplitRecord):
            raise ParametricInputError(
                f"split must be a SplitRecord, got {self.split!r}"
            )
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source
        ):
            raise ParametricInputError(
                f"source must be a non-empty str or None, got {self.source!r}"
            )
        years = _sample_years(self.basis, self.years)
        loss = _loss(self.loss, shape)
        if self.source is None:
            object.__setattr__(
                self, "source", source_token(self.index, frequency, loss, draws)
            )
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "frequency", read_only(frequency))
        object.__setattr__(self, "years", years)
        object.__setattr__(self, "n_draws", draws)
        object.__setattr__(
            self, "n_left_out", check_int("n_left_out", self.n_left_out, low=0)
        )
        if isinstance(self.basis, CatalogueYears):
            measure = self.basis.measure(
                self.index.event_id, frequency, n_draws=draws, years=years
            )
        else:
            measure = self.basis.measure(self.index.event_id, frequency, n_draws=draws)
        object.__setattr__(self, "measure", measure)

    @classmethod
    def from_impact(
        cls,
        index: ParametricIndex,
        impact: Any,
        *,
        units: InsuredUnits,
        basis: Basis,
        scenario: str = "baseline",
    ) -> OutcomeSample:
        """Build a sample from an index and an ``Impact`` (module docstring).

        Parameters
        ----------
        index : ParametricIndex
            The index of the impact's events, matched on ``event_id`` in any
            order, e.g. from ``ParametricIndex.from_hazard(hazard, units)``,
            or ``matched_to(impact)`` for an index from another hazard set.
        impact : climada.engine.Impact
            The loss, computed with ``save_mat=True`` on the units'
            exposure points, e.g. ``ImpactCalc(exposures, impfset,
            hazard).impact(save_mat=True)``.
        units : InsuredUnits
            The insured units, with the index's ``unit_id`` in order. The
            loss of a unit is ``imp_mat`` summed over its points.
        basis : EventBasis, CatalogueYears or YearSets
            What an outcome is. There is no default: the route changes tail
            numbers, so it is part of the setup of every number.
        scenario : str, default "baseline"
            A label for the tables.

        Returns
        -------
        OutcomeSample
            The kept events, in the index's order.

        Raises
        ------
        YearRouteError
            If the basis doesn't fit the kept events, e.g. catalogue years
            whose frequencies are not ``1 / N``.
        ImpactInputError
            If the impact has no ``imp_mat``, its ``frequency_unit`` is not
            ``1/year``, its exposure points are not the units', or a loss
            is negative or not finite.
        EventAlignmentError
            If the impact's events are not the index's, or the basis lacks
            a kept event.
        ParametricInputError
            If an argument is invalid, or no event has a loss or a non-zero
            index.
        """
        _check_parts(index, units, basis)
        return cls._of(impact_losses(index, [impact], units), units, basis, scenario)

    @classmethod
    def from_impact_draws(
        cls,
        index: ParametricIndex,
        impacts: Sequence[Any],
        *,
        units: InsuredUnits,
        basis: Basis,
        scenario: str = "baseline",
    ) -> OutcomeSample:
        """Build a sample from an index and one ``Impact`` per impact-function draw.

        Each impact is the loss of one draw of the impact functions, on the
        same hazard set (:mod:`~climada_petals.engine.parametric.basis_risk`,
        which says what draws can and can't do). The index is the same in
        every draw. An event is kept if it has a loss in some draw or a
        non-zero index at some unit, as in :meth:`from_impact`.

        Parameters
        ----------
        index : ParametricIndex
            As in :meth:`from_impact`.
        impacts : sequence of climada.engine.Impact
            One loss per draw, each computed with ``save_mat=True`` on the
            units' exposure points, with the index's events in any order and
            the same frequencies and loss unit, e.g. one ``ImpactCalc`` per
            set of ``impact_function_draws``.
        units, basis, scenario
            As in :meth:`from_impact`.

        Returns
        -------
        OutcomeSample
            With ``n_draws`` the number of impacts: one row per draw and kept
            event, ordered by draw, then event, each with frequency
            ``f_e / n_draws``. On an annual route an outcome is a year in one
            draw.

        Raises
        ------
        EventAlignmentError
            If an impact's events are not the index's, or two impacts'
            frequencies differ; the error's note names the draw.
        ImpactInputError
            As in :meth:`from_impact`, for any impact, or if the impacts'
            loss units differ.
        YearRouteError, ParametricInputError
            As in :meth:`from_impact`, and if ``impacts`` is not a non-empty
            sequence.
        """
        _check_parts(index, units, basis)
        return cls._of(impact_losses(index, impacts, units), units, basis, scenario)

    @classmethod
    def _of(
        cls, found: ImpactLosses, units: InsuredUnits, basis: Basis, scenario: str
    ) -> OutcomeSample:
        """Return the sample of the kept events of one or several impacts."""
        return cls(
            found.index,
            found.loss,
            found.frequency,
            units,
            basis,
            n_draws=found.n_draws,
            scenario=scenario,
            n_left_out=found.n_left_out,
        )

    @property
    def n_events(self) -> int:
        """Number of events."""
        return self.index.n_events

    @property
    def n_rows(self) -> int:
        """Number of rows, ``n_draws n_events``."""
        return self.index.n_events * self.n_draws

    @property
    def n_units(self) -> int:
        """Number of insured units."""
        return self.index.n_units

    @property
    def unit_id(self) -> tuple[Any, ...]:
        """The units' ids, in column order."""
        return tuple(self.index.unit_id)

    @property
    def event_id(self) -> NDArray[np.int64]:
        """The events' ids."""
        return self.index.event_id

    @property
    def kind(self) -> str:
        """``"event"`` or ``"annual"``."""
        return self.basis.kind

    @property
    def route(self) -> str:
        """``"frequency"``, ``"catalogue"`` or ``"yearsets"``."""
        return self.basis.route

    @property
    def n_years(self) -> int | None:
        """The years the outcomes stand for; None on the event basis."""
        return self.measure.n_years

    @property
    def event_row(self) -> NDArray[np.intp]:
        """The event of each row, as a position in :attr:`index`."""
        return np.tile(np.arange(self.n_events), self.n_draws)

    @property
    def draw(self) -> NDArray[np.intp]:
        """The draw of each row."""
        return np.repeat(np.arange(self.n_draws), self.n_events)

    @property
    def index_values(self) -> NDArray[np.float64]:
        """The index of each row, shape ``(n_rows, n_units)``."""
        return self.index.values[self.event_row]

    @property
    def block(self) -> NDArray[Any]:
        """Each event's block, the unit a split keeps together."""
        return self.basis.blocks(self.event_id)

    @property
    def event_year(self) -> NDArray[np.int64] | None:
        """Each event's simulation year on catalogue years; else None."""
        if isinstance(self.basis, CatalogueYears):
            return self.basis.year_of(self.event_id)
        return None

    @property
    def correction_factor(self) -> float | None:
        """On year sets, ``climada.util.yearsets``' ``correction_fac`` of the total loss.

        The expected annual total loss from the frequencies divided by its
        mean over the sampled years. It is recorded, not applied: one
        factor for every year repairs the mean and not the tail. NaN if no
        sampled year has a loss; None on the other routes.
        """
        if self.route != YEARSETS:
            return None
        total = np.asarray(self.loss.sum(axis=1)).reshape(-1)
        sampled = self.measure.expectation(total)
        expected = float(np.tile(self.frequency, self.n_draws) @ total) / self.n_draws
        return expected / sampled if sampled > 0 else math.nan

    def select(
        self, keys: ArrayLike, *, rng: np.random.Generator | None = None
    ) -> OutcomeSample:
        """Return the part of the sample in ``keys`` (module docstring).

        Parameters
        ----------
        keys : array_like
            On catalogue years, simulation years of the sample. On the other
            routes, block labels of the sample: the part holds their events,
            with frequencies divided by their share of the total.
        rng : numpy.random.Generator, optional
            On year sets, draws the part's ``round(share N)`` years; required
            there and not taken on the other routes.

        Returns
        -------
        OutcomeSample
            With ``split`` method ``"select"``.

        Raises
        ------
        ParametricInputError
            If a key is not the sample's, the part holds no event, or
            ``rng`` is missing on year sets or given on another route.
        ScenarioError
            If the sample mixes a scenario's members on an annual route.
        """
        _check_sample(self, split=True)
        part = Part("select", "selected")
        if isinstance(self.basis, CatalogueYears):
            no_rng(rng)
            return year_part(self, self.basis.check_years(keys), part)
        if isinstance(self.basis, YearSets):
            check_rng(rng)
        elif rng is not None:
            raise ParametricInputError(
                "a selection on the event basis is not random and takes no rng"
            )
        wanted = pd.Index(np.asarray(keys).reshape(-1))
        unknown = wanted[~wanted.isin(self.block)]
        if len(unknown):
            raise ParametricInputError(
                f"keys that are not blocks of the sample, e.g. {list(unknown[:5])}"
            )
        return pool_part(self, np.asarray(pd.Index(self.block).isin(wanted)), rng, part)

    def unit(self, unit_id: Hashable) -> OutcomeSample:
        """Return the sample restricted to one insured unit.

        The part holds every event of the sample, with its frequencies,
        basis, years, draws, label, split record and source token: only the
        index, the loss and the units are the unit's. So a design on the
        part is checked against the sample's other parts like a design on
        the sample (module docstring, "Lineage"). The part's units keep every
        exposure point, so that payouts map back to the points of the loss
        ``Impact``; building them logs the other units' points as in no
        insured unit, and their losses are left out.

        Parameters
        ----------
        unit_id : str or int
            One of :attr:`unit_id`.

        Returns
        -------
        OutcomeSample
            With one unit.

        Raises
        ------
        ParametricInputError
            If the sample has no such unit.
        """
        return unit_part(self, unit_id)


def holdout(
    sample: OutcomeSample,
    *,
    train_fraction: float = 0.7,
    rng: np.random.Generator | None = None,
) -> tuple[OutcomeSample, OutcomeSample]:
    """Split a sample into a training and a test sample (module docstring).

    Parameters
    ----------
    sample : OutcomeSample
        The sample to split.
    train_fraction : float, default 0.7
        The training share of the years (catalogue years) or of the
        frequency (the other routes), in ``(0, 1)``.
    rng : numpy.random.Generator, optional
        Needed on year sets and on the event basis, which pool blocks at
        random; not taken on catalogue years, which cut contiguous years.

    Returns
    -------
    train, test : OutcomeSample
        Disjoint in events and blocks; ``split`` records the side and its
        share.

    Raises
    ------
    LeakageError
        If no contiguous cut of the catalogue years keeps every block on
        one side.
    ParametricInputError
        If ``train_fraction`` is out of range, a side would be empty, or
        ``rng`` is missing or not wanted.
    ScenarioError
        If the sample mixes a scenario's members on an annual route.
    """
    _check_sample(sample, split=True)
    fraction = check_real(
        "train_fraction",
        train_fraction,
        low=0.0,
        high=1.0,
        include_low=False,
        include_high=False,
    )
    train, test = train_test(sample, fraction, rng)
    check_no_leakage(train, test)
    return train, test


def year_block_kfold(
    sample: OutcomeSample, k: int, *, rng: np.random.Generator | None = None
) -> tuple[tuple[OutcomeSample, OutcomeSample], ...]:
    """Split a sample into ``k`` folds of disjoint blocks (module docstring).

    Fold ``i`` tests on its block and trains on the others. On catalogue
    years the folds are contiguous: fold ``i`` holds the years from
    ``floor(i N / k)`` to ``floor((i + 1) N / k)``, each cut moved as in
    :func:`holdout`. On the other routes the folds are pools of ``1 / k`` of
    the frequency each.

    Parameters
    ----------
    sample : OutcomeSample
        The sample to split.
    k : int
        The number of folds, >= 2.
    rng : numpy.random.Generator, optional
        As in :func:`holdout`.

    Returns
    -------
    tuple of (train, test)
        One pair per fold, each disjoint in events and blocks.

    Raises
    ------
    LeakageError
        If the catalogue years can't be cut into ``k`` folds that keep every
        block on one side.
    ParametricInputError
        If ``k`` is invalid, a fold would be empty, or ``rng`` is missing or
        not wanted.
    ScenarioError
        If the sample mixes a scenario's members on an annual route.
    """
    _check_sample(sample, split=True)
    pairs = kfold_pairs(sample, check_int("k", k, low=2), rng)
    for train, test in pairs:
        check_no_leakage(train, test)
    return tuple(pairs)


def check_no_leakage(design: Any, evaluation: OutcomeSample) -> None:
    """Raise if two samples of one event set share an event or a block.

    Use it on samples split from one sample, such as the sides of
    :func:`holdout`. Samples of different hazard sets can share event ids
    or years by chance, so a shared id says nothing about them:
    :func:`check_out_of_sample` compares only samples of one source.

    Parameters
    ----------
    design : OutcomeSample or DesignSample
        The design sample, or a design's record of it
        (:attr:`~climada_petals.engine.parametric.calibrate.CalibrationResult.design_sample`),
        e.g. the training sample.
    evaluation : OutcomeSample
        The other sample, e.g. the test sample.

    Raises
    ------
    LeakageError
        If an event, or a block (an ensemble's parent, or a catalogue year),
        is in both.
    ParametricInputError
        If an argument is neither.
    """
    _check_design(design)
    _check_sample(evaluation)
    events = np.intersect1d(design.event_id, evaluation.event_id)
    if events.size:
        raise LeakageError(
            f"{events.size} events are in both samples, e.g. {events[:5].tolist()}"
        )
    blocks = pd.Index(design.block).intersection(pd.Index(evaluation.block))
    if len(blocks):
        raise LeakageError(
            f"{len(blocks)} blocks are in both samples, e.g. {list(blocks[:5])}: the "
            "events of a block (an ensemble's parent, or a catalogue year) belong "
            "on one side"
        )


def check_out_of_sample(design: Any, evaluation: OutcomeSample) -> None:
    """Raise if a sample of the design sample's source shares an event or a block.

    The check of an out-of-sample evaluation (module docstring, "Lineage").
    The two are compared only if they have one source token: then the
    evaluation sample is the whole sample the design sample was cut from,
    another part of it, or the design sample itself, and
    :func:`check_no_leakage` decides. Samples of other sources are not
    compared, since their event ids and years can coincide by chance.

    Parameters
    ----------
    design : OutcomeSample or DesignSample
        As in :func:`check_no_leakage`. A record without a source token is
        compared with no sample.
    evaluation : OutcomeSample
        The sample of the out-of-sample evaluation.

    Raises
    ------
    LeakageError
        If the two have one source and an event, or a block, is in both.
    ParametricInputError
        If an argument is neither.
    """
    _check_design(design)
    _check_sample(evaluation)
    source = getattr(design, "source", None)
    if source is not None and source == evaluation.source:
        check_no_leakage(design, evaluation)


def _check_design(design: object) -> None:
    """Raise unless ``design`` is a sample or a design's record of one."""
    if not isinstance(design, OutcomeSample) and not (
        isinstance(getattr(design, "event_id", None), np.ndarray)
        and isinstance(getattr(design, "block", None), np.ndarray)
    ):
        raise ParametricInputError(
            "design must be an OutcomeSample or a design's record of one, got "
            f"{type(design).__name__}"
        )


def check_left_out_unpaid(sample: OutcomeSample, paid_at_zero: ArrayLike) -> None:
    """Raise if the sample left events out and a payout pays at an index of 0.

    :meth:`OutcomeSample.from_impact` leaves out the events with no loss and
    an index of 0 at every unit. A payout that pays at an index of 0 would
    pay on them, and the premium, the objective and the metrics would leave
    those payouts out with no error.

    Parameters
    ----------
    sample : OutcomeSample
        The sample a payout is designed or evaluated on.
    paid_at_zero : array_like of float
        What the payout pays at an index of 0, per unit; any positive value
        where it may pay, such as a free level.

    Raises
    ------
    ParametricInputError
        If the sample left events out and a value is above 0.
    """
    if sample.n_left_out and np.any(np.asarray(paid_at_zero, dtype=np.float64) > 0):
        raise ParametricInputError(
            f"the sample left out {sample.n_left_out} events with no loss and an "
            "index of 0 at every unit, and this payout pays, or may pay, at an "
            "index of 0, so it would pay on them while the premium, the objective "
            "and the metrics leave those payouts out. Move the first threshold "
            "above 0, or build the sample with the OutcomeSample constructor from "
            "every event, which leaves none out"
        )


def _check_parts(index: object, units: object, basis: object) -> None:
    """Raise unless the index, units and basis can make a sample.

    Raises
    ------
    ParametricInputError
        If one has the wrong type, the units are not the index's, or the
        index has no event.
    """
    if not isinstance(index, ParametricIndex):
        raise ParametricInputError(f"index must be a ParametricIndex, got {index!r}")
    if not isinstance(units, InsuredUnits) or tuple(units.unit_id) != tuple(
        index.unit_id
    ):
        raise ParametricInputError(
            "units must be the InsuredUnits of the index's unit_id, in order"
        )
    if not isinstance(basis, EventBasis | CatalogueYears | YearSets):
        raise ParametricInputError(
            "basis must be EventBasis(), CatalogueYears or YearSets; there is no "
            f"default, got {basis!r}"
        )
    if index.n_events == 0:
        raise ParametricInputError("the sample holds no event")


def _check_sample(sample: object, *, split: bool = False) -> None:
    """Raise unless ``sample`` is an :class:`OutcomeSample`; to ``split``, not a mixture's."""
    if not isinstance(sample, OutcomeSample):
        raise ParametricInputError(
            f"sample must be an OutcomeSample, got {type(sample).__name__}"
        )
    refusal = getattr(sample.basis, "split_refusal", None) if split else None
    if refusal:
        raise ScenarioError(refusal)


def _sample_years(basis: Basis, years: ArrayLike | None) -> NDArray[np.int64] | None:
    """Return the sample's years: catalogue years only, all of them by default."""
    if not isinstance(basis, CatalogueYears):
        if years is not None:
            raise ParametricInputError("years are taken on catalogue years only")
        return None
    return read_only(basis.years if years is None else basis.check_years(years))


def _loss(loss: Any, shape: tuple[int, int]) -> sp.csr_matrix:
    """Return the loss as a read-only CSR matrix of ``shape``.

    Raises
    ------
    ParametricInputError
        If the shape is wrong or a loss is negative or not finite.
    """
    try:
        matrix = sp.csr_matrix(loss, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ParametricInputError(f"loss must be numeric: {exc}") from exc
    if matrix.shape != shape:
        raise ParametricInputError(
            f"loss must have shape {shape} (n_draws n_events, n_units), got "
            f"{matrix.shape}"
        )
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ParametricInputError("loss must be finite and >= 0")
    matrix.sum_duplicates()
    for part in (matrix.data, matrix.indices, matrix.indptr):
        part.setflags(write=False)
    return matrix
