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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from climada_petals.util.config import LOGGER

from ._checks import check_int, check_real, finite_array, read_only
from ._draws import ImpactLosses, impact_losses, take_events
from .annual import CatalogueYears, YearSets, sample_years
from .errors import LeakageError, ParametricInputError
from .index import ParametricIndex
from .measure import CATALOGUE, YEARSETS, EventBasis, OutcomeMeasure
from .units import InsuredUnits

__all__ = [
    "SPLITS",
    "Basis",
    "OutcomeSample",
    "SplitRecord",
    "check_left_out_unpaid",
    "check_no_leakage",
    "holdout",
    "year_block_kfold",
]

SPLITS: Final[tuple[str, ...]] = ("holdout", "year_block_kfold", "select")
"""The values of :attr:`SplitRecord.method`."""

Basis = EventBasis | CatalogueYears | YearSets
"""What an outcome is: an ``EventBasis``, ``CatalogueYears`` or ``YearSets``."""


@dataclass(frozen=True)
class SplitRecord:
    """How a sample was cut from another.

    Attributes
    ----------
    method : {"holdout", "year_block_kfold", "select"}
        The function or method that made it.
    side : {"train", "test", "selected"}
        Which side it is.
    fold : int
        The fold of ``year_block_kfold``, else 0.
    n_folds : int
        ``k`` of ``year_block_kfold``, else 1.
    share : float
        The side's share of the other sample: of its years on catalogue
        years, of its frequency on the other routes.
    """

    method: str
    side: str
    fold: int
    n_folds: int
    share: float


class _Part(NamedTuple):
    """The labels of a side, before its share is known."""

    method: str
    side: str
    fold: int = 0
    n_folds: int = 1

    def record(self, share: float) -> SplitRecord:
        """Return the side's :class:`SplitRecord`."""
        return SplitRecord(self.method, self.side, self.fold, self.n_folds, share)


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
        years = _sample_years(self.basis, self.years)
        object.__setattr__(self, "loss", _loss(self.loss, shape))
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
        """
        part = _Part("select", "selected")
        if isinstance(self.basis, CatalogueYears):
            _no_rng(rng)
            return _year_part(self, self.basis.check_years(keys), part)
        if isinstance(self.basis, YearSets):
            _check_rng(rng)
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
        return _pool_part(
            self, np.asarray(pd.Index(self.block).isin(wanted)), rng, part
        )


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
    """
    _check_sample(sample)
    fraction = check_real(
        "train_fraction",
        train_fraction,
        low=0.0,
        high=1.0,
        include_low=False,
        include_high=False,
    )
    if sample.route == CATALOGUE:
        _no_rng(rng)
        years = _years(sample)
        (cut,) = _cuts(sample, [math.floor(fraction * years.size + 1e-9)])
        train = _year_part(sample, years[:cut], _Part("holdout", "train"))
        test = _year_part(sample, years[cut:], _Part("holdout", "test"))
    else:
        pool = _pools(sample, [fraction], rng)
        train = _pool_part(sample, pool == 0, rng, _Part("holdout", "train"))
        test = _pool_part(sample, pool == 1, rng, _Part("holdout", "test"))
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
    """
    _check_sample(sample)
    folds = check_int("k", k, low=2)
    pairs = []
    if sample.route == CATALOGUE:
        _no_rng(rng)
        years = _years(sample)
        if years.size < folds:
            raise ParametricInputError(
                f"{folds} folds need at least {folds} years, got {years.size}"
            )
        cuts = _cuts(
            sample, [math.floor(i * years.size / folds + 1e-9) for i in range(1, folds)]
        )
        bounds = [0, *cuts, years.size]
        for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            train_years = np.concatenate([years[:start], years[end:]])
            pairs.append(
                (
                    _year_part(
                        sample,
                        train_years,
                        _Part("year_block_kfold", "train", fold, folds),
                    ),
                    _year_part(
                        sample,
                        years[start:end],
                        _Part("year_block_kfold", "test", fold, folds),
                    ),
                )
            )
    else:
        pool = _pools(sample, [i / folds for i in range(1, folds)], rng)
        for fold in range(folds):
            pairs.append(
                (
                    _pool_part(
                        sample,
                        pool != fold,
                        rng,
                        _Part("year_block_kfold", "train", fold, folds),
                    ),
                    _pool_part(
                        sample,
                        pool == fold,
                        rng,
                        _Part("year_block_kfold", "test", fold, folds),
                    ),
                )
            )
    for train, test in pairs:
        check_no_leakage(train, test)
    return tuple(pairs)


def check_no_leakage(design: Any, evaluation: OutcomeSample) -> None:
    """Raise if two samples of one event set share an event or a block.

    Use it on samples split from one sample, such as the sides of
    :func:`holdout`. Samples of different hazard sets can share event ids
    or years by chance, so a shared id says nothing about them.

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
    if not isinstance(design, OutcomeSample) and not _is_record(design):
        raise ParametricInputError(
            "design must be an OutcomeSample or a design's record of one, got "
            f"{type(design).__name__}"
        )
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


def _is_record(design: object) -> bool:
    """Whether ``design`` is a design's record of its sample: events and blocks."""
    return isinstance(getattr(design, "event_id", None), np.ndarray) and isinstance(
        getattr(design, "block", None), np.ndarray
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


def _check_sample(sample: object) -> None:
    """Raise unless ``sample`` is an :class:`OutcomeSample`."""
    if not isinstance(sample, OutcomeSample):
        raise ParametricInputError(
            f"sample must be an OutcomeSample, got {type(sample).__name__}"
        )


def _no_rng(rng: object) -> None:
    """Raise if an ``rng`` is given where the split takes none."""
    if rng is not None:
        raise ParametricInputError(
            "catalogue years are split into contiguous years and take no rng"
        )


def _sample_years(basis: Basis, years: ArrayLike | None) -> NDArray[np.int64] | None:
    """Return the sample's years: catalogue years only, all of them by default."""
    if not isinstance(basis, CatalogueYears):
        if years is not None:
            raise ParametricInputError("years are taken on catalogue years only")
        return None
    return read_only(basis.years if years is None else basis.check_years(years))


def _years(sample: OutcomeSample) -> NDArray[np.int64]:
    """Return the years of a catalogue sample."""
    return np.asarray(sample.years, dtype=np.int64)


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


def _part(
    sample: OutcomeSample,
    chosen: NDArray[np.bool_],
    *,
    basis: Basis,
    frequency: NDArray[np.float64],
    split: SplitRecord,
    years: NDArray[np.int64] | None = None,
) -> OutcomeSample:
    """Return the sample's events where ``chosen``, on their new basis.

    Raises
    ------
    ParametricInputError
        If no event is chosen.
    """
    if not chosen.any():
        raise ParametricInputError(
            f"the {split.side} side of the split ({split.method}, fold "
            f"{split.fold}) holds no event"
        )
    return OutcomeSample(
        take_events(sample.index, np.flatnonzero(chosen)),
        sample.loss[np.tile(chosen, sample.n_draws)],
        frequency,
        sample.units,
        basis,
        years=years,
        n_draws=sample.n_draws,
        scenario=sample.scenario,
        split=split,
        n_left_out=sample.n_left_out,
    )


def _year_part(
    sample: OutcomeSample, years: NDArray[np.int64], part: _Part
) -> OutcomeSample:
    """Return the catalogue sample's events in ``years``, over those years."""
    chosen = np.isin(np.asarray(sample.event_year), years)
    return _part(
        sample,
        chosen,
        basis=sample.basis,
        frequency=sample.frequency[chosen],
        split=part.record(years.size / _years(sample).size),
        years=years,
    )


def _pool_part(
    sample: OutcomeSample,
    chosen: NDArray[np.bool_],
    rng: Any,
    part: _Part,
) -> OutcomeSample:
    """Return the events of a pool, their frequencies divided by its share.

    On year sets the pool's ``round(share N)`` years (at least one) are
    sampled from its events alone, with ``rng``, which the caller checked.

    Raises
    ------
    ParametricInputError
        If the pool holds no event or no frequency.
    """
    total = float(sample.frequency.sum())
    share = float(sample.frequency[chosen].sum()) / total if total > 0 else 0.0
    if share <= 0:
        raise ParametricInputError(
            f"the {part.side} side of the split ({part.method}) holds no frequency"
        )
    frequency = sample.frequency[chosen] / share
    basis: Basis = sample.basis
    if isinstance(basis, YearSets):
        basis = YearSets(
            sample.event_id[chosen],
            sample_years(frequency, max(1, round(share * basis.n_years)), rng),
            block=sample.block[chosen],
            frequency=frequency,
        )
    return _part(
        sample, chosen, basis=basis, frequency=frequency, split=part.record(share)
    )


def _check_rng(rng: object) -> None:
    """Raise unless ``rng`` is a ``numpy.random.Generator``."""
    if not isinstance(rng, np.random.Generator):
        raise ParametricInputError(
            "year sets and the event basis are split into pools of blocks drawn at "
            "random, and year sets sample each part's years: pass rng, a "
            "numpy.random.Generator such as "
            f"numpy.random.default_rng(seed); got {type(rng).__name__}"
        )


def _pools(sample: OutcomeSample, edges: list[float], rng: Any) -> NDArray[np.intp]:
    """Return each event's pool: blocks in a shuffled order, filled by frequency.

    ``edges`` are the cumulative shares that end the pools but the last. A
    block goes to the pool that holds the middle of its frequency.

    Raises
    ------
    ParametricInputError
        If ``rng`` is missing, the sample holds no frequency, or a pool gets
        no block.
    """
    _check_rng(rng)
    codes, labels = pd.factorize(sample.block)
    weight = np.bincount(codes, weights=sample.frequency, minlength=len(labels))
    total = float(weight.sum())
    if total <= 0:
        raise ParametricInputError("the sample's frequencies sum to 0")
    order = rng.permutation(len(labels))
    cumulative = np.cumsum(weight[order])
    pool = np.empty(len(labels), dtype=np.intp)
    pool[order] = np.searchsorted(
        edges, (cumulative - weight[order] / 2) / total, side="right"
    )
    if np.unique(pool).size < len(edges) + 1:
        raise ParametricInputError(
            f"{len(labels)} blocks can't fill {len(edges) + 1} pools of frequency; "
            "the sample has too few blocks for this split"
        )
    return pool[codes]


def _cuts(sample: OutcomeSample, targets: list[int]) -> list[int]:
    """Return the cut nearest each target that keeps every block on one side.

    A cut ``c`` puts the sample's first ``c`` years on one side. Without
    block labels every year is a block, so every cut is valid.

    Raises
    ------
    ParametricInputError
        If a target leaves a side without a year.
    LeakageError
        If no valid cut is left, or two folds would meet.
    """
    years = _years(sample)
    n_years = years.size
    if any(not 1 <= target <= n_years - 1 for target in targets):
        raise ParametricInputError(
            f"the split of {n_years} years at {targets} leaves a side without a "
            "year; every side needs at least one year"
        )
    position = pd.Index(years).get_indexer(sample.event_year)
    codes, labels = pd.factorize(sample.block)
    low = np.full(len(labels), n_years)
    high = np.full(len(labels), -1)
    np.minimum.at(low, codes, position)
    np.maximum.at(high, codes, position)
    # A cut c splits a block whose years run from low to high iff low < c <= high.
    crossing = np.zeros(n_years + 2, dtype=np.int64)
    np.add.at(crossing, low + 1, 1)
    np.add.at(crossing, high + 1, -1)
    valid = np.flatnonzero(np.cumsum(crossing)[1:n_years] == 0) + 1
    if valid.size == 0:
        raise LeakageError(
            f"every cut of the {n_years} years splits a block: blocks have years "
            "far apart. Number the member-years of one parent next to each other, "
            "e.g. n_members x (calendar_year - first) + member, with the parent as "
            "block (CatalogueYears.from_labels)"
        )
    cuts = [int(valid[np.argmin(np.abs(valid - target))]) for target in targets]
    if np.any(np.diff(cuts) <= 0):
        raise LeakageError(
            f"the blocks leave too few valid cuts of the {n_years} years for "
            f"{len(targets) + 1} sides"
        )
    moved = [(target, cut) for target, cut in zip(targets, cuts) if cut != target]
    if moved:
        LOGGER.info("split: cuts moved to keep the blocks whole: %s", moved)
    return cuts
