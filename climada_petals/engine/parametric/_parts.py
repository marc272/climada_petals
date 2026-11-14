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

Parts of an outcome sample: the sides of a split, a selection, one unit,
and the source token that every part keeps.

:mod:`~climada_petals.engine.parametric.sample` states the rules
(:func:`~climada_petals.engine.parametric.sample.holdout`,
:func:`~climada_petals.engine.parametric.sample.year_block_kfold`,
:meth:`~climada_petals.engine.parametric.sample.OutcomeSample.select` and
:meth:`~climada_petals.engine.parametric.sample.OutcomeSample.unit`); this
module builds the parts. A part is the sample with some of its fields
replaced (``dataclasses.replace``), so it keeps the others: the units, the
draws, the label, the events left out, and the **source token**
(:func:`source_token`), the fingerprint of the whole sample it was cut from.

* :func:`year_part` takes the events of some catalogue years, over those
  years; :func:`cuts` finds contiguous cuts of the years that keep every
  block on one side.
* :func:`pool_part` takes the events of a pool of blocks, with their
  frequencies divided by the pool's share, and on year sets samples the
  pool's years from its own events; :func:`pools` fills the pools.
* :func:`train_test` and :func:`kfold_pairs` return the sides of
  :func:`~climada_petals.engine.parametric.sample.holdout` and
  :func:`~climada_petals.engine.parametric.sample.year_block_kfold`.
* :func:`unit_part` takes one insured unit's column, with every event.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from ._draws import take_events
from .annual import YearSets, sample_years
from .errors import LeakageError, ParametricInputError
from .index import ParametricIndex
from .measure import CATALOGUE
from .units import InsuredUnits

__all__ = [
    "SPLITS",
    "Part",
    "SplitRecord",
    "catalogue_years",
    "check_rng",
    "cuts",
    "kfold_pairs",
    "mixture_token",
    "no_rng",
    "pool_part",
    "pools",
    "source_token",
    "train_test",
    "unit_part",
    "year_part",
]

SPLITS: Final[tuple[str, ...]] = ("holdout", "year_block_kfold", "select")
"""The values of :attr:`SplitRecord.method`."""


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


class Part(NamedTuple):
    """The labels of a side, before its share is known."""

    method: str
    side: str
    fold: int = 0
    n_folds: int = 1

    def record(self, share: float) -> SplitRecord:
        """Return the side's :class:`SplitRecord`."""
        return SplitRecord(self.method, self.side, self.fold, self.n_folds, share)


def _digest_arrays(digest: Any, parts: Sequence[tuple[Any, Any]]) -> None:
    """Feed each array to ``digest`` as ``dtype``, its length first."""
    for values, dtype in parts:
        data = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.int64(data.size).tobytes())
        digest.update(data.tobytes())


def source_token(
    index: ParametricIndex, frequency: NDArray[np.float64], loss: Any, n_draws: int
) -> str:
    """Return the fingerprint of a sample's events: ``"sha256:<hex>"``.

    The SHA-256 of the unit ids and, in the order of the event ids, of the
    events' ids, frequencies, index values and losses per draw. The basis,
    the label and the records are left out: two samples of the same index
    and ``Impact`` have one token on any basis, and two of other hazard
    sets, frequencies, exposures or impact functions have two.

    Parameters
    ----------
    index : ParametricIndex
        The sample's index.
    frequency : numpy.ndarray of float
        Each event's frequency per year.
    loss : scipy.sparse.csr_matrix
        The loss per row and unit, rows ordered by draw, then event.
    n_draws : int
        Draws per event.
    """
    order = np.argsort(index.event_id, kind="stable")
    rows = (np.arange(n_draws)[:, None] * order.size + order[None, :]).reshape(-1)
    matrix = loss[rows].tocsr(copy=True)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256(repr(tuple(index.unit_id)).encode())
    _digest_arrays(
        digest,
        [
            (matrix.shape, np.int64),
            (index.event_id[order], np.int64),
            (frequency[order], np.float64),
            (index.values[order], np.float64),
            (matrix.indptr, np.int64),
            (matrix.indices, np.int64),
            (matrix.data, np.float64),
        ],
    )
    return f"sha256:{digest.hexdigest()}"


def mixture_token(sources: Sequence[str]) -> str:
    """Return the token of a mixture of samples with these tokens, in order.

    The mixtures of one sequence of sources have one token: the mixture of
    the members and the mixtures of their sides, whose events are numbered
    alike (:mod:`~climada_petals.engine.parametric._mixture`).
    """
    text = "mixture\n" + "\n".join(sources)
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def catalogue_years(sample: Any) -> NDArray[np.int64]:
    """Return the years of a catalogue sample."""
    return np.asarray(sample.years, dtype=np.int64)


def no_rng(rng: object) -> None:
    """Raise if an ``rng`` is given where the split takes none."""
    if rng is not None:
        raise ParametricInputError(
            "catalogue years are split into contiguous years and take no rng"
        )


def check_rng(rng: object) -> None:
    """Raise unless ``rng`` is a ``numpy.random.Generator``."""
    if not isinstance(rng, np.random.Generator):
        raise ParametricInputError(
            "year sets and the event basis are split into pools of blocks drawn at "
            "random, and year sets sample each part's years: pass rng, a "
            "numpy.random.Generator such as "
            f"numpy.random.default_rng(seed); got {type(rng).__name__}"
        )


def _event_part(
    sample: Any,
    chosen: NDArray[np.bool_],
    *,
    basis: Any,
    frequency: NDArray[np.float64],
    split: SplitRecord,
    years: NDArray[np.int64] | None = None,
) -> Any:
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
    return dataclasses.replace(
        sample,
        index=take_events(sample.index, np.flatnonzero(chosen)),
        loss=sample.loss[np.tile(chosen, sample.n_draws)],
        frequency=frequency,
        basis=basis,
        years=years,
        split=split,
    )


def year_part(sample: Any, years: NDArray[np.int64], part: Part) -> Any:
    """Return the catalogue sample's events in ``years``, over those years."""
    chosen = np.isin(np.asarray(sample.event_year), years)
    return _event_part(
        sample,
        chosen,
        basis=sample.basis,
        frequency=sample.frequency[chosen],
        split=part.record(years.size / catalogue_years(sample).size),
        years=years,
    )


def pool_part(sample: Any, chosen: NDArray[np.bool_], rng: Any, part: Part) -> Any:
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
    basis = sample.basis
    if isinstance(basis, YearSets):
        basis = YearSets(
            sample.event_id[chosen],
            sample_years(frequency, max(1, round(share * basis.n_years)), rng),
            block=sample.block[chosen],
            frequency=frequency,
        )
    return _event_part(
        sample, chosen, basis=basis, frequency=frequency, split=part.record(share)
    )


def pools(sample: Any, edges: list[float], rng: Any) -> NDArray[np.intp]:
    """Return each event's pool: blocks in a shuffled order, filled by frequency.

    ``edges`` are the cumulative shares that end the pools but the last. A
    block goes to the pool that holds the middle of its frequency.

    Raises
    ------
    ParametricInputError
        If ``rng`` is missing, the sample holds no frequency, or a pool gets
        no block.
    """
    check_rng(rng)
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


def cuts(sample: Any, targets: list[int]) -> list[int]:
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
    years = catalogue_years(sample)
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
    found = [int(valid[np.argmin(np.abs(valid - target))]) for target in targets]
    if np.any(np.diff(found) <= 0):
        raise LeakageError(
            f"the blocks leave too few valid cuts of the {n_years} years for "
            f"{len(targets) + 1} sides"
        )
    moved = [(target, cut) for target, cut in zip(targets, found) if cut != target]
    if moved:
        LOGGER.info("split: cuts moved to keep the blocks whole: %s", moved)
    return found


def train_test(sample: Any, fraction: float, rng: Any) -> tuple[Any, Any]:
    """Return the training and the test side of a holdout split.

    On catalogue years the first ``floor(fraction N)`` years train, the cut
    moved to keep the blocks whole; on the other routes two pools of blocks.
    """
    if sample.route == CATALOGUE:
        no_rng(rng)
        years = catalogue_years(sample)
        (cut,) = cuts(sample, [math.floor(fraction * years.size + 1e-9)])
        return (
            year_part(sample, years[:cut], Part("holdout", "train")),
            year_part(sample, years[cut:], Part("holdout", "test")),
        )
    pool = pools(sample, [fraction], rng)
    return (
        pool_part(sample, pool == 0, rng, Part("holdout", "train")),
        pool_part(sample, pool == 1, rng, Part("holdout", "test")),
    )


def kfold_pairs(sample: Any, folds: int, rng: Any) -> list[tuple[Any, Any]]:
    """Return the training and the test side of each of ``folds`` folds.

    On catalogue years fold ``i`` tests on the years from ``floor(i N / k)``
    to ``floor((i + 1) N / k)``, each cut moved to keep the blocks whole; on
    the other routes on a pool of ``1 / k`` of the frequency.

    Raises
    ------
    ParametricInputError
        If the catalogue has fewer years than folds, or as :func:`cuts` and
        :func:`pools`.
    LeakageError
        As :func:`cuts`.
    """
    pairs = []
    if sample.route == CATALOGUE:
        no_rng(rng)
        years = catalogue_years(sample)
        if years.size < folds:
            raise ParametricInputError(
                f"{folds} folds need at least {folds} years, got {years.size}"
            )
        bounds = [
            0,
            *cuts(
                sample,
                [math.floor(i * years.size / folds + 1e-9) for i in range(1, folds)],
            ),
            years.size,
        ]
        for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            pairs.append(
                (
                    year_part(
                        sample,
                        np.concatenate([years[:start], years[end:]]),
                        Part("year_block_kfold", "train", fold, folds),
                    ),
                    year_part(
                        sample,
                        years[start:end],
                        Part("year_block_kfold", "test", fold, folds),
                    ),
                )
            )
        return pairs
    pool = pools(sample, [i / folds for i in range(1, folds)], rng)
    for fold in range(folds):
        pairs.append(
            (
                pool_part(
                    sample,
                    pool != fold,
                    rng,
                    Part("year_block_kfold", "train", fold, folds),
                ),
                pool_part(
                    sample,
                    pool == fold,
                    rng,
                    Part("year_block_kfold", "test", fold, folds),
                ),
            )
        )
    return pairs


def unit_part(sample: Any, unit_id: Hashable) -> Any:
    """Return the sample restricted to one insured unit, with every event.

    The index, the loss and the units are the unit's column; the units
    keep every exposure point, so payouts map back to the points of the
    loss ``Impact``.

    Raises
    ------
    ParametricInputError
        If the sample has no such unit.
    """
    units = tuple(sample.unit_id)
    if unit_id not in units:
        raise ParametricInputError(
            f"the sample has no unit {unit_id!r}; it has {list(units)[:10]}"
        )
    column = units.index(unit_id)
    index = sample.index
    return dataclasses.replace(
        sample,
        index=ParametricIndex(
            index.values[:, [column]],
            index.event_id,
            (units[column],),
            index.definition,
            event_name=index.event_name,
        ),
        loss=sample.loss[:, [column]],
        units=InsuredUnits(
            sample.units.exposures,
            sample.units.membership[:, [column]],
            (units[column],),
        ),
    )
