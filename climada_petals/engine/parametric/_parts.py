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
(:func:`source_token`), the fingerprint of the event set of the whole
sample it was cut from.

* :func:`year_part` takes the events of some catalogue years, over those
  years; :func:`cuts` finds contiguous cuts of the years that keep every
  block on one side.
* :func:`pool_part` takes the events of a pool of blocks, with their
  frequencies divided by the pool's share, and on year sets samples the
  pool's years from its own events; :func:`pools` fills the pools.
* :func:`train_test` and :func:`kfold_pairs` return the sides of
  :func:`~climada_petals.engine.parametric.sample.holdout` and
  :func:`~climada_petals.engine.parametric.sample.year_block_kfold`.
* :func:`aligned_sides` cuts a sample with the blocks of another sample's
  sides, for samples of one event set, such as a baseline and a set scaled
  by ``apply_climate_scenario_knu``.
* :func:`member_sides` cuts the members of a scenario set in turn: with the
  blocks of an earlier member of the same event set, with the sides of the
  parent storms that earlier members hold (:func:`parent_sides`), or else
  by :func:`~climada_petals.engine.parametric.sample.holdout`.
* :func:`unit_part` takes one insured unit's column, with every event.

Parent storms
-------------
Block labels are mostly **local**: a year, ``event_id // 100`` or a
user's label names a block of one hazard set and says nothing about
another, whose labels can coincide by chance. The members of CLIMADA's perturbed tracks
are different: ``TCTracks.calc_perturbed_trajectories`` names member ``m``
of storm ``sid`` ``<sid>_gen<m>``, so the parent's name, an IBTrACS id say,
names the same storm in every hazard set built from those tracks, such as
the Data API's present and future ``random_walk`` sets of a country. Such
labels are **parent storms** (:func:`parent_labels`), compared across
hazard sets: the rule is that a sample's blocks are its events' parent
storms when

* its index has event names, and some name ends in ``_gen<m>``, and
* every event's block is a ``str``, the text that
  :data:`~climada_petals.engine.parametric.measure.ENSEMBLE_PARENT` reads
  from its name: the blocks of ``labels_from_names(hazard.event_name,
  ENSEMBLE_PARENT)``.

Otherwise its blocks are local, and its events have no parent. A mixture
of a scenario's members (:mod:`~climada_petals.engine.parametric._mixture`)
keeps its members' parents. ``OutcomeSample(..., parents=...)`` states the
parents of a sample the rule doesn't read, such as a set with other names;
labelling its blocks otherwise (with a prefix, say) keeps a sample that
follows the convention local.

Splits keep a parent's events on one side (:func:`split_keys`), and
:func:`member_sides` cuts a member that shares parent storms with earlier
members of a scenario set with their sides.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from ._draws import take_events
from .annual import YearSets, sample_years
from .errors import LeakageError, ParametricInputError, ScenarioError
from .index import ParametricIndex
from .measure import CATALOGUE, ENSEMBLE_PARENT
from .units import InsuredUnits

__all__ = [
    "SPLITS",
    "Part",
    "SplitRecord",
    "aligned_sides",
    "catalogue_years",
    "check_rng",
    "cuts",
    "kfold_pairs",
    "member_sides",
    "mixture_token",
    "no_rng",
    "parent_labels",
    "parent_sides",
    "parents_in",
    "pool_part",
    "pools",
    "source_token",
    "split_keys",
    "train_test",
    "unit_part",
    "year_part",
]

SPLITS: Final[tuple[str, ...]] = ("holdout", "year_block_kfold", "select")
"""The values of :attr:`SplitRecord.method`."""


@dataclass(frozen=True)
class SplitRecord:
    """How a sample was cut from another.

    Parameters
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


def source_token(index: ParametricIndex) -> str:
    """Return the fingerprint of an event set: ``"sha256:<hex>"``.

    The SHA-256 of the unit ids and, in the order of the event ids, of the
    events' ids and index values. The frequencies, the losses, the draws,
    the basis, the label and the records are left out, so a sample of the
    same events with other frequencies (a set scaled by
    ``apply_climate_scenario_knu``), other exposures or impact-function
    draws has the same token, while another hazard set, whose index values
    differ even where its event ids coincide, has another. An index that
    other exposures change, such as one weighted by the values of each
    scenario's own units, gives another token.

    Parameters
    ----------
    index : ParametricIndex
        The index of the events, e.g. the one given to
        ``OutcomeSample.from_impact``, before events are left out.
    """
    order = np.argsort(index.event_id, kind="stable")
    digest = hashlib.sha256(repr(tuple(index.unit_id)).encode())
    for values, dtype in (
        (index.event_id[order], np.int64),
        (index.values[order], np.float64),
    ):
        contiguous = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.int64(contiguous.size).tobytes())
        digest.update(contiguous.tobytes())
    return f"sha256:{digest.hexdigest()}"


def mixture_token(sources: Sequence[str]) -> str:
    """Return the token of a mixture of samples with these tokens, in order.

    The mixtures of one sequence of sources have one token: the mixture of
    the members and the mixtures of their sides, whose events are numbered
    alike (:mod:`~climada_petals.engine.parametric._mixture`).
    """
    text = "mixture\n" + "\n".join(sources)
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


_PARENT: Final[re.Pattern[str]] = re.compile(ENSEMBLE_PARENT)
_MEMBER: Final[re.Pattern[str]] = re.compile(r".+_gen[0-9]+")


def parent_labels(index: ParametricIndex, block: Sequence[Any]) -> dict[int, str]:
    """Return each event's parent storm, by event id, where the blocks name them.

    The rule of the module docstring ("Parent storms"): the blocks are the
    parents' names when the index has event names, some of them of the
    form ``<sid>_gen<m>``, and every block is the ``str`` that
    :data:`~climada_petals.engine.parametric.measure.ENSEMBLE_PARENT` reads
    from its event's name. Otherwise no event has a parent: an empty dict.

    Parameters
    ----------
    index : ParametricIndex
        The sample's index, with its events' names if any.
    block : sequence
        Each event's block, in the index's order.
    """
    names = index.event_name
    if names is None or not any(_MEMBER.fullmatch(name) for name in names):
        return {}
    labels = list(np.asarray(block, dtype=object))
    for name, label in zip(names, labels):
        found = _PARENT.fullmatch(name)
        if not isinstance(label, str) or found is None or found["block"] != label:
            return {}
    return dict(zip(np.asarray(index.event_id).tolist(), labels))


def parents_in(sample: Any) -> set[str]:
    """Return the parent storms of a sample's events, or of a design's record."""
    parents: Mapping[int, str] = getattr(sample, "parents", None) or {}
    return {
        parents[event]
        for event in np.asarray(sample.event_id).tolist()
        if event in parents
    }


def split_keys(sample: Any) -> NDArray[Any]:
    """Return what a split keeps together, per event: its parent, else its block.

    For a sample whose events have no parent, its blocks. Otherwise
    ``("parent", p)`` for an event of parent ``p`` and ``("block", b)`` for
    one of block ``b`` without a parent, so that a mixture keeps the members
    of one storm in its several members together.
    """
    parents: Mapping[int, str] = sample.parents
    block = sample.block
    if not parents:
        return block
    keys = np.empty(block.size, dtype=object)
    for row, (event, label) in enumerate(zip(sample.event_id.tolist(), block.tolist())):
        keys[row] = ("parent", parents[event]) if event in parents else ("block", label)
    return keys


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
    block goes to the pool that holds the middle of its frequency. Events
    with a parent storm form one block per parent (:func:`split_keys`).

    Raises
    ------
    ParametricInputError
        If ``rng`` is missing, the sample holds no frequency, or a pool gets
        no block.
    """
    check_rng(rng)
    codes, labels = pd.factorize(split_keys(sample))
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


def aligned_sides(sample: Any, reference: tuple[Any, Any], rng: Any) -> tuple[Any, Any]:
    """Return the holdout sides of ``sample`` with the blocks of a reference split.

    ``reference`` is the training and the test side of another sample of
    the same event set (the same source token), on the same route. On
    catalogue years each side takes the reference side's years. On the
    other routes each block goes to the side that holds it in the
    reference, and the side's frequencies are divided by its share of the
    sample's own frequency, so the sample keeps its ``lambda``; on year sets
    the side's years are sampled from its events with ``rng``. So each side
    holds the reference side's events, whatever the sample's frequencies.

    Raises
    ------
    ScenarioError
        If the sample can't be cut with the reference's blocks: other
        simulation years, a block on neither side of the reference, or an
        event or block on the other side of the reference.
    ParametricInputError
        If a side would hold no event or no frequency.
    """
    train, test = reference
    if sample.route == CATALOGUE:
        _check_years(sample, train, test, _SAME_SOURCE)
        sides = (
            year_part(sample, np.asarray(train.years), Part("holdout", "train")),
            year_part(sample, np.asarray(test.years), Part("holdout", "test")),
        )
    else:
        on_train, on_test = _sides_of_blocks(sample, train, test)
        sides = (
            pool_part(sample, on_train, rng, Part("holdout", "train")),
            pool_part(sample, on_test, rng, Part("holdout", "test")),
        )
    for one, other in ((train, sides[1]), (sides[0], test)):
        events = np.intersect1d(one.event_id, other.event_id)
        blocks = pd.Index(one.block).intersection(pd.Index(other.block))
        if events.size or len(blocks):
            raise ScenarioError(
                "this sample has the source token of another sample of the set, but "
                f"cut with that sample's blocks it puts {events.size} events and "
                f"{len(blocks)} blocks on the other side of that sample's split: "
                "their year or block labels differ"
            )
    return sides


_SAME_SOURCE: Final[str] = "has the source token of another sample of the set"
_SAME_PARENTS: Final[str] = "shares parent storms with an earlier sample of the set"


def _check_years(sample: Any, train: Any, test: Any, why: str) -> None:
    """Raise unless a catalogue sample's years are a reference split's."""
    years = np.sort(np.concatenate([train.years, test.years]))
    own = catalogue_years(sample)
    if not np.array_equal(own, years):
        raise ScenarioError(
            f"this sample {why}, so they are cut with the same years, but its "
            f"{own.size} simulation years, {own[0]} to {own[-1]}, are not that "
            f"sample's {years.size}, {years[0]} to {years[-1]}"
        )


def _sides_of_blocks(
    sample: Any, train: Any, test: Any
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return which events of ``sample`` have a block of a reference split's sides.

    Raises
    ------
    ScenarioError
        If a block of the sample is on neither side.
    """
    block = pd.Index(sample.block)
    on_train = np.asarray(block.isin(pd.Index(train.block)))
    on_test = np.asarray(block.isin(pd.Index(test.block)))
    missing = block[~(on_train | on_test)]
    if len(missing):
        raise ScenarioError(
            "this sample has the source token of another sample of the set, so "
            f"they are cut with the same blocks, but {len(missing)} of its "
            f"blocks, e.g. {list(missing[:5])}, are on neither side of that "
            "sample's split"
        )
    return on_train, on_test


def member_sides(
    member: Any,
    cut: dict[str | None, tuple[Any, Any]],
    side_of: dict[str, tuple[int, tuple[Any, Any]]],
    *,
    fraction: float,
    rng: Any,
    split: Callable[..., tuple[Any, Any]],
) -> tuple[Any, Any]:
    """Return the holdout sides of a scenario set's member, given the members cut before.

    In this order: a member with the source token of an earlier one is cut
    with its blocks (:func:`aligned_sides`); a member that shares parent
    storms with earlier ones (module docstring) is cut with their sides
    (:func:`parent_sides`); any other member by ``split``, ``holdout``.
    ``cut``, the first sides of each token, and ``side_of``, the side of
    each parent storm (0 for training) with the sides that hold it, are
    updated.

    Raises
    ------
    ScenarioError
        If the member can't be cut with the blocks of an earlier member of
        its token, or with the sides of its parent storms.
    ParametricInputError, LeakageError
        As ``split``, or if a side would hold no event or no frequency.
    """
    shared = sorted(parents_in(member).intersection(side_of))
    if member.source in cut:
        pair = aligned_sides(member, cut[member.source], rng)
    elif shared:
        pair = parent_sides(member, side_of, shared[0], fraction, rng)
    else:
        pair = split(member, train_fraction=fraction, rng=rng)
    cut.setdefault(member.source, pair)
    for position, side in enumerate(pair):
        for parent in parents_in(side):
            side_of.setdefault(parent, (position, pair))
    return pair


def parent_sides(
    sample: Any,
    side_of: Mapping[str, tuple[int, tuple[Any, Any]]],
    first: str,
    fraction: float,
    rng: Any,
) -> tuple[Any, Any]:
    """Return the holdout sides of a sample with the sides of its parent storms.

    ``side_of`` gives each parent storm of earlier samples its side, 0 for
    training, and the sides that hold it; ``first`` is a parent of
    ``sample`` among them. On catalogue years the sample takes the years
    of the sides that hold ``first``. On the other routes each parent storm
    goes to its side; the blocks that no earlier sample holds are pooled by
    frequency in an order shuffled by ``rng``, a ``fraction`` of it on the
    training side, as :func:`pools` pools them. A side's frequencies are
    divided by its share, and on year sets its years are sampled with
    ``rng``.

    Raises
    ------
    ScenarioError
        If the catalogue years are not those sides', or a parent storm
        would land on the other side of an earlier sample's split.
    ParametricInputError
        If a side would hold no event or no frequency.
    """
    if sample.route == CATALOGUE:
        train, test = side_of[first][1]
        _check_years(sample, train, test, _SAME_PARENTS)
        sides = (
            year_part(sample, np.asarray(train.years), Part("holdout", "train")),
            year_part(sample, np.asarray(test.years), Part("holdout", "test")),
        )
    else:
        on_train = _parent_pools(sample, side_of, fraction, rng)
        sides = (
            pool_part(sample, on_train, rng, Part("holdout", "train")),
            pool_part(sample, ~on_train, rng, Part("holdout", "test")),
        )
    for position, side in enumerate(sides):
        crossed = sorted(
            parent
            for parent in parents_in(side)
            if side_of.get(parent, (position,))[0] != position
        )
        if crossed:
            raise ScenarioError(
                f"this sample {_SAME_PARENTS}, but cut with that sample's years it "
                f"puts {len(crossed)} parent storms, e.g. {crossed[:5]}, on the other "
                "side of an earlier sample's split: give the members of one storm "
                "years on one side, or take year sets"
            )
    return sides


def _parent_pools(
    sample: Any,
    side_of: Mapping[str, tuple[int, Any]],
    fraction: float,
    rng: Any,
) -> NDArray[np.bool_]:
    """Return which events of ``sample`` go to the training side (:func:`parent_sides`)."""
    codes, labels = pd.factorize(split_keys(sample))
    side = np.full(len(labels), -1)
    for code, label in enumerate(labels):
        if isinstance(label, tuple) and label[0] == "parent" and label[1] in side_of:
            side[code] = side_of[label[1]][0]
    new = np.flatnonzero(side < 0)
    if new.size:
        weight = np.bincount(codes, weights=sample.frequency, minlength=len(labels))
        order = new[rng.permutation(new.size)]
        total = float(weight[order].sum())
        middle = np.cumsum(weight[order]) - weight[order] / 2
        side[order] = (middle >= fraction * total) & (total > 0)
    return np.asarray(side[codes] == 0)


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
