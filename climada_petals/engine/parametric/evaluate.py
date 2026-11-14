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

Evaluate and compare covers: tables of metrics, and payouts as an ``Impact``.

:func:`evaluate` and :func:`compare` take any **entry**, a cover given as it
is (a resolved :class:`~climada_petals.engine.parametric.cover.ParametricCover`)
or a design (a
:class:`~climada_petals.engine.parametric.calibrate.CalibrationResult`), on
any :class:`~climada_petals.engine.parametric.sample.OutcomeSample` with the
entry's units and index definition. They return a pandas table with one row
per entry, level and unit, the setup in columns and the metrics of
:mod:`~climada_petals.engine.parametric.metrics` as columns.

Rows
----
==========================  ==============================================
Entry                       Rows
==========================  ==============================================
A cover given as it is      One ``"unit"`` row per unit, each unit paying
                            by its own terms, and one ``"total"`` row, the
                            owner's total over the units. ``scope`` is null
A design, scope             One ``"unit"`` row per unit, each its own
``"individual"``            design problem, and one ``"total"`` row
A design, scope             One ``"total"`` row. Per-unit rows only under a
``"joint"``                 stated ``MetricOptions(allocation=...)``:
                            ``"unit_by_value"`` or ``"unit_by_loss"`` rows,
                            with no premium and no net-loss columns
==========================  ==============================================

A **total** row holds the metrics of the owner's total per event,
``L_e = sum_j L_{e,j}`` and ``P_e = sum_j P_{e,j}``: within an event, an
over-payment at one unit offsets a shortfall at another.

A joint design's objective sees only each event's total payout, so other
splits across the units are often optimal too, and its per-unit payouts are
the one split the solver returned. So it has no ``"unit"`` row. An allocated
row is an accounting statement about the total payout, under the rule its
``level`` names: ``"unit_by_value"`` gives unit ``j`` the share
``v_j / sum_k v_k`` of each event's total payout, with ``v`` the units'
exposure values; ``"unit_by_loss"`` the share ``L_{e,j} / L_e`` of the
event's loss, or the value share in an event without loss. Their
``expected_loss`` and ``expected_payout`` add up to the total row's under
both rules, and ``expected_uncovered`` under ``"loss"`` only. Allocation
takes joint designs only, so allocated rows never stand beside the rows of
covers that pay by their own terms.

The premium
-----------
``premium`` is what the contract costs, and ``premium_basis`` says where it
comes from:

* ``"stated"``: the ``premium=`` argument of :func:`evaluate`, the price of
  the whole entry. It is the total row's, and the unit row's of an entry
  with one unit.
* ``"design"``: a design's premium, priced on its design sample. It is the
  contract price, also out of sample. The total row of an individual design
  sums its units' premiums: one contract per unit.
* ``"evaluation"``: a cover given as it is, priced on the evaluation
  sample. Its total row prices the total payout as one contract, and a unit
  row the unit's payout alone.

``premium_repriced`` is the same pricing on the evaluation sample: the
design's own pricing for a design, the ``pricing=`` argument otherwise. The
total row of an individual design sums its units'. Under the pure premium
it equals ``expected_payout``. Allocated rows hold neither.

Setup columns
-------------
Every row carries its setup, so it survives a ``concat`` or a filter, and
:func:`~climada_petals.engine.parametric.metrics.setup_text` renders it as
the text that goes next to a number:

=====================  ===================================================
Column                 Meaning
=====================  ===================================================
``cover_label``        The cover's label, else the first 12 hex digits of
                       its hash
``cover_hash``         ``ParametricCover.definition_hash``
``scope``              ``"individual"`` or ``"joint"``; null for a cover
                       given as it is
``level``,             ``"unit"``, ``"total"`` (null ``unit_id``),
``unit_id``            ``"unit_by_value"`` or ``"unit_by_loss"``; the unit
``scenario``           The sample's label
``basis``, ``route``,  ``"event"`` or ``"annual"``; ``"frequency"``,
``n_years``            ``"catalogue"`` or ``"yearsets"``; the years, null on
                       the event basis
``index_definition``   One line: the index's kind, source, statistic and
                       unit
``pricing``,           The pricing and its parameters; ``"stated"``,
``premium_basis``      ``"design"`` or ``"evaluation"``
``payout_threshold``   ``eps`` of the metrics, in payout units
``split``, ``fold``    ``"in_sample"`` or ``"out_of_sample"``; the fold
``unique``, ``ties``,  From the design, null where it doesn't record them,
``degenerate_tail``    and for a cover given as it is. ``degenerate_tail``
                       is ``"expected_uncovered"`` when the tail held every
                       loss, so the weighted-CVaR design minimised ``E[U]``,
                       or ``"single_outcome"`` when ``1 - beta`` was at most
                       the weight of one outcome with a loss
``comonotone_index``   Whether the index determines the loss almost
                       one-to-one on this sample (below): a unit's flag,
                       on a total row true if every unit's is, false if
                       some unit's is not, else null
=====================  ===================================================

External numbers come from out-of-sample rows on the annual basis.
In-sample rows exist for the overfitting gap.

Index diagnostics
-----------------
:func:`evaluate` and :func:`compare` run
:func:`~climada_petals.engine.parametric.basis_risk.index_diagnostics` on
the evaluation sample, at its defaults, once per call. Where the
index determines the loss almost one-to-one, a cover can replicate the
loss and the basis-risk metrics read close to zero: such units log a
warning and are marked in ``comonotone_index``. Null where the correlation
is undefined, such as a unit whose index or loss takes one value.

Designs
-------
A design is read from its record
(:class:`~climada_petals.engine.parametric.calibrate.CalibrationResult`): it
pays by its resolved cover, ``result.cover``; ``unidentified_share`` reads
the amounts the fill rule set (``result.filled``), and is null for a cover
given as it is. An **out-of-sample** evaluation of a design is checked
against the design sample whenever the evaluation sample has its source
token
(:func:`~climada_petals.engine.parametric.sample.check_out_of_sample`):
the whole sample the design sample was cut from, another part of it (by
:func:`~climada_petals.engine.parametric.sample.holdout`,
:func:`~climada_petals.engine.parametric.sample.year_block_kfold`,
``select`` or ``unit``), or the same events built again, also with other
frequencies, exposures or impact-function draws, such as a set scaled by
``apply_climate_scenario_knu``. A shared event or block raises
:class:`~climada_petals.engine.parametric.errors.LeakageError`. So the
label "out of sample" is always checked. A sample of another source, such
as another hazard set, is not compared, because its event ids and years
can coincide with the design sample's by chance.

CLIMADA objects
---------------
:func:`payout_impact` returns the payouts as a CLIMADA ``Impact``, and
:func:`uncovered_impact` the uncovered losses ``(L_e - P_e)^+``, of one
draw on a sample with impact-function draws, on the exposure points of the
units: a unit's value maps back to its points by value share. Both are
non-negative, so ``calc_freq_curve``, ``eai_exp``,
``local_exceedance_impact`` and ``write_hdf5`` work on them unchanged. A
signed net loss is not an ``Impact``: ``local_exceedance_impact`` would drop
its negative part, and the premium, which is annual, has no place in an
event. The net-loss metrics are in the table.

Samples with left-out events
----------------------------
:meth:`~climada_petals.engine.parametric.sample.OutcomeSample.from_impact`
leaves out the events with no loss and an index of 0 at every unit. A cover
that pays at an index of 0 would pay on them, so every function here
refuses it on such a sample
(:func:`~climada_petals.engine.parametric.sample.check_left_out_unpaid`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.engine import Impact
from numpy.typing import NDArray

from ._checks import check_int, check_real
from .basis_risk import comonotone_of, diagnose
from .calibrate import JOINT, CalibrationResult, DesignSample
from .cover import ParametricCover
from .errors import EventAlignmentError, ParametricInputError
from .index import ParametricIndex
from .metrics import (
    SETUP_COLUMNS,
    SPLITS,
    MetricOptions,
    index_text,
    pricing_text,
    subject_metrics,
)
from .pricing import CostOfCapitalLoad, Pricing, PurePremium
from .sample import OutcomeSample, check_left_out_unpaid, check_out_of_sample

__all__ = [
    "compare",
    "evaluate",
    "payout_impact",
    "uncovered_impact",
]

_INDIVIDUAL: Final[str] = "individual"


class _Entry(NamedTuple):
    """What the evaluation reads from an entry.

    ``pay`` returns the payout per event and unit of an index, in the order
    of ``cover.unit_id``. A design has a ``scope``, its ``pricing`` and one
    premium and degenerate tail per design problem (one per unit, or one),
    ``unique`` and ``ties`` where it records them, ``filled``, which says
    where its payout comes from an amount the fill rule set, and the record
    of its design sample.
    """

    cover: ParametricCover
    pay: Callable[[ParametricIndex], NDArray[np.float64]]
    scope: str | None = None
    pricing: Pricing | None = None
    premiums: tuple[float, ...] = ()
    degenerate: tuple[str | None, ...] = ()
    unique: bool | None = None
    ties: int | None = None
    filled: Callable[[ParametricIndex], NDArray[np.bool_]] | None = None
    design: DesignSample | None = None


class _Context(NamedTuple):
    """The checked arguments of one evaluation, and the sample's index diagnostics."""

    split: str
    fold: int
    pricing: Pricing
    premium: float | None
    options: MetricOptions
    diagnostics: pd.DataFrame | None = None


def _entry(entry: object, sample: OutcomeSample) -> _Entry:
    """Return what the evaluation reads from a cover or a design.

    Raises
    ------
    ParametricInputError
        If ``entry`` is neither, or its units are not the sample's.
    UnresolvedParameterError
        If a cover is a template.
    """
    if isinstance(entry, ParametricCover):
        entry.check_resolved()
        found = _Entry(entry, entry.payout_matrix)
    elif isinstance(entry, CalibrationResult):
        found = _Entry(
            entry.cover,
            entry.cover.payout_matrix,
            scope=entry.scope,
            pricing=entry.pricing,
            premiums=tuple(problem.premium for problem in entry.problems),
            degenerate=tuple(problem.degenerate_tail for problem in entry.problems),
            unique=entry.unique,
            ties=entry.ties,
            filled=entry.filled,
            design=entry.design_sample,
        )
    else:
        raise ParametricInputError(
            "an entry is a resolved ParametricCover or a CalibrationResult, got "
            f"{type(entry).__name__}"
        )
    if tuple(found.cover.unit_id) != sample.unit_id:
        raise ParametricInputError(
            "the entry's units are not the sample's: "
            f"{list(found.cover.unit_id)[:5]} vs {list(sample.unit_id)[:5]}"
        )
    return found


def _check_sample(sample: object) -> OutcomeSample:
    """Return ``sample`` if it is an :class:`OutcomeSample`."""
    if not isinstance(sample, OutcomeSample):
        raise ParametricInputError(
            f"sample must be an OutcomeSample, got {type(sample).__name__}"
        )
    return sample


def _check_left_out(found: _Entry, sample: OutcomeSample) -> None:
    """Raise if the sample left events out and the entry pays at an index of 0."""
    zero = ParametricIndex(
        np.zeros((1, sample.n_units)), [0], sample.unit_id, sample.index.definition
    )
    check_left_out_unpaid(sample, found.pay(zero))


def _context(
    found: _Entry,
    sample: OutcomeSample,
    split: object,
    fold: object,
    stated: tuple[object, object, object],
) -> _Context:
    """Check the arguments of :func:`evaluate`; ``stated`` is the pricing, premium and options.

    Raises
    ------
    ParametricInputError
        If an argument is invalid, the fold is not the sample's, a design
        is given another pricing than its own, or an allocation is asked of
        an entry that is not a joint design.
    LeakageError
        If an out-of-sample evaluation of a design, on a sample of the
        design sample's source, shares an event or a block with it.
    """
    pricing, premium, options = stated
    if split not in SPLITS:
        raise ParametricInputError(
            f"split must be one of {list(SPLITS)}, got {split!r}"
        )
    recorded = 0 if sample.split is None else sample.split.fold
    chosen = recorded if fold is None else check_int("fold", fold, low=0)
    if chosen != recorded:
        raise ParametricInputError(
            f"fold {chosen} is not the sample's: it was cut as fold {recorded}"
        )
    if pricing is not None and not isinstance(pricing, PurePremium | CostOfCapitalLoad):
        raise ParametricInputError(
            f"pricing must be PurePremium() or CostOfCapitalLoad(rate, q), got {pricing!r}"
        )
    if found.pricing is not None and pricing not in (None, found.pricing):
        raise ParametricInputError(
            f"a design is priced with its own pricing, {found.pricing!r}; it was given "
            f"{pricing!r}. Leave pricing out to reprice it with its own"
        )
    if not isinstance(options, MetricOptions):
        raise ParametricInputError(f"options must be MetricOptions, got {options!r}")
    if options.allocation is not None and found.scope != JOINT:
        raise ParametricInputError(
            f"allocation={options.allocation!r} takes joint designs only, so that "
            "allocated rows never stand beside the rows of covers that pay by their "
            "own terms"
        )
    _check_left_out(found, sample)
    if split == "out_of_sample" and found.design is not None:
        check_out_of_sample(found.design, sample)
    return _Context(
        str(split),
        chosen,
        found.pricing or pricing or PurePremium(),
        None if premium is None else check_real("premium", premium, low=0.0),
        options,
    )


class _Subject(NamedTuple):
    """One row's subject: its level, unit, loss and payout per row, premiums."""

    level: str
    unit_id: Hashable | None
    loss: NDArray[np.float64]
    payout: NDArray[np.float64]
    premium: float | None
    premium_basis: str | None
    repriced: float | None
    degenerate: str | None
    filled: NDArray[np.bool_] | None


def _unit_subjects(
    found: _Entry,
    sample: OutcomeSample,
    payouts: tuple[NDArray[np.float64], NDArray[np.bool_] | None],
    context: _Context,
) -> list[_Subject]:
    """Return the ``"unit"`` subjects of a cover or an individual design.

    ``payouts`` holds the payout per row and unit, and where a design's
    payout comes from an amount its fill rule set, or None.
    """
    paid, filled = payouts
    loss = sp.csc_matrix(sample.loss)
    single = sample.n_units == 1 and context.premium is not None
    subjects = []
    for j, unit in enumerate(sample.unit_id):
        repriced = context.pricing.premium(paid[:, j], sample.measure)
        if single:
            premium, basis = context.premium, "stated"
        elif found.scope == _INDIVIDUAL:
            premium, basis = found.premiums[j], "design"
        else:
            premium, basis = repriced, "evaluation"
        subjects.append(
            _Subject(
                "unit",
                unit,
                loss[:, j].toarray().reshape(-1),
                paid[:, j],
                premium,
                basis,
                repriced,
                found.degenerate[j] if found.scope == _INDIVIDUAL else None,
                None if filled is None else filled[:, j],
            )
        )
    return subjects


def _total_subject(
    found: _Entry,
    sample: OutcomeSample,
    payouts: tuple[NDArray[np.float64], NDArray[np.bool_] | None],
    context: _Context,
    units: list[_Subject],
) -> _Subject:
    """Return the ``"total"`` subject: the owner's total over the units."""
    paid, filled = payouts
    total = paid.sum(axis=1)
    if found.scope == _INDIVIDUAL:
        premium = float(sum(found.premiums))
        repriced = float(sum(subject.repriced or 0.0 for subject in units))
        basis, degenerate = "design", None
    else:
        repriced = context.pricing.premium(total, sample.measure)
        premium = found.premiums[0] if found.scope == JOINT else repriced
        basis = "design" if found.scope == JOINT else "evaluation"
        degenerate = found.degenerate[0] if found.scope == JOINT else None
    if context.premium is not None:
        premium, basis = context.premium, "stated"
    return _Subject(
        "total",
        None,
        np.asarray(sample.loss.sum(axis=1), dtype=np.float64).reshape(-1),
        total,
        premium,
        basis,
        repriced,
        degenerate,
        None if filled is None else filled.any(axis=1),
    )


def _allocated_subjects(
    sample: OutcomeSample, total: _Subject, rule: str
) -> list[_Subject]:
    """Return a joint design's rows that allocate its total payout to the units."""
    value_share = sample.units.unit_value / float(sample.units.unit_value.sum())
    loss = sample.loss.toarray()
    share = np.broadcast_to(value_share, loss.shape)
    if rule == "loss":
        lossy = total.loss > 0
        share = np.where(
            lossy[:, None], loss / np.where(lossy, total.loss, 1.0)[:, None], share
        )
    return [
        _Subject(
            f"unit_by_{rule}",
            unit,
            loss[:, j],
            total.payout * share[:, j],
            None,
            None,
            None,
            total.degenerate,
            total.filled,
        )
        for j, unit in enumerate(sample.unit_id)
    ]


def _row(
    found: _Entry, sample: OutcomeSample, subject: _Subject, context: _Context
) -> dict[str, Any]:
    """Return one table row: the setup and the metrics of a subject."""
    cover = found.cover
    digest = cover.definition_hash
    metrics = subject_metrics(
        subject.loss,
        subject.payout,
        sample.measure,
        premium=subject.premium,
        filled=subject.filled,
        options=context.options,
    )
    return {
        "cover_label": cover.label or digest.split(":")[1][:12],
        "cover_hash": digest,
        "scope": found.scope,
        "level": subject.level,
        "unit_id": subject.unit_id,
        "scenario": sample.scenario,
        "basis": sample.kind,
        "route": sample.route,
        "n_years": sample.n_years,
        "index_definition": index_text(cover.index),
        "pricing": pricing_text(context.pricing),
        "premium_basis": subject.premium_basis,
        "payout_threshold": context.options.payout_threshold,
        "split": context.split,
        "fold": context.fold,
        "unique": found.unique,
        "ties": found.ties,
        "degenerate_tail": subject.degenerate,
        "comonotone_index": comonotone_of(
            context.diagnostics,
            sample.unit_id if subject.unit_id is None else [subject.unit_id],
        ),
        "premium": math.nan if subject.premium is None else subject.premium,
        "premium_repriced": math.nan if subject.repriced is None else subject.repriced,
        **metrics,
    }


def _frame(rows: list[dict[str, Any]], options: MetricOptions) -> pd.DataFrame:
    """Return the rows as a table with the setup and metric columns in order."""
    frame = pd.DataFrame(rows, columns=[*SETUP_COLUMNS, *options.columns()])
    for name in ("n_years", "ties"):
        frame[name] = frame[name].astype("Int64")
    frame["fold"] = frame["fold"].astype(np.int64)
    for name in ("unique", "comonotone_index"):
        frame[name] = frame[name].astype(object)
    metric = list(options.columns())
    frame[metric] = frame[metric].astype(np.float64)
    return frame


def evaluate(
    entry: ParametricCover | CalibrationResult,
    sample: OutcomeSample,
    *,
    split: str,
    fold: int | None = None,
    pricing: Pricing | None = None,
    premium: float | None = None,
    options: MetricOptions = MetricOptions(),  # frozen, so safe to share
) -> pd.DataFrame:
    """Return the metrics of a cover or a design on a sample (module docstring).

    Parameters
    ----------
    entry : ParametricCover or CalibrationResult
        A resolved cover given as it is, or a design.
    sample : OutcomeSample
        The evaluation sample, with the entry's units and index definition:
        the design sample (in sample), or a sample the entry was not
        designed on, such as the test side of
        :func:`~climada_petals.engine.parametric.sample.holdout`.
    split : {"in_sample", "out_of_sample"}
        Which the sample is, for the entry. Required.
    fold : int, optional
        The fold, recorded in the table. Default: the fold of the sample's
        split record, or 0 for a whole sample. A fold that differs from the
        record raises.
    pricing : PurePremium or CostOfCapitalLoad, optional
        The pricing of a cover given as it is, on this sample. Default
        ``PurePremium()``. A design is priced with its own pricing; another
        one raises.
    premium : float, optional
        The premium of the whole entry, stated (module docstring).
    options : MetricOptions, default MetricOptions()
        The metrics' thresholds, and the allocation of a joint design.

    Returns
    -------
    pandas.DataFrame
        One row per subject:
        :data:`~climada_petals.engine.parametric.metrics.SETUP_COLUMNS`, then
        ``options.columns()``. Metrics are float, NaN where null.

    Raises
    ------
    ParametricInputError
        If an argument is invalid, the entry's units or index definition
        are not the sample's, or the sample left events out and the entry
        pays at an index of 0.
    UnresolvedParameterError
        If a cover is a template.
    LeakageError
        If ``split="out_of_sample"``, the entry is a design, and the sample
        has the design sample's source token and shares an event or a block
        with it, e.g. the whole sample the design sample was cut from
        (module docstring).

    Examples
    --------
    >>> train, test = holdout(sample, train_fraction=0.7)  # doctest: +SKIP
    >>> table = evaluate(design, test, split="out_of_sample")  # doctest: +SKIP
    >>> print(setup_text(table))  # doctest: +SKIP
    """
    found = _entry(entry, _check_sample(sample))
    context = _context(found, sample, split, fold, (pricing, premium, options))
    return _table(
        found, sample, context._replace(diagnostics=diagnose(sample, "evaluate"))
    )


def _table(found: _Entry, sample: OutcomeSample, context: _Context) -> pd.DataFrame:
    """Return the table of a checked evaluation."""
    options = context.options
    rows = sample.event_row
    payouts = (
        found.pay(sample.index)[rows],
        None if found.filled is None else found.filled(sample.index)[rows],
    )
    if found.scope == JOINT:
        subjects = [_total_subject(found, sample, payouts, context, [])]
        if options.allocation is not None:
            subjects += _allocated_subjects(sample, subjects[0], options.allocation)
    else:
        subjects = _unit_subjects(found, sample, payouts, context)
        subjects.append(_total_subject(found, sample, payouts, context, subjects))
    return _frame([_row(found, sample, s, context) for s in subjects], options)


def compare(
    entries: Sequence[ParametricCover | CalibrationResult],
    sample: OutcomeSample,
    *,
    split: str,
    fold: int | None = None,
    pricing: Pricing | None = None,
    options: MetricOptions = MetricOptions(),  # frozen, so safe to share
) -> pd.DataFrame:
    """Return the metrics of several covers or designs on one sample, in one table.

    Each entry's rows are :func:`evaluate`'s, in the order of ``entries``;
    the index diagnostics run once.
    The table ranks nothing and marks no entry as best. The ``scope`` column
    keeps the rows of each scope apart, and makes no statement about them.

    Parameters
    ----------
    entries : sequence of ParametricCover or CalibrationResult
        Resolved covers given as they are, and designs.
    sample : OutcomeSample
        The evaluation sample.
    split : {"in_sample", "out_of_sample"}
        Which the sample is, for every entry.
    fold : int, optional
        As in :func:`evaluate`.
    pricing : PurePremium or CostOfCapitalLoad, optional
        As in :func:`evaluate`: the pricing of the covers given as they are.
    options : MetricOptions, default MetricOptions()
        As in :func:`evaluate`. An allocation needs every entry to be a
        joint design.

    Returns
    -------
    pandas.DataFrame
        :func:`evaluate`'s columns.

    Raises
    ------
    ParametricInputError
        If ``entries`` is empty or not a sequence, an allocation is asked
        and an entry is not a joint design, or :func:`evaluate` raises.
    LeakageError
        As :func:`evaluate`, for a design.
    """
    if isinstance(entries, ParametricCover | CalibrationResult) or not isinstance(
        entries, Sequence
    ):
        raise ParametricInputError(
            f"entries must be a sequence of covers and designs, got {entries!r}"
        )
    if not entries:
        raise ParametricInputError("entries must hold at least one cover or design")
    if isinstance(options, MetricOptions) and options.allocation is not None:
        others = [
            position
            for position, entry in enumerate(entries)
            if not (isinstance(entry, CalibrationResult) and entry.scope == JOINT)
        ]
        if others:
            raise ParametricInputError(
                f"allocation={options.allocation!r} takes joint designs only; the "
                f"entries at {others} are not"
            )
    checked = _check_sample(sample)
    diagnostics = None
    tables = []
    for entry in entries:
        found = _entry(entry, checked)
        context = _context(found, checked, split, fold, (pricing, None, options))
        if diagnostics is None:
            diagnostics = diagnose(checked, "compare")
        tables.append(_table(found, checked, context._replace(diagnostics=diagnostics)))
    return pd.concat(tables, ignore_index=True)


def _impact_rows(sample: OutcomeSample, impact: Any) -> NDArray[np.intp]:
    """Return the row of each of the sample's events in ``impact``, checked.

    Raises
    ------
    ParametricInputError
        If the sample was cut from another, or the frequencies differ.
    ImpactInputError
        If ``impact`` is not a loss on the units' exposure points.
    EventAlignmentError
        If an event of the sample is not the impact's, or the impact holds
        other events than the sample's left-out ones.
    """
    sample.units.check_impact(impact)
    if sample.split is not None:
        raise ParametricInputError(
            f"impact= takes a whole sample; this one was cut ({sample.split.method}, "
            f"{sample.split.side}), so its events and frequencies are its own: leave "
            "impact out"
        )
    ids = pd.Index(np.asarray(impact.event_id))
    rows = ids.get_indexer(sample.event_id)
    if ids.has_duplicates or np.any(rows < 0):
        raise EventAlignmentError(
            "every event of the sample must be in the impact once, matched on event_id"
        )
    others = np.setdiff1d(np.arange(len(ids)), rows)
    lossy = np.asarray(abs(sample.units.aggregate(impact.imp_mat[others])).sum(axis=1))
    if others.size != sample.n_left_out or np.any(lossy > 0):
        raise EventAlignmentError(
            f"the impact holds {others.size} events beyond the sample's, and the "
            f"sample left out {sample.n_left_out} without loss: pass the impact the "
            "sample was built from"
        )
    frequency = np.asarray(impact.frequency, dtype=np.float64)[rows]
    if not np.allclose(frequency, sample.frequency, rtol=1e-12, atol=0.0):
        raise ParametricInputError("the impact's frequencies are not the sample's")
    return rows


def _as_impact(
    sample: OutcomeSample, per_unit: NDArray[np.float64], impact: Any, unit: str
) -> Impact:
    """Return per-event, per-unit values as an ``Impact`` on the units' points.

    Its ``tot_value`` is the units' value, with or without ``impact``: the
    loss ``Impact``'s own is deprecated in CLIMADA and not read.
    """
    points = sample.units.to_points(per_unit)
    if impact is None:
        names = sample.index.event_name
        source: dict[str, Any] = {
            "event_id": np.asarray(sample.event_id),
            "event_name": (
                [str(e) for e in sample.event_id] if names is None else list(names)
            ),
            "date": np.zeros(sample.n_events, dtype=np.int64),
            "frequency": np.asarray(sample.frequency, dtype=np.float64),
            "coord_exp": np.asarray(sample.units.coords),
            "crs": sample.units.exposures.crs,
            "haz_type": str(
                (sample.index.definition.parameters or {}).get("haz_type", "")
            ),
        }
    else:
        rows = _impact_rows(sample, impact)
        select = sp.csr_matrix(
            (np.ones(rows.size), (rows, np.arange(rows.size))),
            shape=(len(impact.event_id), rows.size),
        )
        points = sp.csr_matrix(select @ points)
        source = {
            "event_id": np.asarray(impact.event_id),
            "event_name": list(impact.event_name),
            "date": np.asarray(impact.date),
            "frequency": np.asarray(impact.frequency, dtype=np.float64),
            "coord_exp": np.asarray(impact.coord_exp),
            "crs": impact.crs,
            "haz_type": impact.haz_type,
        }
    eai_exp = np.asarray(points.T @ source["frequency"], dtype=np.float64)
    return Impact(
        **source,
        tot_value=float(sample.units.value.sum()),
        frequency_unit="1/year",
        eai_exp=eai_exp,
        at_event=np.asarray(points.sum(axis=1), dtype=np.float64).reshape(-1),
        aai_agg=float(eai_exp.sum()),
        unit=unit,
        imp_mat=points,
    )


def payout_impact(
    entry: ParametricCover | CalibrationResult,
    sample: OutcomeSample,
    *,
    impact: Any = None,
) -> Impact:
    """Return the payouts on a sample as a CLIMADA ``Impact`` (module docstring).

    ``imp_mat`` holds the payout per event and exposure point: a unit's
    payout shared among its points by value share
    (:meth:`~climada_petals.engine.parametric.units.InsuredUnits.to_points`).
    ``at_event`` is its sum over the points, ``eai_exp`` the expected annual
    payout per point from the events' frequencies, ``sum_e f_e P_e``, and
    ``aai_agg`` their sum, as CLIMADA defines them. ``unit`` is the cover's
    ``payout_unit``, ``tot_value`` the units' value.

    ``aai_agg`` is the evaluation's ``expected_payout`` of the total row on
    the event basis, and on catalogue years for a whole sample, whose
    events have frequency ``1 / N`` (within the route's tolerance). It is
    not on year sets, whose ``expected_payout`` is the mean over the
    sampled years: the two differ by the sampling error of the years
    (:attr:`~climada_petals.engine.parametric.sample.OutcomeSample.correction_factor`
    gives it for the loss). Nor on a side of a catalogue split, whose years
    weigh ``1 / N_side`` while its events keep the frequency ``1 / N``:
    there ``aai_agg`` is ``N_side / N`` times ``expected_payout``.

    Parameters
    ----------
    entry : ParametricCover or CalibrationResult
        A resolved cover given as it is, or a design.
    sample : OutcomeSample
        The sample, with the entry's units and index definition. Its events
        and frequencies are the ``Impact``'s. A sample holds no dates, so
        every date is 0, no date, unless ``impact`` is given.
    impact : climada.engine.Impact, optional
        The loss ``Impact`` a whole sample was built from. The ``Impact``
        then has its events, in its order, with its dates and names: the
        events the sample left out pay 0, and a ``sampling_vect`` of the
        loss applies to the payouts, e.g. in
        ``climada.util.yearsets.impact_yearset_from_sampling_vect``.

    Returns
    -------
    climada.engine.Impact

    Raises
    ------
    ParametricInputError
        If the entry doesn't fit the sample, or pays at an index of 0 on a
        sample that left events out, or ``impact`` is given with a sample
        cut from another, or with other frequencies.
    ImpactInputError
        If ``impact`` is not a loss on the units' exposure points.
    EventAlignmentError
        If ``impact``'s events are not the sample's and its left-out ones.
    """
    found = _entry(entry, _check_sample(sample))
    _check_left_out(found, sample)
    return _as_impact(sample, found.pay(sample.index), impact, found.cover.payout_unit)


def _draw_rows(sample: OutcomeSample, draw: object) -> slice:
    """Return the rows of one draw of the sample, checked.

    Raises
    ------
    ParametricInputError
        If ``draw`` is missing on a sample with draws, or not one of its
        draws.
    """
    if draw is None:
        if sample.n_draws != 1:
            raise ParametricInputError(
                f"the sample has {sample.n_draws} draws of each event's loss, and an "
                "Impact one loss per event: pass draw=, the draw to return"
            )
        return slice(0, sample.n_events)
    chosen = check_int("draw", draw, low=0)
    if chosen >= sample.n_draws:
        raise ParametricInputError(
            f"draw must be below the sample's {sample.n_draws} draws, got {chosen}"
        )
    return slice(chosen * sample.n_events, (chosen + 1) * sample.n_events)


def uncovered_impact(
    entry: ParametricCover | CalibrationResult,
    sample: OutcomeSample,
    *,
    impact: Any = None,
    draw: int | None = None,
) -> Impact:
    """Return the uncovered losses on a sample as a CLIMADA ``Impact``.

    ``imp_mat`` holds ``(L_{e,j} - P_{e,j})^+``, the shortfall of each unit
    ``j`` in each event, shared among the unit's points by value share, as
    :func:`payout_impact` shares the payouts. ``aai_agg`` is
    ``sum_e f_e U_e``, from the events' frequencies: the sum over the units
    of their ``expected_uncovered`` where :func:`payout_impact`'s
    ``aai_agg`` is the ``expected_payout``, on the event basis and on
    catalogue years for a whole sample, and not on year sets or on a side
    of a catalogue split (:func:`payout_impact`). On a sample with draws it
    is that of one draw, whose mean over the draws is the sum on the event
    basis.

    Parameters
    ----------
    entry, sample, impact
        As in :func:`payout_impact`. On a sample with impact-function
        draws, ``impact`` is the given draw's loss ``Impact``.
    draw : int, optional
        On a sample with impact-function draws, the draw whose uncovered
        losses to return, from 0; its events have their own frequencies.
        Required there; on a sample without draws, None or 0.

    Returns
    -------
    climada.engine.Impact

    Raises
    ------
    ParametricInputError
        As :func:`payout_impact`, and if ``draw`` is missing on a sample
        with draws or not one of its draws.
    ImpactInputError, EventAlignmentError
        As :func:`payout_impact`.
    """
    found = _entry(entry, _check_sample(sample))
    _check_left_out(found, sample)
    rows = _draw_rows(sample, draw)
    shortfall = np.maximum(sample.loss[rows].toarray() - found.pay(sample.index), 0.0)
    return _as_impact(sample, shortfall, impact, found.cover.payout_unit)
