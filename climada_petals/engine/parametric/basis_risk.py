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

Basis risk on CLIMADA objects: where it comes from, and how it is measured.

A cover pays on its index; the owner bears the loss. Basis risk is what the
index doesn't tell about the loss. With an index read from the loss's own
hazard set at the centroid of a unit of one exposure point, and one impact
function, the loss is a function of the index: a design with enough steps
can replicate it, and the metrics then show almost no basis risk. That is
a property of the set-up, not of the cover. So basis risk on CLIMADA objects
comes from a stated source, and the module measures whether there is one.

Sources of basis risk
---------------------
The structural sources come first. They are where a CLIMADA set-up differs
from a real cover's: the index and the loss are different things.

==========================  ==============================================
Source                      How it enters
==========================  ==============================================
An index that is not the    The cat-in-a-circle category of the tracks
footprint at the assets     (``ParametricIndex.from_cat_in_circle``): an
                            event whose track misses the circle has index 0
                            and can't pay. An index from another hazard set,
                            or from a coarser resolution, than the loss's
                            (``from_hazard`` and ``matched_to``). A user's
                            table: station readings, a satellite product
                            (``from_table``)
A unit of several points    A unit that groups points
                            (``InsuredUnits.from_column``,
                            ``from_geometries``, ``whole``), with an index
                            ``index_stat`` of ``"mean"``, ``"max"`` or a
                            percentile over its centroids: the footprint
                            varies within the unit. Its size alone doesn't
                            settle it: a small group behaves like one point
Impact-function draws       Several losses per event, one per draw of the
                            impact functions (:func:`impact_function_draws`,
                            ``OutcomeSample.from_impact_draws``). They come
                            second, with the limit stated below
==========================  ==============================================

The index's definition records the source in every row of a table
(``index_definition``). There is no separate basis-risk model to state: the
index builder is the statement, and the degenerate case is caught by
measurement.

Measuring it
------------
:func:`index_diagnostics` returns, per unit, the frequency-weighted
Spearman correlation of index and loss, the flag ``comonotone_index`` and
the share of the expected loss on events with an index of 0. The flag says
that the index determines the loss almost one-to-one, in the direction the
cover pays. :func:`~climada_petals.engine.parametric.calibrate.calibrate`
and :func:`~climada_petals.engine.parametric.evaluate.evaluate` run it on
their samples at the default threshold: a flagged unit logs a warning
through the module's ``LOGGER`` and sets the ``comonotone_index`` column of
the design's ``summary()`` and of the evaluation's table, so that a reader
of a table sees it.

The default threshold, 0.99, rests on one synthetic tropical-cyclone site,
with designs that minimise the expected uncovered loss on the annual basis,
evaluated out of sample: a unit of 25 points on a 0.1 degree grid (Spearman
0.999) behaved like a single point, its uncovered share falling as steps
were added, and a unit on a 0.5 degree grid (Spearman 0.956) did not. Choose
it for the set-up at hand.

Impact-function draws, and their limit
--------------------------------------
Draws of the impact functions widen the loss at a given index: each draw is
one ``ImpactFuncSet``, every event and point of a draw uses it, and
``OutcomeSample.from_impact_draws`` takes one ``Impact`` per draw. A draw is
uncertainty about the impact functions, common to every event of the draw,
not noise from one event to the next. The draws enter calibration and
evaluation alike.

Their limit: a payout depends on the index only, so noise on the impact
can't change whether an event pays. It can't change whether an event has a
loss either, unless the draws move an impact function's threshold, and
:func:`impact_function_draws` keeps every threshold. So where the index is
the hazard at the unit, the false-negative and false-positive rates at a
loss threshold of 0 are the same with and without these draws. The draws
change how much is lost, and so the uncovered loss, not which events lose.

Uncertainty and sensitivity analysis of cover metrics with
``climada.engine.unsequa`` is not part of this module.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy.sparse as sp
from climada.entity import ImpactFunc, ImpactFuncSet
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from ._checks import check_int, check_real
from .errors import ParametricInputError
from .sample import OutcomeSample

__all__ = [
    "COMONOTONE_THRESHOLD",
    "DIAGNOSTIC_COLUMNS",
    "comonotone_flags",
    "comonotone_of",
    "diagnose",
    "impact_function_draws",
    "index_diagnostics",
]

COMONOTONE_THRESHOLD: Final[float] = 0.99
"""The default threshold of :func:`index_diagnostics` (module docstring)."""

DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "unit_id",
    "n_events",
    "spearman",
    "threshold",
    "comonotone_index",
    "zero_index_loss_share",
)
"""The columns of :func:`index_diagnostics`."""

_ROUNDING: Final[float] = 1e-12
"""What the flag allows below the threshold, for rounding."""


def _midranks(values: NDArray[np.float64], weights: NDArray[np.float64]) -> Any:
    """Return weighted mid-ranks: the weight below each value and half its ties'."""
    distinct, inverse = np.unique(values, return_inverse=True)
    group = np.bincount(inverse, weights=weights, minlength=distinct.size)
    return (np.cumsum(group) - group / 2.0)[inverse]


def _spearman(
    index: NDArray[np.float64], loss: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    """Return the weighted Spearman correlation; NaN if either takes one value.

    The Pearson correlation of the weighted mid-ranks, with the same
    weights. With equal weights it is the usual Spearman correlation, ties
    taking their average rank. Rows of weight 0 are left out.
    """
    positive = weights > 0
    index, loss, weights = index[positive], loss[positive], weights[positive]
    if np.unique(index).size < 2 or np.unique(loss).size < 2:
        return math.nan
    prob = weights / float(weights.sum())
    first = _midranks(index, weights)
    second = _midranks(loss, weights)
    first = first - float(prob @ first)
    second = second - float(prob @ second)
    found = float(prob @ (first * second)) / math.sqrt(
        float(prob @ first**2) * float(prob @ second**2)
    )
    return min(1.0, max(-1.0, found))


def _unit_row(
    unit: Hashable,
    values: NDArray[np.float64],
    loss: NDArray[np.float64],
    weights: NDArray[np.float64],
    rule: tuple[float, float],
) -> dict[str, Any]:
    """Return one unit's diagnostics.

    ``values`` is the unit's index per event, ``loss`` and ``weights`` its
    loss and weight per row (draw, then event), and ``rule`` the threshold
    and the tail's sign.
    """
    level, sign = rule
    n_draws = loss.size // values.size
    index = np.tile(values, n_draws)
    lossy = (loss.reshape(n_draws, values.size) > 0).any(axis=0)
    events = lossy | (values != 0)
    rows = np.tile(events, n_draws)
    rho = _spearman(index[rows], loss[rows], weights[rows])
    expected = float(weights @ loss)
    return {
        "unit_id": unit,
        "n_events": int(events.sum()),
        "spearman": rho,
        "threshold": level,
        "comonotone_index": (
            None if math.isnan(rho) else bool(sign * rho >= level - _ROUNDING)
        ),
        "zero_index_loss_share": (
            float(weights @ (loss * (index == 0))) / expected
            if expected > 0
            else math.nan
        ),
    }


def index_diagnostics(
    sample: OutcomeSample, *, threshold: float = COMONOTONE_THRESHOLD
) -> pd.DataFrame:
    """Return, per unit, how far the index determines the loss (module docstring).

    The relation is read per event, whatever the sample's basis: each row
    (an event, or an event in one draw) weighs its event's frequency per
    year, divided among the draws.

    Parameters
    ----------
    sample : OutcomeSample
        The sample, e.g. a design or an evaluation sample.
    threshold : float, default 0.99
        The Spearman correlation, oriented by the index's tail, from which
        the index counts as determining the loss, in ``(0, 1]``.

    Returns
    -------
    pandas.DataFrame
        One row per unit, in the sample's order, with the columns
        :data:`DIAGNOSTIC_COLUMNS`:

        * ``n_events``: the events with a loss in some draw or a non-zero
          index at the unit, over which the correlation runs, with all
          their draws;
        * ``spearman``: the frequency-weighted Spearman correlation of index
          and loss over them, ties taking their mid-rank; NaN if the index
          or the loss takes one value there;
        * ``threshold``: as given;
        * ``comonotone_index``: whether the correlation, negated on a
          lower-tail index (where the cover pays as the index falls), is at
          least ``threshold``, allowing 1e-12 for rounding; None where the
          correlation is NaN;
        * ``zero_index_loss_share``: the share of the unit's expected loss
          on events with an index of exactly 0, NaN without loss. On an
          upper-tail index these are events a cover that pays nothing at 0
          can't pay on, such as storms whose track misses the circle; on a
          lower-tail index 0 is a reading like any other.

    Raises
    ------
    ParametricInputError
        If ``sample`` is not an ``OutcomeSample`` or ``threshold`` is not in
        ``(0, 1]``.

    Examples
    --------
    >>> table = index_diagnostics(sample)  # doctest: +SKIP
    >>> table[["unit_id", "spearman", "comonotone_index"]]  # doctest: +SKIP
    """
    if not isinstance(sample, OutcomeSample):
        raise ParametricInputError(
            f"sample must be an OutcomeSample, got {type(sample).__name__}"
        )
    level = check_real("threshold", threshold, low=0.0, high=1.0, include_low=False)
    rule = (level, -1.0 if sample.index.definition.tail == "lower" else 1.0)
    weights = np.tile(sample.frequency, sample.n_draws) / sample.n_draws
    loss = sp.csc_matrix(sample.loss)
    frame = pd.DataFrame(
        [
            _unit_row(
                unit,
                sample.index.values[:, j],
                loss[:, j].toarray().reshape(-1),
                weights,
                rule,
            )
            for j, unit in enumerate(sample.unit_id)
        ],
        columns=list(DIAGNOSTIC_COLUMNS),
    )
    frame["comonotone_index"] = frame["comonotone_index"].astype(object)
    return frame


def comonotone_flags(table: pd.DataFrame) -> dict[Hashable, bool | None]:
    """Return the flag of each unit of :func:`index_diagnostics`' table."""
    return {
        unit: (None if pd.isna(flag) else bool(flag))
        for unit, flag in zip(table["unit_id"], table["comonotone_index"])
    }


def comonotone_of(
    table: pd.DataFrame | None, unit_id: Iterable[Hashable]
) -> bool | None:
    """Return whether the index determines the loss of every given unit.

    The flags of the units combine as "and" does with an unknown value:
    true if every unit is flagged, false if some unit is not, else None.
    Each unit paying on its own index, the loss of a set of units is a
    function of their indices when every unit's is.

    Parameters
    ----------
    table : pandas.DataFrame or None
        A table of :func:`index_diagnostics`; None gives None.
    unit_id : iterable of hashable
        Units of the table.

    Returns
    -------
    bool or None
    """
    if table is None:
        return None
    flags: Mapping[Hashable, bool | None] = comonotone_flags(table)
    found = [flags[unit] for unit in unit_id]
    if any(flag is False for flag in found):
        return False
    return True if all(flag is True for flag in found) else None


def diagnose(sample: OutcomeSample, where: str) -> pd.DataFrame:
    """Run :func:`index_diagnostics` at the default threshold; warn of flagged units.

    Parameters
    ----------
    sample : OutcomeSample
        The design or evaluation sample.
    where : str
        What runs it, for the message, e.g. ``"calibrate"``.

    Returns
    -------
    pandas.DataFrame
        The table of :func:`index_diagnostics`.
    """
    table = index_diagnostics(sample)
    flagged = [unit for unit, flag in comonotone_flags(table).items() if flag]
    if flagged:
        LOGGER.warning(
            "%s: at %d of %d units, e.g. %s, the index determines the loss almost "
            "one-to-one (frequency-weighted Spearman correlation of index and loss "
            "at least %g, oriented by the index's tail), so a cover can replicate "
            "the loss and basis risk reads close to zero. Basis risk needs a "
            "stated source: an index from another hazard set or resolution, a "
            "statistic over a group of points, a track-based trigger, a table of "
            "readings, or impact-function draws. The comonotone_index column marks "
            "these units",
            where,
            len(flagged),
            sample.n_units,
            flagged[:5],
            COMONOTONE_THRESHOLD,
        )
    return table


def _curve(impf: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return an impact function's intensities and ``mdd * paa``, checked.

    Raises
    ------
    ParametricInputError
        If the arrays are not finite, of one size, at least one value, with
        intensities that don't fall and ``mdd`` and ``paa`` >= 0.
    """
    intensity = np.asarray(impf.intensity, dtype=np.float64)
    mdd = np.asarray(impf.mdd, dtype=np.float64)
    paa = np.asarray(impf.paa, dtype=np.float64)
    valid = (
        intensity.ndim == 1
        and intensity.size > 0
        and mdd.shape == intensity.shape == paa.shape
        and np.all(np.isfinite(np.concatenate([intensity, mdd, paa])))
    )
    if not valid or np.any(np.diff(intensity) < 0) or np.any(mdd * paa < 0):
        raise ParametricInputError(
            f"impact function {impf.haz_type!r} {impf.id!r}: intensity, mdd and paa "
            "must be finite, of one size and not empty, with intensities that don't "
            "fall and mdd * paa >= 0"
        )
    return intensity, mdd * paa


def _stretched(impf: ImpactFunc, factor: float) -> ImpactFunc:
    """Return a copy of ``impf`` with its intensities above its threshold stretched.

    The threshold is the last intensity of the function's leading zeros of
    ``mdd * paa``, or its first intensity if it has none. An intensity ``x``
    above it becomes ``t + factor (x - t)``; the others stay.
    """
    intensity, mdr = _curve(impf)
    positive = np.flatnonzero(mdr > 0)
    threshold = float(
        intensity[-1] if positive.size == 0 else intensity[max(positive[0] - 1, 0)]
    )
    drawn = copy.deepcopy(impf)
    drawn.intensity = np.where(
        intensity > threshold, threshold + factor * (intensity - threshold), intensity
    )
    return drawn


def _scale(scale: object) -> tuple[float, float]:
    """Return the bounds of the factor, checked.

    Raises
    ------
    ParametricInputError
        If ``scale`` is not a pair ``0 < low <= high`` of finite numbers.
    """
    if not isinstance(scale, tuple | list | np.ndarray) or len(scale) != 2:
        raise ParametricInputError(f"scale must be a pair (low, high), got {scale!r}")
    low = check_real("scale's low", scale[0], low=0.0, include_low=False)
    return low, check_real("scale's high", scale[1], low=low)


def impact_function_draws(
    impf_set: ImpactFuncSet,
    *,
    n_draws: int,
    scale: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[ImpactFuncSet, ...]:
    """Return draws of an impact function set: the curves stretched above their thresholds.

    Each draw takes one factor ``s``, uniform on ``scale``, and applies it to
    every impact function of the set (the draws are uncertainty about the
    functions, common to every event and point of a draw). In each function
    the intensities above its **threshold** ``t``, the last intensity of its
    leading zeros of ``mdd * paa``, are stretched: ``x`` becomes
    ``t + s (x - t)``. So the drawn function at intensity ``v`` above ``t``
    takes the value the original takes at ``t + (v - t) / s``: with
    ``s > 1`` damage grows more slowly with the intensity, with ``s < 1``
    faster. CLIMADA tabulates Emanuel's tropical-cyclone function in steps
    of 5 m/s, so its threshold is 25 m/s, and ``s`` scales ``v_half - 25``:
    ``scale=(0.7, 1.4)`` puts ``v_half`` between about 60 and 95 m/s, around
    the default of 74.7.

    **The limit of these draws.** Every function keeps its threshold, so an
    event has a loss in a draw exactly where it has one without the draws,
    for functions that stay positive above their thresholds. A payout
    depends on the index only, so the draws don't change whether an event
    pays either. Where the index is the hazard at the unit, the
    false-negative and false-positive rates at a loss threshold of 0 are
    therefore the same with and without them: the draws widen the
    uncovered loss and leave unchanged which events lose
    (:mod:`~climada_petals.engine.parametric.basis_risk`). Basis risk in
    which events pay needs a structural source.

    Parameters
    ----------
    impf_set : climada.entity.ImpactFuncSet
        The impact functions, each with intensities that don't fall.
    n_draws : int
        The number of draws, >= 1.
    scale : tuple of (float, float)
        ``(low, high)``, the bounds of the uniform factor, ``0 < low <=
        high``. ``(1, 1)`` reproduces the set.
    rng : numpy.random.Generator
        The source of the factors, ``rng.uniform(low, high, n_draws)`` in
        draw order. Numpy's global random state is not used.

    Returns
    -------
    tuple of climada.entity.ImpactFuncSet
        One set per draw, of copies of the functions; ``impf_set`` is not
        changed.

    Raises
    ------
    ParametricInputError
        If an argument is invalid, the set holds no function, or a
        function's arrays are not finite, of one size, with intensities that
        don't fall and ``mdd * paa >= 0``.

    Examples
    --------
    >>> draws = impact_function_draws(  # doctest: +SKIP
    ...     impfset, n_draws=20, scale=(0.7, 1.4), rng=np.random.default_rng(320)
    ... )
    >>> impacts = [  # doctest: +SKIP
    ...     ImpactCalc(exposures, drawn, hazard).impact(save_mat=True)
    ...     for drawn in draws
    ... ]
    >>> sample = OutcomeSample.from_impact_draws(  # doctest: +SKIP
    ...     index, impacts, units=units, basis=basis
    ... )
    """
    if not isinstance(impf_set, ImpactFuncSet):
        raise ParametricInputError(
            f"impf_set must be an ImpactFuncSet, got {type(impf_set).__name__}"
        )
    functions = [
        impf for by_id in impf_set.get_func().values() for impf in by_id.values()
    ]
    if not functions:
        raise ParametricInputError("impf_set holds no impact function")
    draws = check_int("n_draws", n_draws, low=1)
    low, high = _scale(scale)
    if not isinstance(rng, np.random.Generator):
        raise ParametricInputError(
            "rng must be a numpy.random.Generator, such as "
            f"numpy.random.default_rng(seed); got {type(rng).__name__}"
        )
    for impf in functions:
        _curve(impf)
    return tuple(
        ImpactFuncSet([_stretched(impf, float(factor)) for impf in functions])
        for factor in rng.uniform(low, high, size=draws)
    )
