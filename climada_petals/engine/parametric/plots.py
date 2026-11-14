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

Plots of covers and of their evaluation.

* :func:`plot_payout`: the payout function of a cover's units, against the
  index.
* :func:`plot_payout_vs_loss`: each event's payout against its loss, on a
  sample, with the line where they are equal.
* :func:`plot_exceedance`: exceedance curves, the value against its return
  period, from CLIMADA ``Impact`` objects (such as
  :func:`~climada_petals.engine.parametric.evaluate.payout_impact` and
  :func:`~climada_petals.engine.parametric.evaluate.uncovered_impact`) or
  from the return-period columns of an evaluation table.

Each draws on the axes it is given, or on new ones, and returns them. None
calls ``plt.show()``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from climada.engine import Impact
from numpy.typing import NDArray

from .calibrate import CalibrationResult
from .cover import ParametricCover
from .errors import ParametricInputError
from .evaluate import payout_impact
from .parameters import Free
from .payout import LinearInAmounts
from .sample import OutcomeSample

__all__ = ["plot_exceedance", "plot_payout", "plot_payout_vs_loss"]

_GRID_POINTS: Final[int] = 1001
"""Points of the index grid of :func:`plot_payout`, besides the thresholds."""

_RP_COLUMN: Final[re.Pattern[str]] = re.compile(
    r"(?P<quantity>.+)_rp(?P<period>[0-9_]+)"
)


def _new_axes(axis: Any) -> Any:
    """Return ``axis``, or the axes of a new figure."""
    if axis is None:
        _, axis = plt.subplots()
    return axis


def _thresholds(cover: ParametricCover, units: list[Hashable]) -> NDArray[np.float64]:
    """Return the fixed shape parameters (thresholds) of the units' payouts."""
    found: list[float] = []
    for unit in units:
        payout = cover.payout[unit]
        amounts = (
            set(payout.amount_names()) if isinstance(payout, LinearInAmounts) else set()
        )
        found.extend(
            float(value)
            for name, value in payout.parameters().items()
            if name not in amounts and not isinstance(value, Free)
        )
    return np.unique(np.asarray(found, dtype=np.float64))


def _index_grid(
    cover: ParametricCover, units: list[Hashable], index_range: Any
) -> NDArray[np.float64]:
    """Return the index values to draw: a fine grid, each threshold and its left edge.

    Raises
    ------
    ParametricInputError
        If ``index_range`` is not two increasing numbers, or it is None and
        the payouts state no threshold.
    """
    thresholds = _thresholds(cover, units)
    if index_range is None:
        if thresholds.size == 0:
            raise ParametricInputError(
                "the payouts state no threshold to set the index range from: pass "
                "index_range=(low, high)"
            )
        span = float(np.ptp(thresholds)) or max(abs(float(thresholds[0])), 1.0)
        index_range = (thresholds[0] - 0.25 * span, thresholds[-1] + 0.25 * span)
    try:
        low, high = (float(value) for value in index_range)
    except (TypeError, ValueError) as exc:
        raise ParametricInputError(
            f"index_range must be two numbers, got {index_range!r}"
        ) from exc
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ParametricInputError(
            f"index_range must be two finite numbers, low < high, got {index_range!r}"
        )
    inside = thresholds[(thresholds >= low) & (thresholds <= high)]
    return np.unique(
        np.concatenate(
            [
                np.linspace(low, high, _GRID_POINTS),
                inside,
                np.nextafter(inside, -np.inf),
                np.nextafter(inside, np.inf),
            ]
        )
    )


def plot_payout(
    cover: ParametricCover,
    unit_id: Hashable | None = None,
    axis: Any = None,
    *,
    index_range: tuple[float, float] | None = None,
) -> Any:
    """Plot the payout of a cover's units against the index.

    Parameters
    ----------
    cover : ParametricCover
        A resolved cover.
    unit_id : hashable, optional
        The unit to draw. Default: every unit, one line each.
    axis : matplotlib.axes.Axes, optional
        The axes to draw on. Default: a new figure's.
    index_range : (float, float), optional
        The index values to draw, ``(low, high)``. Default: the payouts'
        thresholds, widened by a quarter of their span on each side.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ParametricInputError
        If ``cover`` is not a cover, ``unit_id`` is not one of its units, or
        ``index_range`` is invalid or needed.
    UnresolvedParameterError
        If the cover is a template.
    """
    if not isinstance(cover, ParametricCover):
        raise ParametricInputError(f"cover must be a ParametricCover, got {cover!r}")
    cover.check_resolved()
    if unit_id is not None and unit_id not in cover.payout:
        raise ParametricInputError(f"the cover has no unit {unit_id!r}")
    units = list(cover.unit_id) if unit_id is None else [unit_id]
    grid = _index_grid(cover, units, index_range)
    axis = _new_axes(axis)
    for unit in units:
        axis.plot(grid, cover.payout[unit].evaluate(grid), label=str(unit))
    axis.set_xlabel(f"Index ({cover.index.unit})")
    axis.set_ylabel(f"Payout ({cover.payout_unit})")
    if len(units) > 1:
        axis.legend(title="Unit")
    return axis


def plot_payout_vs_loss(
    entry: ParametricCover | CalibrationResult,
    sample: OutcomeSample,
    *,
    unit_id: Hashable | None = None,
    axis: Any = None,
) -> Any:
    """Plot each event's payout against its loss, on a sample.

    A point on the line ``payout = loss`` is an event the cover pays
    exactly; below it the payout falls short, above it the cover pays more
    than the loss.

    Parameters
    ----------
    entry : ParametricCover or CalibrationResult
        A resolved cover given as it is, or a design, with the sample's
        units and index definition.
    sample : OutcomeSample
        The sample. A sample with impact-function draws has one point per
        draw and event.
    unit_id : hashable, optional
        The unit to draw. Default: the owner's total over the units.
    axis : matplotlib.axes.Axes, optional
        The axes to draw on. Default: a new figure's.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ParametricInputError
        If ``unit_id`` is not one of the sample's units, or as
        :func:`~climada_petals.engine.parametric.evaluate.payout_impact`.
    """
    paid_impact = payout_impact(entry, sample)
    paid = sample.units.aggregate(paid_impact.imp_mat).toarray()[sample.event_row]
    loss = sample.loss.toarray()
    if unit_id is None:
        x_values, y_values, what = loss.sum(axis=1), paid.sum(axis=1), "total"
    elif unit_id in sample.unit_id:
        column = sample.unit_id.index(unit_id)
        x_values, y_values, what = loss[:, column], paid[:, column], f"unit {unit_id}"
    else:
        raise ParametricInputError(f"the sample has no unit {unit_id!r}")
    axis = _new_axes(axis)
    axis.scatter(x_values, y_values, s=12, label=f"events ({what})")
    top = float(max(x_values.max(initial=0.0), y_values.max(initial=0.0))) or 1.0
    axis.plot([0.0, top], [0.0, top], color="grey", linewidth=1, label="payout = loss")
    axis.set_xlabel(f"Loss ({paid_impact.unit})")
    axis.set_ylabel(f"Payout ({paid_impact.unit})")
    axis.legend()
    return axis


def _table_curves(
    table: pd.DataFrame, quantities: tuple[str, ...]
) -> list[tuple[str, NDArray[np.float64], NDArray[np.float64]]]:
    """Return ``(label, periods, values)`` of each row and quantity of a table.

    Raises
    ------
    ParametricInputError
        If the table has no return-period column of the quantities.
    """
    columns: dict[str, list[tuple[float, str]]] = {}
    for name in table.columns:
        match = _RP_COLUMN.fullmatch(str(name))
        if match and match["quantity"] in quantities:
            period = float(match["period"].replace("_", "."))
            columns.setdefault(match["quantity"], []).append((period, str(name)))
    if not columns:
        raise ParametricInputError(
            f"the table has no return-period column of {list(quantities)}, such as "
            "loss_rp100"
        )
    curves = []
    for _, row in table.iterrows():
        head = f"{row.get('cover_label', '')} {row.get('level', '')}"
        if not pd.isna(row.get("unit_id")):
            head += f" {row['unit_id']}"
        for quantity in quantities:
            pairs = sorted(columns.get(quantity, []))
            values = np.array([row[name] for _, name in pairs], dtype=np.float64)
            if values.size and not np.all(np.isnan(values)):
                periods = np.array([period for period, _ in pairs])
                curves.append((f"{head.strip()} {quantity}", periods, values))
    return curves


def _impact_curve(
    label: str, impact: Any
) -> tuple[str, NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(label, return periods, values)`` of an ``Impact``'s frequency curve."""
    curve = impact.calc_freq_curve()
    return label, np.asarray(curve.return_per), np.asarray(curve.impact)


def plot_exceedance(
    source: Impact | Mapping[str, Impact] | pd.DataFrame,
    *,
    quantities: tuple[str, ...] = ("loss", "payout", "uncovered"),
    axis: Any = None,
) -> Any:
    """Plot exceedance curves: each value against its return period.

    Parameters
    ----------
    source : climada.engine.Impact, mapping of str to Impact, or pandas.DataFrame
        An ``Impact``, such as a loss, payout or uncovered ``Impact``, whose
        ``calc_freq_curve`` is drawn; several, by label; or rows of
        :func:`~climada_petals.engine.parametric.evaluate.evaluate` or
        :func:`~climada_petals.engine.parametric.evaluate.compare`, whose
        ``<quantity>_rp<T>`` columns are drawn, one curve per row and
        quantity. A table's return periods are those of its
        ``MetricOptions``, read on its basis and route.
    quantities : tuple of str, default ("loss", "payout", "uncovered")
        For a table: the quantities to draw, among ``"loss"``,
        ``"payout"``, ``"uncovered"`` and ``"net_loss"``. Null columns,
        such as the net loss on the event basis, are left out.
    axis : matplotlib.axes.Axes, optional
        The axes to draw on. Default: a new figure's.

    Returns
    -------
    matplotlib.axes.Axes
        The return period on a logarithmic x axis.

    Raises
    ------
    ParametricInputError
        If ``source`` is none of these, or a table has no return-period
        column of the quantities.
    """
    if isinstance(source, pd.DataFrame):
        curves = _table_curves(source, tuple(quantities))
    elif isinstance(source, Impact):
        curves = [_impact_curve(source.unit or "impact", source)]
    elif isinstance(source, Mapping) and all(
        isinstance(item, Impact) for item in source.values()
    ):
        curves = [_impact_curve(str(name), item) for name, item in source.items()]
    else:
        raise ParametricInputError(
            "source must be an Impact, a mapping of labels to Impacts, or a table of "
            f"evaluate, got {type(source).__name__}"
        )
    axis = _new_axes(axis)
    for name, periods, values in curves:
        axis.plot(periods, values, marker="o", markersize=3, label=name)
    axis.set_xscale("log")
    axis.set_xlabel("Return period (years)")
    axis.set_ylabel("Value")
    axis.legend()
    return axis
