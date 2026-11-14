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

The metrics of a cover: their definitions, options and column names.

:func:`~climada_petals.engine.parametric.evaluate.evaluate` computes them for
each **subject** of a table row: an insured unit, the owner's total over the
units, or a unit's share of a joint design's total payout. With the outcome
weights ``w_o`` of the sample's measure
(:mod:`~climada_petals.engine.parametric.measure`) and, per outcome, the loss
``L_o``, the payout ``P_o``, the uncovered loss
``U_o = sum_{e in o} (L_e - P_e)^+``, the residual ``R_o = L_o - P_o`` and
the net loss ``N_o = L_o - P_o + premium``, the owner's cost of the outcome:

====================================  ============================================
Column                                Definition
====================================  ============================================
``expected_loss``,                    ``E[L]``, ``E[P]``, ``E[U]``, per year on both
``expected_payout``,                  bases: on the event basis ``expected_loss`` is
``expected_uncovered``                ``Impact.aai_agg`` on the units
``premium``, ``premium_repriced``     The premium used, and the same pricing on the
                                      evaluation sample (``evaluate``)
``coverage_ratio``                    ``E[min(P, L)] / E[L] = 1 - E[U] / E[L]``
``payout_to_loss_ratio``              ``E[P] / E[L]``
``tail_coverage_b<beta>``             ``sum_o t_o (L_o - U_o) / sum_o t_o L_o``, with
                                      ``t`` the CVaR weights of the loss at ``beta``
                                      (``riskmeasures.tail_weights``): the share of
                                      the loss covered in its worst ``1 - beta``
``cvar_uncovered_to_loss_b<beta>``    ``CVaR_beta(U) / CVaR_beta(L)``
``payout_frequency``                  ``P(P > eps)``: on the event basis, paying
                                      events per year
``unidentified_share``                The weight of the outcomes with an event whose
                                      payout comes from an amount the design set by
                                      its fill rule, because no design outcome
                                      reached it: on the annual basis the
                                      probability of such a year, on the event basis
                                      such events per year
``false_negative_rate_l<l>``          ``P(P <= eps, L > l) / P(L > l)``
``false_positive_rate_l<l>``          ``P(P > eps, L <= l) / P(L <= l)``
``rmse``                              ``sqrt(E[R^2])``
``correlation``                       The weighted Pearson correlation of ``L`` and
                                      ``P``
``var_uncovered_b<beta>``,            ``VaR_beta(U)``, ``CVaR_beta(U)``
``cvar_uncovered_b<beta>``
``var_net_loss_b<beta>``,             ``VaR_beta(N)``, ``CVaR_beta(N)``
``cvar_net_loss_b<beta>``
``var_loss_b<beta>``,                 ``VaR_beta(L)``, ``CVaR_beta(L)``, the same for
``cvar_loss_b<beta>``                 every cover on a sample
``loss_rp<T>``, ``payout_rp<T>``,     The value exceeded with weight ``1 / T``:
``uncovered_rp<T>``,                  ``VaR_{1 - 1/T}``. ``T`` is in years: a
``net_loss_rp<T>``                    probability of a year on the annual basis, a
                                      frequency per year on the event basis
====================================  ============================================

VaR and CVaR are those of
:mod:`~climada_petals.engine.parametric.riskmeasures`: on the event basis
``1 - beta`` is a frequency per year, and frequencies that sum to less than
1 leave missing mass, which counts as an outcome without loss or payout.
``beta`` is a probability of a year on the annual basis.

**Null** (NaN) where a value is undefined: a ratio whose denominator is 0,
a correlation where ``L`` or ``P`` doesn't vary, ``unidentified_share`` for a
cover given as it is, and the net-loss columns on the event basis, where
the premium, which is annual, has no place in an event, and on rows with no
premium.

**On the event basis**, ``false_positive_rate_l<l>``'s denominator holds the
sample's events with a loss of at most ``l``, not the missing mass: it
depends on which loss-free events the sample holds. ``correlation`` counts
the missing mass, at ``L = P = 0``.

**Setup columns.** Every table row of
:func:`~climada_petals.engine.parametric.evaluate.evaluate` states its setup
in the columns :data:`SETUP_COLUMNS` (the evaluation's module docstring
says what each holds), and :func:`setup_text` renders them, so that a
number can be shown with its setup.

**Labels.** ``beta``, ``T`` and ``l`` appear in the column names as their
decimal text with the point as an underscore: ``beta = 0.99`` gives
``cvar_uncovered_b0_99``, ``T = 100`` gives ``loss_rp100`` and ``l = 0`` gives
``false_negative_rate_l0``. They and the payout threshold ``eps`` are always
stated, in :class:`MetricOptions`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._checks import check_level, check_real
from .cover import IndexDefinition
from .errors import ParametricInputError
from .measure import EVENT, OutcomeMeasure
from .pricing import Pricing
from .riskmeasures import tail_weights, var_cvar_levels

__all__ = [
    "ALLOCATIONS",
    "LEVELS",
    "PREMIUM_COLUMNS",
    "SETUP_COLUMNS",
    "SPLITS",
    "MetricOptions",
    "index_text",
    "label",
    "pricing_text",
    "setup_text",
    "subject_metrics",
]

ALLOCATIONS: Final[tuple[str, ...]] = ("value", "loss")
"""The rules of :attr:`MetricOptions.allocation`."""

PREMIUM_COLUMNS: Final[tuple[str, ...]] = ("premium", "premium_repriced")
"""The columns that :func:`~climada_petals.engine.parametric.evaluate.evaluate` fills."""


def label(value: float) -> str:
    """Return a threshold's text in a column name: ``0.99`` gives ``"0_99"``.

    Parameters
    ----------
    value : float
        A level, a return period or a loss threshold, >= 0.

    Returns
    -------
    str
        The shortest decimal text that reads back as ``value``, without an
        exponent, with the point as an underscore; a whole number has none.

    Examples
    --------
    >>> label(0.99), label(100.0), label(2.5), label(0.0)
    ('0_99', '100', '2_5', '0')
    """
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return np.format_float_positional(number, trim="-").replace(".", "_")


def _thresholds(name: str, values: object, check: str) -> tuple[float, ...]:
    """Return a non-empty tuple of checked thresholds with distinct labels.

    Raises
    ------
    ParametricInputError
        If ``values`` is not a non-empty sequence of valid values, or two
        give one label.
    """
    if isinstance(values, str) or not isinstance(values, Sequence | np.ndarray):
        raise ParametricInputError(f"{name} must be a sequence, got {values!r}")
    out: list[float] = []
    for value in values:
        if check == "level":
            out.append(check_level(name, value))
        elif check == "period":
            out.append(check_real(name, value, low=1.0, include_low=False))
        else:
            out.append(check_real(name, value, low=0.0))
    labels = [label(value) for value in out]
    if not out or len(set(labels)) != len(labels):
        raise ParametricInputError(
            f"{name} must hold at least one value, with distinct labels; got "
            f"{list(values)}"
        )
    return tuple(out)


@dataclass(frozen=True)
class MetricOptions:
    """The thresholds of the metrics, always stated (module docstring).

    Parameters
    ----------
    betas : sequence of float, default (0.9, 0.95, 0.99)
        The levels of the VaR, CVaR and tail columns, each in ``(0, 1)``.
    return_periods : sequence of float, default (2, 5, 10, 25, 50, 100, 250)
        The return periods ``T`` of the ``_rp<T>`` columns, in years, each
        above 1.
    loss_thresholds : sequence of float, default (0.0,)
        The loss thresholds ``l`` of the false-negative and false-positive
        rates, in loss units, each >= 0.
    payout_threshold : float, default 0.0
        ``eps``, the payout at or below which an outcome counts as paying
        nothing, in payout units, >= 0.
    allocation : {"value", "loss"}, optional
        For joint designs only: per-unit rows that allocate the design's
        total payout to the units, by the units' values or by their share
        of each event's loss
        (:func:`~climada_petals.engine.parametric.evaluate.evaluate`).
        Default None: a joint design has a total row only.

    Raises
    ------
    ParametricInputError
        If a value is out of range, a sequence is empty, two values give one
        column label, or ``allocation`` is unknown.
    """

    betas: Sequence[float] = (0.9, 0.95, 0.99)
    return_periods: Sequence[float] = (2, 5, 10, 25, 50, 100, 250)
    loss_thresholds: Sequence[float] = (0.0,)
    payout_threshold: float = 0.0
    allocation: str | None = None

    def __post_init__(self) -> None:
        """Validate the thresholds and store them as tuples of floats."""
        object.__setattr__(self, "betas", _thresholds("betas", self.betas, "level"))
        object.__setattr__(
            self,
            "return_periods",
            _thresholds("return_periods", self.return_periods, "period"),
        )
        object.__setattr__(
            self,
            "loss_thresholds",
            _thresholds("loss_thresholds", self.loss_thresholds, "amount"),
        )
        object.__setattr__(
            self,
            "payout_threshold",
            check_real("payout_threshold", self.payout_threshold, low=0.0),
        )
        if self.allocation is not None and self.allocation not in ALLOCATIONS:
            raise ParametricInputError(
                f"allocation must be None or one of {list(ALLOCATIONS)}, got "
                f"{self.allocation!r}"
            )

    def columns(self) -> tuple[str, ...]:
        """Return the metric columns, in table order (module docstring).

        Returns
        -------
        tuple of str
            The premium columns among them.
        """
        betas = [label(beta) for beta in self.betas]
        periods = [label(period) for period in self.return_periods]
        losses = [label(level) for level in self.loss_thresholds]
        return (
            "expected_loss",
            "expected_payout",
            "expected_uncovered",
            *PREMIUM_COLUMNS,
            "coverage_ratio",
            "payout_to_loss_ratio",
            *(f"tail_coverage_b{b}" for b in betas),
            *(f"cvar_uncovered_to_loss_b{b}" for b in betas),
            "payout_frequency",
            "unidentified_share",
            *(f"false_negative_rate_l{x}" for x in losses),
            *(f"false_positive_rate_l{x}" for x in losses),
            "rmse",
            "correlation",
            *(
                f"{kind}_{quantity}_b{b}"
                for quantity in ("uncovered", "net_loss", "loss")
                for b in betas
                for kind in ("var", "cvar")
            ),
            *(
                f"{quantity}_rp{t}"
                for quantity in ("loss", "payout", "uncovered", "net_loss")
                for t in periods
            ),
        )


def _ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or NaN if the denominator is 0."""
    return numerator / denominator if denominator > 0 else math.nan


def _correlation(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Weighted Pearson correlation, with missing mass at ``(0, 0)``; NaN if flat."""
    missing = 1.0 - float(weights.sum())
    if missing > 0:
        first = np.append(first, 0.0)
        second = np.append(second, 0.0)
        weights = np.append(weights, missing)
    prob = weights / float(weights.sum())
    mean_x, mean_y = float(prob @ first), float(prob @ second)
    cov = float(prob @ ((first - mean_x) * (second - mean_y)))
    var_x = float(prob @ (first - mean_x) ** 2)
    var_y = float(prob @ (second - mean_y) ** 2)
    if var_x <= 0.0 or var_y <= 0.0:
        return math.nan
    return cov / math.sqrt(var_x * var_y)


def _rows(values: ArrayLike, measure: OutcomeMeasure, name: str) -> NDArray[np.float64]:
    """Return per-row values of the measure, checked."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (measure.n_events,) or not np.all(np.isfinite(arr)):
        raise ParametricInputError(
            f"{name} must hold one finite value per row of the measure "
            f"({measure.n_events}), got shape {arr.shape}"
        )
    return arr


def _tail_columns(
    quantity: str,
    values: NDArray[np.float64] | None,
    weights: NDArray[np.float64],
    options: MetricOptions,
) -> dict[str, float]:
    """Return the VaR, CVaR and return-period columns of one quantity.

    ``values`` None gives null columns (the net loss without a premium).
    """
    levels = [*options.betas, *(1.0 - 1.0 / t for t in options.return_periods)]
    found = (
        [(math.nan, math.nan)] * len(levels)
        if values is None
        else var_cvar_levels(values, weights, levels)
    )
    out: dict[str, float] = {}
    for beta, (var, cvar) in zip(options.betas, found):
        out[f"var_{quantity}_b{label(beta)}"] = var
        out[f"cvar_{quantity}_b{label(beta)}"] = cvar
    for period, (var, _) in zip(options.return_periods, found[len(options.betas) :]):
        out[f"{quantity}_rp{label(period)}"] = var
    return out


def _rates(
    loss: NDArray[np.float64],
    paid: NDArray[np.float64],
    weights: NDArray[np.float64],
    options: MetricOptions,
) -> dict[str, float]:
    """Return the payout frequency and the false-negative and -positive rates."""
    paying = paid > options.payout_threshold
    out = {"payout_frequency": float(weights @ paying)}
    for level in options.loss_thresholds:
        above = loss > level
        out[f"false_negative_rate_l{label(level)}"] = _ratio(
            float(weights @ (above & ~paying)), float(weights @ above)
        )
        out[f"false_positive_rate_l{label(level)}"] = _ratio(
            float(weights @ (~above & paying)), float(weights @ ~above)
        )
    return out


class _Subject(NamedTuple):
    """A subject's outcome values and the outcome weights."""

    loss: NDArray[np.float64]
    payout: NDArray[np.float64]
    uncovered: NDArray[np.float64]
    weights: NDArray[np.float64]


def _subject(loss: ArrayLike, payout: ArrayLike, measure: OutcomeMeasure) -> _Subject:
    """Return a subject's outcomes: rows summed into outcomes, checked."""
    loss_rows = _rows(loss, measure, "loss")
    paid_rows = _rows(payout, measure, "payout")
    return _Subject(
        measure.aggregate(loss_rows),
        measure.aggregate(paid_rows),
        measure.aggregate(np.maximum(loss_rows - paid_rows, 0.0)),
        np.asarray(measure.weights, dtype=np.float64),
    )


def _tail_ratios(
    subject: _Subject, tails: dict[str, float], options: MetricOptions
) -> dict[str, float]:
    """Return the tail coverage and the CVaR of ``U`` against that of ``L``."""
    out: dict[str, float] = {}
    for beta in options.betas:
        tail = tail_weights(subject.loss, subject.weights, beta)
        out[f"tail_coverage_b{label(beta)}"] = _ratio(
            float(tail @ (subject.loss - subject.uncovered)), float(tail @ subject.loss)
        )
    for beta in options.betas:
        out[f"cvar_uncovered_to_loss_b{label(beta)}"] = _ratio(
            tails[f"cvar_uncovered_b{label(beta)}"], tails[f"cvar_loss_b{label(beta)}"]
        )
    return out


def subject_metrics(
    loss: ArrayLike,
    payout: ArrayLike,
    measure: OutcomeMeasure,
    *,
    premium: float | None = None,
    filled: ArrayLike | None = None,
    options: MetricOptions = MetricOptions(),  # frozen, so safe to share
) -> dict[str, float]:
    """Return the metrics of one subject (module docstring), but its premiums.

    Parameters
    ----------
    loss, payout : array_like of float
        The subject's loss and payout per row of the measure (an event, or
        an event and a draw), shape ``(n_rows,)``.
    measure : OutcomeMeasure
        The outcomes of the rows and their weights.
    premium : float, optional
        The premium used, for the net loss. None: the net-loss columns are
        null, as they are on the event basis.
    filled : array_like of bool, optional
        Per row, whether its payout comes from an amount set by a design's
        fill rule. None: ``unidentified_share`` is null.
    options : MetricOptions, default MetricOptions()
        The thresholds.

    Returns
    -------
    dict of str to float
        One value per column of ``options.columns()`` but
        :data:`PREMIUM_COLUMNS`, NaN where null.

    Raises
    ------
    ParametricInputError
        If an array doesn't have one finite value per row of the measure.
    """
    subject = _subject(loss, payout, measure)
    weights = subject.weights
    out: dict[str, float] = {
        "expected_loss": float(weights @ subject.loss),
        "expected_payout": float(weights @ subject.payout),
        "expected_uncovered": float(weights @ subject.uncovered),
    }
    out["coverage_ratio"] = 1.0 - _ratio(
        out["expected_uncovered"], out["expected_loss"]
    )
    out["payout_to_loss_ratio"] = _ratio(out["expected_payout"], out["expected_loss"])
    net = (
        None
        if premium is None or measure.basis == EVENT
        else subject.loss - subject.payout + float(premium)
    )
    tails = {
        **_tail_columns("uncovered", subject.uncovered, weights, options),
        **_tail_columns("net_loss", net, weights, options),
        **_tail_columns("loss", subject.loss, weights, options),
        **_tail_columns("payout", subject.payout, weights, options),
    }
    out.update(_tail_ratios(subject, tails, options))
    out.update(_rates(subject.loss, subject.payout, weights, options))
    out["unidentified_share"] = (
        math.nan
        if filled is None
        else float(weights @ (measure.aggregate(_rows(filled, measure, "filled")) > 0))
    )
    out["rmse"] = math.sqrt(float(weights @ (subject.loss - subject.payout) ** 2))
    out["correlation"] = _correlation(subject.loss, subject.payout, weights)
    out.update(tails)
    return {
        name: out[name] for name in options.columns() if name not in PREMIUM_COLUMNS
    }


SPLITS: Final[tuple[str, ...]] = ("in_sample", "out_of_sample")
"""The values of the ``split`` column."""

LEVELS: Final[tuple[str, ...]] = ("unit", "total", "unit_by_value", "unit_by_loss")
"""The values of the ``level`` column."""

SETUP_COLUMNS: Final[tuple[str, ...]] = (
    "cover_label",
    "cover_hash",
    "scope",
    "level",
    "unit_id",
    "scenario",
    "basis",
    "route",
    "n_years",
    "index_definition",
    "pricing",
    "premium_basis",
    "payout_threshold",
    "split",
    "fold",
    "unique",
    "ties",
    "degenerate_tail",
    "comonotone_index",
)
"""The setup columns of a table, before the metrics.

:mod:`~climada_petals.engine.parametric.evaluate` says what each holds.
"""


def index_text(definition: IndexDefinition) -> str:
    """Return an index definition in one line: kind, source, statistic and unit.

    Parameters
    ----------
    definition : IndexDefinition
        The definition.

    Returns
    -------
    str
        E.g. ``"hazard (TC): mean, value-weighted, in m/s; haz_type=TC"``.

    Examples
    --------
    >>> from climada_petals.engine.parametric import IndexDefinition
    >>> index_text(IndexDefinition("table", unit="m/s", source="stations"))
    'table (stations): in m/s'
    """
    head = definition.kind
    if definition.source is not None:
        head += f" ({definition.source})"
    parts = []
    if isinstance(definition.index_stat, float):
        parts.append(f"percentile {definition.index_stat:g}")
    elif definition.index_stat is not None:
        parts.append(definition.index_stat)
    if definition.weight is not None:
        parts.append(f"{definition.weight}-weighted")
    parts.append(f"in {definition.unit}")
    if definition.tail == "lower":
        parts.append("lower tail")
    text = f"{head}: {', '.join(parts)}"
    if definition.parameters:
        settings = ", ".join(
            f"{key}={definition.parameters[key]}"
            for key in sorted(definition.parameters)
        )
        text += f"; {settings}"
    return text


def pricing_text(pricing: Pricing) -> str:
    """Return a pricing's name and parameters, e.g. ``"cost_of_capital(q=0.99, rate=0.06)"``."""
    document = pricing.to_dict()
    kind = document.pop("kind")
    if not document:
        return str(kind)
    return f"{kind}({', '.join(f'{k}={document[k]}' for k in sorted(document))})"


def _distinct(table: pd.DataFrame, column: str) -> list[Any]:
    """Return the distinct values of a column, in order of appearance; nulls left out."""
    return [value for value in pd.unique(table[column]) if not pd.isna(value)]


def _basis_text(table: pd.DataFrame) -> str:
    """Return the basis, route and years of a table's rows."""
    parts = []
    for (basis, route), rows in table.groupby(["basis", "route"], sort=False):
        if basis == "event":
            parts.append("event basis (frequencies per year as weights)")
            continue
        years = ", ".join(str(n) for n in _distinct(rows, "n_years"))
        what = "catalogue years" if route == "catalogue" else "sampled years"
        parts.append(f"annual basis on {years} {what} (route {route})")
    return "; ".join(parts)


_PREMIUM_TEXT: Final[dict[str, str]] = {
    "stated": "stated",
    "design": "the design premium, priced on the design sample",
    "evaluation": "priced on this sample",
}

_LEVEL_TEXT: Final[dict[str, str]] = {
    "unit": "each unit's own cover",
    "total": "the owner's total over the units",
    "unit_by_value": "a joint design's total payout allocated by value (no premium)",
    "unit_by_loss": "a joint design's total payout allocated by loss (no premium)",
}


def setup_text(table: pd.DataFrame | pd.Series) -> str:
    """Return the setup of a table's rows as text, to print next to its numbers.

    Every aspect lists the values its rows take: the split, the basis and
    route, the scenario, the index, the pricing and premium, the scope, the
    levels and units, and the payout threshold. Scenario labels are separated
    by semicolons, since a label such as ``"rcp45, 2040 to 2080"`` may hold a
    comma.

    Parameters
    ----------
    table : pandas.DataFrame or pandas.Series
        Rows of :func:`evaluate` or :func:`compare`, or one row.

    Returns
    -------
    str
        One line per aspect.

    Raises
    ------
    ParametricInputError
        If the table lacks a setup column or holds no row.
    """
    frame = table.to_frame().T if isinstance(table, pd.Series) else table
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ParametricInputError("setup_text takes rows of evaluate or compare")
    missing = [name for name in SETUP_COLUMNS if name not in frame.columns]
    if missing:
        raise ParametricInputError(f"the table lacks the setup columns {missing}")
    splits = [
        f"{split.replace('_', ' ')}, fold {fold}"
        for split, fold in frame[["split", "fold"]]
        .drop_duplicates()
        .itertuples(index=False)
    ]
    scopes = [
        "a cover given as it is" if pd.isna(scope) else f"{scope} calibration"
        for scope in pd.unique(frame["scope"])
    ]
    premiums = [_PREMIUM_TEXT[basis] for basis in _distinct(frame, "premium_basis")]
    units = [str(unit) for unit in _distinct(frame, "unit_id")]
    lines = [
        f"Split: {'; '.join(splits)}",
        f"Basis: {_basis_text(frame)}",
        f"Scenario: {'; '.join(str(s) for s in _distinct(frame, 'scenario'))}",
        f"Index: {'; '.join(_distinct(frame, 'index_definition'))}",
        f"Pricing: {', '.join(_distinct(frame, 'pricing'))}; premium: "
        f"{', '.join(premiums) or 'none'}",
        f"Scope: {', '.join(scopes)}",
        f"Rows: {'; '.join(_LEVEL_TEXT[level] for level in _distinct(frame, 'level'))}"
        + (f"; units {', '.join(units)}" if units else ""),
        "Payout threshold: "
        + ", ".join(f"{eps:g}" for eps in _distinct(frame, "payout_threshold")),
    ]
    return "\n".join(lines)
