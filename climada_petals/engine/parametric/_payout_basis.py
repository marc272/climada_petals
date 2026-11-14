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

The building blocks of the payout families: index values, severity, the
bases' matrices and the amounts.

A family of :mod:`~climada_petals.engine.parametric.payout` works on the
severity of the index, ``s = x`` for ``tail="upper"`` and ``s = -x`` for
``tail="lower"``, with its thresholds mapped the same way (that module's
"Severity"). This module checks index values (:func:`index_values`), maps
one unit's index values and a family's fixed thresholds to severity
(:func:`severity_1d`, :func:`fixed_thresholds`), and builds the sparse
bases of the step, the piecewise-linear and the binary families from the
severities (:func:`step_basis`, :func:`piecewise_basis`,
:func:`binary_basis`): one row per index value, one column per amount.
For a payout that is linear in its amounts, :func:`fixed_amounts` and
:func:`amount_bounds` return the amounts and their bounds under a limit,
the defaults of
:class:`~climada_petals.engine.parametric.payout.LinearInAmounts`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from .errors import ParametricInputError, UnresolvedParameterError
from .parameters import AmountConstraints, Free, Parameter, non_negative

if TYPE_CHECKING:
    from .payout import LinearInAmounts, PayoutFunction

__all__ = [
    "amount_bounds",
    "binary_basis",
    "fixed_amounts",
    "fixed_thresholds",
    "index_values",
    "piecewise_basis",
    "severity_1d",
    "step_basis",
]


def index_values(index: ArrayLike) -> NDArray[np.float64]:
    """Return index values as a finite float array of any shape.

    Parameters
    ----------
    index : array_like of float
        Index values, any shape.

    Returns
    -------
    numpy.ndarray
        The values as float, of the same shape.

    Raises
    ------
    ParametricInputError
        If a value is not numeric or not finite.
    """
    try:
        values = np.asarray(index, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ParametricInputError(f"index values must be numeric: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise ParametricInputError("index values must be finite")
    return values


def severity_1d(payout: PayoutFunction, index: ArrayLike) -> NDArray[np.float64]:
    """Return the severity of one unit's index values, which must be 1-D.

    Parameters
    ----------
    payout : PayoutFunction
        The payout whose ``severity`` maps the values.
    index : array_like of float
        Index values of one unit, shape ``(n_events,)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_events,)``.

    Raises
    ------
    ParametricInputError
        If ``index`` is not 1-D, not numeric or not finite.
    """
    severity = payout.severity(index)
    if severity.ndim != 1:
        raise ParametricInputError(
            f"a basis takes one unit's index values, 1-D; got shape {severity.shape}"
        )
    return severity


def fixed_thresholds(
    payout: PayoutFunction, name: str, values: Sequence[Parameter]
) -> NDArray[np.float64]:
    """Return fixed thresholds in severity; raise if one is free.

    Parameters
    ----------
    payout : PayoutFunction
        The payout, for its ``kind`` and ``tail``.
    name : str
        The thresholds' name, for the error.
    values : sequence of float or Free
        The thresholds, in index units.

    Returns
    -------
    numpy.ndarray
        The thresholds in severity: as they are for ``tail="upper"``,
        negated for ``tail="lower"``.

    Raises
    ------
    UnresolvedParameterError
        If a threshold is free.
    """
    if any(isinstance(value, Free) for value in values):
        raise UnresolvedParameterError(
            f"the {payout.kind} payout's {name} must be fixed for its basis: resolve "
            "them with with_values"
        )
    thresholds = np.array(values, dtype=np.float64)
    return -thresholds if payout.tail == "lower" else thresholds


def step_basis(
    thresholds: NDArray[np.float64], severity: NDArray[np.float64]
) -> sp.csr_matrix:
    """Return the one-hot step of each severity.

    Parameters
    ----------
    thresholds : numpy.ndarray
        The breakpoints in severity, increasing.
    severity : numpy.ndarray
        One unit's severities, shape ``(n_events,)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_events, n_steps)``; a row is empty below the first
        breakpoint.
    """
    step = np.searchsorted(thresholds, severity, side="right") - 1
    rows = np.flatnonzero(step >= 0)
    return sp.csr_matrix(
        (np.ones(rows.size), (rows, step[rows])),
        shape=(severity.size, thresholds.size),
    )


def piecewise_basis(
    knots: NDArray[np.float64], severity: NDArray[np.float64]
) -> sp.csr_matrix:
    """Return the interpolation weights of each severity on the knots.

    Parameters
    ----------
    knots : numpy.ndarray
        The knots in severity, increasing, at least two.
    severity : numpy.ndarray
        One unit's severities, shape ``(n_events,)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_events, n_knots)``, without stored zeros: ``1 - t`` and
        ``t`` on the knots around a severity, 1 on the last knot from it
        on, and nothing below the first.
    """
    last = knots.size - 1
    segment = np.searchsorted(knots, severity, side="right") - 1
    inner = np.flatnonzero((segment >= 0) & (segment < last))
    top = np.flatnonzero(segment == last)
    left = segment[inner]
    share = (severity[inner] - knots[left]) / (knots[left + 1] - knots[left])
    rows = np.concatenate([inner, inner, top])
    cols = np.concatenate([left, left + 1, np.full(top.size, last)])
    weights = np.concatenate([1.0 - share, share, np.ones(top.size)])
    matrix = sp.csr_matrix((weights, (rows, cols)), shape=(severity.size, knots.size))
    matrix.eliminate_zeros()
    return matrix


def binary_basis(trigger: float, severity: NDArray[np.float64]) -> sp.csr_matrix:
    """Return 1 where the severity reaches the trigger, as one column.

    Parameters
    ----------
    trigger : float
        The trigger in severity.
    severity : numpy.ndarray
        One unit's severities, shape ``(n_events,)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_events, 1)``.
    """
    rows = np.flatnonzero(severity >= trigger)
    return sp.csr_matrix(
        (np.ones(rows.size), (rows, np.zeros(rows.size, dtype=np.intp))),
        shape=(severity.size, 1),
    )


def fixed_amounts(payout: LinearInAmounts) -> NDArray[np.float64]:
    """Return a payout's amounts, in its amounts' order; raise if one is free.

    Parameters
    ----------
    payout : LinearInAmounts
        The payout.

    Returns
    -------
    numpy.ndarray
        As
        :meth:`~climada_petals.engine.parametric.payout.LinearInAmounts.amounts`.

    Raises
    ------
    UnresolvedParameterError
        If an amount is free.
    """
    parameters = payout.parameters()
    names = payout.amount_names()
    free = [name for name in names if isinstance(parameters[name], Free)]
    if free:
        raise UnresolvedParameterError(
            f"the {payout.kind} payout has free amounts {free}: resolve them with "
            "with_values, or calibrate the cover"
        )
    return np.array([parameters[name] for name in names], dtype=np.float64)


def amount_bounds(payout: LinearInAmounts, limit: float | None) -> AmountConstraints:
    """Return the bounds of a payout's amounts, under a limit.

    Parameters
    ----------
    payout : LinearInAmounts
        The payout.
    limit : float or None
        The most one event pays at the unit, >= 0, or no limit.

    Returns
    -------
    AmountConstraints
        As
        :meth:`~climada_petals.engine.parametric.payout.LinearInAmounts.amount_constraints`.

    Raises
    ------
    CoverDefinitionError
        If ``limit`` is not a finite number >= 0.
    """
    cap = math.inf if limit is None else non_negative("limit", limit)
    parameters = payout.parameters()
    lower: list[float] = []
    upper: list[float] = []
    for name in payout.amount_names():
        value = parameters[name]
        if isinstance(value, Free):
            lower.append(value.lower)
            upper.append(min(cap, math.inf if value.upper is None else value.upper))
        else:
            lower.append(value)
            upper.append(min(value, cap))
    return AmountConstraints(np.array(lower), np.array(upper))
