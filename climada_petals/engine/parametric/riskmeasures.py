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

VaR, CVaR and tail weights of outcomes with weights.

The weights are those of an
:class:`~climada_petals.engine.parametric.measure.OutcomeMeasure`: a
probability per simulated year on the annual basis, or a frequency per year
on the event basis, where they need not sum to 1.

* ``VaR_beta(X)`` is the smallest ``x`` whose exceedance weight
  ``sum_{X_o > x} w_o`` is at most ``1 - beta``. With probabilities that is
  the lower ``beta``-quantile, the smallest ``x`` with ``P(X <= x) >= beta``;
  on the event basis ``1 - beta`` is a frequency per year, and the VaR is
  the value of CLIMADA's event exceedance curve at the return period
  ``1 / (1 - beta)``.
* ``CVaR_beta(X) = VaR_beta(X) + sum_o w_o (X_o - VaR_beta(X))^+ / (1 - beta)``,
  the Rockafellar-Uryasev value: the mean of the worst ``1 - beta`` of the
  weight, taking only as much of the atom at the VaR as it needs. It is the
  quantity the weighted-CVaR objective and the cost-of-capital load minimise
  or bound, so the metrics and the design use one definition. It does not
  depend on which of two adjacent atoms the VaR takes.
* Weights that sum to less than 1 leave **missing mass**, ``1 - sum w``: it
  counts as one more outcome of value 0, as on the event basis, where the
  missing mass is the part of a year with no event. Weights that sum to 1
  or more are used as they are, never divided by their sum: ``beta`` would
  then depend on which loss-free events an event set holds.

On the annual basis every year has the same weight ``1 / N``, so a level
such as ``beta = 0.99`` with 10,000 years lies exactly on a step of the
distribution. The exceedance weight is then compared with ``1 - beta``
within :data:`CDF_ATOL`, which gives the exact-arithmetic VaR, the lower
atom, whatever the summation order.

References
----------
* Rockafellar, R. T. and Uryasev, S. (2002). Conditional value-at-risk for
  general loss distributions. Journal of Banking & Finance, 26(7),
  1443-1471. https://doi.org/10.1016/S0378-4266(02)00271-6
* Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional
  value-at-risk. Journal of Risk, 2(3), 21-41.
  https://doi.org/10.21314/JOR.2000.038
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._checks import check_level, finite_array
from .errors import ParametricInputError
from .measure import PROBABILITY_ATOL

__all__ = ["CDF_ATOL", "tail_weights", "var_cvar", "var_cvar_levels"]

CDF_ATOL: Final[float] = 1e-10
"""An exceedance weight within this of ``1 - beta`` counts as reaching it."""


class _Outcomes(NamedTuple):
    """Outcome values in ascending order, with their weights and cumulative weights."""

    values: NDArray[np.float64]
    weights: NDArray[np.float64]
    cumulative: NDArray[np.float64]
    total: float


def _outcomes(values: ArrayLike, weights: ArrayLike) -> _Outcomes:
    """Return the outcomes in ascending order, with the missing mass added.

    Raises
    ------
    ParametricInputError
        If the arrays are not one weight >= 0 per finite value, or the
        weights sum to 0.
    """
    vals = finite_array(values, "values", 1)
    wts = finite_array(weights, "weights", 1)
    if vals.size == 0 or wts.shape != vals.shape or np.any(wts < 0):
        raise ParametricInputError(
            "values and weights must be non-empty 1-D arrays of one length, "
            "with weights >= 0"
        )
    total = float(wts.sum())
    if total <= 0:
        raise ParametricInputError("the weights must have a positive sum")
    if abs(total - 1.0) <= PROBABILITY_ATOL:
        # A probability vector: divided by its sum, which is 1 up to rounding.
        wts = wts / total
        total = 1.0
    elif total < 1.0:
        vals = np.append(vals, 0.0)
        wts = np.append(wts, 1.0 - total)
        total = 1.0
    order = np.argsort(vals, kind="stable")
    sorted_wts = wts[order]
    return _Outcomes(vals[order], sorted_wts, np.cumsum(sorted_wts), total)


def _var_cvar(outcomes: _Outcomes, level: float) -> tuple[float, float]:
    """Return the VaR and CVaR at ``level`` of sorted outcomes."""
    # The exceedance weight above position k is total - cumulative[k]; the
    # VaR is the first value at which it is at most 1 - level.
    pos = int(
        np.searchsorted(
            outcomes.cumulative, outcomes.total - 1.0 + level - CDF_ATOL, side="left"
        )
    )
    var = float(outcomes.values[min(pos, outcomes.values.size - 1)])
    first = int(np.searchsorted(outcomes.values, var, side="right"))
    tail_weight = float(outcomes.weights[first:].sum())
    if tail_weight <= 0:
        return var, var
    tail = float(outcomes.weights[first:] @ outcomes.values[first:])
    return var, (tail + (1.0 - level - tail_weight) * var) / (1.0 - level)


def var_cvar(values: ArrayLike, weights: ArrayLike, beta: float) -> tuple[float, float]:
    """Return the VaR and CVaR at ``beta`` of weighted outcomes (module docstring).

    Parameters
    ----------
    values : array_like of float
        Outcome values, shape ``(n_outcomes,)``. Finite.
    weights : array_like of float
        Outcome weights, same shape, >= 0 with a positive sum: probabilities
        that sum to 1, or frequencies per year, which may sum to more or
        less than 1.
    beta : float
        Confidence level in ``(0, 1)``.

    Returns
    -------
    var, cvar : float
        ``VaR_beta`` and the Rockafellar-Uryasev ``CVaR_beta``.

    Raises
    ------
    ParametricInputError
        If ``beta`` is not in ``(0, 1)``, or the arrays are not one weight
        >= 0 per finite value with a positive sum.

    Examples
    --------
    >>> var_cvar([0.0, 1.0, 2.0, 10.0], [0.25, 0.25, 0.25, 0.25], 0.5)
    (1.0, 6.0)

    Frequencies per year: the worst 0.125 per year of an event set in which
    the value 10 has frequency 0.0625 and 4 has 0.5.

    >>> var_cvar([4.0, 10.0], [0.5, 0.0625], 0.875)
    (4.0, 7.0)
    """
    return var_cvar_levels(values, weights, [beta])[0]


def var_cvar_levels(
    values: ArrayLike, weights: ArrayLike, betas: Sequence[float]
) -> list[tuple[float, float]]:
    """Return :func:`var_cvar` at several levels, sorting the outcomes once.

    Parameters
    ----------
    values, weights : array_like of float
        As in :func:`var_cvar`.
    betas : sequence of float
        Confidence levels, each in ``(0, 1)``.

    Returns
    -------
    list of (float, float)
        ``(VaR_beta, CVaR_beta)`` per level, in the order of ``betas``.

    Raises
    ------
    ParametricInputError
        As :func:`var_cvar`.
    """
    levels = [check_level("beta", beta) for beta in betas]
    outcomes = _outcomes(values, weights)
    return [_var_cvar(outcomes, level) for level in levels]


def tail_weights(
    values: ArrayLike, weights: ArrayLike, beta: float
) -> NDArray[np.float64]:
    """Return the CVaR weights of each outcome: its share of the worst ``1 - beta``.

    ``w_o`` where ``X_o > VaR_beta(X)``; where ``X_o = VaR_beta(X)``, a share
    of what is left of ``1 - beta``, pro rata to ``w_o``; 0 elsewhere. So
    ``CVaR_beta(X) = sum_o tail_weights_o X_o / (1 - beta)``. The weights sum
    to ``1 - beta``, less the share of missing mass at a VaR of 0 (module
    docstring), which holds the value 0.

    Parameters
    ----------
    values, weights : array_like of float
        As in :func:`var_cvar`.
    beta : float
        Confidence level in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_outcomes,)``, in the order of ``values``.

    Raises
    ------
    ParametricInputError
        As :func:`var_cvar`.

    Examples
    --------
    >>> tail_weights([0.0, 1.0, 2.0, 10.0], [0.25, 0.25, 0.25, 0.25], 0.625).tolist()
    [0.0, 0.0, 0.125, 0.25]
    """
    level = check_level("beta", beta)
    outcomes = _outcomes(values, weights)
    var, _ = _var_cvar(outcomes, level)
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    out = np.where(vals > var, wts, 0.0)
    at_var = vals == var
    atom = float(outcomes.weights[outcomes.values == var].sum())
    left = max(0.0, 1.0 - level - float(out.sum()))
    if atom > 0:
        out[at_var] = left * wts[at_var] / atom
    return out
