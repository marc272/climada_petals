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

VaR and CVaR of a discrete distribution.

``VaR_beta`` is the lower ``beta``-quantile: the smallest value ``v`` with
``P(X <= v) >= beta``. ``CVaR_beta`` is the Rockafellar-Uryasev value
``VaR + E[(X - VaR)^+] / (1 - beta)``, the mean of the worst ``1 - beta``
of the probability mass, taking only as much of the atom at the VaR as it
needs. It is the quantity the weighted-CVaR objective and the
cost-of-capital load minimise or bound, so the metrics and the design use
one definition.

On the annual basis every year has the same probability ``1 / |S|``, so a
level such as ``beta = 0.99`` with 10,000 years lies exactly on a step of
the distribution. The cumulative probability is then compared with
``beta`` within :data:`CDF_ATOL`, which gives the exact-arithmetic lower
quantile whatever the summation order. The CVaR does not depend on which
of two adjacent atoms is taken.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from ._checks import check_level, finite_array
from .errors import ParametricInputError
from .measure import PROBABILITY_ATOL

__all__ = ["CDF_ATOL", "var_cvar"]

CDF_ATOL: Final[float] = 1e-10
"""A cumulative probability within this of ``beta`` counts as reaching it."""


def var_cvar(
    values: ArrayLike, probability: ArrayLike, beta: float
) -> tuple[float, float]:
    """Return the VaR and CVaR at ``beta`` of a discrete distribution.

    Parameters
    ----------
    values : array_like of float
        Outcome values, shape ``(n_outcomes,)``. Finite.
    probability : array_like of float
        Outcome probabilities, same shape, >= 0 and summing to one within
        :data:`~climada_petals.engine.parametric.measure.PROBABILITY_ATOL`.
    beta : float
        Confidence level in ``(0, 1)``.

    Returns
    -------
    var, cvar : float
        The lower ``beta``-quantile and the Rockafellar-Uryasev CVaR.

    Raises
    ------
    ParametricInputError
        If ``beta`` is not in ``(0, 1)`` or the arrays are not a discrete
        distribution.

    Examples
    --------
    >>> var_cvar([0.0, 1.0, 2.0, 10.0], [0.25, 0.25, 0.25, 0.25], 0.5)
    (1.0, 6.0)
    """
    level = check_level("beta", beta)
    vals = finite_array(values, "values", 1)
    prob = finite_array(probability, "probability", 1)
    if vals.size == 0 or prob.shape != vals.shape or np.any(prob < 0):
        raise ParametricInputError(
            "values and probability must be non-empty 1-D arrays of one length, "
            "with probability >= 0"
        )
    total = float(prob.sum())
    if abs(total - 1.0) > PROBABILITY_ATOL:
        raise ParametricInputError(f"probability must sum to 1, got {total!r}")
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    sorted_prob = prob[order] / total
    cdf = np.cumsum(sorted_prob)
    pos = int(np.searchsorted(cdf, level - CDF_ATOL, side="left"))
    var = float(sorted_vals[min(pos, sorted_vals.size - 1)])
    tail = sorted_vals > var
    tail_prob = float(sorted_prob[tail].sum())
    if tail_prob <= 0:
        return var, var
    return var, (
        float(sorted_prob[tail] @ sorted_vals[tail]) + (1.0 - level - tail_prob) * var
    ) / (1.0 - level)
