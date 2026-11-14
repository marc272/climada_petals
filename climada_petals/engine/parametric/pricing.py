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

Pricing: the premium of a payout.

Both pricings price the payout of one design problem (summed over its
units), aggregated into outcomes: ``P_o = sum_{e in o} P_e``.

* :class:`PurePremium`: ``E[P]``.
* :class:`CostOfCapitalLoad`: ``(1 - r) E[P] + r CVaR_q(P)``. The load
  ``r (CVaR_q(P) - E[P])`` prices the capital a writer holds against the
  payout tail, so ``E[P] <= premium <= CVaR_q(P)``. Convex in the payout for
  ``r`` in ``[0, 1]``, so the premium budget
  ``premium <= budget * E[L]`` keeps the design problem an LP (a QP with the
  mean squared error). ``r = 0`` is :class:`PurePremium`.

A pricing is given once, to :func:`~climada_petals.engine.parametric.calibrate.calibrate`:
the budget constrains it and the cover records the premium of its payout
under it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from ._checks import check_level, check_real
from .measure import OutcomeMeasure
from .riskmeasures import var_cvar

__all__ = ["CostOfCapitalLoad", "Pricing", "PurePremium"]


@dataclass(frozen=True)
class PurePremium:
    """Expected payout: ``premium = E[P] = sum_o p_o P_o``."""

    name: ClassVar[str] = "pure"

    def premium(self, payout: ArrayLike, measure: OutcomeMeasure) -> float:
        """Return the premium of a realised payout.

        Parameters
        ----------
        payout : array_like of float
            Per-event payout of the design problem, shape ``(n_events,)``.
        measure : OutcomeMeasure
            Measure over the same events.

        Returns
        -------
        float
            ``E[P]``, in the payout's unit.
        """
        return measure.expectation(payout)


@dataclass(frozen=True)
class CostOfCapitalLoad:
    """Expected payout plus a cost-of-capital load on the payout tail.

    ``premium = (1 - rate) E[P] + rate CVaR_q(P)``.

    Parameters
    ----------
    rate : float
        Cost-of-capital rate ``r`` in ``[0, 1]``, e.g. 0.06 to 0.10.
    q : float
        Confidence level of the payout tail, in ``(0, 1)``, e.g. 0.99.

    Raises
    ------
    ParametricInputError
        If ``rate`` is not in ``[0, 1]`` or ``q`` not in ``(0, 1)``.
    """

    name: ClassVar[str] = "cost_of_capital"

    rate: float
    q: float

    def __post_init__(self) -> None:
        """Validate ``rate`` and ``q``."""
        object.__setattr__(
            self, "rate", check_real("rate", self.rate, low=0.0, high=1.0)
        )
        object.__setattr__(self, "q", check_level("q", self.q))

    def premium(self, payout: ArrayLike, measure: OutcomeMeasure) -> float:
        """Return the loaded premium of a realised payout.

        Parameters
        ----------
        payout : array_like of float
            Per-event payout of the design problem, shape ``(n_events,)``.
        measure : OutcomeMeasure
            Measure over the same events.

        Returns
        -------
        float
            ``(1 - rate) E[P] + rate CVaR_q(P)`` over the measure's outcomes,
            with the exact discrete CVaR.
        """
        values = np.asarray(payout, dtype=np.float64)
        expected = measure.expectation(values)
        if self.rate == 0.0:
            return expected
        _, cvar = var_cvar(measure.aggregate(values), measure.probability, self.q)
        return (1.0 - self.rate) * expected + self.rate * cvar


Pricing = PurePremium | CostOfCapitalLoad
"""The pricings :func:`~climada_petals.engine.parametric.calibrate.calibrate` accepts."""
