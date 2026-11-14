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

Both pricings price a payout per event, aggregated into the outcomes of an
:class:`~climada_petals.engine.parametric.measure.OutcomeMeasure`:
``P_o = sum_{e in o} P_e``.

* :class:`PurePremium`: ``E[P]``.
* :class:`CostOfCapitalLoad`: ``(1 - r) E[P] + r CVaR_q(P)``. The load
  ``r (CVaR_q(P) - E[P])`` prices the capital a writer holds against the
  payout tail, so ``E[P] <= premium <= CVaR_q(P)``. Convex in the payout for
  ``r`` in ``[0, 1]``, so the premium budget
  ``premium <= budget * E[L]`` keeps the design problem an LP (a QP with the
  mean squared error). ``r = 0`` is :class:`PurePremium`.

On the annual basis the premium is that of a year's payout. On the event
basis ``E[P]`` is the expected payout per year, ``sum_e f_e P_e``, and the
CVaR is that of the worst ``1 - q`` of frequency per year
(:mod:`~climada_petals.engine.parametric.riskmeasures`), whatever the
frequencies sum to.

The calibration prices its design with the pricing it is given, and the
budget constrains that premium.
:func:`~climada_petals.engine.parametric.evaluate.evaluate` prices a cover
given as it is, and reprices a design, on the evaluation sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

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
            Payout per row of the measure (an event, or an event and a
            draw), summed over the units priced, shape ``(n_rows,)``.
        measure : OutcomeMeasure
            Measure over the same rows.

        Returns
        -------
        float
            ``E[P]``, in the payout's unit: per year on both bases.
        """
        return measure.expectation(payout)

    def to_dict(self) -> dict[str, Any]:
        """Return the pricing's document.

        Returns
        -------
        dict
            ``{"kind": "pure"}``.
        """
        return {"kind": self.name}


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
    q: float  # pylint: disable=invalid-name

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
            Payout per row of the measure (an event, or an event and a
            draw), summed over the units priced, shape ``(n_rows,)``.
        measure : OutcomeMeasure
            Measure over the same rows.

        Returns
        -------
        float
            ``(1 - rate) E[P] + rate CVaR_q(P)`` over the measure's outcomes,
            with the exact discrete CVaR
            (:func:`~climada_petals.engine.parametric.riskmeasures.var_cvar`).
        """
        values = np.asarray(payout, dtype=np.float64)
        expected = measure.expectation(values)
        if self.rate == 0.0:
            return expected
        _, cvar = var_cvar(measure.aggregate(values), measure.weights, self.q)
        return (1.0 - self.rate) * expected + self.rate * cvar

    def to_dict(self) -> dict[str, Any]:
        """Return the pricing's document.

        Returns
        -------
        dict
            ``{"kind": "cost_of_capital", "q": ..., "rate": ...}``.
        """
        return {"kind": self.name, "q": self.q, "rate": self.rate}


Pricing = PurePremium | CostOfCapitalLoad
"""The pricings of the calibration and the evaluation."""
