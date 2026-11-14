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

Objectives of a design problem: what the calibration minimises.

With outcome probabilities ``p_o``, outcome loss ``L_o`` and outcome payout
``P_o`` (sums over the events of outcome ``o`` and over the units of the
design problem; see :mod:`~climada_petals.engine.parametric.measure`), and the uncovered
loss ``U_o = sum_{e in o} (L_e - P_e)^+``:

* :class:`ExpectedUncovered`: ``E[U] = sum_o p_o U_o``. Over-payment is
  not rewarded. A linear program (LP).
* :class:`WeightedCVaR`: ``alpha * CVaR_beta(U) + (1 - alpha) * E[U]``:
  **alpha weights the CVaR term**, so ``alpha = 0`` is ``E[U]`` and
  ``alpha = 1`` the CVaR of ``U`` alone. With the Rockafellar-Uryasev form
  of the CVaR it is an LP too. It is a mixture of the mean and the CVaR of
  the uncovered loss, not a CVaR with weights on the outcomes.
* :class:`MeanSquaredError`: ``sum_o p_o (L_o - P_o)^2``, the signed
  residual per outcome, so over-payment counts like under-payment. A
  quadratic program (QP): it needs the optional dependency cvxpy.

The LPs are solved with scipy's HiGHS (:mod:`~climada_petals.engine.parametric.solvers`).
Each class also evaluates its objective for given losses and payouts with
numpy, which is how a solution is checked and how the basis-risk metrics
are computed.

When ``beta`` is at most the probability of the outcomes without loss,
``P(L_o = 0)``, every design has ``VaR_beta(U) = 0`` and the weighted-CVaR
objective is ``(alpha / (1 - beta) + 1 - alpha) * E[U]``: it has the same
minimisers as ``E[U]``. :func:`~climada_petals.engine.parametric.calibrate.calibrate`
records this case as ``degenerate_tail`` and logs a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._checks import check_level, check_real
from .measure import OutcomeMeasure
from .riskmeasures import var_cvar

__all__ = ["ExpectedUncovered", "MeanSquaredError", "Objective", "WeightedCVaR"]


def _outcome_uncovered(
    loss: ArrayLike, payout: ArrayLike, measure: OutcomeMeasure
) -> NDArray[np.float64]:
    """Return ``U_o = sum_{e in o} (L_e - P_e)^+`` per outcome."""
    shortfall = np.maximum(np.asarray(loss) - np.asarray(payout), 0.0)
    return measure.aggregate(shortfall)


@dataclass(frozen=True)
class MeanSquaredError:
    """Mean squared signed residual per outcome, ``E[(L - P)^2]``.

    On the annual basis the residual of a year is the year's loss minus its
    payout, summed over its events before squaring: the owner's net
    position per year. A QP: :func:`~climada_petals.engine.parametric.calibrate.calibrate`
    solves it with cvxpy (optional dependency) and raises
    :class:`~climada_petals.engine.parametric.errors.MissingDependencyError` without it.
    """

    name: ClassVar[str] = "mse"
    is_linear: ClassVar[bool] = False

    def evaluate(
        self, loss: ArrayLike, payout: ArrayLike, measure: OutcomeMeasure
    ) -> float:
        """Return the objective for per-event losses and payouts.

        Parameters
        ----------
        loss, payout : array_like of float
            Per-event loss and payout of the design problem (summed over its
            units), shape ``(n_events,)``.
        measure : OutcomeMeasure
            Measure over the same events.

        Returns
        -------
        float
            ``sum_o p_o (L_o - P_o)^2``.
        """
        residual = measure.aggregate(np.asarray(loss) - np.asarray(payout))
        return float(measure.probability @ residual**2)


@dataclass(frozen=True)
class ExpectedUncovered:
    """Expected uncovered loss, ``E[U] = sum_o p_o U_o``. An LP."""

    name: ClassVar[str] = "expected_uncovered"
    is_linear: ClassVar[bool] = True

    def evaluate(
        self, loss: ArrayLike, payout: ArrayLike, measure: OutcomeMeasure
    ) -> float:
        """Return the objective for per-event losses and payouts.

        Parameters
        ----------
        loss, payout : array_like of float
            Per-event loss and payout of the design problem, shape
            ``(n_events,)``.
        measure : OutcomeMeasure
            Measure over the same events.

        Returns
        -------
        float
            ``E[U]``.
        """
        return float(measure.probability @ _outcome_uncovered(loss, payout, measure))


@dataclass(frozen=True)
class WeightedCVaR:
    """Weighted CVaR of the uncovered loss, ``alpha CVaR_beta(U) + (1 - alpha) E[U]``.

    An LP in the Rockafellar-Uryasev form:
    ``CVaR_beta(U) = min_zeta zeta + E[(U - zeta)^+] / (1 - beta)``.

    Parameters
    ----------
    alpha : float
        Weight of the CVaR term, in ``[0, 1]``. ``alpha = 0`` is ``E[U]``.
    beta : float
        CVaR confidence level, in ``(0, 1)``. On the annual basis,
        ``beta = 0.99`` is the mean of the worst 1 % of years.

    Raises
    ------
    ParametricInputError
        If ``alpha`` is not in ``[0, 1]`` or ``beta`` not in ``(0, 1)``.
    """

    name: ClassVar[str] = "weighted_cvar"
    is_linear: ClassVar[bool] = True

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        """Validate ``alpha`` and ``beta``."""
        object.__setattr__(
            self, "alpha", check_real("alpha", self.alpha, low=0.0, high=1.0)
        )
        object.__setattr__(self, "beta", check_level("beta", self.beta))

    def evaluate(
        self, loss: ArrayLike, payout: ArrayLike, measure: OutcomeMeasure
    ) -> float:
        """Return the objective for per-event losses and payouts.

        Parameters
        ----------
        loss, payout : array_like of float
            Per-event loss and payout of the design problem, shape
            ``(n_events,)``.
        measure : OutcomeMeasure
            Measure over the same events.

        Returns
        -------
        float
            ``alpha * CVaR_beta(U) + (1 - alpha) * E[U]``, with the exact
            discrete CVaR (:func:`~climada_petals.engine.parametric.riskmeasures.var_cvar`).
        """
        uncovered = _outcome_uncovered(loss, payout, measure)
        expected = float(measure.probability @ uncovered)
        if self.alpha == 0.0:
            return expected
        _, cvar = var_cvar(uncovered, measure.probability, self.beta)
        return self.alpha * cvar + (1.0 - self.alpha) * expected


Objective = MeanSquaredError | ExpectedUncovered | WeightedCVaR
"""The objectives :func:`~climada_petals.engine.parametric.calibrate.calibrate` accepts."""
