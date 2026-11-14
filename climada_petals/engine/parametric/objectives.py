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

With the outcome weights ``w_o`` of the design sample's measure
(:mod:`~climada_petals.engine.parametric.measure`): a probability per
simulated year on the annual basis, or the event's frequency per year on the
event basis, which need not sum to 1. Per outcome, ``L_o`` and ``P_o`` are
the loss and the payout (sums over the events of outcome ``o`` and over the
units of the design problem), and ``U_o = sum_{e in o} (L_e - P_e)^+`` is the
uncovered loss:

* :class:`ExpectedUncovered`: ``E[U] = sum_o w_o U_o``. Over-payment is
  not rewarded. A linear program (LP).
* :class:`WeightedCVaR`: ``alpha * CVaR_beta(U) + (1 - alpha) * E[U]``:
  **alpha weights the CVaR term**, so ``alpha = 0`` is ``E[U]`` and
  ``alpha = 1`` the CVaR of ``U`` alone. With the Rockafellar-Uryasev form
  of the CVaR it is an LP too. It is a mixture of the mean and the CVaR of
  the uncovered loss, not a CVaR with weights on the outcomes.
* :class:`MSE`: ``sum_o w_o (L_o - P_o)^2``, the signed residual per
  outcome, so over-payment counts like under-payment. A quadratic program
  (QP), solved with scipy's ``SLSQP``.
* :class:`UserObjective`: any function ``function(loss, payout, weight)`` of
  the outcome arrays ``L_o``, ``P_o`` and ``w_o`` that returns a float. It
  is evaluated, not compiled into a program, so the calibration takes it on
  the grid route: every free parameter on a stated grid, in numpy.

On the event basis ``1 - beta`` is a frequency per year, and the CVaR is
the mean of the worst ``1 - beta`` of frequency, the missing mass counting
as zero (:mod:`~climada_petals.engine.parametric.riskmeasures`). The LP
needs no probability: in the Rockafellar-Uryasev form
``CVaR_beta(U) = min_{zeta >= 0} zeta + sum_o w_o (U_o - zeta)^+ / (1 - beta)``
the bound ``zeta >= 0`` gives this CVaR for any weights, because ``U`` is
never negative.

The LPs are solved with scipy's HiGHS, the QP with scipy's ``SLSQP``
(:mod:`~climada_petals.engine.parametric.solvers`). Each class also
evaluates its objective for given losses and payouts with numpy: the
calibration recomputes every solution's objective that way, and checks it
against the solver's. The evaluation's metrics have their own kernels
(:mod:`~climada_petals.engine.parametric.metrics`). Each class has a
document, :meth:`to_dict`, which a calibration result records.

When the tail holds every outcome with a loss, ``1 - beta`` at least the
weight of the outcomes with ``L_o > 0`` (on the annual basis, ``beta`` at
most ``P(L_o = 0)``), every design has ``VaR_beta(U) = 0`` and the
weighted-CVaR objective is ``(alpha / (1 - beta) + 1 - alpha) * E[U]``: it has
the same minimisers as ``E[U]``.
:func:`~climada_petals.engine.parametric.calibrate.calibrate` records this
case as ``degenerate_tail="expected_uncovered"`` and logs a warning.

References
----------
* Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional
  value-at-risk. Journal of Risk, 2(3), 21-41.
  https://doi.org/10.21314/JOR.2000.038
* Rockafellar, R. T. and Uryasev, S. (2002). Conditional value-at-risk for
  general loss distributions. Journal of Banking & Finance, 26(7),
  1443-1471. https://doi.org/10.1016/S0378-4266(02)00271-6
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._checks import check_level, check_real
from .errors import ParametricInputError
from .measure import OutcomeMeasure
from .riskmeasures import CDF_ATOL, var_cvar

__all__ = [
    "MSE",
    "ExpectedUncovered",
    "Objective",
    "UserObjective",
    "WeightedCVaR",
    "tail_degeneracy",
]


def _outcome_uncovered(
    loss: ArrayLike, payout: ArrayLike, measure: OutcomeMeasure
) -> NDArray[np.float64]:
    """Return ``U_o = sum_{e in o} (L_e - P_e)^+`` per outcome."""
    shortfall = np.maximum(np.asarray(loss) - np.asarray(payout), 0.0)
    return measure.aggregate(shortfall)


@dataclass(frozen=True)
class MSE:
    """Mean squared signed residual per outcome, ``E[(L - P)^2]``.

    On the annual basis the residual of a year is the year's loss minus its
    payout, summed over its events before squaring: the owner's net
    position per year. A QP, which
    :func:`~climada_petals.engine.parametric.calibrate.calibrate` solves
    with scipy's ``SLSQP`` on the amounts, with cutting planes under a
    cost-of-capital load, within stated limits.
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
            ``sum_o w_o (L_o - P_o)^2``.
        """
        residual = measure.aggregate(np.asarray(loss) - np.asarray(payout))
        return float(measure.probability @ residual**2)

    def to_dict(self) -> dict[str, Any]:
        """Return the objective's document, ``{"kind": "mse"}``."""
        return {"kind": self.name}


@dataclass(frozen=True)
class ExpectedUncovered:
    """Expected uncovered loss, ``E[U] = sum_o w_o U_o``. An LP."""

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

    def to_dict(self) -> dict[str, Any]:
        """Return the objective's document, ``{"kind": "expected_uncovered"}``."""
        return {"kind": self.name}


@dataclass(frozen=True)
class WeightedCVaR:
    """Weighted CVaR of the uncovered loss, ``alpha CVaR_beta(U) + (1 - alpha) E[U]``.

    An LP in the Rockafellar-Uryasev form, with ``zeta >= 0`` (module
    docstring): ``CVaR_beta(U) = min_zeta zeta + E[(U - zeta)^+] / (1 - beta)``.

    Parameters
    ----------
    alpha : float
        Weight of the CVaR term, in ``[0, 1]``. ``alpha = 0`` is ``E[U]``.
    beta : float
        CVaR confidence level, in ``(0, 1)``. On the annual basis,
        ``beta = 0.99`` is the mean of the worst 1 % of years; on the event
        basis, of the worst 0.01 of frequency per year.

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

    def to_dict(self) -> dict[str, Any]:
        """Return the objective's document, with ``alpha`` and ``beta``."""
        return {"kind": self.name, "alpha": self.alpha, "beta": self.beta}


@dataclass(frozen=True)
class UserObjective:
    """A user's objective, ``function(loss, payout, weight) -> float``.

    The calibration minimises it. It is evaluated on the outcome arrays, not
    compiled into a linear or quadratic program, so
    :func:`~climada_petals.engine.parametric.calibrate.calibrate` takes it on
    the grid route: every free parameter of the template needs a grid (the
    amounts too), the scope is individual, and each candidate's payout is
    evaluated in numpy. Candidates whose premium is above the budget, or
    whose payout is above the limit, are not chosen.

    Parameters
    ----------
    function : callable
        ``function(loss, payout, weight)`` with three float arrays of shape
        ``(n_outcomes,)``: the loss ``L_o`` and the payout ``P_o`` per
        outcome (sums over its events, as for the other objectives) and the
        outcome weights ``w_o`` (a probability per simulated year on the
        annual basis, where the years without an event may be pooled into
        one outcome of their summed weight; the frequency on the event
        basis). It returns a finite real number, in its own units. The
        arrays are copies.
    name : str
        The objective's name in the record and in logs. Keyword-only.
    parameters : mapping of str to float, int, str, bool or None, optional
        The function's own parameters, recorded in the document
        (:meth:`to_dict`) so that a design states what it minimised; the
        function is not called with them. Keyword-only.

    Raises
    ------
    ParametricInputError
        If ``function`` is not callable, ``name`` is not a non-empty string,
        or ``parameters`` is not a mapping of names to such values.

    Notes
    -----
    A result whose objective is a ``UserObjective`` pickles only if the
    function does, e.g. a function defined at a module's top level.

    Examples
    --------
    The expected squared shortfall, which leaves over-payment unpenalised:

    >>> def shortfall_squared(loss, payout, weight):
    ...     return float(weight @ np.maximum(loss - payout, 0.0) ** 2)
    >>> objective = UserObjective(shortfall_squared, name="shortfall_squared")
    >>> objective.to_dict()
    {'kind': 'user', 'name': 'shortfall_squared', 'parameters': {}}
    """

    kind: ClassVar[str] = "user"
    is_linear: ClassVar[bool] = False

    function: Callable[..., Any]
    name: str = field(kw_only=True)
    parameters: Mapping[str, Any] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the fields; store the parameters read-only."""
        if not callable(self.function):
            raise ParametricInputError(
                f"UserObjective's function must be callable, got {self.function!r}"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ParametricInputError(
                f"UserObjective's name must be a non-empty string, got {self.name!r}"
            )
        parameters = {} if self.parameters is None else self.parameters
        if not isinstance(parameters, Mapping) or not all(
            isinstance(key, str) and _is_document_value(value)
            for key, value in parameters.items()
        ):
            raise ParametricInputError(
                "UserObjective's parameters must map names to finite numbers, "
                f"strings, bools or None, got {self.parameters!r}"
            )
        object.__setattr__(self, "parameters", MappingProxyType(dict(parameters)))

    def __hash__(self) -> int:
        """Hash the function, the name and the parameters."""
        return hash((self.function, self.name, frozenset(self._items())))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy through the constructor, with the parameters as a dict."""
        return (_user_objective, (self.function, self.name, dict(self._items())))

    def _items(self) -> tuple[tuple[str, Any], ...]:
        """Return the parameters as ``(name, value)`` pairs, sorted by name."""
        return tuple(sorted(dict(self.parameters or {}).items()))

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
            ``function(L_o, P_o, w_o)``.

        Raises
        ------
        ParametricInputError
            If the function doesn't return a finite real number.
        """
        value = self.function(
            measure.aggregate(loss),
            measure.aggregate(payout),
            np.array(measure.probability, dtype=np.float64),
        )
        if isinstance(value, bool | np.bool_) or not isinstance(
            value, int | float | np.number
        ):
            raise ParametricInputError(
                f"the objective {self.name!r} must return a real number, got {value!r}"
            )
        if not math.isfinite(float(value)):
            raise ParametricInputError(
                f"the objective {self.name!r} returned {value!r}, which is not finite"
            )
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        """Return the objective's document: ``kind``, ``name`` and ``parameters``."""
        return {
            "kind": self.kind,
            "name": self.name,
            "parameters": dict(self._items()),
        }


def _is_document_value(value: object) -> bool:
    """Whether a value can stand in a document: a finite number, a string, a bool or None."""
    if value is None or isinstance(value, str | bool | np.bool_):
        return True
    return isinstance(value, int | float | np.number) and math.isfinite(float(value))


def _user_objective(
    function: Callable[..., Any], name: str, parameters: dict[str, Any]
) -> UserObjective:
    """Rebuild a user objective from its fields (pickling)."""
    return UserObjective(function, name=name, parameters=parameters)


Objective = MSE | ExpectedUncovered | WeightedCVaR | UserObjective
"""The objectives :func:`~climada_petals.engine.parametric.calibrate.calibrate` accepts.

The first three are solved exactly where the template allows it; a
:class:`UserObjective` always takes the grid route."""


def tail_degeneracy(
    objective: Objective, loss: ArrayLike, measure: OutcomeMeasure
) -> tuple[float, str | None]:
    """Return the weight of the outcomes with a loss, and the degenerate tail.

    Parameters
    ----------
    objective : Objective
        The design's objective.
    loss : array_like of float
        The design problem's loss per row of the measure.
    measure : OutcomeMeasure
        The design sample's measure.

    Returns
    -------
    loss_weight : float
        The weight of the outcomes with ``L_o > 0``.
    degenerate_tail : {"expected_uncovered", "single_outcome"} or None
        For a weighted CVaR with ``alpha > 0``: ``"expected_uncovered"`` when
        ``1 - beta`` is at least ``loss_weight`` (module docstring),
        ``"single_outcome"`` when ``1 - beta`` is at most the smallest weight
        of an outcome with a loss, so the CVaR is the worst outcome alone;
        both within :data:`~climada_petals.engine.parametric.riskmeasures.CDF_ATOL`.
        Else None.
    """
    lossy = measure.aggregate(loss) > 0.0
    weight = float(measure.weights[lossy].sum())
    if not isinstance(objective, WeightedCVaR) or objective.alpha == 0.0:
        return weight, None
    tail = 1.0 - objective.beta
    if tail >= weight - CDF_ATOL:
        return weight, "expected_uncovered"
    if tail <= float(measure.weights[lossy].min()) + CDF_ATOL:
        return weight, "single_outcome"
    return weight, None
