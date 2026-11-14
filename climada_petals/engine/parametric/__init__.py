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

Parametric insurance design on CLIMADA objects.

Designs index-based (parametric) covers for the assets of one risk owner
from CLIMADA's ``Hazard``, ``TCTracks``, ``Exposures`` and ``Impact``:

* **Covers:** :class:`ParametricCover` holds the terms of a cover and
  nothing else: what it pays on (:class:`IndexDefinition`), one payout
  function per insured unit with its parameters, the payout unit and the
  most one event pays at a unit (:class:`Limit`). A cover with a
  :class:`Free` parameter is a template. A cover round-trips through JSON,
  so it can be stored and applied to another hazard set.
* **Payouts:** :class:`StepPayout`, :class:`LinearPayout`,
  :class:`PiecewiseLinearPayout` and :class:`BinaryPayout`, on the upper or
  the lower tail of the index, and user-defined families through the
  :class:`PayoutFunction` protocol and :func:`register_payout`. Nothing is
  paid below the first threshold.
* **Index:** :class:`ParametricIndex`, the hazard intensity at each exposure
  point (:meth:`~ParametricIndex.from_hazard`) or the cat-in-a-circle
  category from tracks (:meth:`~ParametricIndex.from_cat_in_circle`), with
  each event's simulation year, read from the hazard's dates or event names
  or given; :func:`split_years` for training and test years.
* **Objectives:** :class:`WeightedCVaR`, the tail objective
  ``alpha CVaR_beta(U) + (1 - alpha) E[U]`` of the uncovered loss ``U``,
  next to :class:`ExpectedUncovered` (``E[U]``) and
  :class:`MeanSquaredError`. The first two are linear programs solved with
  scipy's HiGHS; the mean squared error is a quadratic program and needs the
  optional dependency cvxpy.
* **Calibration scope:** :func:`calibrate` chooses the levels of a step
  payout template (:meth:`StepPayout.uniform`) with ``scope="individual"``
  (one design problem per exposure point) or ``scope="portfolio"`` (one
  joint problem on the owner's total loss, so the units' payouts can offset
  each other's shortfalls), and returns a :class:`CalibrationResult`.
  Pricing: :class:`PurePremium` or :class:`CostOfCapitalLoad`, under a
  premium budget.
* **Basis:** ``"annual"`` (outcomes are simulation years, so
  ``CVaR_beta`` has a return-period meaning) or ``"event"``.
* **Metrics:** :func:`basis_risk`, and the payouts as a CLIMADA ``Impact``
  (:meth:`CalibrationResult.payout_impact`).

The module depends only on CLIMADA, numpy, scipy and pandas, plus cvxpy for
the mean squared error.
"""

from .calibrate import CalibrationResult, CoverProblem, calibrate
from .cover import IndexDefinition, Limit, ParametricCover
from .errors import (
    CalibrationError,
    CoverDefinitionError,
    MissingDependencyError,
    ParametricError,
    ParametricInputError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from .index import ParametricIndex, split_years
from .index_tracks import SAFFIR_SIMPSON_KN
from .measure import OutcomeMeasure, outcome_measure
from .metrics import basis_risk
from .objectives import ExpectedUncovered, MeanSquaredError, WeightedCVaR
from .parameters import AmountConstraints, Free
from .payout import (
    BinaryPayout,
    LinearInAmounts,
    LinearPayout,
    PayoutFunction,
    PiecewiseLinearPayout,
    StepPayout,
    register_payout,
)
from .pricing import CostOfCapitalLoad, PurePremium
from .riskmeasures import var_cvar

__all__ = [
    "SAFFIR_SIMPSON_KN",
    "AmountConstraints",
    "BinaryPayout",
    "CalibrationError",
    "CalibrationResult",
    "CostOfCapitalLoad",
    "CoverDefinitionError",
    "CoverProblem",
    "ExpectedUncovered",
    "Free",
    "IndexDefinition",
    "Limit",
    "LinearInAmounts",
    "LinearPayout",
    "MeanSquaredError",
    "MissingDependencyError",
    "OutcomeMeasure",
    "ParametricCover",
    "ParametricError",
    "ParametricIndex",
    "ParametricInputError",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "PurePremium",
    "StepPayout",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "WeightedCVaR",
    "basis_risk",
    "calibrate",
    "outcome_measure",
    "register_payout",
    "split_years",
    "var_cvar",
]
