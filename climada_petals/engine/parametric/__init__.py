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
* **Units:** :class:`InsuredUnits`, the exposure points that share one
  payout function: one point, a group (a column such as ``region_id``, or
  polygons) or the whole exposure set. A unit's loss sums ``imp_mat`` over
  its points, and its payout maps back to them by value share.
* **Index:** :class:`ParametricIndex`, one value per event and unit with its
  :class:`IndexDefinition`, on any hazard: a statistic of a ``Hazard``'s
  footprint over each unit (:meth:`~ParametricIndex.from_hazard`), the
  cat-in-a-circle category from tracks
  (:meth:`~ParametricIndex.from_cat_in_circle`), a user's table
  (:meth:`~ParametricIndex.from_table`), or a cover's index on another
  hazard set (:meth:`~ParametricIndex.from_definition`). Events are matched
  on ``event_id`` (:meth:`~ParametricIndex.matched_to`). An index holds no
  years.
* **Outcomes:** :class:`OutcomeSample` joins an index, an ``Impact``'s loss
  per unit and a stated basis, which says what an outcome is:
  :class:`EventBasis` (one outcome per event, weighted by its frequency, for
  any hazard), or one simulated year on :class:`CatalogueYears` (a year
  label per event, from the dates, the event names or an array) or
  :class:`YearSets` (years sampled from the frequencies with a seeded
  generator, as ``climada.util.yearsets`` does, or a given
  ``sampling_vect``). A route that doesn't fit the events raises
  :class:`YearRouteError`.
* **Splits:** :func:`holdout` and :func:`year_block_kfold` split a sample by
  blocks: contiguous years on catalogue years, pools of events or of
  ensemble parents on the other routes, with no event or parent on both
  sides (:func:`check_no_leakage`, :class:`LeakageError`).
* **Objectives:** :class:`WeightedCVaR`, the tail objective
  ``alpha CVaR_beta(U) + (1 - alpha) E[U]`` of the uncovered loss ``U``,
  next to :class:`ExpectedUncovered` (``E[U]``) and
  :class:`MeanSquaredError`. The first two are linear programs solved with
  scipy's HiGHS; the mean squared error is a quadratic program and needs the
  optional dependency cvxpy.
* **Calibration scope:** :func:`calibrate` chooses the levels of a step
  payout template (:meth:`StepPayout.uniform`) with ``scope="individual"``
  (one design problem per unit) or ``scope="portfolio"`` (one
  joint problem on the owner's total loss, so the units' payouts can offset
  each other's shortfalls), and returns a :class:`CalibrationResult`.
  Pricing: :class:`PurePremium` or :class:`CostOfCapitalLoad`, under a
  premium budget.
* **Metrics:** :func:`basis_risk`, and the payouts as a CLIMADA ``Impact``
  (:meth:`CalibrationResult.payout_impact`).

The module depends only on CLIMADA, numpy, scipy and pandas, plus cvxpy for
the mean squared error.
"""

from .annual import CatalogueYears, YearSets, sample_years
from .calibrate import CalibrationResult, CoverProblem, calibrate
from .cover import IndexDefinition, Limit, ParametricCover
from .errors import (
    CalibrationError,
    CoverDefinitionError,
    EventAlignmentError,
    ImpactInputError,
    LeakageError,
    MissingDependencyError,
    ParametricError,
    ParametricInputError,
    UnitMappingError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
    YearRouteError,
)
from .index import ParametricIndex
from .index_tracks import SAFFIR_SIMPSON_KN
from .measure import ENSEMBLE_PARENT, EventBasis, OutcomeMeasure, labels_from_names
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
from .sample import (
    OutcomeSample,
    SplitRecord,
    check_no_leakage,
    holdout,
    year_block_kfold,
)
from .units import InsuredUnits

__all__ = [
    "ENSEMBLE_PARENT",
    "SAFFIR_SIMPSON_KN",
    "AmountConstraints",
    "BinaryPayout",
    "CalibrationError",
    "CalibrationResult",
    "CatalogueYears",
    "CostOfCapitalLoad",
    "CoverDefinitionError",
    "CoverProblem",
    "EventAlignmentError",
    "EventBasis",
    "ExpectedUncovered",
    "Free",
    "ImpactInputError",
    "IndexDefinition",
    "InsuredUnits",
    "LeakageError",
    "Limit",
    "LinearInAmounts",
    "LinearPayout",
    "MeanSquaredError",
    "MissingDependencyError",
    "OutcomeMeasure",
    "OutcomeSample",
    "ParametricCover",
    "ParametricError",
    "ParametricIndex",
    "ParametricInputError",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "PurePremium",
    "SplitRecord",
    "StepPayout",
    "UnitMappingError",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "WeightedCVaR",
    "YearRouteError",
    "YearSets",
    "basis_risk",
    "calibrate",
    "check_no_leakage",
    "holdout",
    "labels_from_names",
    "register_payout",
    "sample_years",
    "var_cvar",
    "year_block_kfold",
]
