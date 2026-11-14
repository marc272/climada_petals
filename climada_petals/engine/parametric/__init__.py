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
  next to :class:`ExpectedUncovered` (``E[U]``) and :class:`MSE`. The first
  two are linear programs solved with scipy's HiGHS; the mean squared error
  is a quadratic program solved with scipy's ``SLSQP``, within stated
  limits. A :class:`UserObjective` is any function of the outcomes' losses,
  payouts and weights, evaluated on a grid.
* **Calibration:** :func:`calibrate` chooses the free amounts of a template
  cover on a design sample, with ``scope="individual"`` (one design problem
  per unit) or ``scope="joint"`` (one problem on the owner's total loss, so
  the units' payouts can offset each other's shortfalls), and returns a
  :class:`CalibrationResult`, which holds the resolved cover and how it was
  found. Pricing: :class:`PurePremium` or :class:`CostOfCapitalLoad`, under
  a premium :class:`Budget`; options in :class:`CalibrationOptions`. Free
  shape parameters (such as a free attachment), payouts that are not linear
  in their amounts and user objectives take a stated grid, in the
  individual scope. :func:`calibrate_frontier` calibrates at several
  budgets: the budget frontier, a measurement with no best point.
* **Evaluation:** :func:`evaluate` and :func:`compare` give the metrics of
  any cover, designed or given, on a sample, in or out of sample, as a
  pandas table: coverage, tail coverage, basis risk, VaR and CVaR of the
  uncovered loss and of the net loss, return periods and more, per unit
  and for the owner's total. Every row carries its setup (basis and route,
  split, index, pricing, scope), which :func:`setup_text` renders. The
  thresholds are always stated, in :class:`MetricOptions`.
  :func:`payout_impact` and :func:`uncovered_impact` give the payouts and
  the uncovered losses as CLIMADA ``Impact`` objects, so that
  ``calc_freq_curve``, ``local_exceedance_impact`` and ``write_hdf5`` work
  on them. Plots: :func:`plot_payout`, :func:`plot_payout_vs_loss` and
  :func:`plot_exceedance`.
* **Basis risk:** it comes from a stated source. The structural ones lead:
  an index that is not the footprint at the assets (cat-in-a-circle,
  another or a coarser hazard set, a table of readings) and units of
  several points. :func:`index_diagnostics` measures, per unit, how far the
  index determines the loss; :func:`calibrate` and :func:`evaluate` run it
  and warn where it does almost one-to-one, so that basis risk would read
  close to zero, and the tables mark such units (``comonotone_index``).
  Impact-function draws come second (:func:`impact_function_draws`,
  :meth:`OutcomeSample.from_impact_draws`): they widen the uncovered loss
  but don't change which events pay or lose.
* **Scenarios:** :class:`ScenarioSet` maps labels to scenarios, other
  hazard sets and possibly other exposures (a future climate from the Data
  API, a set scaled by ``apply_climate_scenario_knu``), all on one route.
  A label may hold several samples, which form one scenario as an
  equal-weight mixture: the windows of a pathway or the members of an
  ensemble. :func:`evaluate_scenarios` evaluates one cover on every
  scenario, with the label in the ``scenario`` column, and
  :func:`calibrate_per_scenario` designs one per scenario
  (:func:`designs_table`). The tables rank no scenario and no cover.

The module depends only on CLIMADA, numpy, scipy, pandas and matplotlib.
"""

from .annual import CatalogueYears, YearSets, sample_years
from .basis_risk import impact_function_draws, index_diagnostics
from .calibrate import (
    CalibrationOptions,
    CalibrationResult,
    CoverProblem,
    DesignSample,
    calibrate,
    calibrate_frontier,
)
from .constraints import Budget
from .cover import IndexDefinition, Limit, ParametricCover
from .errors import (
    CalibrationError,
    CalibrationLimitError,
    CoverDefinitionError,
    EventAlignmentError,
    ImpactInputError,
    InfeasibleCalibrationError,
    LeakageError,
    MissingDependencyError,
    ParametricError,
    ParametricInputError,
    ScenarioError,
    UnboundedCalibrationError,
    UnitMappingError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
    YearRouteError,
)
from .evaluate import compare, evaluate, payout_impact, uncovered_impact
from .index import ParametricIndex
from .index_tracks import SAFFIR_SIMPSON_KN
from .measure import ENSEMBLE_PARENT, EventBasis, OutcomeMeasure, labels_from_names
from .metrics import MetricOptions, setup_text
from .objectives import MSE, ExpectedUncovered, UserObjective, WeightedCVaR
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
from .plots import plot_exceedance, plot_payout, plot_payout_vs_loss
from .pricing import CostOfCapitalLoad, PurePremium
from .riskmeasures import var_cvar
from .sample import (
    OutcomeSample,
    SplitRecord,
    check_no_leakage,
    holdout,
    year_block_kfold,
)
from .scenarios import (
    ScenarioSet,
    calibrate_per_scenario,
    designs_table,
    evaluate_scenarios,
)
from .units import InsuredUnits

__all__ = [
    "ENSEMBLE_PARENT",
    "MSE",
    "SAFFIR_SIMPSON_KN",
    "AmountConstraints",
    "BinaryPayout",
    "Budget",
    "CalibrationError",
    "CalibrationLimitError",
    "CalibrationOptions",
    "CalibrationResult",
    "CatalogueYears",
    "CostOfCapitalLoad",
    "CoverDefinitionError",
    "CoverProblem",
    "DesignSample",
    "EventAlignmentError",
    "EventBasis",
    "ExpectedUncovered",
    "Free",
    "ImpactInputError",
    "IndexDefinition",
    "InfeasibleCalibrationError",
    "InsuredUnits",
    "LeakageError",
    "Limit",
    "LinearInAmounts",
    "LinearPayout",
    "MetricOptions",
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
    "ScenarioError",
    "ScenarioSet",
    "SplitRecord",
    "StepPayout",
    "UnboundedCalibrationError",
    "UnitMappingError",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "UserObjective",
    "WeightedCVaR",
    "YearRouteError",
    "YearSets",
    "calibrate",
    "calibrate_frontier",
    "calibrate_per_scenario",
    "check_no_leakage",
    "compare",
    "designs_table",
    "evaluate",
    "evaluate_scenarios",
    "holdout",
    "impact_function_draws",
    "index_diagnostics",
    "labels_from_names",
    "payout_impact",
    "plot_exceedance",
    "plot_payout",
    "plot_payout_vs_loss",
    "register_payout",
    "sample_years",
    "setup_text",
    "uncovered_impact",
    "var_cvar",
    "year_block_kfold",
]
