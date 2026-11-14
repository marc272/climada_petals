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

* **Covers:** :class:`.ParametricCover` holds the terms of a cover and
  nothing else: what it pays on (:class:`.IndexDefinition`), one payout
  function per insured unit with its parameters, the payout unit and the
  most one event pays at a unit (:class:`.Limit`). A cover with a
  :class:`.Free` parameter is a template. A cover round-trips through JSON,
  so it can be stored and applied to another hazard set.
* **Payouts:** :class:`.StepPayout`, :class:`.LinearPayout`,
  :class:`.PiecewiseLinearPayout` and :class:`.BinaryPayout`, on the upper or
  the lower tail of the index, and user-defined families through the
  :class:`.PayoutFunction` protocol and :func:`.register_payout`. Nothing is
  paid below the first threshold.
* **Units:** :class:`.InsuredUnits`, the exposure points that share one
  payout function: one point, a group (a column such as ``region_id``,
  polygons, or the subareas of ``cat_bonds``) or the whole exposure set. A
  unit's loss sums ``imp_mat`` over its points, and its payout maps back to
  them by value share.
* **Index:** :class:`.ParametricIndex`, one value per event and unit with its
  :class:`.IndexDefinition`, on any hazard: a statistic of a ``Hazard``'s
  footprint over each unit (:meth:`~.ParametricIndex.from_hazard`), the
  cat-in-a-circle category from tracks
  (:meth:`~.ParametricIndex.from_cat_in_circle`), a user's table
  (:meth:`~.ParametricIndex.from_table`), or a cover's index on another
  hazard set (:meth:`~.ParametricIndex.from_definition`). Events are matched
  on ``event_id`` (:meth:`~.ParametricIndex.matched_to`). An index holds no
  years.
* **Outcomes:** :class:`.OutcomeSample` joins an index, an ``Impact``'s loss
  per unit and a stated basis, which says what an outcome is:
  :class:`.EventBasis` (one outcome per event, weighted by its frequency, for
  any hazard), or one simulated year on :class:`.CatalogueYears` (a year
  label per event, from the dates, the event names or an array) or
  :class:`.YearSets` (years sampled from the frequencies with a seeded
  generator, as ``climada.util.yearsets`` does, or a given
  ``sampling_vect``). A route that doesn't fit the events raises
  :class:`.YearRouteError`.
* **Splits:** :func:`.holdout` and :func:`.year_block_kfold` split a sample by
  blocks: contiguous years on catalogue years, pools of events or of
  ensemble parents on the other routes, with no event or parent on both
  sides (:func:`.check_no_leakage`, :class:`.LeakageError`). Every part of a
  sample, a side or :meth:`~.OutcomeSample.unit`, keeps its source token
  (:attr:`~.OutcomeSample.source`), so an out-of-sample evaluation of a
  design is always checked against the sample it was cut from. Parent
  storms read from the names of CLIMADA's perturbed tracks
  (:attr:`~.OutcomeSample.parents`) are checked across hazard sets too, such
  as a present and a future set of one country.
* **Objectives:** :class:`.WeightedCVaR`, the tail objective
  ``alpha CVaR_beta(U) + (1 - alpha) E[U]`` of the uncovered loss ``U``,
  next to :class:`.ExpectedUncovered` (``E[U]``) and :class:`.MSE`. The first
  two are linear programs solved with scipy's HiGHS; the mean squared error
  is a quadratic program solved with scipy's ``SLSQP``, within stated
  limits. A :class:`.UserObjective` is any function of the outcomes' losses,
  payouts and weights, evaluated on a grid.
* **Calibration:** :func:`.calibrate` chooses the free amounts of a template
  cover on a design sample, with ``scope="individual"`` (one design problem
  per unit) or ``scope="joint"`` (one problem on the owner's total loss, so
  the units' payouts can offset each other's shortfalls), and returns a
  :class:`.CalibrationResult`, which holds the resolved cover and how it was
  found. Pricing: :class:`.PurePremium` or :class:`.CostOfCapitalLoad`, under
  a premium :class:`.Budget`; options in :class:`.CalibrationOptions`. Free
  shape parameters (such as a free attachment), payouts that are not linear
  in their amounts and user objectives take a stated grid, in the
  individual scope. :func:`.calibrate_frontier` calibrates at several
  budgets: the budget frontier, a measurement with no best point.
* **Evaluation:** :func:`.evaluate` and :func:`.compare` give the metrics of
  any cover, designed or given, on a sample, in or out of sample, as a
  pandas table: coverage, tail coverage, basis risk, VaR and CVaR of the
  uncovered loss and of the net loss, return periods and more, per unit
  and for the owner's total. Every row carries its setup (basis and route,
  split, index, pricing, scope), which :func:`.setup_text` renders. The
  thresholds are always stated, in :class:`.MetricOptions`.
  :func:`.payout_impact` and :func:`.uncovered_impact` give the payouts and
  the uncovered losses as CLIMADA ``Impact`` objects, so that
  ``calc_freq_curve``, ``local_exceedance_impact`` and ``write_hdf5`` work
  on them. Plots: :func:`.plot_payout`, :func:`.plot_payout_vs_loss` and
  :func:`.plot_exceedance`.
* **Basis risk:** it comes from a stated source. The structural ones lead:
  an index that is not the footprint at the assets (cat-in-a-circle,
  another or a coarser hazard set, a table of readings) and units of
  several points. :func:`.index_diagnostics` measures, per unit, how far the
  index determines the loss; :func:`.calibrate` and :func:`.evaluate` run it
  and warn where it does almost one-to-one, so that basis risk would read
  close to zero, and the tables mark such units (``comonotone_index``).
  Impact-function draws come second (:func:`.impact_function_draws`,
  :meth:`.OutcomeSample.from_impact_draws`): they widen the uncovered loss
  but don't change which events pay or lose.
* **Scenarios:** :class:`.ScenarioSet` maps labels to scenarios, other
  hazard sets and possibly other exposures (a future climate from the Data
  API, a set scaled by ``apply_climate_scenario_knu``), all on one route.
  A label may hold several samples, which form one scenario as an
  equal-weight mixture: the windows of a pathway or the members of an
  ensemble. :func:`.evaluate_scenarios` evaluates one cover on every
  scenario, with the label in the ``scenario`` column, and
  :func:`.calibrate_per_scenario` designs one per scenario
  (:func:`.designs_table`). The tables rank no scenario and no cover.

Working with ``cat_bonds``
--------------------------
``climada_petals.engine.cat_bonds`` designs parametric cat bonds for a
sovereign: one linear payout per subarea of a country's exposure, with its
two triggers fitted by COBYLA, then the pooling of countries into bonds and
the bonds' pricing. This module designs covers for one risk owner, with any
payout family and objective. The two fit together:

* **Units.** :meth:`.InsuredUnits.from_subareas` makes one unit of each
  subarea of a ``Subareas``, named by its ``subarea_letter``. A
  ``Subareas`` maps no point to a subarea, so the points are joined to its
  polygons, and the subareas must cover every point of positive value.
* **Index.** ``index_stat`` keeps its name and values: ``"mean"`` or a
  percentile, in :meth:`.ParametricIndex.from_hazard`. ``cat_bonds`` takes
  the statistic over the hazard's centroids inside a subarea's polygon,
  this module over the centroids of the unit's points; the two agree when
  the centroids are the exposure points. Events are matched on
  ``event_id`` here, and by position in ``cat_bonds``.
* **Payout.** ``LinearPayout.from_triggers(min_trig, max_trig, max_pay)``
  pays what ``calc_payout(min_trig, max_trig, haz_int, max_pay)`` pays on
  a float index. On an integer-typed index ``calc_payout`` truncates the
  payouts, and this module doesn't. With ``min_trig > max_trig``,
  ``calc_payout`` pays ``max_pay`` from ``max_trig`` on, as
  ``BinaryPayout(max_trig, max_pay)`` does.
* **Fit.** ``cat_bonds`` fits each subarea's triggers, unconstrained, to
  the sum over the events of the squared error between the subarea's loss
  and its payout, with ``max_pay`` fixed at the smaller of the subarea's
  largest loss and the principal. :class:`.MSE` on :class:`.EventBasis`
  weights each event by its frequency, so with equal frequencies ``f`` it
  is ``f`` times that sum. A template
  ``LinearPayout.from_triggers(Free(..., grid=...), Free(..., grid=...),
  max_pay)``, calibrated with ``scope="individual"``, searches the
  triggers on the stated grids; with a free limit under
  ``Limit.largest_loss()`` it also solves for the payout at the
  exhaustion, up to the unit's largest loss.
* **Bonds.** Bond simulation, pooling and pricing stay in ``cat_bonds``,
  and have no counterpart here.

The terms of ``cat_bonds``, and their counterparts here:

* A subarea and its ``subarea_letter``: an insured unit and its
  ``unit_id``.
* ``min_trig``, ``max_trig`` and ``max_pay``: the ``attachment`` and the
  ``exhaustion``, in index units, and the ``limit``, in payout units, of a
  :class:`.LinearPayout`.
* ``attachment_point`` and ``exhaustion_point``: levels of loss that set a
  bond's principal, with no counterpart. An attachment here is an index
  threshold.
* ``principal``: it caps each subarea's ``max_pay`` and each event's total
  payout. A :class:`.Limit` caps one unit's payout per event, and nothing
  here caps the total over the units.
* "Basis risk": there, the event-level squared error that the fit
  minimises, :class:`.MSE` here. Here, basis risk is a family of metrics of
  :func:`.evaluate`.
* "Pooling": there, grouping countries into one bond, with no counterpart.
  ``scope="joint"`` calibrates the units of one owner together.
* ``SubareaCalculations.create_pay_vs_dam`` and its ``pay_vs_dam`` table:
  :func:`.calibrate`, which returns a :class:`.CalibrationResult`, and
  :func:`.payout_impact`, the payouts per event and exposure point.

The module depends only on CLIMADA, numpy, scipy, pandas and matplotlib.

References
----------
* Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional
  value-at-risk. *Journal of Risk*, 2(3), 21-41.
  https://doi.org/10.21314/JOR.2000.038. The CVaR as a minimum over a
  threshold, which keeps the weighted-CVaR objective, and a premium budget
  under a cost-of-capital load, linear
  (:mod:`~climada_petals.engine.parametric.objectives`,
  :mod:`~climada_petals.engine.parametric.solvers`).
* Rockafellar, R. T. and Uryasev, S. (2002). Conditional value-at-risk for
  general loss distributions. *Journal of Banking & Finance*, 26(7),
  1443-1471. https://doi.org/10.1016/S0378-4266(02)00271-6. The CVaR of a
  distribution with atoms, as the metrics, the pricing and the objectives
  compute it (:mod:`~climada_petals.engine.parametric.riskmeasures`).
* Steinmann, C. B., Guillod, B. P., Fairless, C. and Bresch, D. N. (2023).
  A generalized framework for designing open-source natural hazard
  parametric insurance. *Environment Systems and Decisions*, 43(4),
  555-568. https://doi.org/10.1007/s10669-023-09934-x. Parametric
  insurance designed with CLIMADA, and the cat-in-a-circle index from
  tropical cyclone tracks
  (:mod:`~climada_petals.engine.parametric.index_tracks`).
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
