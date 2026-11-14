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

Calibrate a parametric cover: choose the payout levels of every unit.

:func:`calibrate` fits a :class:`~climada_petals.engine.parametric.payout.StepPayout` to
the losses of an :class:`~climada_petals.engine.parametric.sample.OutcomeSample`: an
index, the loss of a CLIMADA ``Impact`` per event and unit, and a basis. The
**scope** says which units share a design problem:

``"individual"``
    One design problem per unit (an exposure point or a group of them,
    :class:`~climada_petals.engine.parametric.units.InsuredUnits`): its own
    objective on its own loss and payout, its own budget
    ``premium_j <= budget * E[L_j]`` and its own premium.
``"portfolio"``
    One joint design problem for one risk owner: one payout function per
    unit, but the objective is on the owner's **total** loss and payout,
    ``L_e = sum_j L_{e,j}`` and ``P_e = sum_j P_{e,j}``, under one budget
    on the total premium, ``premium <= budget * E[sum_j L_j]``. Within an
    event, an over-payment at one unit offsets a shortfall at another
    (cross-coverage).

With the same ``budget``, the individual budgets add up to the portfolio
budget. The individual designs taken together satisfy the portfolio
problem's constraints (under a cost-of-capital load too, because the loaded
premium of a total is at most the sum of the loaded premiums), so on the
calibration sample the portfolio optimum is at most their objective on the
total. That is a property of the optimisation; nothing here says which scope
gives the better cover on other samples.

**Per-unit levels of a portfolio cover are not unique.** The objective of a
joint design problem depends on the units' payouts only through their total
per event, so other feasible levels with the same totals are optimal too,
and the linear program can have several optimal solutions besides. The
levels a portfolio cover holds are the one optimal split the solver
returned. Read them, and per-unit metrics of a portfolio cover, as that.

The objective, the premium and the budget all use the sample's outcome
measure (:mod:`~climada_petals.engine.parametric.measure`). On an annual route the
events of a year are summed: the CVaR is a CVaR over years. On the event
basis the measure is read as a probability over one year, so the
frequencies must sum to at most 1 there
(:meth:`~climada_petals.engine.parametric.measure.OutcomeMeasure.as_probability`).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from ._checks import check_real, finite_array, read_only
from .errors import ParametricInputError
from .index import ParametricIndex
from .measure import OutcomeMeasure
from .objectives import ExpectedUncovered, MeanSquaredError, Objective, WeightedCVaR
from .parameters import Free
from .payout import StepPayout
from .pricing import CostOfCapitalLoad, Pricing, PurePremium
from .sample import OutcomeSample
from .solvers import DesignData, SolverInfo, solve_design
from .units import InsuredUnits

__all__ = [
    "PORTFOLIO",
    "SCOPES",
    "CalibrationResult",
    "CoverProblem",
    "Scope",
    "calibrate",
]

Scope = Literal["individual", "portfolio"]
"""Which units share a design problem."""

SCOPES: Final[tuple[str, ...]] = get_args(Scope)
"""The values of :data:`Scope`: ``("individual", "portfolio")``."""

PORTFOLIO: Final[str] = "portfolio"
"""``problem_id`` of the portfolio scope's single design problem."""


@dataclass(frozen=True)
class CoverProblem:
    """The solution of one design problem of a cover.

    Attributes
    ----------
    problem_id : str
        The unit id (individual scope) or ``"portfolio"``.
    unit_id : tuple of hashable
        The units of the problem.
    objective_value : float
        The optimal objective value, in loss units (squared for the mean
        squared error), on the calibration sample.
    premium : float
        The premium of the problem's payout under the cover's pricing, on
        the calibration sample.
    expected_loss : float
        ``E[L]`` of the problem's total loss, on the calibration sample.
    budget_bound : float or None
        ``budget * expected_loss``, or None without a budget.
    solver : SolverInfo
        Solver, status and timing.
    diagnostics : mapping of str to float or bool
        ``zero_loss_probability``: ``P(L_o = 0)`` over the outcomes.
        For the weighted CVaR also ``degenerate_tail`` (``beta`` at most
        ``P(L_o = 0)``, so the design minimises ``E[U]``) and
        ``tail_within_one_outcome`` (``1 - beta`` below the smallest
        outcome probability, so the CVaR is the worst outcome alone).
    """

    problem_id: str
    unit_id: tuple[Hashable, ...]
    objective_value: float
    premium: float
    expected_loss: float
    budget_bound: float | None
    solver: SolverInfo
    diagnostics: Mapping[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    """A calibration's result: the payout levels of every unit and how they were found.

    Attributes
    ----------
    payout : StepPayout
        The step payout template of every unit: its breakpoints. The levels
        are in ``levels``.
    objective : Objective
        The objective the levels minimise.
    pricing : Pricing
        The pricing of the premium and the budget.
    scope : {"individual", "portfolio"}
        Which units shared a design problem.
    basis : {"annual", "event"}
        The basis of the calibration sample: of the objective, the premium
        and the budget.
    budget : float or None
        ``kappa`` in ``premium <= kappa E[L]``, or None.
    unit_id : tuple of hashable
        The units, in the index's order.
    levels : numpy.ndarray
        ``theta``, the payout levels, shape ``(n_units, n_steps)``, loss
        units. Not unique per unit in the portfolio scope (module
        docstring).
    problems : tuple of CoverProblem
        One per design problem: one per unit, or one for the portfolio.
    calibration_years : numpy.ndarray of int or None
        The simulation years of a calibration sample on catalogue years;
        None on the other routes.
    """

    payout: StepPayout
    objective: Objective
    pricing: Pricing
    scope: str
    basis: str
    budget: float | None
    unit_id: tuple[Hashable, ...]
    levels: NDArray[np.float64]
    problems: tuple[CoverProblem, ...]
    calibration_years: NDArray[np.int64] | None

    def __post_init__(self) -> None:
        """Validate the shapes and store read-only copies."""
        levels = finite_array(self.levels, "levels", 2)
        if levels.shape != (len(self.unit_id), self.payout.n_steps):
            raise ParametricInputError(
                f"levels must have shape ({len(self.unit_id)}, "
                f"{self.payout.n_steps}), got {levels.shape}"
            )
        if self.scope not in SCOPES:
            raise ParametricInputError(
                f"scope must be one of {list(SCOPES)}, got {self.scope!r}"
            )
        object.__setattr__(self, "levels", read_only(levels))
        object.__setattr__(self, "unit_id", tuple(self.unit_id))
        object.__setattr__(self, "problems", tuple(self.problems))
        if self.calibration_years is not None:
            object.__setattr__(
                self, "calibration_years", read_only(np.asarray(self.calibration_years))
            )

    @property
    def premium(self) -> float:
        """Sum of the design problems' premiums, on the calibration sample.

        In the portfolio scope this is the one problem's premium. In the
        individual scope it is the price of buying each unit's cover
        separately; under a cost-of-capital load the premium of the combined
        payout can be lower, because the load is sub-additive.
        """
        return float(sum(problem.premium for problem in self.problems))

    def payout_matrix(self, index: ParametricIndex) -> NDArray[np.float64]:
        """Return the payout of every event and unit of ``index``.

        Parameters
        ----------
        index : ParametricIndex
            An index with the cover's units, e.g. a sample's ``index``, or
            one on another hazard set.

        Returns
        -------
        numpy.ndarray
            Shape ``(index.n_events, n_units)``, loss units.

        Raises
        ------
        ParametricInputError
            If the index's units are not the cover's.
        """
        if tuple(index.unit_id) != self.unit_id:
            raise ParametricInputError(
                "the index's units are not the cover's: "
                f"{list(index.unit_id)[:5]} vs {list(self.unit_id)[:5]}"
            )
        return _paid(self.payout, index.values, self.levels)

    def payout_impact(
        self, index: ParametricIndex, impact: Any, *, units: InsuredUnits
    ) -> Any:
        """Return the payouts as a CLIMADA ``Impact``, aligned with ``impact``.

        Parameters
        ----------
        index : ParametricIndex
            The index the payouts are computed from, with every event of
            ``impact``.
        impact : climada.engine.Impact
            The loss impact of the same events, on the units' exposure
            points, with ``imp_mat``. Its events, dates, frequencies,
            coordinates and unit are copied.
        units : InsuredUnits
            The insured units, with the cover's ``unit_id`` in order.

        Returns
        -------
        climada.engine.Impact
            ``imp_mat`` holds the payout per event and exposure point, a
            unit's payout shared among its points by value share
            (:meth:`~climada_petals.engine.parametric.units.InsuredUnits.to_points`),
            in the impact's event order; ``at_event`` is its sum over the
            points, ``eai_exp`` the expected annual payout per point and
            ``aai_agg`` their sum, so that CLIMADA's tools
            (``calc_freq_curve``, ``impact_per_year``, the plots) work on
            the payouts as on the loss.

        Raises
        ------
        ParametricInputError
            If ``units`` are not the cover's.
        ImpactInputError
            If ``impact`` is not a loss on the units' exposure points
            (:meth:`~climada_petals.engine.parametric.units.InsuredUnits.check_impact`).
        EventAlignmentError
            If the impact's events are not the index's.
        """
        if not isinstance(units, InsuredUnits) or tuple(units.unit_id) != self.unit_id:
            raise ParametricInputError(
                "units must be the InsuredUnits of the cover's unit_id, in order"
            )
        units.check_impact(impact)
        payout = np.empty((index.n_events, len(self.unit_id)))
        payout[index.rows_in(impact)] = self.payout_matrix(index)
        points = units.to_points(payout)
        frequency = np.asarray(impact.frequency, dtype=float)
        eai_exp = np.asarray(points.T @ frequency, dtype=float)
        impact_class = type(impact)
        return impact_class(
            event_id=np.asarray(impact.event_id),
            event_name=list(impact.event_name),
            date=np.asarray(impact.date),
            frequency=frequency,
            frequency_unit=impact.frequency_unit,
            coord_exp=np.asarray(impact.coord_exp),
            crs=impact.crs,
            eai_exp=eai_exp,
            at_event=np.asarray(points.sum(axis=1), dtype=float).reshape(-1),
            tot_value=impact.tot_value,
            aai_agg=float(eai_exp.sum()),
            unit=impact.unit,
            imp_mat=points,
            haz_type=impact.haz_type,
        )

    def levels_frame(self) -> pd.DataFrame:
        """Return the payout levels as a table.

        Returns
        -------
        pandas.DataFrame
            One row per unit and step: ``unit_id``, ``step``, ``lower`` and
            ``upper`` (the step's index range, ``[lower, upper)``; ``inf``
            for the last step, which has no upper edge) and ``level`` (the
            payout, loss units).
        """
        edges = np.append(np.asarray(self.payout.breakpoints, dtype=float), np.inf)
        n_steps = self.payout.n_steps
        return pd.DataFrame(
            {
                "unit_id": np.repeat(np.array(self.unit_id, dtype=object), n_steps),
                "step": np.tile(np.arange(n_steps), len(self.unit_id)),
                "lower": np.tile(edges[:-1], len(self.unit_id)),
                "upper": np.tile(edges[1:], len(self.unit_id)),
                "level": self.levels.reshape(-1),
            }
        )

    def summary(self) -> pd.DataFrame:
        """Return one row per design problem: objective, premium, budget, solver.

        Returns
        -------
        pandas.DataFrame
            Columns ``problem_id``, ``objective``, ``objective_value``,
            ``premium``, ``expected_loss``, ``budget_bound``, ``solver``,
            ``status`` and the diagnostics.
        """
        rows = []
        for problem in self.problems:
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "objective": self.objective.name,
                    "objective_value": problem.objective_value,
                    "premium": problem.premium,
                    "expected_loss": problem.expected_loss,
                    "budget_bound": problem.budget_bound,
                    "solver": problem.solver.solver,
                    "status": problem.solver.status,
                    **problem.diagnostics,
                }
            )
        return pd.DataFrame(rows)


def _check_objective(objective: object) -> Objective:
    """Return ``objective`` if it is one of the module's objectives."""
    if not isinstance(objective, MeanSquaredError | ExpectedUncovered | WeightedCVaR):
        raise ParametricInputError(
            "objective must be MeanSquaredError(), ExpectedUncovered() or "
            f"WeightedCVaR(alpha, beta), got {objective!r}"
        )
    return objective


def _check_pricing(pricing: object) -> Pricing:
    """Return ``pricing`` if it is one of the module's pricings."""
    if not isinstance(pricing, PurePremium | CostOfCapitalLoad):
        raise ParametricInputError(
            "pricing must be PurePremium() or CostOfCapitalLoad(rate, q), "
            f"got {pricing!r}"
        )
    return pricing


def _diagnostics(
    objective: Objective, total_loss: NDArray[np.float64], measure: OutcomeMeasure
) -> dict[str, float | bool]:
    """Record when the tail objective degenerates (``CoverProblem.diagnostics``)."""
    outcome_loss = measure.aggregate(total_loss)
    zero = float(measure.probability[outcome_loss <= 0.0].sum())
    out: dict[str, float | bool] = {"zero_loss_probability": zero}
    if isinstance(objective, WeightedCVaR) and objective.alpha > 0.0:
        smallest = float(measure.probability[measure.probability > 0].min())
        out["degenerate_tail"] = objective.beta <= zero
        out["tail_within_one_outcome"] = 1.0 - objective.beta < smallest
    return out


def _problems(
    scope: str, unit_id: tuple[Hashable, ...]
) -> list[tuple[str, NDArray[np.intp]]]:
    """Return ``(problem_id, unit positions)`` for each design problem."""
    if scope == PORTFOLIO:
        return [(PORTFOLIO, np.arange(len(unit_id)))]
    return [(str(unit), np.array([j])) for j, unit in enumerate(unit_id)]


def _check_arguments(
    sample: object,
    payout: object,
    objective: object,
    pricing: object,
    *,
    scope: object,
) -> tuple[Objective, Pricing]:
    """Check the arguments of :func:`calibrate`; return the objective and pricing.

    Raises
    ------
    ParametricInputError
        If an argument is not one the module accepts.
    """
    if not isinstance(sample, OutcomeSample):
        raise ParametricInputError(
            f"sample must be an OutcomeSample, got {type(sample).__name__}"
        )
    if not isinstance(payout, StepPayout):
        raise ParametricInputError(f"payout must be a StepPayout, got {payout!r}")
    if (
        payout.tail != "upper"
        or not payout.monotone
        or any(isinstance(edge, Free) for edge in payout.breakpoints)
        or any(level != Free() for level in payout.levels)
    ):
        raise ParametricInputError(
            "calibrate takes a StepPayout with fixed breakpoints, tail='upper', "
            "monotone=True and every level Free(), such as "
            f"StepPayout.uniform(attachment, exhaustion, n_steps); got {payout!r}"
        )
    checked_objective = _check_objective(objective)
    checked_pricing = _check_pricing(pricing)
    if scope not in SCOPES:
        raise ParametricInputError(
            f"scope must be one of {list(SCOPES)}, got {scope!r}"
        )
    return checked_objective, checked_pricing


@dataclass(frozen=True, eq=False)
class _Design:
    """What every design problem of one calibration shares."""

    payout: StepPayout
    objective: Objective
    pricing: Pricing
    budget: float | None
    measure: OutcomeMeasure


def _cover_problem(
    design: _Design, data: DesignData, unit_id: tuple[Hashable, ...]
) -> tuple[CoverProblem, NDArray[np.float64]]:
    """Solve one design problem; return its record and its levels."""
    solution = solve_design(
        data, design.payout, design.objective, design.pricing, design.budget
    )
    paid = _paid(design.payout, data.index, solution.levels).sum(axis=1)
    expected_loss = design.measure.expectation(data.total_loss)
    diagnostics = _diagnostics(design.objective, data.total_loss, design.measure)
    problem = CoverProblem(
        problem_id=data.problem_id,
        unit_id=unit_id,
        objective_value=solution.objective_value,
        premium=design.pricing.premium(paid, design.measure),
        expected_loss=expected_loss,
        budget_bound=None if design.budget is None else design.budget * expected_loss,
        solver=solution.info,
        diagnostics=diagnostics,
    )
    return problem, solution.levels


def _solve_problems(
    design: _Design, scope: str, sample: OutcomeSample
) -> tuple[NDArray[np.float64], list[CoverProblem]]:
    """Solve every design problem of the scope; return the levels and the records."""
    levels = np.zeros((sample.n_units, design.payout.n_steps))
    problems: list[CoverProblem] = []
    index = sample.index_values
    loss = sample.loss.toarray()
    for problem_id, units in _problems(scope, sample.unit_id):
        data = DesignData(
            problem_id=problem_id,
            index=index[:, units],
            loss=loss[:, units],
            measure=design.measure,
        )
        problem, levels[units] = _cover_problem(
            design, data, tuple(sample.unit_id[j] for j in units)
        )
        problems.append(problem)
        _log(problem, design.objective)
    return levels, problems


def calibrate(
    sample: OutcomeSample,
    *,
    payout: StepPayout,
    objective: Objective,
    scope: str = "individual",
    pricing: Pricing = PurePremium(),  # frozen, so safe to share
    budget: float | None = None,
) -> CalibrationResult:
    """Fit the payout levels of every unit (module docstring).

    Parameters
    ----------
    sample : OutcomeSample
        The calibration sample: the index, the loss to cover and the basis,
        e.g. the training side of
        :func:`~climada_petals.engine.parametric.sample.holdout`.
    payout : StepPayout
        The payout template of every unit: fixed breakpoints, ``tail="upper"``,
        ``monotone=True`` and every level ``Free()``, as
        :meth:`StepPayout.uniform` gives it.
    objective : MeanSquaredError, ExpectedUncovered or WeightedCVaR
        What the levels minimise. The mean squared error needs cvxpy.
    scope : {"individual", "portfolio"}, default "individual"
        One design problem per unit, or one joint problem on the total.
    pricing : PurePremium or CostOfCapitalLoad, default PurePremium()
        The premium the budget constrains and the cover records.
    budget : float, optional
        ``kappa >= 0`` in ``premium <= kappa * E[L]``, per design problem.
        Default: no budget.

    Returns
    -------
    CalibrationResult
        The levels, one :class:`CoverProblem` per design problem, and the
        setup.

    Raises
    ------
    ParametricInputError
        If an argument is invalid, or the sample is on the event basis with
        frequencies that sum to more than 1 per year.
    CalibrationError
        If a design problem is not solved to optimality. No problem is
        skipped.
    MissingDependencyError
        If the objective is the mean squared error and cvxpy is missing.

    Examples
    --------
    A tail objective calibrated jointly across the units:

    >>> train, test = holdout(sample, train_fraction=0.7)  # doctest: +SKIP
    >>> cover = calibrate(  # doctest: +SKIP
    ...     train,
    ...     payout=StepPayout.uniform(attachment=1, exhaustion=6, n_steps=5),
    ...     objective=WeightedCVaR(alpha=0.9, beta=0.99),
    ...     scope="portfolio",
    ...     budget=1.0,
    ... )
    """
    objective, pricing = _check_arguments(
        sample, payout, objective, pricing, scope=scope
    )
    kappa = None if budget is None else check_real("budget", budget, low=0.0)
    measure = sample.measure.as_probability()
    levels, problems = _solve_problems(
        _Design(payout, objective, pricing, kappa, measure), scope, sample
    )
    return CalibrationResult(
        payout=payout,
        objective=objective,
        pricing=pricing,
        scope=scope,
        basis=measure.basis,
        budget=kappa,
        unit_id=sample.unit_id,
        levels=levels,
        problems=tuple(problems),
        calibration_years=sample.years,
    )


def _paid(
    payout: StepPayout, index: NDArray[np.float64], levels: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``out[e, j] = basis(index[:, j]) @ levels[j]``, the payout per unit."""
    return np.column_stack(
        [payout.basis(index[:, j]) @ levels[j] for j in range(index.shape[1])]
    )


def _log(problem: CoverProblem, objective: Objective) -> None:
    """Log a solved design problem, and warn when its tail degenerates."""
    LOGGER.info(
        "design problem %r: %s = %.6g, premium %.6g (%s, %.2f s)",
        problem.problem_id,
        objective.name,
        problem.objective_value,
        problem.premium,
        problem.solver.solver,
        problem.solver.seconds,
    )
    if problem.diagnostics.get("degenerate_tail"):
        LOGGER.warning(
            "design problem %r: beta = %s is at most P(L = 0) = %.4g, so the "
            "weighted-CVaR design minimises E[U] only",
            problem.problem_id,
            getattr(objective, "beta", None),
            problem.diagnostics["zero_loss_probability"],
        )
    if problem.diagnostics.get("tail_within_one_outcome"):
        LOGGER.warning(
            "design problem %r: 1 - beta is below the smallest outcome "
            "probability, so the CVaR is the single worst outcome",
            problem.problem_id,
        )
