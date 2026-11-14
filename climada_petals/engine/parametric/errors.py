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

Errors of the parametric insurance module.

Every error derives from :class:`ParametricError`. Input errors are also
:class:`ValueError` and a missing optional dependency is also an
:class:`ImportError`, so callers that catch the built-in types keep working.
"""

from __future__ import annotations

__all__ = [
    "CalibrationError",
    "CalibrationLimitError",
    "CoverDefinitionError",
    "EventAlignmentError",
    "ImpactInputError",
    "InfeasibleCalibrationError",
    "LeakageError",
    "MissingDependencyError",
    "ParametricError",
    "ParametricInputError",
    "ScenarioError",
    "UnboundedCalibrationError",
    "UnitMappingError",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "YearRouteError",
]


class ParametricError(Exception):
    """Base class of the module's errors."""


class ParametricInputError(ParametricError, ValueError):
    """An argument is invalid, or two inputs do not match.

    Examples: index values that are not finite, an ``Impact`` whose events
    are not the index's events, or frequencies that do not fit the annual
    basis.
    """


class UnitMappingError(ParametricInputError):
    """Exposure points don't map onto insured units as a cover needs.

    Examples: a unit with no point or with zero total value, a point in two
    units, a polygon that holds no exposure point, or a point whose assigned
    hazard centroid is not a column of the hazard's intensity matrix.
    """


class EventAlignmentError(ParametricInputError):
    """An index and an ``Impact`` don't hold the same events.

    Events are matched on ``event_id`` (or on a stated key), never by
    position.
    """


class ImpactInputError(ParametricInputError):
    """A CLIMADA ``Impact``, or what it was computed from, can't be used as it is.

    Examples: an ``Impact`` without ``imp_mat`` (compute it with
    ``save_mat=True``), a ``frequency_unit`` other than ``1/year``, exposures
    with ``cover`` or ``deductible`` (their losses are already net of an
    indemnity layer), or negative losses.
    """


class YearRouteError(ParametricInputError):
    """An annual route doesn't fit the events.

    Examples: catalogue years whose events don't all have the frequency
    ``1 / N`` (a set scaled by ``apply_climate_scenario_knu``, or dates that
    don't carry the years), an event name that doesn't match its pattern, or
    a ``sampling_vect`` that holds event ids instead of positions.
    """


class LeakageError(ParametricInputError):
    """Two samples that must be disjoint share an event or a block.

    Examples: a training and a test sample with a common event, or with
    members of one parent storm on both sides, or a split of catalogue years
    that no contiguous cut can make without splitting a block.
    """


class ScenarioError(ParametricInputError):
    """Scenarios that a comparison or a mixture puts together don't fit.

    Examples: two scenarios of a set on other routes (the event basis and
    catalogue years, say), on other insured units or index definitions, or
    the members of one scenario with other exposures, other numbers of
    draws, or from other splits.
    """


class CoverDefinitionError(ParametricError, ValueError):
    """A payout, a parameter or a cover is invalid.

    Examples: breakpoints out of order, a negative amount, a free threshold
    without a grid, a payout whose ``tail`` is not its index's, or a cover
    document with an unknown field.
    """


class UnknownPayoutKindError(CoverDefinitionError):
    """A cover document names a payout ``kind`` that no class is registered for.

    A user-defined family is registered with
    :func:`~climada_petals.engine.parametric.payout.register_payout`.
    """


class UnresolvedParameterError(ParametricError, ValueError):
    """A template is used where a resolved payout or cover is needed.

    A template has a free parameter, or a limit stated as a rule, such as
    the largest loss on a design sample. ``with_values`` or the calibration
    resolves it.
    """


class CalibrationError(ParametricError, RuntimeError):
    """A design problem could not be solved, or its solution failed a check.

    The base of :class:`InfeasibleCalibrationError`,
    :class:`UnboundedCalibrationError` and :class:`CalibrationLimitError`.
    Raised itself when the solver reports another failure, such as numerical
    trouble, or when the solution breaks a constraint or doesn't reproduce
    the solver's objective value.

    Parameters
    ----------
    message : str
        What went wrong.
    problem_id : str
        The design problem: a unit id, or ``"joint"``.
    status : str, optional
        The solver's status, if it returned one.
    solver_message : str, optional
        The solver's own message, if it returned one.
    """

    def __init__(
        self,
        message: str,
        *,
        problem_id: str,
        status: str | None = None,
        solver_message: str | None = None,
    ) -> None:
        super().__init__(f"design problem {problem_id!r}: {message}")
        self.problem_id = problem_id
        self.status = status
        self.solver_message = solver_message


class InfeasibleCalibrationError(CalibrationError):
    """The solver found no amounts that meet every constraint.

    For example, fixed amounts whose premium is above the budget.
    """


class UnboundedCalibrationError(CalibrationError):
    """The solver found the objective unbounded below."""


class CalibrationLimitError(CalibrationError):
    """A stated limit of the calibration was reached, such as ``time_limit``."""


class MissingDependencyError(ParametricError, ImportError):
    """An optional dependency that the requested feature needs is missing."""
