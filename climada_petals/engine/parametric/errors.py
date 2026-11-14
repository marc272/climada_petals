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
    "CoverDefinitionError",
    "EventAlignmentError",
    "ImpactInputError",
    "MissingDependencyError",
    "ParametricError",
    "ParametricInputError",
    "UnitMappingError",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
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
    """A design problem could not be solved.

    Parameters
    ----------
    message : str
        What went wrong.
    problem_id : str
        The design problem: a unit id, or ``"portfolio"``.
    status : str, optional
        The solver's status, if it returned one.
    """

    def __init__(
        self, message: str, *, problem_id: str, status: str | None = None
    ) -> None:
        super().__init__(f"design problem {problem_id!r}: {message}")
        self.problem_id = problem_id
        self.status = status


class MissingDependencyError(ParametricError, ImportError):
    """An optional dependency that the requested feature needs is missing."""
