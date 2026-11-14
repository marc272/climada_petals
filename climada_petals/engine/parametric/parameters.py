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

Parameters of a payout: a fixed number, or free within bounds.

Every parameter of a payout family is either a number, which fixes it, or
:class:`Free`, which leaves it to the calibration within stated bounds. A
payout or a cover with a free parameter is a template; one without is
resolved.

* **Amounts** are in payout units: a step's level, a linear payout's limit,
  a binary payout's amount. A free amount has ``lower >= 0`` and may have
  ``upper=None``; the cover's
  :class:`~climada_petals.engine.parametric.cover.Limit` or the premium
  budget then bounds it.
* **Shape parameters** are in index units: breakpoints, knots, an
  attachment, an exhaustion or a trigger. A free shape parameter needs
  finite bounds and a grid. The calibration tries the points of the grid,
  with an exact solve for the amounts at each.

In a document (``to_dict``), a fixed parameter is a number and a free one is
``{"free": {"lower": ..., "upper": ..., "grid": ...}}``.

:class:`AmountConstraints` holds the linear constraints on a payout's
amounts that the calibration's exact routes use. The functions without an
entry in ``__all__`` are the checks the payout families and the covers
share.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._checks import read_only
from .errors import CoverDefinitionError

__all__ = [
    "AmountConstraints",
    "Free",
    "Parameter",
]


def real_number(name: str, value: object) -> float:
    """Return ``value`` as a float if it is a finite real number.

    Parameters
    ----------
    name : str
        Parameter name, for the message.
    value : object
        The value. ``bool`` is refused.

    Returns
    -------
    float
        ``value``.

    Raises
    ------
    CoverDefinitionError
        If ``value`` is not a finite real number.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.number
    ):
        raise CoverDefinitionError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise CoverDefinitionError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class Free:
    """A parameter left to the calibration, within bounds.

    Parameters
    ----------
    lower : float, default 0.0
        Smallest value. An amount needs ``lower >= 0``.
    upper : float, optional
        Largest value, at least ``lower``. None: no bound; allowed for an
        amount only.
    grid : int or sequence of float, optional, keyword-only
        The candidates of a shape parameter: explicit values, strictly
        increasing and within the bounds, or a number of uniform points
        from ``lower`` to ``upper`` (at least 2, and ``upper`` must be
        given). Stored as explicit values. Required for a free shape
        parameter; an amount takes one only for a route that searches every
        free parameter on a grid.

    Raises
    ------
    CoverDefinitionError
        If a bound is not a finite number, ``upper < lower``, or the grid is
        invalid.

    Examples
    --------
    A free level, and a free attachment tried at 25, 30 and 35 m/s:

    >>> Free()
    Free(lower=0.0, upper=None, grid=None)
    >>> Free(25.0, 35.0, grid=3).grid
    (25.0, 30.0, 35.0)
    """

    lower: float = 0.0
    upper: float | None = None
    grid: tuple[float, ...] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the bounds and the grid; store floats and a tuple."""
        lower = real_number("Free.lower", self.lower)
        upper = None if self.upper is None else real_number("Free.upper", self.upper)
        if upper is not None and upper < lower:
            raise CoverDefinitionError(
                f"Free.upper must be at least Free.lower, got {upper} < {lower}"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "grid", _grid(self.grid, lower, upper))

    def check(self, name: str, value: object) -> float:
        """Return ``value`` as a float if it lies within the bounds.

        Parameters
        ----------
        name : str
            Parameter name, for the message.
        value : object
            A value for this parameter, e.g. from ``with_values``. It need
            not be a point of the grid.

        Returns
        -------
        float
            ``value``.

        Raises
        ------
        CoverDefinitionError
            If ``value`` is not a finite number in ``[lower, upper]``.
        """
        number = real_number(name, value)
        upper = math.inf if self.upper is None else self.upper
        if not self.lower <= number <= upper:
            raise CoverDefinitionError(
                f"{name} = {number} is outside its bounds [{self.lower}, {upper}]"
            )
        return number

    def to_dict(self) -> dict[str, Any]:
        """Return the document of this parameter.

        Returns
        -------
        dict
            ``{"free": {"grid": [...] or None, "lower": ..., "upper": ...}}``.
        """
        grid = None if self.grid is None else list(self.grid)
        return {"free": {"grid": grid, "lower": self.lower, "upper": self.upper}}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Free:
        """Read a free parameter from its document (:meth:`to_dict`).

        Parameters
        ----------
        document : mapping
            ``{"free": {"lower": ..., "upper": ..., "grid": ...}}``. The
            three inner keys are optional and default to ``Free()``'s.

        Returns
        -------
        Free

        Raises
        ------
        CoverDefinitionError
            If the document has another form or an unknown key.
        """
        if not isinstance(document, Mapping) or set(document) != {"free"}:
            raise CoverDefinitionError(
                f"a free parameter is {{'free': {{...}}}}, got {document!r}"
            )
        inner = document["free"]
        if not isinstance(inner, Mapping) or not set(inner) <= {
            "lower",
            "upper",
            "grid",
        }:
            raise CoverDefinitionError(
                "a free parameter takes the keys 'lower', 'upper' and 'grid', "
                f"got {inner!r}"
            )
        return cls(inner.get("lower", 0.0), inner.get("upper"), grid=inner.get("grid"))


Parameter = float | Free
"""A parameter of a payout, a fixed number or a :class:`Free`."""


def _grid(grid: object, lower: float, upper: float | None) -> tuple[float, ...] | None:
    """Return the grid as explicit, checked values (see :class:`Free`)."""
    if grid is None:
        return None
    if isinstance(grid, int | np.integer) and not isinstance(grid, bool | np.bool_):
        if grid < 2 or upper is None:
            raise CoverDefinitionError(
                "a grid given as a number of points needs at least 2 points and a "
                f"finite upper bound, got grid={grid} and upper={upper}"
            )
        return tuple(np.linspace(lower, upper, int(grid)).tolist())
    try:
        values = np.asarray(grid, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CoverDefinitionError(f"Free.grid must be numeric: {exc}") from exc
    top = math.inf if upper is None else upper
    valid = values.ndim == 1 and values.size > 0
    if not (
        valid
        and np.all(np.isfinite(values))
        and np.all(np.diff(values) > 0)
        and lower <= values[0]
        and values[-1] <= top
    ):
        raise CoverDefinitionError(
            "Free.grid must be a number of points, or finite values, strictly "
            f"increasing, within [{lower}, {top}]; got {grid!r}"
        )
    return tuple(values.tolist())


def check_flag(name: str, value: object) -> bool:
    """Return ``value`` if it is a bool."""
    if not isinstance(value, bool | np.bool_):
        raise CoverDefinitionError(f"{name} must be a bool, got {value!r}")
    return bool(value)


def non_negative(name: str, value: object) -> float:
    """Return ``value`` as a float if it is a finite number >= 0."""
    number = real_number(name, value)
    if number < 0:
        raise CoverDefinitionError(f"{name} must be >= 0, got {number}")
    return number


def shape_parameter(name: str, value: object) -> Parameter:
    """Return a shape parameter: a finite number, or Free with bounds and a grid."""
    if isinstance(value, Free):
        if value.upper is None or value.grid is None:
            raise CoverDefinitionError(
                f"the free shape parameter {name} needs a finite upper bound and a "
                f"grid, got {value!r}"
            )
        return value
    return real_number(name, value)


def amount_parameter(name: str, value: object) -> Parameter:
    """Return an amount: a finite number >= 0, or Free with ``lower >= 0``."""
    if isinstance(value, Free):
        if value.lower < 0:
            raise CoverDefinitionError(
                f"the free amount {name} needs lower >= 0, got {value!r}"
            )
        return value
    return non_negative(name, value)


def parameter_tuple(
    name: str,
    values: object,
    check: Callable[[str, object], Parameter],
    minimum: int,
) -> tuple[Parameter, ...]:
    """Return ``values`` as a tuple of checked parameters, at least ``minimum``."""
    if isinstance(values, str | Mapping) or not isinstance(
        values, Sequence | np.ndarray
    ):
        raise CoverDefinitionError(f"{name} must be a sequence, got {values!r}")
    out = tuple(check(f"{name}[{k}]", value) for k, value in enumerate(values))
    if len(out) < minimum:
        raise CoverDefinitionError(
            f"{name} needs at least {minimum} values, got {len(out)}"
        )
    return out


def check_lengths(
    first: tuple[str, Sequence[Parameter]], second: tuple[str, Sequence[Parameter]]
) -> None:
    """Raise if two parameter sequences differ in length."""
    if len(first[1]) != len(second[1]):
        raise CoverDefinitionError(
            f"{first[0]} and {second[0]} must have one length, got "
            f"{len(first[1])} and {len(second[1])}"
        )


def check_order(name: str, values: Sequence[Parameter], tail: str) -> None:
    """Raise unless the fixed thresholds strictly increase in severity."""
    sign = -1.0 if tail == "lower" else 1.0
    fixed = [sign * value for value in values if not isinstance(value, Free)]
    if any(later <= earlier for earlier, later in pairwise(fixed)):
        direction = "decreasing" if tail == "lower" else "increasing"
        raise CoverDefinitionError(
            f"{name} must be strictly {direction} for tail={tail!r}, got "
            f"{list(values)}"
        )


def check_monotone(name: str, values: Sequence[Parameter]) -> None:
    """Raise unless the fixed values are non-decreasing."""
    fixed = [value for value in values if not isinstance(value, Free)]
    if any(later < earlier for earlier, later in pairwise(fixed)):
        raise CoverDefinitionError(
            f"{name} must be non-decreasing with monotone=True, got {list(values)}"
        )


@dataclass(frozen=True, eq=False)
class AmountConstraints:
    """Linear constraints on the amounts of one unit's payout.

    Parameters
    ----------
    lower, upper : array_like of float
        Bounds per amount, shape ``(n_amounts,)``. ``upper`` may be
        ``inf``. A fixed amount ``v`` has ``lower = v``.
    a_ub, b_ub : array_like of float, optional
        Rows ``a_ub @ amounts <= b_ub``, shapes ``(n_rows, n_amounts)`` and
        ``(n_rows,)``, e.g. monotone levels. Default: no rows.

    Raises
    ------
    CoverDefinitionError
        If the shapes do not fit or a value is not finite (``upper`` may be
        ``inf``).
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    a_ub: NDArray[np.float64] | None = None
    b_ub: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Check the shapes and store read-only arrays."""
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        count = lower.size
        a_ub = np.zeros((0, count)) if self.a_ub is None else self.a_ub
        b_ub = np.zeros(0) if self.b_ub is None else self.b_ub
        a_ub = np.asarray(a_ub, dtype=np.float64)
        b_ub = np.asarray(b_ub, dtype=np.float64)
        if (
            lower.shape != (count,)
            or upper.shape != (count,)
            or a_ub.shape != (b_ub.size, count)
            or b_ub.shape != (b_ub.size,)
        ):
            raise CoverDefinitionError(
                "amount constraints need lower and upper of one shape (n,), a_ub of "
                f"shape (r, n) and b_ub of shape (r,); got {lower.shape}, "
                f"{upper.shape}, {a_ub.shape} and {b_ub.shape}"
            )
        finite = np.isfinite(lower).all() and np.isfinite(a_ub).all()
        if not (finite and np.isfinite(b_ub).all() and not np.isnan(upper).any()):
            raise CoverDefinitionError("amount constraints must be finite")
        for name, value in (
            ("lower", lower),
            ("upper", upper),
            ("a_ub", a_ub),
            ("b_ub", b_ub),
        ):
            object.__setattr__(self, name, read_only(value))

    @property
    def bounds_feasible(self) -> bool:
        """Whether ``lower <= upper`` holds for every amount."""
        return bool(np.all(self.lower <= self.upper))

    def with_monotone_rows(self) -> AmountConstraints:
        """Return these bounds with the rows ``x_k - x_{k+1} <= 0``, and no other rows."""
        count = self.lower.size
        rows = np.zeros((count - 1, count))
        position = np.arange(count - 1)
        rows[position, position] = 1.0
        rows[position, position + 1] = -1.0
        return AmountConstraints(
            self.lower, self.upper, a_ub=rows, b_ub=np.zeros(count - 1)
        )


def is_parameter(value: object) -> bool:
    """Return whether a field value is a parameter: a real number or Free."""
    if isinstance(value, bool | np.bool_):
        return False
    return isinstance(value, Free | int | float | np.number)


def field_to_document(value: object, name: str) -> Any:
    """Return a field value as it is written in a document."""
    if isinstance(value, Free):
        return value.to_dict()
    if isinstance(value, tuple | list):
        return [field_to_document(entry, name) for entry in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value)
    raise CoverDefinitionError(
        f"field {name!r} holds {value!r}, which a document can't hold: override "
        "to_dict and from_dict"
    )


def field_from_document(document: object) -> Any:
    """Return a field value from its document: Free, a tuple, or as written."""
    if isinstance(document, Mapping):
        return Free.from_dict(document)
    if isinstance(document, list):
        return tuple(field_from_document(entry) for entry in document)
    return document


def constructor_fields(target: object) -> list[dataclasses.Field[Any]]:
    """Return the constructor fields of a dataclass, or a dataclass class."""
    if not dataclasses.is_dataclass(target):
        raise CoverDefinitionError(
            f"{target!r} is not a dataclass: override parameters, with_values, "
            "to_dict and from_dict"
        )
    return [item for item in dataclasses.fields(target) if item.init]


def required_field(item: dataclasses.Field[Any]) -> bool:
    """Return whether a dataclass field has no default."""
    return (
        item.default is dataclasses.MISSING
        and item.default_factory is dataclasses.MISSING
    )
