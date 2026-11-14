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

Constraints of a design problem: the premium budget, the amounts and the limit.

* **The budget**, :class:`Budget`, bounds the premium of each design
  problem on the design sample: ``Budget(ratio=kappa)`` means
  ``premium <= kappa E[L]``, with ``E[L]`` the problem's expected loss, and
  ``Budget(amount=B)`` means ``premium <= B``, in payout units. In the
  individual scope, an amount takes a mapping by unit id, one budget per
  unit; in the joint scope, one number for the owner's total premium. At
  one ``kappa`` and under the pure premium, both scopes spend the same
  total.
* **The amounts** of a payout family that is linear in its amounts
  (:class:`~climada_petals.engine.parametric.payout.LinearInAmounts`) have
  their bounds, ``lower >= 0`` and the free amount's ``upper``, and the
  family's own rows, such as monotone levels
  (:meth:`~climada_petals.engine.parametric.payout.LinearInAmounts.amount_constraints`).
* **The limit** is a term of the cover
  (:class:`~climada_petals.engine.parametric.cover.Limit`): the most one
  event pays at one unit. ``Limit.largest_loss()`` is resolved on the design
  sample into each unit's largest event loss (:func:`resolve_limit`), and
  the resolved cover holds those numbers.

On a design sample, :func:`unit_amounts` gives each unit of a template its
payout's basis, its constraints with the limit, and which free amounts are
unidentified: no row of the sample with a weight reaches them
(:func:`unit_amount` does it for one unit's payout). It refuses what the
exact route doesn't take: a payout that is not linear in its amounts, or a
free shape parameter (a threshold). The calibration takes those on the
grid route, in the individual scope; a joint design never has them.

After a solve, :func:`snap_amounts` moves the solver's amounts onto their
bounds and their monotone order, which a solver meets only to its
tolerance, and :func:`fill_amounts` sets the unidentified amounts by the
stated fill rule (:func:`final_amounts` does both per unit).
:func:`resolved_cover` writes them, and the resolved limit, into the cover.
:func:`violation` measures how far amounts break their constraints, for
the checks after the solve.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import linprog

from ._checks import check_real
from .cover import Limit, ParametricCover
from .errors import CoverDefinitionError, ParametricInputError
from .parameters import AmountConstraints, Free
from .payout import LinearInAmounts
from .sample import OutcomeSample, check_left_out_unpaid

__all__ = [
    "FILLS",
    "Budget",
    "UnitAmounts",
    "check_bounded",
    "check_unpaid_at_zero",
    "fill_amounts",
    "final_amounts",
    "payout_at_zero",
    "resolve_limit",
    "resolved_cover",
    "snap_amounts",
    "unit_amount",
    "unit_amounts",
    "violation",
]

FILLS: Final[tuple[str, ...]] = ("lowest", "highest")
"""The fill rules of unidentified amounts (:func:`fill_amounts`)."""

_JOINT: Final[str] = "joint"


@dataclass(frozen=True, kw_only=True)
class Budget:
    """The premium budget of each design problem (module docstring).

    Exactly one of ``ratio`` and ``amount`` is given, as a keyword.

    Parameters
    ----------
    ratio : float, optional
        ``kappa >= 0`` in ``premium <= kappa E[L]``, with ``E[L]`` the
        design problem's expected loss on the design sample.
    amount : float or mapping of unit id to float, optional
        ``B >= 0`` in ``premium <= B``, in payout units: one number, or in
        the individual scope one per unit id, for every unit.

    Raises
    ------
    ParametricInputError
        If both or neither are given, or a value is negative or not finite.

    Examples
    --------
    >>> Budget(ratio=0.8).to_dict()
    {'ratio': 0.8}
    >>> Budget(amount={"parish_1": 1e5, "parish_2": 2e5}).to_dict()
    {'amount': {'parish_1': 100000.0, 'parish_2': 200000.0}}
    """

    ratio: float | None = None
    amount: float | Mapping[Hashable, float] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate the budget; store an amount by unit read-only."""
        if (self.ratio is None) == (self.amount is None):
            raise ParametricInputError(
                "a Budget takes ratio= or amount=, exactly one; got "
                f"ratio={self.ratio!r} and amount={self.amount!r}"
            )
        if self.ratio is not None:
            object.__setattr__(self, "ratio", check_real("ratio", self.ratio, low=0.0))
        elif isinstance(self.amount, Mapping):
            if not self.amount:
                raise ParametricInputError("Budget.amount must not be an empty mapping")
            amounts = {
                unit: check_real(f"Budget.amount[{unit!r}]", value, low=0.0)
                for unit, value in self.amount.items()
            }
            object.__setattr__(self, "amount", MappingProxyType(amounts))
        else:
            object.__setattr__(
                self, "amount", check_real("Budget.amount", self.amount, low=0.0)
            )

    def __hash__(self) -> int:
        """Hash the ratio or the amounts."""
        if isinstance(self.amount, Mapping):
            return hash(frozenset(self.amount.items()))
        return hash((self.ratio, self.amount))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy through the constructor."""
        amount = dict(self.amount) if isinstance(self.amount, Mapping) else self.amount
        return (_budget, (self.ratio, amount))

    def check(self, scope: str, unit_id: Sequence[Hashable]) -> None:
        """Raise unless the budget fits the scope and the units.

        Parameters
        ----------
        scope : {"individual", "joint"}
            The calibration scope.
        unit_id : sequence of str or int
            The units of the design sample.

        Raises
        ------
        ParametricInputError
            If an amount by unit is given in the joint scope, or not for
            exactly the sample's units, or one amount is given in the
            individual scope with several units.
        """
        if not isinstance(self.amount, Mapping):
            if self.amount is not None and scope != _JOINT and len(unit_id) > 1:
                raise ParametricInputError(
                    "in the individual scope Budget(amount=...) takes a mapping by "
                    f"unit id, one budget per design problem; got {self.amount!r} "
                    f"for {len(unit_id)} units. Use Budget(ratio=...) for one "
                    "ratio of each unit's expected loss"
                )
            return
        if scope == _JOINT:
            raise ParametricInputError(
                "the joint scope has one budget on the total premium: give "
                "Budget(amount=B) one number, or Budget(ratio=...)"
            )
        if set(self.amount) != set(unit_id):
            raise ParametricInputError(
                "Budget.amount by unit must have one amount per unit id; got "
                f"{list(self.amount)[:10]} for units {list(unit_id)[:10]}"
            )

    def bound(self, unit_id: Sequence[Hashable], expected_loss: float) -> float:
        """Return the largest premium of one design problem.

        Parameters
        ----------
        unit_id : sequence of str or int
            The problem's units: one in the individual scope.
        expected_loss : float
            ``E[L]`` of the problem's loss on the design sample.

        Returns
        -------
        float
            ``kappa E[L]``, ``B``, or ``B`` of the problem's unit.
        """
        if self.ratio is not None:
            return self.ratio * float(expected_loss)
        if isinstance(self.amount, Mapping):
            (unit,) = unit_id
            return float(self.amount[unit])
        return float(self.amount or 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"ratio": kappa}`` or ``{"amount": ...}``, keyed by ``str(unit)``."""
        if self.ratio is not None:
            return {"ratio": self.ratio}
        if isinstance(self.amount, Mapping):
            return {"amount": {str(unit): value for unit, value in self.amount.items()}}
        return {"amount": self.amount}


def _budget(ratio: float | None, amount: Any) -> Budget:
    """Rebuild a budget from its fields (pickling)."""
    return Budget(ratio=ratio, amount=amount)


def resolve_limit(
    limit: Limit | None, unit_id: Sequence[Hashable], loss: Any
) -> Limit | None:
    """Return the cover's limit, with ``Limit.largest_loss()`` resolved.

    Parameters
    ----------
    limit : Limit or None
        The template's limit.
    unit_id : sequence of str or int
        The units, in the columns' order of ``loss``.
    loss : scipy.sparse matrix
        The design sample's loss per row and unit, ``(n_rows, n_units)``.

    Returns
    -------
    Limit or None
        ``limit`` itself when it is stated or None; under the rule, each
        unit's largest loss in one row of the design sample, by unit id.
    """
    if limit is None or limit.rule is None:
        return limit
    largest = np.asarray(sp.csc_matrix(loss).max(axis=0).toarray(), dtype=np.float64)
    return Limit(
        {unit: float(value) for unit, value in zip(unit_id, largest.reshape(-1))}
    )


def _is_chain(constraints: AmountConstraints) -> bool:
    """Whether the rows are exactly ``x_k - x_{k+1} <= 0``, monotone amounts."""
    count = constraints.lower.size
    if constraints.a_ub.shape != (count - 1, count) or np.any(constraints.b_ub):
        return False
    expected = np.eye(count - 1, count) - np.eye(count - 1, count, k=1)
    return bool(np.array_equal(constraints.a_ub, expected))


def snap_amounts(
    values: NDArray[np.float64], constraints: AmountConstraints
) -> NDArray[np.float64]:
    """Return a solver's amounts of one unit on their bounds and monotone order.

    A solver meets bounds and rows only to its tolerance, so an amount can
    be ``-1e-16``, or a level a few ulps below the one before; a cover
    refuses both. Each amount is clipped to its bounds. If the rows are the
    monotone ones, the bounds are first tightened to the order
    (``lower_k`` to the largest lower bound up to ``k``, ``upper_k`` to the
    smallest upper bound from ``k`` on) and the clipped amounts are then
    made non-decreasing, which keeps them within those bounds. Other rows
    are left to the checks after the solve.

    Parameters
    ----------
    values : numpy.ndarray
        The amounts, shape ``(n_amounts,)``.
    constraints : AmountConstraints
        Their constraints.

    Returns
    -------
    numpy.ndarray
        The snapped amounts.
    """
    lower, upper = constraints.lower, constraints.upper
    if not _is_chain(constraints):
        return np.clip(values, lower, upper)
    low = np.maximum.accumulate(lower)
    high = np.minimum.accumulate(upper[::-1])[::-1]
    return np.maximum.accumulate(np.clip(values, low, high))


def violation(values: NDArray[np.float64], constraints: AmountConstraints) -> float:
    """Return how far amounts break their bounds or rows, in payout units.

    Returns
    -------
    float
        The largest of ``lower - x``, ``x - upper`` and ``a_ub @ x - b_ub``,
        or 0 if every constraint holds.
    """
    gaps = [constraints.lower - values, values - constraints.upper]
    if constraints.b_ub.size:
        gaps.append(constraints.a_ub @ values - constraints.b_ub)
    return max(0.0, float(max(np.max(gap, initial=0.0) for gap in gaps)))


def fill_amounts(
    values: NDArray[np.float64],
    constraints: AmountConstraints,
    unidentified: NDArray[np.bool_],
    rule: str,
) -> NDArray[np.float64]:
    """Return the amounts with the unidentified ones set by the fill rule.

    An unidentified amount changes no payout on the design sample, so the
    objective and the premium hold whatever value it takes within its
    constraints. ``"lowest"`` takes the smallest sum of the unidentified
    amounts that the constraints allow with the others as they are, and
    ``"highest"`` the largest. Under monotone levels that is the previous
    level (or the lower bound), and the next level (or the upper bound).

    Parameters
    ----------
    values : numpy.ndarray
        The amounts after the solve, shape ``(n_amounts,)``.
    constraints : AmountConstraints
        Their constraints.
    unidentified : numpy.ndarray of bool
        Which amounts are unidentified.
    rule : {"lowest", "highest"}
        The fill rule.

    Returns
    -------
    numpy.ndarray
        The amounts, snapped (:func:`snap_amounts`).

    Raises
    ------
    ParametricInputError
        If ``rule="highest"`` and an unidentified amount has no upper bound.
    CoverDefinitionError
        If no value of the unidentified amounts meets the constraints with
        the others as they are.
    """
    if not unidentified.any():
        return values
    known = ~unidentified
    a_ub = constraints.a_ub
    sign = 1.0 if rule == "lowest" else -1.0
    result = linprog(
        np.full(int(unidentified.sum()), sign),
        A_ub=a_ub[:, unidentified] if a_ub.size else None,
        b_ub=(constraints.b_ub - a_ub[:, known] @ values[known]) if a_ub.size else None,
        bounds=np.column_stack(
            [constraints.lower[unidentified], constraints.upper[unidentified]]
        ),
        method="highs",
    )
    if result.status == 3:
        raise ParametricInputError(
            "fill='highest' needs an upper bound on every unidentified amount: a "
            "Limit, an upper bound of Free, or monotone levels with an identified "
            "level above it"
        )
    if result.status != 0:
        raise CoverDefinitionError(
            f"the unidentified amounts can't be set within their constraints: "
            f"{result.message}"
        )
    out = values.copy()
    out[unidentified] = result.x
    return snap_amounts(out, constraints)


class UnitAmounts(NamedTuple):
    """The amounts of one unit of a template, on a design sample.

    Parameters
    ----------
    unit_id : str or int
        The unit.
    payout : LinearInAmounts
        Its payout, with every shape parameter fixed.
    constraints : AmountConstraints
        The bounds and rows of its amounts, with the cover's limit.
    basis : scipy.sparse.csr_matrix
        The payout's basis on the sample's rows, ``(n_rows, n_amounts)``.
    free : numpy.ndarray of bool
        Which amounts are free in the template.
    unidentified : numpy.ndarray of bool
        Which free amounts no row of the sample with a weight reaches.
    """

    unit_id: Hashable
    payout: LinearInAmounts
    constraints: AmountConstraints
    basis: sp.csr_matrix
    free: NDArray[np.bool_]
    unidentified: NDArray[np.bool_]

    @property
    def unidentified_names(self) -> tuple[str, ...]:
        """The names of the unidentified amounts."""
        names = self.payout.amount_names()
        return tuple(name for name, flag in zip(names, self.unidentified) if flag)


def _route_payout(template: ParametricCover, unit: Hashable, scope: str) -> Any:
    """Return a unit's payout if the template route takes it.

    Raises
    ------
    ParametricInputError
        If the payout is not linear in its amounts, or has a free shape
        parameter.
    """
    payout = template.payout[unit]
    if not isinstance(payout, LinearInAmounts):
        raise ParametricInputError(
            f"the {payout.kind} payout of unit {unit!r} is not linear in its amounts "
            "(LinearInAmounts), which the exact route needs; such a payout takes "
            "the grid route, in the individual scope"
        )
    shapes = [
        name
        for name in payout.free_parameters()
        if name not in set(payout.amount_names())
    ]
    if shapes and scope == _JOINT:
        raise ParametricInputError(
            f"thresholds are fixed in a joint design; unit {unit!r} has the free "
            f"shape parameters {shapes}. Fix them, and compare thresholds by "
            "running the design several times"
        )
    if shapes:
        raise ParametricInputError(
            f"unit {unit!r} has the free shape parameters {shapes}, which the exact "
            "route can't solve for; they need the grid route"
        )
    return payout


def unit_amount(
    unit_id: Hashable,
    payout: LinearInAmounts,
    values: NDArray[np.float64],
    weighted: NDArray[np.float64],
    cap: float | None,
) -> UnitAmounts:
    """Return one unit's payout, constraints, basis and unidentified amounts.

    Parameters
    ----------
    unit_id : str or int
        The unit.
    payout : LinearInAmounts
        Its payout, with every shape parameter fixed.
    values : numpy.ndarray
        The unit's index per row of the design sample.
    weighted : numpy.ndarray
        1 for a row with a weight, else 0.
    cap : float or None
        The unit's limit, resolved.

    Returns
    -------
    UnitAmounts

    Raises
    ------
    CoverDefinitionError
        If a fixed amount, or the lower bound of a free one, is above ``cap``.
    """
    constraints = payout.amount_constraints(cap)
    if not constraints.bounds_feasible:
        raise CoverDefinitionError(
            f"the {payout.kind} payout of unit {unit_id!r} can't stay within its "
            f"limit {cap}: a fixed amount, or the lower bound of a free one, is "
            "above it"
        )
    basis = sp.csr_matrix(payout.basis(values))
    free = _free_amounts(payout)
    reached = np.asarray(abs(basis).T @ weighted).reshape(-1) > 0
    return UnitAmounts(unit_id, payout, constraints, basis, free, free & ~reached)


def unit_amounts(
    template: ParametricCover,
    sample: OutcomeSample,
    *,
    scope: str,
    limit: Limit | None,
    positions: Sequence[int] | None = None,
) -> list[UnitAmounts]:
    """Return every unit's payout, constraints, basis and unidentified amounts.

    Parameters
    ----------
    template : ParametricCover
        The template, on the sample's units.
    sample : OutcomeSample
        The design sample.
    scope : {"individual", "joint"}
        The calibration scope.
    limit : Limit or None
        The template's limit, resolved (:func:`resolve_limit`).
    positions : sequence of int, optional
        The units to build, by their position in the sample. Default: every
        unit.

    Returns
    -------
    list of UnitAmounts
        One per unit of ``positions``, in their order.

    Raises
    ------
    ParametricInputError
        If a payout is not linear in its amounts, or has a free shape
        parameter (which needs the grid route; in a joint design thresholds
        are fixed).
    CoverDefinitionError
        If a fixed amount is above the unit's limit.
    """
    values = sample.index_values
    weighted = (sample.measure.event_weight > 0).astype(np.float64)
    units = []
    if positions is None:
        positions = range(len(sample.unit_id))
    for position in positions:
        unit = sample.unit_id[position]
        payout = _route_payout(template, unit, scope)
        cap = None if limit is None else limit.amount_of(unit)
        units.append(unit_amount(unit, payout, values[:, position], weighted, cap))
    return units


def _free_amounts(payout: LinearInAmounts) -> NDArray[np.bool_]:
    """Return which amounts of a payout are free."""
    parameters = payout.parameters()
    return np.array(
        [isinstance(parameters[name], Free) for name in payout.amount_names()],
        dtype=bool,
    )


def check_bounded(units: Sequence[UnitAmounts], objective_name: str) -> None:
    """Raise if a free amount has no upper bound, for an objective without a budget.

    ``E[U]`` and the weighted CVaR don't penalise over-payment, so without a
    budget they need bounded amounts.

    Raises
    ------
    ParametricInputError
        If a free amount has no upper bound.
    """
    unbounded = [
        unit.unit_id
        for unit in units
        if np.any(unit.free & ~np.isfinite(unit.constraints.upper))
    ]
    if unbounded:
        raise ParametricInputError(
            f"{objective_name} needs a budget or bounded amounts: without "
            "either, any payout at least as large as the loss is optimal. The "
            f"free amounts of units {unbounded[:5]} have no upper bound; give a "
            "Budget, a Limit (such as Limit.largest_loss()) or Free upper bounds"
        )


def payout_at_zero(payout: LinearInAmounts, constraints: AmountConstraints) -> float:
    """Return the most a payout may pay at an index of 0, its amounts at their upper bounds.

    An amount without an upper bound counts as 1: any positive value says
    that the payout may pay there.
    """
    at_zero = sp.csr_matrix(payout.basis(np.zeros(1))).toarray()[0]
    upper = constraints.upper
    return float(at_zero @ np.where(np.isfinite(upper), upper, 1.0))


def check_unpaid_at_zero(sample: OutcomeSample, units: Sequence[UnitAmounts]) -> None:
    """Raise if the sample left events out and a payout may pay at an index of 0.

    See :func:`~climada_petals.engine.parametric.sample.check_left_out_unpaid`.
    """
    check_left_out_unpaid(
        sample, [payout_at_zero(unit.payout, unit.constraints) for unit in units]
    )


def final_amounts(
    units: Sequence[UnitAmounts], amounts: NDArray[np.float64], fill: str
) -> dict[Hashable, NDArray[np.float64]]:
    """Return each unit's amounts, snapped, with the unidentified ones filled.

    Parameters
    ----------
    units : sequence of UnitAmounts
        The units of one design problem.
    amounts : numpy.ndarray
        The solver's amounts of those units, side by side.
    fill : {"lowest", "highest"}
        The fill rule (:func:`fill_amounts`).

    Returns
    -------
    dict of unit id to numpy.ndarray
    """
    out = {}
    start = 0
    for unit in units:
        count = unit.constraints.lower.size
        part = snap_amounts(amounts[start : start + count], unit.constraints)
        start += count
        out[unit.unit_id] = fill_amounts(
            part, unit.constraints, unit.unidentified, fill
        )
    return out


def resolved_cover(
    template: ParametricCover,
    units: Sequence[UnitAmounts],
    amounts: Mapping[Hashable, NDArray[np.float64]],
    limit: Limit | None,
    others: Mapping[Hashable, Mapping[str, float]] | None = None,
) -> ParametricCover:
    """Return the template with its free parameters set and its limit resolved.

    ``units`` and ``amounts`` set the free amounts of the units solved on
    the exact route; ``others`` sets every free parameter of the other
    units, by unit id, such as those designed on a grid.
    """
    values: dict[Hashable, Mapping[str, float]] = {
        unit.unit_id: {
            name: float(value)
            for name, value, free in zip(
                unit.payout.amount_names(), amounts[unit.unit_id], unit.free
            )
            if free
        }
        for unit in units
    }
    values.update(others or {})
    return replace(template.with_values(values), limit=limit)
