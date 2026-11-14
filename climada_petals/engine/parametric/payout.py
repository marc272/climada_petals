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

Payout functions: what a cover pays at one unit for an index value.

A payout function maps index values to payouts, vectorised over numpy
arrays. Four families are built in. A user family implements the same
protocol, and :func:`register_payout` lets a cover document name it.

Severity
--------
``tail="upper"``: larger index values are more severe, as for wind or
rainfall. ``tail="lower"``: smaller values are more severe, as for a drought
index. A family works on the severity ``s = x`` (upper) or ``s = -x``
(lower), with its thresholds mapped the same way. Thresholds are given in
index units, in order of increasing severity: increasing for ``"upper"``,
decreasing for ``"lower"``.

Families
--------
With ``s`` the severity, and the thresholds in severity:

* :class:`StepPayout`, breakpoints ``b_0 < ... < b_{n-1}`` and levels
  ``theta_0 ... theta_{n-1}``: ``theta_k`` for ``b_k <= s < b_{k+1}``, and
  ``theta_{n-1}`` for ``s >= b_{n-1}``.
* :class:`LinearPayout`, attachment ``a``, exhaustion ``e`` and limit
  ``l``: ``l * clip((s - a) / (e - a), 0, 1)``. Where the parameters
  coincide and the index is float, this is ``cat_bonds``' ``calc_payout``
  (:meth:`LinearPayout.from_triggers`).
* :class:`PiecewiseLinearPayout`, knots ``c_0 < ... < c_{m-1}`` and levels
  ``y_0 ... y_{m-1}``: the linear interpolation of the points
  ``(c_k, y_k)`` on ``[c_0, c_{m-1}]``, and ``y_{m-1}`` for
  ``s >= c_{m-1}``.
* :class:`BinaryPayout`, trigger ``t`` and amount ``A``: ``A`` for
  ``s >= t``.

Every family follows these rules:

* **Nothing is paid below the first threshold** (``b_0``, ``a``, ``c_0`` or
  ``t``). An event whose index is below it at every unit adds nothing to
  any outcome, so dropping such events from a hazard set changes no design
  and no premium.
* A value equal to a threshold belongs to the more severe side.
* Index values are converted to float first, so an integer-typed index is
  not truncated, and they must be finite.
* Amounts are at least 0. With ``monotone=True``, the default of the step
  and piecewise-linear families, the levels are non-decreasing, so the
  payout never falls as the severity rises. The linear and binary families
  are monotone by construction.

The protocol
------------
:class:`PayoutFunction` is the protocol of every payout, and
:class:`LinearInAmounts` its opt-in extension for a payout that is linear in
its amounts, ``basis(index) @ amounts``, which the calibration's exact
solvers need. The four families implement both.
"""

from __future__ import annotations

import abc
import dataclasses
import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal, TypeVar, get_args

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from .errors import (
    CoverDefinitionError,
    ParametricInputError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from .parameters import (
    AmountConstraints,
    Free,
    Parameter,
    amount_parameter,
    check_flag,
    check_lengths,
    check_monotone,
    check_order,
    constructor_fields,
    field_from_document,
    field_to_document,
    is_parameter,
    non_negative,
    parameter_tuple,
    real_number,
    required_field,
    shape_parameter,
)

__all__ = [
    "TAILS",
    "BinaryPayout",
    "LinearInAmounts",
    "LinearPayout",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "StepPayout",
    "Tail",
    "payout_from_dict",
    "register_payout",
]

Tail = Literal["upper", "lower"]
"""Which side of the index is severe."""

TAILS: Final[tuple[str, ...]] = get_args(Tail)
"""The values of :data:`Tail`: ``("upper", "lower")``."""

_P = TypeVar("_P", bound="PayoutFunction")
_REGISTRY: dict[str, type[PayoutFunction]] = {}


class PayoutFunction(abc.ABC):
    """The protocol of a payout function.

    A payout maps index values to payouts, in payout units, vectorised over
    numpy arrays. A subclass sets :attr:`kind`, has a ``tail`` and
    implements :meth:`evaluate`. The other methods have defaults for a
    frozen dataclass whose fields are its constructor's arguments, as the
    built-in families are:

    * its parameters are the fields that hold a number or
      :class:`~climada_petals.engine.parametric.parameters.Free`, or a tuple
      of them, named ``"field"`` or ``"field[k]"``;
    * :meth:`with_values` replaces free parameters and builds the payout
      again, so its own checks run;
    * its document holds ``kind`` and every field.

    Another class overrides :meth:`parameters`, :meth:`with_values`,
    :meth:`to_dict` and :meth:`from_dict`. A user family that wants the
    calibration's exact solvers implements :class:`LinearInAmounts`.
    :func:`register_payout` lets a cover document name it.
    """

    kind: ClassVar[str] = ""
    """The family's name in a document, e.g. ``"step"``. Class attribute."""

    index_names: ClassVar[tuple[str, ...]] = ("index",)
    """The index components the payout reads, ``("index",)`` in this
    version. Class attribute."""

    tail: str
    """Which side of the index is severe, ``"upper"`` or ``"lower"`` (module
    docstring)."""

    @abc.abstractmethod
    def evaluate(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return the payout at each index value.

        Parameters
        ----------
        index : array_like of float
            Index values, any shape. Finite.

        Returns
        -------
        numpy.ndarray
            The payouts, in payout units, of the same shape as ``index``.
            At least 0.

        Raises
        ------
        UnresolvedParameterError
            If the payout has a free parameter.
        ParametricInputError
            If an index value is not numeric or not finite.
        """

    def parameters(self) -> dict[str, Parameter]:
        """Return every parameter by name, in the order of the fields.

        Returns
        -------
        dict of str to float or Free
            For example ``{"breakpoints[0]": 25.0, ..., "levels[0]":
            Free(), ...}``.
        """
        out: dict[str, Parameter] = {}
        for item in constructor_fields(self):
            value = getattr(self, item.name)
            if is_parameter(value):
                out[item.name] = value
            elif isinstance(value, tuple) and all(is_parameter(v) for v in value):
                for position, entry in enumerate(value):
                    out[f"{item.name}[{position}]"] = entry
        return out

    def free_parameters(self) -> tuple[str, ...]:
        """Return the names of the free parameters, in parameter order."""
        return tuple(
            name for name, value in self.parameters().items() if isinstance(value, Free)
        )

    @property
    def is_resolved(self) -> bool:
        """Whether every parameter is fixed."""
        return not self.free_parameters()

    def check_resolved(self) -> None:
        """Raise if a parameter is free.

        Raises
        ------
        UnresolvedParameterError
            If a parameter is free.
        """
        free = self.free_parameters()
        if free:
            raise UnresolvedParameterError(
                f"the {self.kind} payout has free parameters {list(free)}: resolve "
                "them with with_values, or calibrate the cover"
            )

    def with_values(self: _P, values: Mapping[str, float]) -> _P:
        """Return the payout with values for some or all free parameters.

        Parameters
        ----------
        values : mapping of str to float
            A value per free parameter, by name (:meth:`parameters`). Each
            must lie within its parameter's bounds, not necessarily on its
            grid. Parameters not named stay as they are.

        Returns
        -------
        PayoutFunction
            A new payout of the same family, checked again.

        Raises
        ------
        CoverDefinitionError
            If a name is not a free parameter, a value is out of bounds or
            not a finite number, or the new payout is invalid (thresholds
            out of order, levels not monotone).
        """
        if not isinstance(values, Mapping):
            raise CoverDefinitionError(
                f"values must map parameter names to numbers, got {values!r}"
            )
        current = self.parameters()
        changes: dict[str, Any] = {}
        for name, value in values.items():
            parameter = current.get(name)
            if not isinstance(parameter, Free):
                raise CoverDefinitionError(
                    f"{name!r} is not a free parameter of the {self.kind} payout; its "
                    f"free parameters are {list(self.free_parameters())}"
                )
            number = parameter.check(name, value)
            field_name, _, position = name.partition("[")
            if position:
                entries = changes.setdefault(
                    field_name, list(getattr(self, field_name))
                )
                entries[int(position[:-1])] = number
            else:
                changes[field_name] = number
        for field_name, change in changes.items():
            if isinstance(change, list):
                changes[field_name] = tuple(change)
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the payout's document: ``kind`` and every field.

        Returns
        -------
        dict
            JSON-ready. A fixed parameter is a number, a free one
            ``{"free": {...}}``, a tuple a list.
        """
        document: dict[str, Any] = {"kind": self.kind}
        for item in constructor_fields(self):
            document[item.name] = field_to_document(getattr(self, item.name), item.name)
        return document

    @classmethod
    def from_dict(cls: type[_P], document: Mapping[str, Any]) -> _P:
        """Read a payout of this family from its document (:meth:`to_dict`).

        Parameters
        ----------
        document : mapping
            ``kind`` and the fields. Fields with a default may be left out.

        Returns
        -------
        PayoutFunction

        Raises
        ------
        CoverDefinitionError
            If the kind is not this family's, a key is unknown or missing,
            or a value is invalid.
        """
        if not isinstance(document, Mapping) or document.get("kind") != cls.kind:
            raise CoverDefinitionError(
                f"a {cls.__name__} document has kind {cls.kind!r}, got {document!r}"
            )
        items = {item.name: item for item in constructor_fields(cls)}
        unknown = set(document) - set(items) - {"kind"}
        missing = {name for name, item in items.items() if required_field(item)}
        missing -= set(document)
        if unknown or missing:
            raise CoverDefinitionError(
                f"a {cls.kind} payout document has unknown keys {sorted(unknown)} "
                f"and misses {sorted(missing)}"
            )
        kwargs = {
            name: field_from_document(document[name])
            for name in items
            if name in document
        }
        return cls(**kwargs)

    def severity(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return the severity of index values: ``x`` (upper) or ``-x`` (lower).

        The values are converted to float, so an integer index is not
        truncated.

        Raises
        ------
        ParametricInputError
            If a value is not numeric or not finite.
        """
        values = _index_values(index)
        return -values if self.tail == "lower" else values


class LinearInAmounts(PayoutFunction):
    """A payout that is linear in its amounts: ``basis(index) @ amounts``.

    The opt-in extension of :class:`PayoutFunction` for the calibration's
    exact routes, a linear or a quadratic program in the amounts. Shape
    parameters, such as thresholds, fix the basis; the amounts, in payout
    units, weight its columns. A subclass implements :meth:`amount_names`
    and :meth:`basis`; :meth:`evaluate` and :meth:`amount_constraints` have
    defaults built on them.
    """

    @abc.abstractmethod
    def amount_names(self) -> tuple[str, ...]:
        """Return the names of the amounts, in the order of the basis columns."""

    @abc.abstractmethod
    def basis(self, index: ArrayLike) -> sp.csr_matrix:
        """Return the basis of one unit's index values.

        Parameters
        ----------
        index : array_like of float
            Index values of one unit, shape ``(n_events,)``. Finite.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_events, n_amounts)``, non-negative: the payout is
            ``basis(index) @ amounts`` for any amounts. The rows of the
            built-in families sum to at most 1, so their payout is at most
            their largest amount.

        Raises
        ------
        UnresolvedParameterError
            If a shape parameter is free. Free amounts are allowed.
        ParametricInputError
            If ``index`` is not 1-D, not numeric or not finite.
        """

    def amounts(self) -> NDArray[np.float64]:
        """Return the amounts, shape ``(n_amounts,)``, in :meth:`amount_names` order.

        Raises
        ------
        UnresolvedParameterError
            If an amount is free.
        """
        parameters = self.parameters()
        names = self.amount_names()
        free = [name for name in names if isinstance(parameters[name], Free)]
        if free:
            raise UnresolvedParameterError(
                f"the {self.kind} payout has free amounts {free}: resolve them with "
                "with_values, or calibrate the cover"
            )
        return np.array([parameters[name] for name in names], dtype=np.float64)

    def evaluate(self, index: ArrayLike) -> NDArray[np.float64]:
        """Return ``basis(index) @ amounts``, in the shape of ``index``.

        See :meth:`PayoutFunction.evaluate`.
        """
        self.check_resolved()
        values = _index_values(index)
        paid = self.basis(values.reshape(-1)) @ self.amounts()
        return np.asarray(paid, dtype=np.float64).reshape(values.shape)

    def amount_constraints(self, limit: float | None = None) -> AmountConstraints:
        """Return the linear constraints on the amounts.

        Parameters
        ----------
        limit : float, optional
            The most one event pays at the unit, >= 0: the cover's
            :class:`~climada_petals.engine.parametric.cover.Limit`.

        Returns
        -------
        AmountConstraints
            Bounds: a free amount's ``lower``, and the smaller of its
            ``upper`` and ``limit``; a fixed amount ``v`` has ``lower = v``
            and ``upper = min(v, limit)``, so ``lower > upper`` flags a
            fixed amount above the limit. No rows here; a family adds its
            own, such as monotone levels. Bounding every amount by the limit
            bounds the payout when the basis rows sum to at most 1, as in
            the built-in families; another family overrides this.

        Raises
        ------
        CoverDefinitionError
            If ``limit`` is not a finite number >= 0.
        """
        cap = math.inf if limit is None else non_negative("limit", limit)
        parameters = self.parameters()
        lower: list[float] = []
        upper: list[float] = []
        for name in self.amount_names():
            value = parameters[name]
            if isinstance(value, Free):
                lower.append(value.lower)
                upper.append(min(cap, math.inf if value.upper is None else value.upper))
            else:
                lower.append(value)
                upper.append(min(value, cap))
        return AmountConstraints(np.array(lower), np.array(upper))


def register_payout(cls: type[_P]) -> type[_P]:
    """Register a payout family, so that a cover document can name its ``kind``.

    Use it as a class decorator. The four built-in families are registered.

    Parameters
    ----------
    cls : type
        A concrete subclass of :class:`PayoutFunction` with a ``kind``.

    Returns
    -------
    type
        ``cls``, unchanged.

    Raises
    ------
    CoverDefinitionError
        If ``cls`` is not a concrete subclass of :class:`PayoutFunction`,
        its ``kind`` is not a non-empty string, or another class is
        registered under its ``kind``. Registering a class again, or a new
        definition of it, as a notebook cell run twice makes, replaces it.
    """
    if not (inspect.isclass(cls) and issubclass(cls, PayoutFunction)) or (
        inspect.isabstract(cls)
    ):
        raise CoverDefinitionError(
            f"a payout family is a concrete subclass of PayoutFunction, got {cls!r}"
        )
    kind = cls.kind
    if not isinstance(kind, str) or not kind:
        raise CoverDefinitionError(
            f"{cls.__qualname__}.kind must be a non-empty string, got {kind!r}"
        )
    known = _REGISTRY.get(kind)
    if known is not None and _name(known) != _name(cls):
        raise CoverDefinitionError(
            f"payout kind {kind!r} is registered for {_name(known)}"
        )
    _REGISTRY[kind] = cls
    return cls


def _name(cls: type) -> str:
    """Return a class's module and qualified name."""
    return f"{cls.__module__}.{cls.__qualname__}"


def payout_from_dict(document: Mapping[str, Any]) -> PayoutFunction:
    """Read a payout from its document, through the registry.

    Parameters
    ----------
    document : mapping
        A payout document (:meth:`PayoutFunction.to_dict`), whose ``kind``
        names a registered family.

    Returns
    -------
    PayoutFunction

    Raises
    ------
    UnknownPayoutKindError
        If no family is registered for the ``kind``.
    CoverDefinitionError
        If the document is not a mapping with a ``kind``, or is invalid.
    """
    if not isinstance(document, Mapping) or not isinstance(document.get("kind"), str):
        raise CoverDefinitionError(
            f"a payout document is a mapping with a 'kind', got {document!r}"
        )
    family = _REGISTRY.get(document["kind"])
    if family is None:
        raise UnknownPayoutKindError(
            f"no payout family is registered for kind {document['kind']!r} "
            f"(registered: {sorted(_REGISTRY)}); a user-defined family is "
            "registered with register_payout"
        )
    return family.from_dict(document)


@register_payout
@dataclass(frozen=True)
class StepPayout(LinearInAmounts):
    """Step payout: one level per step of the index.

    Parameters
    ----------
    breakpoints : sequence of float or Free
        ``b_0 ... b_{n-1}``, ``n >= 1``, in index units, strictly increasing
        in severity. Step ``k`` covers ``b_k <= s < b_{k+1}``; the last has
        no upper edge. A free breakpoint needs finite bounds and a grid.
    levels : sequence of float or Free
        ``theta_0 ... theta_{n-1}``, the payout of each step, in payout
        units, at least 0.
    tail : {"upper", "lower"}, default "upper"
        Which side of the index is severe (module docstring).
    monotone : bool, default True
        Whether the levels are non-decreasing. Checked on the fixed levels
        here, and a constraint on the free ones in the calibration.

    Raises
    ------
    CoverDefinitionError
        If a parameter is invalid: no breakpoints, lengths that differ,
        breakpoints out of order, a negative or non-finite value, fixed
        levels that fall under ``monotone=True``, or a free breakpoint
        without finite bounds and a grid.

    Examples
    --------
    >>> payout = StepPayout([25.0, 35.0, 50.0], [10.0, 40.0, 100.0])
    >>> payout.evaluate([20.0, 25.0, 34.9, 35.0, 80.0]).tolist()
    [0.0, 10.0, 10.0, 40.0, 100.0]

    A drought index, more severe as it falls:

    >>> drought = StepPayout([-1.0, -2.0], [5.0, 20.0], tail="lower")
    >>> drought.evaluate([0.0, -1.0, -1.5, -2.5]).tolist()
    [0.0, 5.0, 5.0, 20.0]
    """

    kind: ClassVar[str] = "step"

    breakpoints: tuple[Parameter, ...]
    levels: tuple[Parameter, ...]
    tail: str = field(default="upper", kw_only=True)
    monotone: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the parameters and store them as tuples of floats or Free."""
        tail = check_tail(self.tail)
        breakpoints = parameter_tuple(
            "breakpoints", self.breakpoints, shape_parameter, 1
        )
        check_order("breakpoints", breakpoints, tail)
        levels = parameter_tuple("levels", self.levels, amount_parameter, 1)
        check_lengths(("breakpoints", breakpoints), ("levels", levels))
        monotone = check_flag("monotone", self.monotone)
        if monotone:
            check_monotone("levels", levels)
        _store(
            self, tail=tail, breakpoints=breakpoints, levels=levels, monotone=monotone
        )

    @classmethod
    def uniform(
        cls,
        attachment: float,
        exhaustion: float,
        n_steps: int,
        *,
        levels: Sequence[Parameter] | None = None,
        tail: str = "upper",
        monotone: bool = True,
    ) -> StepPayout:
        """Return a step payout with ``n_steps`` steps of equal width.

        The breakpoints are ``attachment + k (exhaustion - attachment) /
        n_steps`` for ``k = 0 ... n_steps - 1``. ``exhaustion`` is the right
        edge of the last step, which pays its level for every value beyond
        it, so it changes no payout and isn't stored.

        Parameters
        ----------
        attachment, exhaustion : float
            The first breakpoint and the last step's right edge, in index
            units. ``exhaustion`` is more severe: larger for
            ``tail="upper"``, smaller for ``tail="lower"``.
        n_steps : int
            Number of steps, >= 1.
        levels : sequence of float or Free, optional
            The levels. Default: ``Free()`` for every step, a template.
        tail : {"upper", "lower"}, default "upper"
        monotone : bool, default True

        Returns
        -------
        StepPayout

        Raises
        ------
        CoverDefinitionError
            If ``n_steps < 1``, ``exhaustion`` is not more severe than
            ``attachment``, or the payout is invalid.

        Examples
        --------
        >>> StepPayout.uniform(20.0, 70.0, 4).breakpoints
        (20.0, 32.5, 45.0, 57.5)
        """
        if isinstance(n_steps, bool | np.bool_) or not isinstance(
            n_steps, int | np.integer
        ):
            raise CoverDefinitionError(f"n_steps must be an int, got {n_steps!r}")
        if n_steps < 1:
            raise CoverDefinitionError(f"n_steps must be >= 1, got {n_steps}")
        side = check_tail(tail)
        first = real_number("attachment", attachment)
        last = real_number("exhaustion", exhaustion)
        check_order("attachment and exhaustion", (first, last), side)
        edges = np.linspace(first, last, int(n_steps) + 1)[:-1]
        chosen = [Free()] * int(n_steps) if levels is None else levels
        return cls(tuple(edges.tolist()), chosen, tail=side, monotone=monotone)

    @property
    def n_steps(self) -> int:
        """Number of steps."""
        return len(self.breakpoints)

    def amount_names(self) -> tuple[str, ...]:
        """Return ``("levels[0]", ..., "levels[n-1]")``."""
        return tuple(f"levels[{k}]" for k in range(self.n_steps))

    def basis(self, index: ArrayLike) -> sp.csr_matrix:
        """Return the one-hot step of each value (:meth:`LinearInAmounts.basis`).

        A row is empty where the index is below the first breakpoint.
        """
        thresholds = _shape(self, "breakpoints", self.breakpoints)
        severity = _severity_1d(self, index)
        step = np.searchsorted(thresholds, severity, side="right") - 1
        rows = np.flatnonzero(step >= 0)
        return sp.csr_matrix(
            (np.ones(rows.size), (rows, step[rows])),
            shape=(severity.size, self.n_steps),
        )

    def amount_constraints(self, limit: float | None = None) -> AmountConstraints:
        """Return the bounds, and ``theta_k <= theta_{k+1}`` if monotone.

        See :meth:`LinearInAmounts.amount_constraints`.
        """
        constraints = super().amount_constraints(limit)
        return constraints.with_monotone_rows() if self.monotone else constraints


@register_payout
@dataclass(frozen=True)
class LinearPayout(LinearInAmounts):
    """Linear payout from an attachment to an exhaustion, up to a limit.

    The payout is ``limit * clip((s - a) / (e - a), 0, 1)`` with ``s``, ``a``
    and ``e`` the severities of the index, the attachment and the
    exhaustion: nothing at or below the attachment, ``limit`` at and beyond
    the exhaustion.

    Parameters
    ----------
    attachment, exhaustion : float or Free
        Index units. ``exhaustion`` is more severe than ``attachment``. A
        free one needs finite bounds and a grid.
    limit : float or Free
        The payout at and beyond the exhaustion, in payout units, >= 0. Not
        the cover's :class:`~climada_petals.engine.parametric.cover.Limit`,
        which bounds it.
    tail : {"upper", "lower"}, default "upper"

    Raises
    ------
    CoverDefinitionError
        If a parameter is invalid, as for :class:`StepPayout`.

    Examples
    --------
    >>> payout = LinearPayout(attachment=30.0, exhaustion=50.0, limit=1000.0)
    >>> payout.evaluate([20.0, 30.0, 40.0, 50.0, 70.0]).tolist()
    [0.0, 0.0, 500.0, 1000.0, 1000.0]
    """

    kind: ClassVar[str] = "linear"

    attachment: Parameter
    exhaustion: Parameter
    limit: Parameter
    tail: str = field(default="upper", kw_only=True)

    def __post_init__(self) -> None:
        """Validate the parameters and store floats or Free."""
        tail = check_tail(self.tail)
        attachment = shape_parameter("attachment", self.attachment)
        exhaustion = shape_parameter("exhaustion", self.exhaustion)
        check_order("attachment and exhaustion", (attachment, exhaustion), tail)
        limit = amount_parameter("limit", self.limit)
        _store(
            self, tail=tail, attachment=attachment, exhaustion=exhaustion, limit=limit
        )

    @classmethod
    def from_triggers(
        cls,
        min_trig: Parameter,
        max_trig: Parameter,
        max_pay: Parameter,
        *,
        tail: str = "upper",
    ) -> LinearPayout:
        """Return ``LinearPayout(min_trig, max_trig, max_pay, tail=tail)``.

        ``cat_bonds``' ``calc_payout`` calls the attachment, the exhaustion
        and the limit ``min_trig``, ``max_trig`` and ``max_pay``; on a float
        index the payouts are equal. ``calc_payout`` truncates an
        integer-typed index, and this payout does not. ``cat_bonds``'
        ``attachment_point`` and ``exhaustion_point`` are levels of loss,
        not these.
        """
        return cls(min_trig, max_trig, max_pay, tail=tail)

    def amount_names(self) -> tuple[str, ...]:
        """Return ``("limit",)``."""
        return ("limit",)

    def basis(self, index: ArrayLike) -> sp.csr_matrix:
        """Return ``clip((s - a) / (e - a), 0, 1)`` as one column.

        See :meth:`LinearInAmounts.basis`.
        """
        start, end = _shape(
            self, "attachment and exhaustion", (self.attachment, self.exhaustion)
        )
        severity = _severity_1d(self, index)
        share = np.clip((severity - start) / (end - start), 0.0, 1.0)
        return sp.csr_matrix(share[:, None])


@register_payout
@dataclass(frozen=True)
class PiecewiseLinearPayout(LinearInAmounts):
    """Piecewise-linear payout through given points.

    Parameters
    ----------
    knots : sequence of float or Free
        ``c_0 ... c_{m-1}``, ``m >= 2``, in index units, strictly
        increasing in severity. A free knot needs finite bounds and a grid.
    levels : sequence of float or Free
        ``y_0 ... y_{m-1}``, the payout at each knot, in payout units, at
        least 0. The payout is 0 below ``c_0``, interpolated linearly
        between the knots, and ``y_{m-1}`` at and beyond ``c_{m-1}``.
    tail : {"upper", "lower"}, default "upper"
    monotone : bool, default True
        Whether the levels are non-decreasing, as for :class:`StepPayout`.

    Raises
    ------
    CoverDefinitionError
        If a parameter is invalid, as for :class:`StepPayout`, or ``m < 2``.

    Examples
    --------
    >>> payout = PiecewiseLinearPayout([50.0, 80.0, 120.0], [50.0, 250.0, 700.0])
    >>> payout.evaluate([40.0, 50.0, 65.0, 100.0, 200.0]).tolist()
    [0.0, 50.0, 150.0, 475.0, 700.0]
    """

    kind: ClassVar[str] = "piecewise_linear"

    knots: tuple[Parameter, ...]
    levels: tuple[Parameter, ...]
    tail: str = field(default="upper", kw_only=True)
    monotone: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the parameters and store them as tuples of floats or Free."""
        tail = check_tail(self.tail)
        knots = parameter_tuple("knots", self.knots, shape_parameter, 2)
        check_order("knots", knots, tail)
        levels = parameter_tuple("levels", self.levels, amount_parameter, 2)
        check_lengths(("knots", knots), ("levels", levels))
        monotone = check_flag("monotone", self.monotone)
        if monotone:
            check_monotone("levels", levels)
        _store(self, tail=tail, knots=knots, levels=levels, monotone=monotone)

    def amount_names(self) -> tuple[str, ...]:
        """Return ``("levels[0]", ..., "levels[m-1]")``."""
        return tuple(f"levels[{k}]" for k in range(len(self.knots)))

    def basis(self, index: ArrayLike) -> sp.csr_matrix:
        """Return the interpolation weights of each value on the knots.

        Between ``c_k`` and ``c_{k+1}`` the weights are ``1 - t`` and ``t``,
        ``t = (s - c_k) / (c_{k+1} - c_k)``; from ``c_{m-1}`` on, 1 on the
        last level; below ``c_0``, none. See :meth:`LinearInAmounts.basis`.
        """
        knots = _shape(self, "knots", self.knots)
        severity = _severity_1d(self, index)
        rows, cols, weights = _interpolation(knots, severity)
        matrix = sp.csr_matrix(
            (weights, (rows, cols)), shape=(severity.size, knots.size)
        )
        matrix.eliminate_zeros()
        return matrix

    def amount_constraints(self, limit: float | None = None) -> AmountConstraints:
        """Return the bounds, and ``y_k <= y_{k+1}`` if monotone.

        See :meth:`LinearInAmounts.amount_constraints`.
        """
        constraints = super().amount_constraints(limit)
        return constraints.with_monotone_rows() if self.monotone else constraints


@register_payout
@dataclass(frozen=True)
class BinaryPayout(LinearInAmounts):
    """Binary payout: a fixed amount once the index reaches a trigger.

    Parameters
    ----------
    trigger : float or Free
        Index units. A free trigger needs finite bounds and a grid.
    amount : float or Free
        The payout at and beyond the trigger, in payout units, >= 0.
    tail : {"upper", "lower"}, default "upper"

    Raises
    ------
    CoverDefinitionError
        If a parameter is invalid, as for :class:`StepPayout`.

    Examples
    --------
    >>> BinaryPayout(trigger=210.0, amount=1800.0).evaluate([200.0, 210.0]).tolist()
    [0.0, 1800.0]
    """

    kind: ClassVar[str] = "binary"

    trigger: Parameter
    amount: Parameter
    tail: str = field(default="upper", kw_only=True)

    def __post_init__(self) -> None:
        """Validate the parameters and store floats or Free."""
        tail = check_tail(self.tail)
        trigger = shape_parameter("trigger", self.trigger)
        amount = amount_parameter("amount", self.amount)
        _store(self, tail=tail, trigger=trigger, amount=amount)

    def amount_names(self) -> tuple[str, ...]:
        """Return ``("amount",)``."""
        return ("amount",)

    def basis(self, index: ArrayLike) -> sp.csr_matrix:
        """Return 1 where the severity reaches the trigger, as one column.

        See :meth:`LinearInAmounts.basis`.
        """
        (trigger,) = _shape(self, "trigger", (self.trigger,))
        severity = _severity_1d(self, index)
        rows = np.flatnonzero(severity >= trigger)
        return sp.csr_matrix(
            (np.ones(rows.size), (rows, np.zeros(rows.size, dtype=np.intp))),
            shape=(severity.size, 1),
        )


def _store(payout: PayoutFunction, **values: Any) -> None:
    """Set the fields of a frozen payout in its ``__post_init__``."""
    for name, value in values.items():
        object.__setattr__(payout, name, value)


def _index_values(index: ArrayLike) -> NDArray[np.float64]:
    """Return index values as a finite float array of any shape."""
    try:
        values = np.asarray(index, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ParametricInputError(f"index values must be numeric: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise ParametricInputError("index values must be finite")
    return values


def _severity_1d(payout: PayoutFunction, index: ArrayLike) -> NDArray[np.float64]:
    """Return the severity of one unit's index values, which must be 1-D."""
    severity = payout.severity(index)
    if severity.ndim != 1:
        raise ParametricInputError(
            f"a basis takes one unit's index values, 1-D; got shape {severity.shape}"
        )
    return severity


def _shape(
    payout: PayoutFunction, name: str, values: Sequence[Parameter]
) -> NDArray[np.float64]:
    """Return fixed thresholds in severity; raise if one is free."""
    if any(isinstance(value, Free) for value in values):
        raise UnresolvedParameterError(
            f"the {payout.kind} payout's {name} must be fixed for its basis: resolve "
            "them with with_values"
        )
    thresholds = np.array(values, dtype=np.float64)
    return -thresholds if payout.tail == "lower" else thresholds


def _interpolation(
    knots: NDArray[np.float64], severity: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Return rows, columns and weights of a piecewise-linear basis."""
    last = knots.size - 1
    segment = np.searchsorted(knots, severity, side="right") - 1
    inner = np.flatnonzero((segment >= 0) & (segment < last))
    top = np.flatnonzero(segment == last)
    left = segment[inner]
    share = (severity[inner] - knots[left]) / (knots[left + 1] - knots[left])
    return (
        np.concatenate([inner, inner, top]),
        np.concatenate([left, left + 1, np.full(top.size, last)]),
        np.concatenate([1.0 - share, share, np.ones(top.size)]),
    )


def check_tail(tail: object) -> str:
    """Return ``tail`` if it is ``"upper"`` or ``"lower"``."""
    if not isinstance(tail, str) or tail not in TAILS:
        raise CoverDefinitionError(f"tail must be one of {list(TAILS)}, got {tail!r}")
    return tail
