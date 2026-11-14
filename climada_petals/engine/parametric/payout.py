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

Step payout: a monotone, capped step function of the index.

:class:`StepPayout` pays the level ``theta[j, k]`` when unit ``j``'s index
falls in step ``k``:

``payout[e, j] = sum_k theta[j, k] * 1[index[e, j] in B_k]``,

with ``0 <= theta[j, 0] <= ... <= theta[j, K - 1] <= cap[j]``. The
calibration (:func:`~climada_petals.engine.parametric.calibrate.calibrate`) chooses the
levels; the cap of a unit is its largest loss in the calibration events.

Steps
-----
* The steps are half-open, ``B_k = [b_k, b_{k + 1})``, on breakpoints
  ``attachment = b_0 < b_1 < ... < b_K = limit`` (uniform unless given).
  A value equal to a breakpoint belongs to the step above it.
* An index at or above the limit is in the last step. Values are clipped to
  ``limit - 1e-12`` before the lookup, so the last step must be at least
  ``1e-12`` wide.
* Below the attachment (``below_attachment``):

  - ``"zero"`` (the default): the index is in no step and pays nothing. A
    loss-free event whose index is below the attachment at every unit then
    adds nothing to any outcome, so dropping such events from a hazard set
    changes neither the design nor its premium.
  - ``"first_bin"``: the index falls into the first step and pays
    ``theta[j, 0]``. It exists to compare with designs made in this mode.
    A design with ``theta[j, 0] > 0`` then pays on every event the hazard
    set holds, so it depends on which events were kept.

* Non-finite index values raise
  :class:`~climada_petals.engine.parametric.errors.ParametricInputError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, get_args

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from ._checks import check_real, finite_array
from .errors import ParametricInputError

__all__ = [
    "BELOW_ATTACHMENT_MODES",
    "NO_STEP",
    "UPPER_EPS",
    "BelowAttachment",
    "StepPayout",
]

BelowAttachment = Literal["zero", "first_bin"]
"""What a :class:`StepPayout` pays for an index below the attachment."""

BELOW_ATTACHMENT_MODES: Final[tuple[str, ...]] = get_args(BelowAttachment)
"""The values of :data:`BelowAttachment`: ``("zero", "first_bin")``."""

NO_STEP: Final[int] = -1
"""Step number of an index in no step (below the attachment, ``"zero"``)."""

UPPER_EPS: Final[float] = 1e-12
"""Offset below the limit that index values are clipped to."""


@dataclass(frozen=True)
class StepPayout:
    """Monotone, capped step payout on fixed breakpoints.

    Parameters
    ----------
    attachment : float
        Lowest breakpoint. With ``below_attachment="zero"`` nothing is paid
        below it.
    limit : float
        Highest breakpoint, > ``attachment``. An index at or above it is
        in the last step.
    n_steps : int
        Number of steps (payout levels per unit), >= 1.
    breakpoints : tuple of float, optional
        Explicit breakpoints: ``n_steps + 1`` strictly increasing values
        from ``attachment`` to ``limit``. Uniform if not given.
    below_attachment : {"zero", "first_bin"}, default "zero"
        What an index below the attachment pays: nothing, or the first
        step's level (module docstring).

    Raises
    ------
    ParametricInputError
        If a parameter is invalid: ``n_steps < 1``, ``attachment >= limit``,
        non-finite values, explicit breakpoints of the wrong length, not
        increasing or not running from ``attachment`` to ``limit``, a last
        step narrower than :data:`UPPER_EPS`, or an unknown
        ``below_attachment``.

    Examples
    --------
    Saffir-Simpson categories 1 to 5 in their own steps; category 0 (no
    storm in the circle, or a tropical storm) is below the attachment and
    pays nothing:

    >>> payout = StepPayout(attachment=1, limit=6, n_steps=5)
    >>> payout.edges.tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> payout.step_of([0.0, 1.0, 2.0, 5.0, 7.0]).tolist()
    [-1, 0, 1, 4, 4]
    """

    attachment: float
    limit: float
    n_steps: int
    breakpoints: tuple[float, ...] | None = None
    below_attachment: BelowAttachment = "zero"
    _edges: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the parameters and store the breakpoints read-only."""
        if isinstance(self.n_steps, bool) or not isinstance(
            self.n_steps, int | np.integer
        ):
            raise ParametricInputError(f"n_steps must be an int, got {self.n_steps!r}")
        if self.n_steps < 1:
            raise ParametricInputError(f"n_steps must be >= 1, got {self.n_steps}")
        attachment = check_real("attachment", self.attachment)
        limit = check_real("limit", self.limit)
        if attachment >= limit:
            raise ParametricInputError(
                f"attachment must be < limit, got {attachment} and {limit}"
            )
        if self.below_attachment not in BELOW_ATTACHMENT_MODES:
            raise ParametricInputError(
                f"below_attachment must be one of {list(BELOW_ATTACHMENT_MODES)}, "
                f"got {self.below_attachment!r}"
            )
        object.__setattr__(self, "attachment", attachment)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "n_steps", int(self.n_steps))
        if self.breakpoints is None:
            edges = np.linspace(attachment, limit, self.n_steps + 1)
        else:
            edges = self._check_breakpoints(self.breakpoints)
            object.__setattr__(self, "breakpoints", tuple(edges.tolist()))
        if edges[-1] - UPPER_EPS < edges[-2]:
            raise ParametricInputError(
                f"the last step [{edges[-2]}, {edges[-1]}) must be at least "
                f"{UPPER_EPS} wide"
            )
        edges.setflags(write=False)
        object.__setattr__(self, "_edges", edges)

    def _check_breakpoints(self, breakpoints: object) -> NDArray[np.float64]:
        """Validate explicit breakpoints and return them as an array."""
        try:
            edges = np.array(breakpoints, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ParametricInputError(f"breakpoints must be numeric: {exc}") from exc
        if edges.shape != (self.n_steps + 1,):
            raise ParametricInputError(
                f"breakpoints must hold n_steps + 1 = {self.n_steps + 1} values, "
                f"got shape {edges.shape}"
            )
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ParametricInputError(
                "breakpoints must be finite and strictly increasing"
            )
        if edges[0] != self.attachment or edges[-1] != self.limit:
            raise ParametricInputError(
                f"breakpoints must run from attachment={self.attachment} to "
                f"limit={self.limit}, got {edges[0]} to {edges[-1]}"
            )
        return edges

    @property
    def edges(self) -> NDArray[np.float64]:
        """Breakpoints ``b_0 < ... < b_K``, shape ``(n_steps + 1,)``, read-only."""
        return self._edges

    def step_of(self, index: ArrayLike) -> NDArray[np.intp]:
        """Return the step of each index value.

        Parameters
        ----------
        index : array_like of float
            Index values, any shape. Finite.

        Returns
        -------
        numpy.ndarray of int
            Same shape as ``index``, values in ``[0, n_steps - 1]``, and
            :data:`NO_STEP` (-1) below the attachment under
            ``below_attachment="zero"``.

        Raises
        ------
        ParametricInputError
            If ``index`` is not numeric or not finite.
        """
        values = finite_array(index, "index", np.ndim(index))
        edges = self._edges
        clipped = np.clip(values, edges[0], edges[-1] - UPPER_EPS)
        step = np.searchsorted(edges, clipped, side="right") - 1
        step = np.clip(step, 0, self.n_steps - 1).astype(np.intp)
        if self.below_attachment == "zero":
            step = np.where(values < edges[0], NO_STEP, step).astype(np.intp)
        return step

    def basis_matrix(self, index: ArrayLike) -> sp.csr_matrix:
        """Return the one-hot step membership of one unit's index.

        Parameters
        ----------
        index : array_like of float
            Index values of one unit, shape ``(n_events,)``. Finite.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_events, n_steps)``: entry ``(e, k)`` is 1 if event
            ``e`` is in step ``k``. A row is empty when the index is in no
            step, so that event pays nothing for any levels. The unit's
            payout is ``basis_matrix(index) @ theta[j]``.

        Raises
        ------
        ParametricInputError
            If ``index`` is not 1-D or not finite.
        """
        values = finite_array(index, "index", 1)
        step = self.step_of(values)
        rows = np.flatnonzero(step != NO_STEP)
        return sp.csr_matrix(
            (np.ones(rows.size), (rows, step[rows])),
            shape=(values.shape[0], self.n_steps),
        )

    def evaluate(self, index: ArrayLike, levels: ArrayLike) -> NDArray[np.float64]:
        """Return the payout per event and unit.

        Parameters
        ----------
        index : array_like of float
            Shape ``(n_events, n_units)``. Finite.
        levels : array_like of float
            ``theta``, shape ``(n_units, n_steps)``. Finite. Not checked
            against monotonicity or the caps, so any levels can be evaluated.

        Returns
        -------
        numpy.ndarray
            ``out[e, j] = levels[j, step_of(index[e, j])]``, and 0 where the
            index is in no step. Shape ``(n_events, n_units)``.

        Raises
        ------
        ParametricInputError
            If the shapes do not fit or a value is not finite.
        """
        values = finite_array(index, "index", 2)
        theta = finite_array(levels, "levels", 2)
        want = (values.shape[1], self.n_steps)
        if theta.shape != want:
            raise ParametricInputError(
                f"levels must have shape {want}, got {theta.shape}"
            )
        step = self.step_of(values)
        unit = np.arange(values.shape[1])[None, :]
        unpaid = step == NO_STEP
        return np.where(unpaid, 0.0, theta[unit, np.where(unpaid, 0, step)])
