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

The loss of one ``Impact``, or of one per draw, on the insured units.

:meth:`~climada_petals.engine.parametric.sample.OutcomeSample.from_impact`
and :meth:`~climada_petals.engine.parametric.sample.OutcomeSample.from_impact_draws`
read their impacts here, with one rule for the events they keep: an event
with a loss in some draw or a non-zero index at some unit, or every event
on a lower-tail index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from climada_petals.util.config import LOGGER

from .errors import EventAlignmentError, ImpactInputError, ParametricInputError
from .index import ParametricIndex
from .units import InsuredUnits

__all__ = ["ImpactLosses", "impact_losses", "take_events"]


class ImpactLosses(NamedTuple):
    """The kept events of one or several impacts, as a sample takes them.

    Attributes
    ----------
    index : ParametricIndex
        The index of the kept events, in the index's order.
    loss : scipy.sparse.csr_matrix
        The loss per row and unit, ``(n_draws n_kept, n_units)``: rows are
        ordered by draw, then event.
    frequency : numpy.ndarray of float
        Each kept event's frequency per year.
    n_draws : int
        The impacts, one per draw.
    n_left_out : int
        The events left out.
    """

    index: ParametricIndex
    loss: sp.csr_matrix
    frequency: NDArray[np.float64]
    n_draws: int
    n_left_out: int


def take_events(index: ParametricIndex, rows: NDArray[np.intp]) -> ParametricIndex:
    """Return the index of the events at ``rows``."""
    names = index.event_name
    return ParametricIndex(
        index.values[rows],
        index.event_id[rows],
        index.unit_id,
        index.definition,
        event_name=None if names is None else [names[row] for row in rows],
    )


def _checked(impacts: object) -> list[Any]:
    """Return the impacts as a list, if they are a non-empty sequence.

    Raises
    ------
    ParametricInputError
        If ``impacts`` is not a non-empty sequence.
    """
    if isinstance(impacts, str) or not isinstance(impacts, Sequence) or not impacts:
        raise ParametricInputError(
            "impacts must be a non-empty sequence of Impact, one per draw; for one "
            f"Impact, use OutcomeSample.from_impact. Got {type(impacts).__name__}"
        )
    return list(impacts)


def _draw_loss(
    index: ParametricIndex, impact: Any, units: InsuredUnits
) -> tuple[sp.csr_matrix, NDArray[np.float64]]:
    """Return one impact's loss per event and unit, and frequencies, in the index's order."""
    units.check_impact(impact)
    rows = index.rows_in(impact)
    loss = units.aggregate(impact.imp_mat)[rows]
    return loss, np.asarray(impact.frequency, dtype=np.float64)[rows]


def _aligned(
    draw: int, found: tuple[Any, NDArray[np.float64]], first: tuple[Any, Any]
) -> None:
    """Raise unless draw ``draw`` has the first draw's frequencies and loss unit.

    ``found`` is the draw's impact and frequencies; ``first`` the first
    draw's.

    Raises
    ------
    EventAlignmentError
        If the frequencies differ.
    ImpactInputError
        If the impacts' units differ.
    """
    impact, frequency = found
    if not np.allclose(frequency, first[1], rtol=1e-12, atol=0.0):
        raise EventAlignmentError(
            f"draw {draw}'s frequencies are not draw 0's: the draws must be impacts "
            "of one hazard set, with its events and their frequencies"
        )
    if str(impact.unit) != str(first[0].unit):
        raise ImpactInputError(
            f"draw {draw}'s loss is in {impact.unit!r} and draw 0's in "
            f"{first[0].unit!r}"
        )


def impact_losses(
    index: ParametricIndex, impacts: Sequence[Any], units: InsuredUnits
) -> ImpactLosses:
    """Return the kept events' index, loss per draw and unit, and frequencies.

    Parameters
    ----------
    index : ParametricIndex
        The index of the impacts' events, matched on ``event_id`` in any
        order.
    impacts : sequence of climada.engine.Impact
        One per draw, each on the units' exposure points, with the index's
        events and the same frequencies.
    units : InsuredUnits
        The insured units, with the index's ``unit_id`` in order.

    Returns
    -------
    ImpactLosses

    Raises
    ------
    ImpactInputError
        If an impact can't be read as a loss on the units (``check_impact``),
        or the impacts' units differ.
    EventAlignmentError
        If an impact's events are not the index's, or two impacts'
        frequencies differ. With several draws, the error's note names the
        draw.
    ParametricInputError
        If ``impacts`` is not a non-empty sequence, or no event has a loss or
        a non-zero index.
    """
    found = []
    for draw, impact in enumerate(_checked(impacts)):
        try:
            loss, frequency = _draw_loss(index, impact, units)
            if found:
                _aligned(draw, (impact, frequency), (impacts[0], found[0][1]))
        except ParametricInputError as error:
            if len(impacts) > 1:
                error.add_note(f"raised by the impact of draw {draw}")
            raise
        found.append((loss, frequency))
    lossy = np.zeros(index.n_events, dtype=bool)
    for loss, _ in found:
        lossy |= np.asarray(abs(loss).sum(axis=1)).reshape(-1) > 0
    keep = lossy | np.any(index.values != 0, axis=1)
    if index.definition.tail == "lower":
        keep[:] = True
    if not keep.any():
        raise ParametricInputError(
            "no event of the impact has a loss or a non-zero index"
        )
    left_out = int((~keep).sum())
    if left_out:
        LOGGER.info(
            "outcome sample: %d of %d events have no loss and an index of 0 at "
            "every unit, and are left out",
            left_out,
            keep.size,
        )
    kept = np.flatnonzero(keep)
    return ImpactLosses(
        take_events(index, kept),
        sp.vstack([loss[kept] for loss, _ in found], format="csr"),
        found[0][1][kept],
        len(found),
        left_out,
    )
