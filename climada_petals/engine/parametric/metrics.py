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

Basis-risk metrics of a calibrated cover.

:func:`basis_risk` evaluates a cover on an
:class:`~climada_petals.engine.parametric.sample.OutcomeSample`, usually the test
side of a split that the cover was not calibrated on. With outcome probabilities
``p_o`` (:mod:`~climada_petals.engine.parametric.measure`) and, for each subject, the
outcome loss ``L_o``, payout ``P_o``, uncovered loss
``U_o = sum_{e in o} (L_e - P_e)^+`` and residual ``R_o = L_o - P_o``:

==========================  ===============================================
``expected_loss``           ``E[L]``
``expected_payout``         ``E[P]``
``expected_uncovered``      ``E[U]``
``var_uncovered``           ``VaR_beta(U)``, the lower ``beta``-quantile
``cvar_uncovered``          ``CVaR_beta(U)`` (Rockafellar-Uryasev)
``rmse_residual``           ``sqrt(E[R^2])``
``payout_to_loss_ratio``    ``E[P] / E[L]``; NaN if ``E[L] = 0``
``correlation``             correlation of ``L_o`` and ``P_o`` weighted with
                            ``p_o``; NaN if either has no variance
``premium``                 the premium of the subject's payout under the
                            cover's pricing; NaN for a unit of a portfolio
                            cover, whose payout is part of one design problem
==========================  ===============================================

Subjects: each unit (``level = "unit"``) and the owner's total over every
unit (``level = "portfolio"``). The metrics of a unit of a **portfolio**
cover describe the one optimal split of the payout that the solver
returned, which is not unique (:mod:`~climada_petals.engine.parametric.calibrate`); their
``split_identified`` column is False. The ``"portfolio"`` row is identified
in both scopes.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._checks import check_level
from .calibrate import PORTFOLIO, CalibrationResult
from .errors import ParametricInputError
from .measure import OutcomeMeasure
from .riskmeasures import var_cvar
from .sample import OutcomeSample

__all__ = ["METRICS", "basis_risk", "subject_metrics"]

METRICS: Final[tuple[str, ...]] = (
    "expected_loss",
    "expected_payout",
    "expected_uncovered",
    "var_uncovered",
    "cvar_uncovered",
    "rmse_residual",
    "payout_to_loss_ratio",
    "correlation",
)
"""The metrics of every subject, in column order (``premium`` follows)."""


def _weighted_correlation(
    first: NDArray[np.float64], second: NDArray[np.float64], prob: NDArray[np.float64]
) -> float:
    """Correlation of two outcome vectors weighted with ``prob``; NaN if flat."""
    mean_x, mean_y = float(prob @ first), float(prob @ second)
    cov = float(prob @ ((first - mean_x) * (second - mean_y)))
    var_x = float(prob @ (first - mean_x) ** 2)
    var_y = float(prob @ (second - mean_y) ** 2)
    if var_x <= 0.0 or var_y <= 0.0:
        return math.nan
    return cov / math.sqrt(var_x * var_y)


def subject_metrics(
    loss: ArrayLike, payout: ArrayLike, measure: OutcomeMeasure, *, beta: float
) -> dict[str, float]:
    """Return the metrics of one subject (module docstring), without the premium.

    Parameters
    ----------
    loss, payout : array_like of float
        Per-event loss and payout of the subject, shape ``(n_events,)``.
    measure : OutcomeMeasure
        Measure over the same events.
    beta : float
        Level of ``var_uncovered`` and ``cvar_uncovered``, in ``(0, 1)``.

    Returns
    -------
    dict of str to float
        One value per name in :data:`METRICS`.
    """
    level = check_level("beta", beta)
    loss_e = np.asarray(loss, dtype=np.float64)
    paid_e = np.asarray(payout, dtype=np.float64)
    prob = measure.probability
    loss_o = measure.aggregate(loss_e)
    paid_o = measure.aggregate(paid_e)
    uncovered = measure.aggregate(np.maximum(loss_e - paid_e, 0.0))
    var, cvar = var_cvar(uncovered, prob, level)
    expected_loss = float(prob @ loss_o)
    expected_payout = float(prob @ paid_o)
    return {
        "expected_loss": expected_loss,
        "expected_payout": expected_payout,
        "expected_uncovered": float(prob @ uncovered),
        "var_uncovered": var,
        "cvar_uncovered": cvar,
        "rmse_residual": math.sqrt(float(prob @ (loss_o - paid_o) ** 2)),
        "payout_to_loss_ratio": (
            expected_payout / expected_loss if expected_loss > 0 else math.nan
        ),
        "correlation": _weighted_correlation(loss_o, paid_o, prob),
    }


def _unit_rows(
    cover: CalibrationResult,
    loss: NDArray[np.float64],
    payout: NDArray[np.float64],
    measure: OutcomeMeasure,
    *,
    unit_id: Sequence[Hashable],
    beta: float,
) -> list[dict[str, Any]]:
    """Return the rows of :func:`basis_risk` of each unit."""
    joint = cover.scope == PORTFOLIO
    rows: list[dict[str, Any]] = []
    for j, unit in enumerate(unit_id):
        premium = math.nan if joint else cover.pricing.premium(payout[:, j], measure)
        rows.append(
            {
                "level": "unit",
                "subject": unit,
                **subject_metrics(loss[:, j], payout[:, j], measure, beta=beta),
                "premium": premium,
                "split_identified": not joint,
            }
        )
    return rows


def basis_risk(
    sample: OutcomeSample, cover: CalibrationResult, *, beta: float
) -> pd.DataFrame:
    """Return the basis-risk metrics of a cover (module docstring).

    Parameters
    ----------
    sample : OutcomeSample
        The evaluation sample, with the cover's units, e.g. the test side
        of :func:`~climada_petals.engine.parametric.sample.holdout`. Its basis
        is the metrics' basis.
    cover : CalibrationResult
        The calibrated cover.
    beta : float
        Level of ``var_uncovered`` and ``cvar_uncovered``, in ``(0, 1)``.
        On the annual basis, ``beta = 0.99`` is the 100-year level.

    Returns
    -------
    pandas.DataFrame
        One row per unit and one ``"portfolio"`` row, with the columns
        ``level``, ``subject``, ``basis``, ``beta``, ``n_years`` (None on
        the event basis), the :data:`METRICS`, ``premium`` and
        ``split_identified``.

    Raises
    ------
    ParametricInputError
        If ``sample`` is not an ``OutcomeSample``, its units are not the
        cover's, or it is on the event basis with frequencies that sum to
        more than 1 per year.
    """
    if not isinstance(sample, OutcomeSample):
        raise ParametricInputError(
            f"sample must be an OutcomeSample, got {type(sample).__name__}"
        )
    level = check_level("beta", beta)
    payout = cover.payout_matrix(sample.index)[sample.event_row]
    loss = sample.loss.toarray()
    measure = sample.measure.as_probability()
    rows = _unit_rows(cover, loss, payout, measure, unit_id=sample.unit_id, beta=level)
    total_paid = payout.sum(axis=1)
    rows.append(
        {
            "level": PORTFOLIO,
            "subject": PORTFOLIO,
            **subject_metrics(loss.sum(axis=1), total_paid, measure, beta=level),
            "premium": cover.pricing.premium(total_paid, measure),
            "split_identified": True,
        }
    )
    frame = pd.DataFrame(rows)
    frame.insert(2, "basis", measure.basis)
    frame.insert(3, "beta", level)
    frame.insert(4, "n_years", measure.n_years)
    return frame
