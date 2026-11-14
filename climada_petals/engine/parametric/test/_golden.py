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

The golden cases of the tests, read from ``data/``.

The cases in ``data/cases`` and their samples in ``data/samples`` are copied
unchanged from the parity fixtures of the parametric-insurance-design
repository (``parity/climada/``): small synthetic samples, and the values a
thesis-faithful reference implementation, or a closed form, gives on them.
They are dedicated to the public domain (CC0-1.0); every sample records it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from climada.entity import Exposures

from climada_petals.engine.parametric import (
    MSE,
    Budget,
    CatalogueYears,
    CostOfCapitalLoad,
    EventBasis,
    ExpectedUncovered,
    IndexDefinition,
    InsuredUnits,
    MetricOptions,
    OutcomeSample,
    ParametricIndex,
    PurePremium,
    WeightedCVaR,
)

DATA = Path(__file__).parent / "data"

#: The levels and return periods the golden cases list.
GOLDEN = MetricOptions(betas=(0.9, 0.99), return_periods=(2, 10, 50))

#: Metrics without a unit: their absolute floor is 1e-9, not 1e-9 x the scale.
UNITLESS = (
    "coverage_ratio",
    "payout_to_loss_ratio",
    "payout_frequency",
    "correlation",
    "tail_coverage_",
    "cvar_uncovered_to_loss_",
    "false_negative_rate_",
    "false_positive_rate_",
)


def golden_sample(case: dict[str, Any]) -> tuple[OutcomeSample, float]:
    """Return a case's sample, every event kept, and the case's scale."""
    raw = json.loads((DATA / case["sample"]).read_text(encoding="utf-8"))
    n_units = len(raw["unit_id"])
    exp = Exposures(
        lat=np.full(n_units, 10.0),
        lon=np.linspace(0.0, 1.0, n_units),
        value=np.ones(n_units),
    )
    exp.gdf.index = raw["unit_id"]
    units = InsuredUnits.points(exp)
    definition = IndexDefinition.from_dict(case["cover"]["index"])
    index = ParametricIndex(raw["index"], raw["event_id"], units.unit_id, definition)
    if case["basis"]["kind"] == "event":
        basis: Any = EventBasis()
    else:
        basis = CatalogueYears.from_labels(
            raw["event_id"], raw["year"], n_years=case["basis"]["n_years"]
        )
    loss = np.asarray(raw["loss"], dtype=float)
    sample = OutcomeSample(index, loss, raw["frequency"], units, basis)
    return sample, max(1.0, float(loss.sum(axis=1).max()))


def golden_pricing(document: dict[str, Any]) -> Any:
    """Return a case's pricing."""
    if document["kind"] == "pure":
        return PurePremium()
    return CostOfCapitalLoad(rate=document["rate"], q=document["q"])


def golden_objective(document: dict[str, Any]) -> Any:
    """Return a case's objective."""
    if document["kind"] == "mse":
        return MSE()
    if document["kind"] == "expected_uncovered":
        return ExpectedUncovered()
    return WeightedCVaR(alpha=document["alpha"], beta=document["beta"])


def golden_budget(document: dict[str, Any] | None) -> Budget | None:
    """Return a case's budget."""
    return None if document is None else Budget(ratio=document["ratio"])


def golden_cases(kind: str) -> list[dict[str, Any]]:
    """Return the golden cases of one kind."""
    paths = sorted((DATA / "cases").glob("*.json"))
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return [case for case in cases if case["kind"] == kind]


def within(expected: Any, got: float, floor: float) -> bool:
    """Whether a value is within 1e-6 relative of a case's value, or ``floor``.

    ``expected`` is a number, or ``{"either": [lower, upper]}`` for a
    quantile on a step, where either passes.
    """
    candidates = expected["either"] if isinstance(expected, dict) else [expected]
    return any(abs(got - c) <= max(1e-6 * abs(c), floor) for c in candidates)


def metric_floor(name: str, scale: float) -> float:
    """Return the absolute floor of a metric's tolerance."""
    return 1e-9 if name.startswith(UNITLESS) else 1e-9 * scale


def is_null(value: float) -> bool:
    """Whether a table value is null."""
    return math.isnan(value)
