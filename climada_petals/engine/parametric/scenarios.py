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

Scenarios: a cover on other hazard sets, and possibly other exposures.

In CLIMADA a **scenario** is another ``Hazard``, and possibly other
``Exposures`` or impact functions: a future climate from the Data API, a set
scaled by ``TropCyclone.apply_climate_scenario_knu``, an exposure
projection. A cover holds terms and nothing else, no event ids and no
frequencies, so it pays on any of them;
:meth:`~climada_petals.engine.parametric.index.ParametricIndex.from_definition`
rebuilds its index on another hazard set.

* :class:`ScenarioSet` maps labels to scenarios, each an
  :class:`~climada_petals.engine.parametric.sample.OutcomeSample` or several
  of them, which form one scenario (below). It builds the samples from an
  index and an ``Impact`` per scenario (:meth:`ScenarioSet.from_impacts`),
  or from the ``Hazard``, the units' ``Exposures`` and the ``ImpactFuncSet``
  (:meth:`ScenarioSet.from_hazards`).
* :func:`evaluate_scenarios` evaluates one cover, designed or given as it
  is, on every scenario: :func:`~climada_petals.engine.parametric.evaluate.evaluate`'s
  table per scenario, with the label in the ``scenario`` column.
* :func:`calibrate_per_scenario` designs one cover per scenario with
  :func:`~climada_petals.engine.parametric.calibrate.calibrate`, and
  :func:`designs_table` puts the designs side by side.

These are measurements. No table ranks the scenarios or the covers, and
nothing here names a best one. Robust design over a set of scenarios and
CLIMADA's ``trajectory`` module are not part of this module yet.

One route across a comparison
-----------------------------
Every sample of a set is on the same basis and route: the event basis,
catalogue years or year sets. The route changes tail numbers, so a
comparison across two routes would compare the routes too; a set that
holds two raises
:class:`~climada_petals.engine.parametric.errors.ScenarioError`. The samples
also share the insured units (their ids) and the index definition, since
one cover pays on them all. The number of years may differ, and every row
of a table records it (``n_years``).

======================================  =====================================
Source                                  Route
======================================  =====================================
A set scaled by                         The event basis, or year sets
``apply_climate_scenario_knu``          (``YearSets.from_frequencies``): it
                                        changes frequencies only, so they are
                                        no longer ``1 / N``, and catalogue
                                        years raise ``YearRouteError``
Future tropical cyclones of the Data    Year sets on the whole file, or the
API, ``random_walk`` (properties        event basis: the frequencies are not
``climate_scenario`` and ``ref_year``)  uniform, and differ between the
                                        members of one parent. Catalogue
                                        years only on events of frequency
                                        ``1 / N``, as for the present
STORM per climate model                 Catalogue years from the event names,
                                        as for the present
Any other hazard set                    The route its present-day set takes
                                        (:mod:`~climada_petals.engine.parametric.measure`)
======================================  =====================================

On the event basis, a frequency-scaled set keeps the events, so its
difference from the baseline holds no sampling noise. On year sets every
scenario has its own sampling noise: sample at least 10,000 years.

Windows and members: several samples under one label
----------------------------------------------------
One horizon of one model can be noisy. Several samples under one label form
**one scenario, an equal-weight mixture** (:mod:`~climada_petals.engine.parametric._mixture`):
the windows of a pathway, such as one hazard set per reference year, or the
members of an ensemble, such as one set per climate model::

    ScenarioSet({"rcp45, 2040 to 2080": [s_2040, s_2060, s_2080]})

On the event basis each member's frequencies are divided by ``m``, the
number of members; on an annual route each year of member ``i`` has the
weight ``1 / (m N_i)``, so the mixture's ``n_years`` is the sum of the
members'. The members have the same units, exposures included, the same
number of impact-function draws, and come from the same side of one split,
or none.

Other exposures, and value weights
----------------------------------
A scenario with other exposures has its own
:class:`~climada_petals.engine.parametric.units.InsuredUnits`, with the
same unit ids: its losses and the value shares of its payouts follow its
exposures. An index with ``weight="value"`` weights the centroids by the
values of the units it is computed on. :meth:`ScenarioSet.from_hazards`
computes each scenario's index on that scenario's units unless
``index_units`` is given: the units at inception keep the weights of the
contract's index, as a contract states them, while the scenario's own
units weight it by the scenario's exposures. Both are a stated choice; the
index definition, and so every table, is the same for both.

Splits and leakage
------------------
Split the members, not a mixture: :meth:`ScenarioSet.holdout` splits each
member and returns a training and a test set with the same labels; members
of one source, such as a baseline and its frequency-scaled set, are cut
with the same blocks, and members that share parent storms, such as the
Data API's present and future ``random_walk`` sets, with the same sides
of those storms. On an annual route a split of the mixture itself would
not mix the members' sides (on year sets, one year would hold the events
of several members), so ``holdout``, ``year_block_kfold`` and ``select``
raise :class:`~climada_petals.engine.parametric.errors.ScenarioError` on
it. On the event basis a mixture is one set of events, which they split as
any other.

:func:`evaluate_scenarios` checks an out-of-sample evaluation of a design
for leakage as :func:`~climada_petals.engine.parametric.evaluate.evaluate`
does, by the samples' source tokens
(:func:`~climada_petals.engine.parametric.sample.check_out_of_sample`):
every scenario's sample, and every member of a mixture, of the design
sample's source is compared with it, whether whole or cut. A mixture has
the token of its members' tokens, so a design on a mixture is compared
with the mixtures of the same members' sides, or of the whole members. The
members of a mixture are different hazard sets, whose event ids and blocks
say nothing about each other, so the mixture keeps them apart
(:mod:`~climada_petals.engine.parametric._mixture`): the same storm in two
members counts as two blocks. Parent storms are the exception: they are
compared whatever the source, so a design on the present set is out of
sample only on future sets cut with its sides.

A set scaled by ``apply_climate_scenario_knu`` holds its baseline's events:
a design calibrated on the baseline has seen every footprint of it. It has
the baseline's source token (the token identifies the events, not their
frequencies, exposures or impact functions), so an out-of-sample
evaluation of a baseline design on it is compared, and raises on the
events the design was calibrated on. :meth:`ScenarioSet.holdout` cuts the
members of one source with the same blocks, so a design on the baseline's
training side is out of sample on the scaled set's test side, which shows
the cover under other frequencies on unseen events.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from climada.engine import ImpactCalc
from climada.hazard import Hazard

from ._mixture import (
    as_sequence,
    labelled_items,
    mixture_sample,
    scenario_members,
    where,
)
from ._parts import member_sides
from .calibrate import CalibrationOptions, CalibrationResult, DesignSample, calibrate
from .constraints import Budget
from .cover import IndexDefinition, ParametricCover
from .errors import ParametricError, ParametricInputError
from .evaluate import evaluate
from .index import ParametricIndex
from .metrics import MetricOptions
from .objectives import Objective
from .pricing import Pricing, PurePremium
from .sample import Basis, OutcomeSample, check_no_leakage, check_out_of_sample, holdout
from .units import InsuredUnits

__all__ = [
    "ScenarioSet",
    "calibrate_per_scenario",
    "designs_table",
    "evaluate_scenarios",
]

BasisSource = Basis | Callable[[Any], Basis]
"""A basis for every member, or a function of a member's hazard or impact."""


def _for_label(value: Any, label: str, what: str) -> Any:
    """Return ``value``, or its entry for ``label`` if it is a mapping by label.

    Raises
    ------
    ParametricInputError
        If a mapping has no entry for ``label``.
    """
    if not isinstance(value, Mapping):
        return value
    if label not in value:
        raise ParametricInputError(f"{what} has no entry for scenario {label!r}")
    return value[label]


def _basis_for(basis: BasisSource, source: Any) -> Any:
    """Return the basis, or the basis a function gives for a hazard or an impact."""
    if callable(basis):
        return basis(source)
    return basis


def _is_pair(value: object) -> bool:
    """Whether ``value`` is an index and an impact, as a tuple."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], ParametricIndex)
    )


@dataclass(frozen=True)
class _HazardSetup:
    """The arguments of :meth:`ScenarioSet.from_hazards` but the hazards."""

    definition: IndexDefinition
    units: Any
    impact_funcs: Any
    basis: BasisSource
    index_units: InsuredUnits | None


def _hazard_sample(hazard: Any, label: str, *, setup: _HazardSetup) -> OutcomeSample:
    """Return the sample of one hazard set: its index, its impact and its basis.

    Raises
    ------
    ParametricInputError
        If ``hazard`` is not a ``Hazard``, or ``index_units`` has other unit
        ids than the scenario's units.
    """
    if not isinstance(hazard, Hazard):
        raise ParametricInputError(
            f"a member is a climada Hazard, got {type(hazard).__name__}"
        )
    units = _for_label(setup.units, label, "units")
    index_units = units if setup.index_units is None else setup.index_units
    if tuple(index_units.unit_id) != tuple(units.unit_id):
        raise ParametricInputError(
            f"index_units has the units {list(index_units.unit_id)[:5]} and the "
            f"scenario {list(units.unit_id)[:5]}: they must be the same unit ids, in "
            "order"
        )
    index = ParametricIndex.from_definition(setup.definition, hazard, index_units)
    impact = ImpactCalc(
        units.exposures.copy(),
        _for_label(setup.impact_funcs, label, "impact_funcs"),
        hazard,
    ).impact(save_mat=True)
    return OutcomeSample.from_impact(
        index,
        impact,
        units=units,
        basis=_basis_for(setup.basis, hazard),
        scenario=label,
    )


def _impact_sample(
    pair: Any, label: str, *, units: Any, basis: BasisSource
) -> OutcomeSample:
    """Return the sample of one index and impact.

    Raises
    ------
    ParametricInputError
        If ``pair`` is not a tuple of an index and an impact.
    """
    if not _is_pair(pair):
        raise ParametricInputError(
            "a member is a tuple (index, impact), with a ParametricIndex first; "
            f"got {type(pair).__name__}"
        )
    return OutcomeSample.from_impact(
        pair[0],
        pair[1],
        units=_for_label(units, label, "units"),
        basis=_basis_for(basis, pair[1]),
        scenario=label,
    )


def _built(
    members: object,
    is_one: Callable[[object], bool],
    build: Callable[[Any, str], OutcomeSample],
    method: str,
) -> dict[str, list[OutcomeSample]]:
    """Return each label's samples, ``build(item, label)`` for each of its items.

    A label maps to one item (``is_one``) or a non-empty sequence of them.
    An error has a note naming the method, the scenario and the member.
    """
    samples: dict[str, list[OutcomeSample]] = {}
    for label, value in labelled_items(members):
        items = (value,) if is_one(value) else as_sequence(value, f"scenario {label!r}")
        for position, item in enumerate(items):
            try:
                samples.setdefault(label, []).append(build(item, label))
            except (ParametricError, ValueError) as error:
                error.add_note(
                    f"raised by ScenarioSet.{method} at "
                    + where(label, position, len(items))
                )
                raise
    return samples


@dataclass(frozen=True, eq=False, repr=False)
class ScenarioSet:
    """Labels mapped to scenarios, each one sample or a mixture (module docstring).

    Parameters
    ----------
    members : mapping of str to OutcomeSample or sequence of OutcomeSample
        Each label to its sample, or to several samples that form one
        scenario as an equal-weight mixture: the windows of a pathway or
        the members of an ensemble. Every sample is on one basis and route,
        with the same unit ids and index definition (module docstring).
        Each sample takes the label as its ``scenario``.

    Attributes
    ----------
    members : mapping of str to tuple of OutcomeSample
        The samples of each label, read-only.

    Raises
    ------
    ScenarioError
        If two samples are on other routes, units or index definitions, or
        the members of a mixture have other exposures, draws or splits.
    ParametricInputError
        If ``members`` is not a non-empty mapping of non-empty string labels
        to samples or non-empty sequences of them.

    Examples
    --------
    >>> scenarios = ScenarioSet(  # doctest: +SKIP
    ...     {
    ...         "baseline": test,
    ...         "rcp45, 2040 to 2080": [s_2040, s_2060, s_2080],
    ...     }
    ... )
    >>> evaluate_scenarios(design, scenarios, split="out_of_sample")  # doctest: +SKIP
    """

    members: Mapping[str, Any]
    _samples: Mapping[str, OutcomeSample] = field(init=False)
    _folds: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Check the samples, label them and build the mixtures."""
        found = scenario_members(self.members)
        samples = {
            label: group[0] if len(group) == 1 else mixture_sample(label, group)
            for label, group in found.items()
        }
        object.__setattr__(self, "members", MappingProxyType(found))
        object.__setattr__(self, "_samples", MappingProxyType(samples))
        object.__setattr__(
            self,
            "_folds",
            MappingProxyType(
                {
                    label: 0 if group[0].split is None else group[0].split.fold
                    for label, group in found.items()
                }
            ),
        )

    def __repr__(self) -> str:
        """Return the labels, the basis and the route."""
        return (
            f"ScenarioSet(labels={list(self.labels)!r}, basis={self.basis!r}, "
            f"route={self.route!r})"
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """The scenarios' labels, in order."""
        return tuple(self.members)

    @property
    def basis(self) -> str:
        """The samples' basis, ``"event"`` or ``"annual"``."""
        return self._samples[self.labels[0]].kind

    @property
    def route(self) -> str:
        """The samples' route, ``"frequency"``, ``"catalogue"`` or ``"yearsets"``."""
        return self._samples[self.labels[0]].route

    @property
    def unit_id(self) -> tuple[Hashable, ...]:
        """The insured units' ids, in order."""
        return self._samples[self.labels[0]].unit_id

    @property
    def definition(self) -> IndexDefinition:
        """The index definition of every sample, the one a cover pays on."""
        return self._samples[self.labels[0]].index.definition

    def sample(self, label: str) -> OutcomeSample:
        """Return a scenario's sample: its only member, or its members' mixture.

        Parameters
        ----------
        label : str
            The scenario.

        Returns
        -------
        OutcomeSample
            With ``scenario`` the label. A mixture is a whole sample of the
            members' events, numbered and labelled by member
            (:mod:`~climada_petals.engine.parametric._mixture`).

        Raises
        ------
        ParametricInputError
            If the set has no such scenario.
        """
        if label not in self._samples:
            raise ParametricInputError(
                f"the set has no scenario {label!r}; it has {list(self.labels)}"
            )
        return self._samples[label]

    def fold(self, label: str) -> int:
        """Return the fold a scenario's members were cut as; 0 for whole samples.

        Raises
        ------
        ParametricInputError
            If the set has no such scenario.
        """
        self.sample(label)
        return self._folds[label]

    def summary(self) -> pd.DataFrame:
        """Return one row per scenario: its members and its sample's setup.

        Returns
        -------
        pandas.DataFrame
            Columns ``scenario``, ``members`` (the number of samples),
            ``basis``, ``route``, ``n_years`` (of the mixture: the sum of the
            members'; null on the event basis), ``n_events``, ``n_draws``,
            ``split`` (the members' side of a split, or None) and ``fold``.
        """
        rows = []
        for label, group in self.members.items():
            sample = self._samples[label]
            rows.append(
                {
                    "scenario": label,
                    "members": len(group),
                    "basis": sample.kind,
                    "route": sample.route,
                    "n_years": sample.n_years,
                    "n_events": sample.n_events,
                    "n_draws": sample.n_draws,
                    "split": None if group[0].split is None else group[0].split.side,
                    "fold": self._folds[label],
                }
            )
        frame = pd.DataFrame(rows)
        frame["n_years"] = frame["n_years"].astype("Int64")
        return frame

    def holdout(
        self, *, train_fraction: float = 0.7, rng: np.random.Generator | None = None
    ) -> tuple[ScenarioSet, ScenarioSet]:
        """Split every member of every scenario; return a training and a test set.

        Each member is split by
        :func:`~climada_petals.engine.parametric.sample.holdout`, in the
        order of the labels and members, with the same ``rng``, except one
        that holds what a member split before it holds:

        * its events, by the source token: a set scaled by
          ``apply_climate_scenario_knu`` and its baseline, say. It is cut
          with the blocks of the first one's sides (on catalogue years, its
          years);
        * parent storms (:mod:`~climada_petals.engine.parametric.sample`,
          "Parent storms"): the Data API's future ``random_walk`` sets hold
          the present set's, say. Each goes to its side there, and other
          blocks are pooled as ``holdout`` pools them; on catalogue years
          it takes the years of the earlier member's sides.

        So a design on one side is out of sample on the other side of every
        such member. Its frequencies are divided by its own share on each
        side, and on year sets its years are sampled from its side's
        events. The two sets have the labels of this one, and each scenario
        its members' sides, so a mixture stays a mixture of its members'
        sides.

        Parameters
        ----------
        train_fraction : float, default 0.7
            As in :func:`~climada_petals.engine.parametric.sample.holdout`.
        rng : numpy.random.Generator, optional
            As in :func:`~climada_petals.engine.parametric.sample.holdout`:
            needed on year sets and on the event basis, not taken on
            catalogue years.

        Returns
        -------
        train, test : ScenarioSet

        Raises
        ------
        ParametricInputError, LeakageError
            As :func:`~climada_petals.engine.parametric.sample.holdout`; the
            error's note names the scenario and the member.
        ScenarioError
            If a member with the source token of an earlier one can't be
            cut with its blocks (other simulation years, other block labels,
            or blocks that the earlier one doesn't hold), or one that shares
            its parent storms with its years (catalogue years).
        """
        sides: tuple[dict[str, list[OutcomeSample]], dict[str, list[OutcomeSample]]]
        sides = ({}, {})
        cut: dict[Any, Any] = {}  # the first sides of each source token
        side_of: dict[str, Any] = {}  # each parent storm's side, and those sides
        for label, group in self.members.items():
            for position, member in enumerate(group):
                try:
                    pair = member_sides(
                        member,
                        cut,
                        side_of,
                        fraction=train_fraction,
                        rng=rng,
                        split=holdout,
                    )
                    check_no_leakage(*pair)
                except ParametricError as error:
                    error.add_note(
                        "raised by ScenarioSet.holdout at "
                        + where(label, position, len(group))
                    )
                    raise
                for side, part in zip(sides, pair):
                    side.setdefault(label, []).append(part)
        return ScenarioSet(sides[0]), ScenarioSet(sides[1])

    @classmethod
    def from_impacts(
        cls,
        members: Mapping[str, Any],
        *,
        units: InsuredUnits | Mapping[str, InsuredUnits],
        basis: BasisSource,
    ) -> ScenarioSet:
        """Build a set from an index and an ``Impact`` per scenario.

        Parameters
        ----------
        members : mapping of str to (ParametricIndex, Impact) or a sequence of them
            Each label to the index and the loss of its hazard set, as
            :meth:`OutcomeSample.from_impact` takes them, or to several such
            pairs, which form one scenario (module docstring). The index is
            on the cover's definition, e.g. from
            :meth:`ParametricIndex.from_definition`.
        units : InsuredUnits or mapping of str to InsuredUnits
            The insured units, or each label's units, for scenarios with
            other exposures.
        basis : EventBasis, CatalogueYears, YearSets, or a function
            The basis of every sample, e.g. ``EventBasis()``, or a function
            that takes an ``Impact`` and returns its basis, e.g.
            ``lambda impact: YearSets.from_frequencies(impact,
            n_years=10_000, rng=rng)``.

        Returns
        -------
        ScenarioSet

        Raises
        ------
        ParametricInputError
            If a member is not a pair of an index and an impact, a mapping
            lacks a label, or as :meth:`OutcomeSample.from_impact` (the
            error's note names the scenario and the member).
        ScenarioError
            As the class.
        """
        return cls(
            _built(
                members,
                _is_pair,
                partial(_impact_sample, units=units, basis=basis),
                "from_impacts",
            )
        )

    @classmethod
    def from_hazards(
        cls,
        members: Mapping[str, Any],
        *,
        definition: IndexDefinition,
        units: InsuredUnits | Mapping[str, InsuredUnits],
        impact_funcs: Any,
        basis: BasisSource,
        index_units: InsuredUnits | None = None,
    ) -> ScenarioSet:
        """Build a set from the hazard sets, exposures and impact functions of each scenario.

        For each hazard set: the cover's index,
        ``ParametricIndex.from_definition(definition, hazard, units)``; the
        loss, ``ImpactCalc(units.exposures, impact_funcs,
        hazard).impact(save_mat=True)``, on a copy of the exposures; and the
        sample, ``OutcomeSample.from_impact(index, impact, units=units,
        basis=basis)``.

        Parameters
        ----------
        members : mapping of str to Hazard or sequence of Hazard
            Each label to its hazard set, e.g. a future climate from the Data
            API or ``hazard.apply_climate_scenario_knu(...)``, or to several,
            which form one scenario (module docstring), e.g. one per
            ``ref_year``.
        definition : IndexDefinition
            The index a cover pays on, e.g. ``cover.index``: a hazard index
            (``ParametricIndex.from_hazard``). A cat-in-a-circle index needs
            its tracks: build it with ``ParametricIndex.from_definition(...,
            tracks=...)`` and use :meth:`from_impacts`.
        units : InsuredUnits or mapping of str to InsuredUnits
            The insured units and their exposures, or each label's, for
            scenarios with other exposures (the same unit ids). The
            exposures have the impact-function column of the hazard type.
        impact_funcs : climada.entity.ImpactFuncSet or mapping of str to it
            The impact functions, or each label's.
        basis : EventBasis, CatalogueYears, YearSets, or a function
            The basis of every sample, e.g. ``EventBasis()``, or a function
            that takes a ``Hazard`` and returns its basis, e.g. ``lambda
            hazard: YearSets.from_frequencies(hazard, n_years=10_000,
            rng=rng)``.
        index_units : InsuredUnits, optional
            The units the index is computed on, with the same unit ids, e.g.
            the units at inception, so that a value-weighted index keeps the
            contract's weights. Default: each scenario's own units, so that
            the weights follow its exposures (module docstring).

        Returns
        -------
        ScenarioSet

        Raises
        ------
        ParametricInputError
            If a member is not a ``Hazard``, a mapping lacks a label,
            ``index_units`` has other unit ids, or as
            :meth:`ParametricIndex.from_definition` and
            :meth:`OutcomeSample.from_impact`.
        YearRouteError
            If the basis doesn't fit a hazard set, e.g. catalogue years on a
            frequency-scaled set.
        ScenarioError
            As the class.

        Each error from a hazard set has a note naming the scenario and the
        member.

        Examples
        --------
        Future tropical cyclones of the Data API, by reference year:

        >>> client = Client()  # doctest: +SKIP
        >>> futures = [  # doctest: +SKIP
        ...     client.get_hazard(
        ...         "tropical_cyclone",
        ...         properties={
        ...             "country_iso3alpha": "JAM",
        ...             "climate_scenario": "rcp45",
        ...             "ref_year": str(year),
        ...             "nb_synth_tracks": "10",
        ...         },
        ...     )
        ...     for year in (2040, 2060, 2080)
        ... ]
        >>> rng = np.random.default_rng(321)  # doctest: +SKIP
        >>> scenarios = ScenarioSet.from_hazards(  # doctest: +SKIP
        ...     {"present": present, "rcp45, 2040 to 2080": futures},
        ...     definition=cover.index,
        ...     units=units,
        ...     impact_funcs=impfset,
        ...     basis=lambda hazard: YearSets.from_frequencies(
        ...         hazard, n_years=10_000, rng=rng
        ...     ),
        ... )
        """
        if index_units is not None and not isinstance(index_units, InsuredUnits):
            raise ParametricInputError(
                f"index_units must be InsuredUnits or None, got {index_units!r}"
            )
        setup = _HazardSetup(definition, units, impact_funcs, basis, index_units)
        return cls(
            _built(
                members,
                lambda value: isinstance(value, Hazard),
                partial(_hazard_sample, setup=setup),
                "from_hazards",
            )
        )


def _check_set(scenarios: object) -> ScenarioSet:
    """Return ``scenarios`` if it is a :class:`ScenarioSet`."""
    if not isinstance(scenarios, ScenarioSet):
        raise ParametricInputError(
            f"scenarios must be a ScenarioSet, got {type(scenarios).__name__}"
        )
    return scenarios


def _splits(split: object, labels: tuple[str, ...]) -> dict[str, Any]:
    """Return the split of each scenario: one for all, or a mapping by label.

    Raises
    ------
    ParametricInputError
        If a mapping lacks a label or has others.
    """
    if not isinstance(split, Mapping):
        return {label: split for label in labels}
    missing = [label for label in labels if label not in split]
    others = [label for label in split if label not in labels]
    if missing or others:
        raise ParametricInputError(
            f"split must give every scenario of the set and no other; it lacks "
            f"{missing} and has {others}"
        )
    return {label: split[label] for label in labels}


def _check_mixture_leakage(
    design: DesignSample, scenarios: ScenarioSet, label: str
) -> None:
    """Check a mixture's members against a design (module docstring).

    Every member is checked as ``evaluate`` checks a sample: by its source
    token. ``evaluate`` checks the mixture itself, whose events and blocks
    carry their member.

    Raises
    ------
    LeakageError
        If a member of the design sample's source shares an event or a
        block with it.
    """
    group = scenarios.members[label]
    if len(group) == 1:
        return
    for member in group:
        check_out_of_sample(design, member)


def evaluate_scenarios(
    entry: ParametricCover | CalibrationResult,
    scenarios: ScenarioSet,
    *,
    split: str | Mapping[str, str],
    pricing: Pricing | None = None,
    premium: float | None = None,
    options: MetricOptions = MetricOptions(),  # frozen, so safe to share
) -> pd.DataFrame:
    """Return the metrics of one cover or design on every scenario, in one table.

    The rows of each scenario are those of
    ``evaluate(entry, scenarios.sample(label), split=..., pricing=pricing,
    premium=premium, options=options)``, in the order of the labels, with
    the label in the ``scenario`` column. The cover is the same on every
    scenario: a design keeps its design premium, the contract price, in
    ``premium``, and ``premium_repriced`` prices its payouts on each
    scenario. The table ranks nothing and names no scenario as best.

    Parameters
    ----------
    entry : ParametricCover or CalibrationResult
        A resolved cover given as it is, or a design.
    scenarios : ScenarioSet
        The scenarios, all on one route.
    split : {"in_sample", "out_of_sample"} or mapping of str to it
        Which each scenario's sample is for the entry: one for every
        scenario, or one per label, e.g. a design's own scenario in sample
        and the others out of sample.
    pricing, premium, options
        As in :func:`~climada_petals.engine.parametric.evaluate.evaluate`.

    Returns
    -------
    pandas.DataFrame
        :func:`~climada_petals.engine.parametric.evaluate.evaluate`'s
        columns. ``fold`` is the fold a scenario's members were cut as.

    Raises
    ------
    ParametricInputError
        If ``scenarios`` is not a ``ScenarioSet``, ``split`` lacks a label or
        has others, or as
        :func:`~climada_petals.engine.parametric.evaluate.evaluate`.
    LeakageError
        If an out-of-sample evaluation of a design meets a sample, or a
        member of a mixture, of the design sample's source that shares an
        event or a block with it (module docstring).

    Each error from a scenario has a note naming it.

    Examples
    --------
    >>> table = evaluate_scenarios(  # doctest: +SKIP
    ...     design, scenarios, split="out_of_sample"
    ... )
    >>> print(setup_text(table))  # doctest: +SKIP
    """
    checked = _check_set(scenarios)
    splits = _splits(split, checked.labels)
    design = entry.design_sample if isinstance(entry, CalibrationResult) else None
    tables = []
    for label in checked.labels:
        try:
            if splits[label] == "out_of_sample" and design is not None:
                _check_mixture_leakage(design, checked, label)
            table = evaluate(
                entry,
                checked.sample(label),
                split=splits[label],
                pricing=pricing,
                premium=premium,
                options=options,
            )
        except ParametricError as error:
            error.add_note(f"raised by evaluate_scenarios at scenario {label!r}")
            raise
        table["fold"] = np.int64(checked.fold(label))
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def calibrate_per_scenario(
    template: ParametricCover,
    scenarios: ScenarioSet,
    objective: Objective,
    *,
    scope: str,
    pricing: Pricing = PurePremium(),  # frozen, so safe to share
    budget: Budget | None = None,
    options: CalibrationOptions = CalibrationOptions(),  # frozen, so safe to share
) -> dict[str, CalibrationResult]:
    """Calibrate a template on every scenario: one design per scenario.

    Each scenario's design is ``calibrate(template, scenarios.sample(label),
    objective, scope=scope, pricing=pricing, budget=budget,
    options=options)``, solved on its own, so a design doesn't depend on the
    other scenarios. A ``Budget(ratio=kappa)`` bounds each design's premium
    by ``kappa`` times the expected loss of its own scenario, and a
    ``Budget(amount=B)`` by the same amount on every scenario. The designs
    are measurements on their scenarios' samples: evaluate each on other
    samples (:func:`evaluate_scenarios`) for out-of-sample numbers, and put
    them side by side with :func:`designs_table`. Nothing here names a best
    design or scenario. Robust design over the set is not part of this
    module yet.

    Parameters
    ----------
    template : ParametricCover
        The cover to design, as for
        :func:`~climada_petals.engine.parametric.calibrate.calibrate`.
    scenarios : ScenarioSet
        The design samples, one per scenario, e.g. the training set of
        :meth:`ScenarioSet.holdout`.
    objective : MSE, ExpectedUncovered, WeightedCVaR or UserObjective
        What every design minimises.
    scope : {"individual", "joint"}
        As for :func:`~climada_petals.engine.parametric.calibrate.calibrate`.
        Required.
    pricing, budget, options
        As for :func:`~climada_petals.engine.parametric.calibrate.calibrate`,
        for every scenario.

    Returns
    -------
    dict of str to CalibrationResult
        One design per scenario, in the order of the labels.

    Raises
    ------
    ParametricInputError
        If ``scenarios`` is not a ``ScenarioSet``, or as
        :func:`~climada_petals.engine.parametric.calibrate.calibrate`.
    CalibrationError
        As :func:`~climada_petals.engine.parametric.calibrate.calibrate`, at
        a scenario the template can't be designed on: no scenario is
        skipped. The error's note names the scenario.

    Examples
    --------
    >>> train, test = scenarios.holdout(rng=np.random.default_rng(320))  # doctest: +SKIP
    >>> designs = calibrate_per_scenario(  # doctest: +SKIP
    ...     template,
    ...     train,
    ...     WeightedCVaR(alpha=0.9, beta=0.99),
    ...     scope="joint",
    ...     budget=Budget(ratio=1.0),
    ... )
    >>> designs_table(designs)  # doctest: +SKIP
    """
    checked = _check_set(scenarios)
    settings = {
        "scope": scope,
        "pricing": pricing,
        "budget": budget,
        "options": options,
    }
    designs = {}
    for label in checked.labels:
        try:
            designs[label] = calibrate(
                template, checked.sample(label), objective, **settings
            )
        except ParametricError as error:
            error.add_note(f"raised by calibrate_per_scenario at scenario {label!r}")
            raise
    return designs


def designs_table(designs: Mapping[str, CalibrationResult]) -> pd.DataFrame:
    """Return several designs, e.g. one per scenario, as one table.

    Parameters
    ----------
    designs : mapping of str to CalibrationResult
        Each label to its design, e.g. the result of
        :func:`calibrate_per_scenario`.

    Returns
    -------
    pandas.DataFrame
        One row per design problem of each design (one per unit, or one for
        the joint scope), design by design: ``scenario`` (the label), the
        columns of
        :meth:`~climada_petals.engine.parametric.calibrate.CalibrationResult.summary`,
        and every parameter of the problem's units in the resolved cover,
        as ``"<unit_id>:<name>"`` (NaN on the rows of other units). The
        objective values and premiums are on each design's own sample. No
        column ranks the designs.

    Raises
    ------
    ParametricInputError
        If ``designs`` is not a non-empty mapping of labels to designs.
    """
    items = labelled_items(designs, "designs")
    frames = []
    for label, result in items:
        if not isinstance(result, CalibrationResult):
            raise ParametricInputError(
                f"designs[{label!r}] must be a CalibrationResult, got "
                f"{type(result).__name__}"
            )
        rows = []
        for problem in result.problems:
            row: dict[str, Any] = {}
            for unit in problem.unit_id:
                row.update(
                    (f"{unit}:{name}", float(value))
                    for name, value in result.cover.payout[unit].parameters().items()
                )
            rows.append(row)
        summary = result.summary()
        frames.append(
            pd.concat(
                [
                    pd.DataFrame({"scenario": [label] * len(summary)}),
                    summary,
                    pd.DataFrame(rows),
                ],
                axis=1,
            )
        )
    return pd.concat(frames, ignore_index=True)
