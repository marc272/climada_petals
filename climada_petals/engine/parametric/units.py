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

Insured units: the exposure points that share one payout function.

A unit is what gets one payout function of a cover: one exposure point
(:meth:`InsuredUnits.points`), a group of points
(:meth:`InsuredUnits.from_column`, by a column of ``Exposures.gdf`` such as
``region_id``, or :meth:`InsuredUnits.from_geometries`, by polygons), or the
whole exposure set (:meth:`InsuredUnits.whole`). LitPop exposures hold
thousands of points, so a unit per point is rarely the contract anyone
wants.

* A point belongs to at most one unit. Points in no unit are counted and
  logged, and their losses are left out of every unit's loss.
* A unit with no point, or with zero total value, raises
  :class:`~climada_petals.engine.parametric.errors.UnitMappingError`.
* The loss of a unit is ``Impact.imp_mat`` summed over the unit's points,
  sparse throughout (:meth:`InsuredUnits.aggregate`).
* A unit's payout maps back to its points by value share,
  ``pi[e, i] = pi[e, u] * v_i / V_u``, where ``V_u`` is the unit's total
  value (:meth:`InsuredUnits.to_points`). So payouts on the units have the
  exposure points of the loss ``Impact``.
* Exposures with a ``cover`` or ``deductible`` column raise
  :class:`~climada_petals.engine.parametric.errors.ImpactInputError`:
  ``ImpactCalc`` then computes losses that are already net of an indemnity
  layer. Drop the columns to go on.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
from climada.util.constants import DEF_FREQ_UNIT
from climada.util.coordinates import equal_crs
from numpy.typing import ArrayLike, NDArray

from climada_petals.util.config import LOGGER

from ._checks import finite_array, read_only
from .errors import ImpactInputError, ParametricInputError, UnitMappingError

__all__ = ["InsuredUnits"]

_COORD_ATOL: Final[float] = 1e-9
"""Absolute tolerance, in degrees, of an impact's points against the units'."""

_POLYGONS: Final[frozenset[str]] = frozenset({"Polygon", "MultiPolygon"})


@dataclass(frozen=True, eq=False)
class InsuredUnits:
    """Insured units: a grouping of exposure points (module docstring).

    Use :meth:`points`, :meth:`from_column`, :meth:`from_geometries` or
    :meth:`whole`; the constructor takes the membership directly.

    Parameters
    ----------
    exposures : climada.entity.Exposures
        The exposure points, with a ``value`` column and no ``cover`` or
        ``deductible`` column. The units keep a copy.
    membership : array_like or scipy.sparse matrix
        Shape ``(n_points, n_units)``: 1 where a point is in a unit, else 0.
        Stored as a CSR matrix.
    unit_id : sequence of str or int
        One id per unit, unique, also as strings.

    Raises
    ------
    UnitMappingError
        If a point is in two units, a unit holds no point or has zero total
        value, a value of a point in a unit is negative or not finite, the
        exposures are not points or have no ``value`` column, or the unit
        ids are invalid.
    ImpactInputError
        If the exposures have a ``cover`` or ``deductible`` column.
    ParametricInputError
        If ``exposures`` is not an ``Exposures`` object.
    """

    exposures: Any
    membership: Any
    unit_id: Sequence[Hashable]
    _value: NDArray[np.float64] = field(init=False, repr=False)
    _coords: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the grouping; keep a copy of the exposures."""
        exposures = _copied(self.exposures)
        n_points = int(exposures.gdf.shape[0])
        membership = _membership(self.membership, n_points)
        units = _unit_ids(self.unit_id)
        if len(units) != membership.shape[1]:
            raise UnitMappingError(
                f"membership has {membership.shape[1]} units and unit_id {len(units)}"
            )
        per_point = np.diff(membership.indptr)
        _refuse(per_point > 1, exposures, "are in two or more units")
        inside = per_point == 1
        value = _values(exposures, inside)
        counts = np.asarray(membership.sum(axis=0)).ravel()
        unit_value = membership.T @ value
        for wrong, what in ((counts == 0, "no point"), (unit_value <= 0, "zero value")):
            if wrong.any():
                labels = [units[k] for k in np.flatnonzero(wrong)[:5]]
                raise UnitMappingError(
                    f"{int(wrong.sum())} units hold {what}, e.g. {labels}; a unit "
                    "needs at least one point and a positive total value"
                )
        if not inside.all():
            LOGGER.warning(
                "%d of %d exposure points are in no insured unit; their losses "
                "are left out",
                int(n_points - inside.sum()),
                n_points,
            )
        object.__setattr__(self, "exposures", exposures)
        object.__setattr__(self, "membership", membership)
        object.__setattr__(self, "unit_id", units)
        object.__setattr__(self, "_value", read_only(value))
        object.__setattr__(self, "_coords", read_only(_coordinates(exposures)))

    @property
    def n_points(self) -> int:
        """Number of exposure points, in a unit or not."""
        return int(self.membership.shape[0])

    @property
    def n_units(self) -> int:
        """Number of insured units."""
        return int(self.membership.shape[1])

    @property
    def value(self) -> NDArray[np.float64]:
        """Value of each exposure point, shape ``(n_points,)``; 0 outside units."""
        return self._value

    @property
    def coords(self) -> NDArray[np.float64]:
        """Latitude and longitude of each point, shape ``(n_points, 2)``."""
        return self._coords

    @property
    def unit_value(self) -> NDArray[np.float64]:
        """Total value of each unit, shape ``(n_units,)``."""
        return np.asarray(self.membership.T @ self.value, dtype=np.float64)

    @property
    def point_unit(self) -> NDArray[np.intp]:
        """Position of each point's unit, shape ``(n_points,)``; -1 for none."""
        out = np.full(self.n_points, -1, dtype=np.intp)
        rows = np.repeat(np.arange(self.n_points), np.diff(self.membership.indptr))
        out[rows] = self.membership.indices
        return out

    @classmethod
    def points(cls, exposures: Any) -> InsuredUnits:
        """One unit per exposure point, named by the label of ``exposures.gdf``.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            Every point is a unit, so every value must be positive: drop
            points of zero value first.

        Returns
        -------
        InsuredUnits
        """
        n_points = _rows(exposures)
        return cls(
            exposures, sp.identity(n_points, format="csr"), list(exposures.gdf.index)
        )

    @classmethod
    def from_column(cls, exposures: Any, column: str) -> InsuredUnits:
        """One unit per value of a column of ``exposures.gdf``, e.g. ``region_id``.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            The exposure points.
        column : str
            A column of integers or strings; whole floats count as
            integers. A point with a missing value (NaN or None) is in no
            unit.

        Returns
        -------
        InsuredUnits
            The units in the order of their sorted ids, integers first.

        Raises
        ------
        UnitMappingError
            If the column is missing, a value is neither an integer nor a
            string, or the column holds no value at all.
        """
        n_points = _rows(exposures)
        if column not in exposures.gdf.columns:
            raise UnitMappingError(f"the exposures have no column {column!r}")
        cells = exposures.gdf[column]
        present = np.asarray(cells.notna(), dtype=bool)
        keys = [_unit_key(cell, column) for cell in cells[present]]
        unit_id = sorted(set(keys), key=lambda key: (isinstance(key, str), key))
        position = {key: k for k, key in enumerate(unit_id)}
        membership = sp.csr_matrix(
            (
                np.ones(len(keys)),
                (np.flatnonzero(present), [position[key] for key in keys]),
            ),
            shape=(n_points, len(unit_id)),
        )
        return cls(exposures, membership, unit_id)

    @classmethod
    def from_geometries(
        cls, exposures: Any, gdf: gpd.GeoDataFrame, *, id_column: str
    ) -> InsuredUnits:
        """One unit per polygon: the exposure points it holds.

        A point on a polygon's boundary is in it (``intersects``), so a
        point on the border of two polygons is in both, and raises.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            The exposure points.
        gdf : geopandas.GeoDataFrame
            Polygons or multipolygons, with a coordinate reference system.
            They are projected to the exposures' if it differs.
        id_column : str
            The column of ``gdf`` that names each unit: integers or strings,
            unique.

        Returns
        -------
        InsuredUnits
            The units in the order of the rows of ``gdf``.

        Raises
        ------
        UnitMappingError
            If a geometry is not a polygon, ``gdf`` has no coordinate
            reference system or no ``id_column``, a polygon holds no point,
            or a point is in two polygons.
        """
        n_points = _rows(exposures)
        if not isinstance(gdf, gpd.GeoDataFrame) or id_column not in gdf.columns:
            raise UnitMappingError(
                f"gdf must be a GeoDataFrame with the column {id_column!r}"
            )
        kinds = set(gdf.geometry.geom_type) - _POLYGONS
        if kinds or gdf.crs is None:
            raise UnitMappingError(
                "gdf must hold polygons or multipolygons in a stated crs; got the "
                f"geometries {sorted(map(str, kinds))} and crs {gdf.crs}"
            )
        unit_id = [_unit_key(cell, id_column) for cell in gdf[id_column]]
        polygons = gpd.GeoDataFrame(
            {"unit_position": np.arange(len(gdf))},
            geometry=gdf.geometry.values,
            crs=gdf.crs,
        )
        if not equal_crs(polygons.crs, exposures.crs):
            polygons = polygons.to_crs(exposures.crs)
        located = gpd.GeoDataFrame(
            {"point_position": np.arange(n_points)},
            geometry=exposures.gdf.geometry.values,
            crs=exposures.crs,
        )
        pairs = gpd.sjoin(located, polygons, how="inner", predicate="intersects")
        membership = sp.csr_matrix(
            (
                np.ones(len(pairs)),
                (pairs["point_position"].to_numpy(), pairs["unit_position"].to_numpy()),
            ),
            shape=(n_points, len(unit_id)),
        )
        return cls(exposures, membership, unit_id)

    @classmethod
    def whole(cls, exposures: Any, *, unit_id: str | int = "total") -> InsuredUnits:
        """One unit holding every exposure point: one payout for the whole set.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            The exposure points.
        unit_id : str or int, default "total"
            The unit's id.

        Returns
        -------
        InsuredUnits
        """
        n_points = _rows(exposures)
        return cls(exposures, sp.csr_matrix(np.ones((n_points, 1))), [unit_id])

    def aggregate(self, imp_mat: Any) -> sp.csr_matrix:
        """Sum a matrix over each unit's points, e.g. the loss of ``Impact.imp_mat``.

        Parameters
        ----------
        imp_mat : array_like or scipy.sparse matrix
            Shape ``(n_events, n_points)``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_events, n_units)``. Points in no unit are left out.

        Raises
        ------
        ParametricInputError
            If ``imp_mat`` does not have one column per exposure point.
        """
        matrix = _sparse(imp_mat, self.n_points, "imp_mat")
        return sp.csr_matrix(matrix @ self.membership)

    def to_points(self, payout: Any) -> sp.csr_matrix:
        """Share each unit's payout among its points by value share.

        Parameters
        ----------
        payout : array_like or scipy.sparse matrix
            Shape ``(n_events, n_units)``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_events, n_points)``: ``payout[e, u] * v_i / V_u`` at
            each point ``i`` of unit ``u``, 0 at points in no unit. Each
            event's total over the points is its total over the units.

        Raises
        ------
        ParametricInputError
            If ``payout`` does not have one column per unit.
        """
        matrix = _sparse(payout, self.n_units, "payout")
        share = self.membership.multiply(self.value[:, None]).multiply(
            1.0 / self.unit_value[None, :]
        )
        return sp.csr_matrix(matrix @ sp.csr_matrix(share).T)

    def check_impact(self, impact: Any) -> None:
        """Check that ``impact`` is a loss on these units' exposure points.

        Parameters
        ----------
        impact : climada.engine.Impact
            Computed with ``save_mat=True`` on the units' exposures, e.g.
            ``ImpactCalc(exposures, impfset, hazard).impact(save_mat=True)``.

        Raises
        ------
        ImpactInputError
            If the impact has no ``imp_mat``, its ``frequency_unit`` is not
            ``1/year``, its exposure points (``coord_exp``) are not the
            units', or a loss is negative or not finite.
        """
        matrix = getattr(impact, "imp_mat", None)
        n_events = len(impact.event_id)
        if matrix is None or (n_events and not np.prod(matrix.shape)):
            raise ImpactInputError(
                "the impact has no imp_mat; compute it with save_mat=True, e.g. "
                "ImpactCalc(exposures, impfset, hazard).impact(save_mat=True)"
            )
        if impact.frequency_unit != DEF_FREQ_UNIT:
            raise ImpactInputError(
                f"the impact's frequency_unit is {impact.frequency_unit!r}; the "
                f"module reads frequencies per year ({DEF_FREQ_UNIT!r})"
            )
        if tuple(matrix.shape) != (n_events, self.n_points):
            raise ImpactInputError(
                f"impact.imp_mat has shape {tuple(matrix.shape)}, expected "
                f"({n_events}, {self.n_points}): one column per exposure point of "
                "the units. Compute the impact with save_mat=True"
            )
        coords = np.asarray(impact.coord_exp, dtype=float)
        if coords.shape != self.coords.shape or not np.allclose(
            coords, self.coords, rtol=0.0, atol=_COORD_ATOL
        ):
            raise ImpactInputError(
                "the impact's exposure coordinates (coord_exp) are not the units' "
                "exposure points"
            )
        losses = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
        if not np.all(np.isfinite(losses)):
            raise ImpactInputError("the impact holds losses that are not finite")
        if np.any(losses < 0):
            raise ImpactInputError("the impact holds negative losses")

    def centroids(self, hazard: Any) -> NDArray[np.intp]:
        """Return each point's centroid in ``hazard``: its column of the intensity.

        The points are matched to the hazard's centroids on a copy of the
        exposures, as ``ImpactCalc`` does by default
        (``Exposures.assign_centroids`` with its defaults). A centroid
        column already in the exposures is not used, since it may belong
        to another hazard of the same type.

        Parameters
        ----------
        hazard : climada.hazard.Hazard
            The hazard.

        Returns
        -------
        numpy.ndarray of int
            Shape ``(n_points,)``; -1 for a point in no unit and with no
            centroid.

        Raises
        ------
        UnitMappingError
            If a point in a unit has no centroid within CLIMADA's matching
            threshold, the hazard and the exposures have different
            coordinate reference systems, or the hazard's intensity matrix
            does not have one valid column per centroid. The last happens
            when a ``Centroids`` object holds the same coordinates twice:
            ``TropCyclone.from_tracks`` then returns column indices of -1.
            Build the centroids from unique coordinates.
        """
        intensity = hazard.intensity
        n_centroids = int(hazard.centroids.size)
        if not _one_column_per_centroid(intensity, n_centroids):
            raise UnitMappingError(
                f"the hazard's intensity must be a CSR matrix with one valid column "
                f"per centroid ({n_centroids}); got {type(intensity).__name__} of "
                f"shape {getattr(intensity, 'shape', None)}. Centroids that hold the "
                "same coordinates twice make TropCyclone.from_tracks return column "
                "indices of -1: build the centroids from unique coordinates"
            )
        assigned = self.exposures.copy()
        try:
            assigned.assign_centroids(hazard, overwrite=True)
        except ValueError as exc:
            raise UnitMappingError(
                f"cannot match the points to centroids: {exc}"
            ) from exc
        centroid = np.asarray(
            assigned.gdf[f"centr_{hazard.haz_type}"], dtype=np.intp
        ).copy()
        _refuse(
            (centroid < 0) & (self.point_unit >= 0),
            self.exposures,
            "have no hazard centroid within CLIMADA's matching threshold",
        )
        return centroid


def _one_column_per_centroid(intensity: Any, n_centroids: int) -> bool:
    """Whether ``intensity`` is a CSR matrix with valid columns, one per centroid."""
    if not sp.issparse(intensity) or intensity.format != "csr":
        return False
    indices = np.asarray(intensity.indices)
    return intensity.shape[1] == n_centroids and not (
        indices.size and (indices.min() < 0 or indices.max() >= n_centroids)
    )


def _rows(exposures: Any) -> int:
    """Return the number of exposure points, refusing other objects."""
    gdf = getattr(exposures, "gdf", None)
    if gdf is None or not hasattr(exposures, "copy"):
        raise ParametricInputError(
            f"exposures must be a climada Exposures, got {type(exposures).__name__}"
        )
    return int(gdf.shape[0])


def _copied(exposures: Any) -> Any:
    """Return a copy of the exposures after checking their columns."""
    _rows(exposures)
    insured = [name for name in ("cover", "deductible") if name in exposures.gdf]
    if insured:
        raise ImpactInputError(
            f"the exposures have the columns {insured}, so ImpactCalc computes "
            "losses net of an indemnity layer; drop the columns to design a "
            "parametric cover on the gross loss"
        )
    if "value" not in exposures.gdf:
        raise UnitMappingError("the exposures have no value column")
    return exposures.copy()


def _membership(membership: Any, n_points: int) -> sp.csr_matrix:
    """Return the membership as a canonical CSR matrix of ones."""
    matrix = sp.csr_matrix(membership, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape[0] != n_points or not np.all(matrix.data == 1.0):
        raise UnitMappingError(
            f"membership must have {n_points} rows, one per exposure point, and "
            f"hold only 0 and 1; got shape {matrix.shape}"
        )
    matrix.sort_indices()
    return matrix


def _values(exposures: Any, inside: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return the points' values; those of points in units must be finite, >= 0."""
    try:
        value = np.asarray(exposures.gdf["value"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UnitMappingError(f"the exposures' values must be numbers: {exc}") from exc
    _refuse(
        inside & ~(np.isfinite(value) & (value >= 0)),
        exposures,
        "are in a unit and have a value that is negative or not finite",
    )
    return np.where(inside, value, 0.0)


def _coordinates(exposures: Any) -> NDArray[np.float64]:
    """Return the points' latitude and longitude."""
    try:
        coords = np.column_stack([exposures.latitude, exposures.longitude])
    except ValueError as exc:
        raise UnitMappingError(f"the exposures must be points: {exc}") from exc
    return finite_array(coords, "the exposures' coordinates", 2)


def _refuse(wrong: NDArray[np.bool_], exposures: Any, what: str) -> None:
    """Raise :class:`UnitMappingError` naming the points where ``wrong`` holds."""
    if wrong.any():
        labels = [exposures.gdf.index[i] for i in np.flatnonzero(wrong)[:5]]
        raise UnitMappingError(
            f"{int(wrong.sum())} exposure points {what}, e.g. {labels}"
        )


def _unit_key(cell: object, column: str) -> str | int:
    """Return a unit id from a cell: an integer (whole floats too) or a string."""
    if isinstance(cell, int | np.integer) and not isinstance(cell, bool | np.bool_):
        return int(cell)
    if isinstance(cell, float | np.floating) and float(cell).is_integer():
        return int(cell)
    if isinstance(cell, str) and cell:
        return cell
    raise UnitMappingError(
        f"the column {column!r} must hold integers or non-empty strings, got {cell!r}"
    )


def _unit_ids(values: Sequence[Hashable]) -> tuple[str | int, ...]:
    """Return the unit ids: integers or strings, unique, also as strings."""
    units = tuple(_unit_key(value, "unit_id") for value in values)
    if not units or len({str(unit) for unit in units}) != len(units):
        raise UnitMappingError(
            "unit_id must hold at least one unit, each unique, also as a string; "
            f"got {list(units)[:10]}"
        )
    return units


def _sparse(values: ArrayLike, n_columns: int, name: str) -> sp.csr_matrix:
    """Return a 2-D matrix with ``n_columns`` columns as a CSR matrix."""
    matrix = values if sp.issparse(values) else np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != n_columns:
        raise ParametricInputError(
            f"{name} must have shape (n_events, {n_columns}), got {matrix.shape}"
        )
    return sp.csr_matrix(matrix, dtype=np.float64)
