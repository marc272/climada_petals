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

Cat-in-a-circle index from tropical cyclone tracks, and its track geometry.

:func:`cat_in_circle` computes the index that
:meth:`~climada_petals.engine.parametric.index.ParametricIndex.from_cat_in_circle`
builds: the highest Saffir-Simpson category that a storm's track reaches
within ``radius_km`` of an insured unit. It reads the tracks, not the wind
field that drives the loss, so index and loss are decoupled as in a real
contract. Steinmann et al. (2023, Environment Systems and Decisions
43:555-568) build such products in CLIMADA, with radii of 10 to 50 km.

Cat-in-a-circle, per storm and exposure point:

1. Take the storm's track points (centre position and maximum sustained
   wind). Interpolate the track linearly in time to ``track_step_hours``;
   the original points are kept. Without interpolation, a small circle can
   miss a fast storm that crosses it between two track points.
2. Keep the points within ``radius_km`` of the exposure point (haversine
   distance on a sphere of 6,371 km, inclusive).
3. Take the highest wind of those points, or 0 if there are none.

A unit's wind is the highest over its points, so a group of points reads
the track within ``radius_km`` of any of them: its ``index_stat`` is
``"max"``. Then ``metric="category"`` gives the number of breakpoints this
wind reaches, 0 to 5 on the Saffir-Simpson scale, and ``metric="max_wind"``
the wind itself, in m/s.

The default breakpoints, :data:`SAFFIR_SIMPSON_KN`, are the NHC lower bounds
of categories 1 to 5 in knots of 1-minute sustained wind, the values of
``climada.hazard.tc_tracks.SAFFIR_SIM_CAT[1:6]``. Category 0 covers tropical
storms, depressions and storms that never enter the circle.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from ._checks import check_real
from .cover import IndexDefinition
from .errors import ParametricInputError

if TYPE_CHECKING:
    from .units import InsuredUnits

__all__ = [
    "CAT_IN_CIRCLE",
    "DEFAULT_TRACK_STEP_HOURS",
    "EARTH_RADIUS_KM",
    "MS_TO_KN",
    "SAFFIR_SIMPSON_KN",
    "Metric",
    "cat_in_circle",
    "cat_in_circle_arguments",
    "category_of",
    "densify_track",
    "haversine_km",
    "max_wind_within_radius",
]

CAT_IN_CIRCLE: Final[str] = "cat_in_circle"
"""The ``kind`` of a cat-in-a-circle :class:`IndexDefinition`."""

SAFFIR_SIMPSON_KN: Final[tuple[float, ...]] = (64.0, 83.0, 96.0, 113.0, 137.0)
"""Lower bounds of Saffir-Simpson categories 1-5, 1-minute sustained wind, kn."""

MS_TO_KN: Final[float] = 1.9438444924406
"""Knots per m/s."""

EARTH_RADIUS_KM: Final[float] = 6371.0
"""Radius of the sphere of the haversine distance."""

DEFAULT_TRACK_STEP_HOURS: Final[float] = 0.5
"""Default time step, in hours, that tracks are interpolated to."""

Metric = Literal["category", "max_wind"]
"""What the cat-in-a-circle index reports: the category or the wind."""

_PARAMETERS: Final[frozenset[str]] = frozenset(
    {"breakpoints_kn", "metric", "radius_km", "track_step_hours"}
)


def cat_in_circle(
    tracks: Any,
    units: InsuredUnits,
    hazard: Any,
    *,
    radius_km: float,
    metric: Metric = "category",
    breakpoints_kn: Sequence[float] = SAFFIR_SIMPSON_KN,
    track_step_hours: float | None = DEFAULT_TRACK_STEP_HOURS,
) -> tuple[NDArray[np.float64], IndexDefinition]:
    """Return the cat-in-a-circle index of every hazard event and insured unit.

    Parameters
    ----------
    tracks : climada.hazard.TCTracks
        The storms' tracks, wind in knots. Every hazard event needs
        exactly one track whose ``sid`` is the event's name (as
        ``TropCyclone.from_tracks`` names its events).
    units : InsuredUnits
        The insured units; a unit reads the track near any of its points.
    hazard : climada.hazard.Hazard
        Sets the events and their order.
    radius_km : float
        Circle radius in km, > 0.
    metric : {"category", "max_wind"}, default "category"
        The category (0 to ``len(breakpoints_kn)``), or the wind in m/s.
    breakpoints_kn : sequence of float, default SAFFIR_SIMPSON_KN
        Strictly increasing, positive lower bounds of categories 1, 2,
        ..., in knots. Used by ``"category"`` only.
    track_step_hours : float or None, default 0.5
        Interpolate the tracks to this time step before testing the
        distance; None uses the track points as they are.

    Returns
    -------
    values : numpy.ndarray
        The index, shape ``(n_events, n_units)``, in categories or m/s.
    definition : IndexDefinition
        Kind ``"cat_in_circle"``, ``index_stat="max"``, and the arguments
        above as ``parameters`` (``breakpoints_kn`` with ``"category"``
        only).

    Raises
    ------
    ParametricInputError
        If a parameter is out of range, a hazard event has no track or
        two, or a track's wind is not in knots.
    """
    radius, bounds = _check_parameters(
        radius_km, metric, breakpoints_kn, track_step_hours
    )
    unit_wind = _unit_wind(
        tracks, units, hazard, radius=radius, track_step_hours=track_step_hours
    )
    parameters: dict[str, Any] = {
        "metric": metric,
        "radius_km": radius,
        "track_step_hours": (
            None if track_step_hours is None else float(track_step_hours)
        ),
    }
    if metric == "category":
        parameters["breakpoints_kn"] = list(bounds)
        unit = "Saffir-Simpson category" if bounds == SAFFIR_SIMPSON_KN else "category"
        values = category_of(unit_wind, bounds)
    else:
        unit, values = "m/s", unit_wind / MS_TO_KN
    definition = IndexDefinition(
        CAT_IN_CIRCLE, unit=unit, index_stat="max", parameters=parameters
    )
    return values, definition


def _unit_wind(
    tracks: Any,
    units: InsuredUnits,
    hazard: Any,
    *,
    radius: float,
    track_step_hours: float | None,
) -> NDArray[np.float64]:
    """Return the highest wind within the radius of each unit's points, in kn.

    Each distinct location of a point in a unit is read once.
    """
    unit_of = units.point_unit
    inside = np.flatnonzero(unit_of >= 0)
    order = inside[np.argsort(unit_of[inside], kind="stable")]
    locations, location_of = np.unique(units.coords[order], axis=0, return_inverse=True)
    wind_kn = _max_wind_kn(
        tracks, hazard, locations, radius=radius, track_step_hours=track_step_hours
    )
    starts = np.searchsorted(unit_of[order], np.arange(units.n_units))
    return np.maximum.reduceat(
        wind_kn[:, np.asarray(location_of).reshape(-1)], starts, axis=1
    )


def cat_in_circle_arguments(definition: IndexDefinition) -> dict[str, Any]:
    """Return the arguments of :func:`cat_in_circle` that a definition states.

    Parameters
    ----------
    definition : IndexDefinition
        A definition of kind ``"cat_in_circle"``.

    Returns
    -------
    dict
        ``radius_km``, ``metric``, ``track_step_hours`` and, for
        ``"category"``, ``breakpoints_kn``.

    Raises
    ------
    ParametricInputError
        If the definition is of another kind, or its parameters are not
        the ones :func:`cat_in_circle` states.
    """
    parameters: Mapping[str, Any] = definition.parameters or {}
    names = set(parameters)
    needed = {"metric", "radius_km", "track_step_hours"}
    if definition.kind != CAT_IN_CIRCLE or not needed <= names <= _PARAMETERS:
        raise ParametricInputError(
            f"a cat-in-a-circle definition has kind {CAT_IN_CIRCLE!r} and the "
            f"parameters {sorted(needed)}, and breakpoints_kn for the category; "
            f"got kind {definition.kind!r} and {sorted(names)}"
        )
    return {name: parameters[name] for name in names}


def _check_parameters(
    radius_km: float,
    metric: str,
    breakpoints_kn: Sequence[float],
    track_step_hours: float | None,
) -> tuple[float, tuple[float, ...]]:
    """Return the radius and the breakpoints, checked (:func:`cat_in_circle`).

    Raises
    ------
    ParametricInputError
        If a parameter is out of range.
    """
    radius = check_real("radius_km", radius_km, low=0.0, include_low=False)
    if metric not in ("category", "max_wind"):
        raise ParametricInputError(
            f"metric must be 'category' or 'max_wind', got {metric!r}"
        )
    bounds = tuple(float(b) for b in breakpoints_kn)
    if (
        not bounds
        or not all(math.isfinite(b) and b > 0 for b in bounds)
        or any(b >= c for b, c in itertools.pairwise(bounds))
    ):
        raise ParametricInputError(
            "breakpoints_kn must be non-empty, finite, positive and strictly "
            f"increasing, got {list(breakpoints_kn)}"
        )
    if track_step_hours is not None:
        check_real("track_step_hours", track_step_hours, low=0.0, include_low=False)
    return radius, bounds


def _max_wind_kn(
    tracks: Any,
    hazard: Any,
    locations: NDArray[np.float64],
    *,
    radius: float,
    track_step_hours: float | None,
) -> NDArray[np.float64]:
    """Return the highest wind in the circle per event and location, in kn.

    Raises
    ------
    ParametricInputError
        If a hazard event has no track or two, or a track's wind is not in
        knots.
    """
    by_sid = _tracks_by_sid(tracks)
    names = [str(name) for name in hazard.event_name]
    missing = [name for name in names if name not in by_sid]
    if missing:
        raise ParametricInputError(
            f"{len(missing)} hazard events have no track, e.g. {missing[:5]}"
        )
    wind_kn = np.zeros((len(names), locations.shape[0]))
    for row, name in enumerate(names):
        lat, lon, wind, hours = _track_arrays(by_sid[name])
        lat, lon, wind = densify_track(lat, lon, wind, hours, track_step_hours)
        wind_kn[row] = max_wind_within_radius(
            lat, lon, wind, locations[:, 0], locations[:, 1], radius_km=radius
        )
    return wind_kn


def _tracks_by_sid(tracks: Any) -> dict[str, Any]:
    """Map each track's ``sid`` to the track; refuse repeated ids."""
    out: dict[str, Any] = {}
    for track in tracks.data:
        sid = str(track.attrs.get("sid"))
        if sid in out:
            raise ParametricInputError(f"two tracks have the sid {sid!r}")
        out[sid] = track
    return out


def _track_arrays(
    track: Any,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Return ``(lat, lon, wind in kn, hours)`` of a CLIMADA track (xarray.Dataset).

    Raises
    ------
    ParametricInputError
        If the track's wind is not in knots.
    """
    wind_unit = track.attrs.get("max_sustained_wind_unit")
    if wind_unit != "kn":
        raise ParametricInputError(
            f"track {track.attrs.get('sid')!r}: max_sustained_wind_unit is "
            f"{wind_unit!r}, expected 'kn'"
        )
    time = np.asarray(track["time"].values)
    hours = (time - time[0]) / np.timedelta64(1, "h")
    return (
        np.asarray(track["lat"].values, dtype=np.float64),
        np.asarray(track["lon"].values, dtype=np.float64),
        np.asarray(track["max_sustained_wind"].values, dtype=np.float64),
        np.asarray(hours, dtype=np.float64),
    )


def haversine_km(
    lat: NDArray[np.float64], lon: NDArray[np.float64], lat0: float, lon0: float
) -> NDArray[np.float64]:
    """Return the great-circle distance from each point to one point.

    Parameters
    ----------
    lat, lon : numpy.ndarray
        Points, in degrees.
    lat0, lon0 : float
        The reference point, in degrees.

    Returns
    -------
    numpy.ndarray
        Distances in km on a sphere of radius :data:`EARTH_RADIUS_KM`.
    """
    rad_lat, rad_lat0 = np.radians(lat), np.radians(lat0)
    dlat = np.radians(lat0 - lat)
    dlon = np.radians(lon0 - lon)
    hav = (
        np.sin(dlat / 2) ** 2
        + np.cos(rad_lat) * np.cos(rad_lat0) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))


def category_of(
    wind_kn: NDArray[np.float64], breakpoints_kn: Sequence[float]
) -> NDArray[np.float64]:
    """Return the category of each wind: the number of breakpoints it reaches.

    Parameters
    ----------
    wind_kn : numpy.ndarray
        Winds in knots.
    breakpoints_kn : sequence of float
        Strictly increasing lower bounds of categories 1, 2, ...

    Returns
    -------
    numpy.ndarray
        Categories as floats, 0 to ``len(breakpoints_kn)``; a wind equal to a
        breakpoint is in the higher category.
    """
    bounds = np.asarray(breakpoints_kn, dtype=np.float64)
    wind = np.asarray(wind_kn, dtype=np.float64)
    return np.searchsorted(bounds, wind, side="right").astype(np.float64)


def densify_track(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    wind: NDArray[np.float64],
    hours: NDArray[np.float64],
    step_hours: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate a track linearly in time to at most ``step_hours`` apart.

    Each segment is split into ``ceil(dt / step_hours)`` equal parts, and
    latitude, longitude and wind are interpolated linearly, as CLIMADA's
    ``TCTracks.equal_timestep`` does. The original points are kept exactly.
    Longitudes are unwrapped first, so a segment that crosses the
    antimeridian takes the short way.

    Parameters
    ----------
    lat, lon, wind : numpy.ndarray
        The track points, in time order.
    hours : numpy.ndarray
        Time of each point in hours, strictly increasing.
    step_hours : float or None
        Largest time between interpolated points; None returns the points.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lat, lon, wind)`` of the interpolated track.

    Raises
    ------
    ParametricInputError
        If the arrays differ in length or the times do not increase.
    """
    lat, lon, wind, hours = (
        np.asarray(a, dtype=np.float64) for a in (lat, lon, wind, hours)
    )
    n_points = len(lat)
    if not len(lon) == len(wind) == len(hours) == n_points:
        raise ParametricInputError("lat, lon, wind and hours must have one length")
    if step_hours is None or n_points < 2:
        return lat, lon, wind
    gaps = np.diff(hours)
    if not (gaps > 0).all():
        raise ParametricInputError("track times must increase strictly")
    parts = np.maximum(np.ceil(gaps / step_hours), 1).astype(np.int64)
    segment = np.repeat(np.arange(n_points - 1), parts)
    first = np.cumsum(parts) - parts
    fraction = (np.arange(int(parts.sum())) - np.repeat(first, parts)) / np.repeat(
        parts, parts
    )
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon)))

    def interp(values: NDArray[np.float64]) -> NDArray[np.float64]:
        start, end = values[segment], values[segment + 1]
        return np.append(start + fraction * (end - start), values[-1])

    return interp(lat), interp(lon_unwrapped), interp(wind)


def max_wind_within_radius(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    wind: NDArray[np.float64],
    location_lat: NDArray[np.float64],
    location_lon: NDArray[np.float64],
    *,
    radius_km: float,
) -> NDArray[np.float64]:
    """Return, per location, the highest wind of the track points in the circle.

    Parameters
    ----------
    lat, lon, wind : numpy.ndarray
        Track points.
    location_lat, location_lon : numpy.ndarray
        The locations, in degrees.
    radius_km : float
        Circle radius in km, keyword-only; a point at exactly ``radius_km``
        is inside.

    Returns
    -------
    numpy.ndarray
        One value per location: the highest wind of the points within
        ``radius_km`` (haversine distance), or 0 if there are none.
    """
    lat, lon, wind = (np.asarray(a, dtype=np.float64) for a in (lat, lon, wind))
    out = np.zeros(len(location_lat), dtype=np.float64)
    for j, (lat0, lon0) in enumerate(zip(location_lat, location_lon, strict=True)):
        inside = haversine_km(lat, lon, float(lat0), float(lon0)) <= radius_km
        if inside.any():
            out[j] = float(wind[inside].max())
    return out
