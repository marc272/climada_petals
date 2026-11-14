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

Small synthetic CLIMADA objects for the tests. Nothing is downloaded."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import xarray as xr
from climada.engine import Impact, ImpactCalc
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet, ImpfTropCyclone
from climada.hazard import Centroids, Hazard, TCTracks, TropCyclone
from numpy.typing import ArrayLike

from climada_petals.engine.parametric import (
    IndexDefinition,
    InsuredUnits,
    ParametricIndex,
)

#: The definition of the indices that :func:`array_index` builds.
TABLE_WIND = IndexDefinition("table", unit="m/s", source="synthetic")

#: Three insured points around Miami (latitude, longitude, value in USD).
POINTS: tuple[tuple[float, float, float], ...] = (
    (25.7617, -80.1918, 1.0e9),
    (25.7907, -80.1300, 2.0e9),
    (25.9500, -80.1400, 5.0e8),
)


def straight_track(
    sid: str,
    lat: float,
    lon_from: float,
    lon_to: float,
    wind_kn: float,
    *,
    n_points: int = 13,
    year: int = 2000,
) -> xr.Dataset:
    """Return a CLIMADA track moving along one latitude, 3 h between points."""
    time = np.datetime64(f"{year}-09-01T00:00") + np.arange(n_points) * np.timedelta64(
        3, "h"
    )
    return xr.Dataset(
        data_vars={
            "time_step": ("time", np.full(n_points, 3.0)),
            "radius_max_wind": ("time", np.full(n_points, 20.0)),
            "radius_oci": ("time", np.full(n_points, 150.0)),
            "max_sustained_wind": ("time", np.full(n_points, float(wind_kn))),
            "central_pressure": ("time", np.full(n_points, 960.0)),
            "environmental_pressure": ("time", np.full(n_points, 1013.0)),
            "basin": ("time", np.full(n_points, "NA")),
        },
        coords={
            "time": ("time", time),
            "lat": ("time", np.full(n_points, float(lat))),
            "lon": ("time", np.linspace(lon_from, lon_to, n_points)),
        },
        attrs={
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "hPa",
            "name": sid,
            "sid": sid,
            "orig_event_flag": True,
            "data_provider": "synthetic",
            "id_no": 0,
            "category": 0,
        },
    )


def three_tracks() -> TCTracks:
    """Return the tracks ``A``, ``B`` and ``C`` near :data:`POINTS`.

    Track ``A`` runs along latitude 25.75 from 77 to 83 W, 0.5 degrees (about
    50 km) between points, at 120.5 kn (category 4), in 2001; ``B`` along
    26.9 N at 73.9 kn (category 1), 105.6 km from the third point and about
    125 km from the others, in 2003; ``C`` along 29.5 N at 37 kn (category 0),
    in 2003. The first point lies 1.3 km, the second 4.5 km and the third
    22.2 km from A's path.
    """
    return TCTracks(
        [
            straight_track("A", 25.75, -77.0, -83.0, 120.5, year=2001),
            straight_track("B", 26.9, -77.0, -83.0, 73.9, year=2003),
            straight_track("C", 29.5, -78.0, -81.0, 37.0, n_points=6, year=2003),
        ]
    )


def exposures(points: Sequence[tuple[float, float, float]] = POINTS) -> Exposures:
    """Return one exposure point per insured unit, with impact function 1."""
    table = np.array(points, dtype=float)
    lat, lon, value = table[:, 0], table[:, 1], table[:, 2]
    exp = Exposures(lat=lat, lon=lon, value=value)
    exp.gdf["impf_TC"] = 1
    return exp


def hazard_and_impact(
    tracks: TCTracks,
    exp: Exposures,
    *,
    frequency: float | None = None,
    date: ArrayLike | None = None,
) -> tuple[Any, Impact]:
    """Return the wind hazard at the exposure points and the impact on them.

    The impact is computed on a copy of ``exp``, so ``exp`` gets no centroid
    column. ``frequency`` (one value for every event) and ``date`` replace
    what ``TropCyclone.from_tracks`` sets, before the impact is computed.
    """
    centroids = Centroids(
        lat=np.array(exp.latitude, dtype=float),
        lon=np.array(exp.longitude, dtype=float),
    )
    hazard = TropCyclone.from_tracks(
        tracks, centroids=centroids, ignore_distance_to_coast=True
    )
    if frequency is not None:
        hazard.frequency = np.full(hazard.size, float(frequency))
    if date is not None:
        hazard.date = np.asarray(date, dtype=np.int64)
    impfset = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])
    impact = ImpactCalc(exp.copy(), impfset, hazard).impact(save_mat=True)
    return hazard, impact


def catalogue_tracks(storms: Sequence[tuple[int, int, float, float]]) -> TCTracks:
    """Return tracks of a synthetic catalogue in the IRIS layout.

    Each storm ``(year, number, latitude, wind_kn)`` moves west along its
    latitude and is named ``NA_<year>_<number>``. As in IRIS, every track
    starts on one date (1 September 2000), so the dates carry no year.
    """
    return TCTracks(
        [
            straight_track(f"NA_{year}_{number}", lat, -77.0, -83.0, wind, year=2000)
            for year, number, lat, wind in storms
        ]
    )


def array_index(
    values: ArrayLike,
    loss: ArrayLike,
    event_year: ArrayLike,
    n_years: int,
    *,
    frequency: ArrayLike | None = None,
    definition: IndexDefinition = TABLE_WIND,
) -> tuple[ParametricIndex, Impact]:
    """Return an index and an ``Impact`` built from arrays.

    Every event has frequency ``1 / n_years`` unless given; the years are
    ``0 .. n_years - 1``; the units are one exposure point each, of value 1,
    named ``U0, U1, ...``.
    """
    index_values = np.asarray(values, dtype=float)
    n_units = index_values.shape[1]
    points = np.column_stack(
        [np.full(n_units, 25.0), np.linspace(-81.0, -80.0, n_units)]
    )
    impact = _array_impact(
        np.asarray(loss, dtype=float),
        (
            np.full(index_values.shape[0], 1.0 / n_years)
            if frequency is None
            else np.asarray(frequency, dtype=float)
        ),
        points,
    )
    exp = Exposures(lat=points[:, 0], lon=points[:, 1], value=np.ones(n_units))
    exp.gdf.index = [f"U{j}" for j in range(n_units)]
    units = InsuredUnits.points(exp)
    index = ParametricIndex(
        index_values,
        impact.event_id,
        units.unit_id,
        definition,
        event_name=impact.event_name,
        units=units,
    )
    return (
        index.with_years(
            impact, event_year=np.asarray(event_year), years=np.arange(n_years)
        ),
        impact,
    )


def _array_impact(
    losses: np.ndarray, frequency: np.ndarray, points: np.ndarray
) -> Impact:
    """Return an ``Impact`` with the losses of events 1, 2, ... at the points."""
    n_events = losses.shape[0]
    return Impact(
        event_id=np.arange(1, n_events + 1),
        event_name=[f"e{i}" for i in range(n_events)],
        date=np.ones(n_events, dtype=int),
        frequency=frequency,
        coord_exp=points,
        eai_exp=frequency @ losses,
        at_event=losses.sum(axis=1),
        aai_agg=float((frequency @ losses).sum()),
        imp_mat=sp.csr_matrix(losses),
        unit="USD",
    )


def random_portfolio(
    seed: int, *, n_events: int = 40, n_units: int = 3, n_years: int = 60
) -> tuple[ParametricIndex, Impact]:
    """Return a random index and impact: several events in some years, none in others.

    Losses grow with the index above 25 (as a wind damage function would)
    with noise, so that index and loss are related but not one-to-one.
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 80.0, size=(n_events, n_units))
    noise = rng.lognormal(0.0, 0.5, size=(n_events, n_units))
    loss = np.where(values > 25.0, (values - 25.0) ** 2 * noise, 0.0)
    loss[rng.random((n_events, n_units)) < 0.15] = 0.0
    years = np.sort(rng.integers(0, n_years, size=n_events))
    return array_index(values, loss, years, n_years)


#: Six points near the four centroids of :func:`grid_hazard`: latitude,
#: longitude, value and region. The first two share a centroid; the last,
#: of value 0, is in no region.
GRID_POINTS: tuple[tuple[float, float, float, float], ...] = (
    (10.0, 20.0, 1.0, 1),
    (10.001, 20.0, 3.0, 1),
    (10.0, 21.0, 2.0, 1),
    (11.0, 20.0, 4.0, 2),
    (11.0, 21.0, 4.0, 2),
    (11.0, 21.001, 0.0, np.nan),
)

#: Flood depths (m) of the events 30, 10 and 20 at the four centroids.
FLOOD_DEPTH: tuple[tuple[float, ...], ...] = (
    (0.0, 2.0, 1.0, 0.0),
    (1.0, 3.0, 0.0, 0.0),
    (0.5, 0.0, 4.0, 2.0),
)


def grid_exposures(
    points: Sequence[tuple[float, float, float, float]] = GRID_POINTS,
) -> Exposures:
    """Return the points of :data:`GRID_POINTS`, with a ``region_id`` column."""
    table = np.array(points, dtype=float)
    exp = Exposures(lat=table[:, 0], lon=table[:, 1], value=table[:, 2])
    exp.gdf["region_id"] = table[:, 3]
    exp.gdf["impf_FL"] = 1
    exp.gdf["impf_DR"] = 1
    return exp


def grid_hazard(
    intensity: ArrayLike = FLOOD_DEPTH,
    *,
    haz_type: str = "FL",
    units: str = "m",
    event_id: ArrayLike = (30, 10, 20),
    event_name: Sequence[str] = ("a", "b", "c"),
) -> Hazard:
    """Return a hazard on four centroids at 10-11 N, 20-21 E, 1 degree apart.

    The events have frequency 1 / 10 and dates in 2001, 2002, 2003.
    """
    matrix = sp.csr_matrix(np.asarray(intensity, dtype=float))
    n_events = matrix.shape[0]
    return Hazard(
        haz_type=haz_type,
        units=units,
        centroids=Centroids(
            lat=np.array([10.0, 10.0, 11.0, 11.0]),
            lon=np.array([20.0, 21.0, 20.0, 21.0]),
        ),
        event_id=np.asarray(event_id),
        event_name=list(event_name),
        frequency=np.full(n_events, 0.1),
        date=np.array([dt.date(2001 + k, 6, 1).toordinal() for k in range(n_events)]),
        intensity=matrix,
        fraction=sp.csr_matrix(matrix.shape),
    )


def grid_impact(hazard: Hazard, exp: Exposures) -> Impact:
    """Return the impact of ``hazard`` on ``exp`` (a copy), with ``imp_mat``.

    The impact function maps 0 to 5 (any unit) linearly to 0 to 100 % of
    the value; negative intensities do no harm.
    """
    impf = ImpactFunc(
        haz_type=hazard.haz_type,
        id=1,
        intensity=np.array([-10.0, 0.0, 5.0]),
        mdd=np.array([0.0, 0.0, 1.0]),
        paa=np.ones(3),
        intensity_unit=hazard.units,
        name="linear",
    )
    return ImpactCalc(exp.copy(), ImpactFuncSet([impf]), hazard).impact(save_mat=True)


def shuffled(hazard: Hazard, order: Sequence[int]) -> Hazard:
    """Return ``hazard`` with its events in another order."""
    rows = np.asarray(order)
    return Hazard(
        haz_type=hazard.haz_type,
        units=hazard.units,
        centroids=hazard.centroids,
        event_id=hazard.event_id[rows],
        event_name=[hazard.event_name[k] for k in rows],
        frequency=hazard.frequency[rows],
        date=hazard.date[rows],
        intensity=hazard.intensity[rows],
        fraction=hazard.fraction[rows],
    )
