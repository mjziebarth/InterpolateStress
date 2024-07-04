# The main stress tensor API.
#
# Author: Malte J. Ziebarth (mjz.science@fmvkb.de)
#
# Copyright (C) 2022 Malte J. Ziebarth,
#               2024 Technische Universität München
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

import numpy as np
from numpy.typing import NDArray
from typing import Literal
from math import sqrt
from .table import StressTable
from .kernel import GaussianKernel
from .ellipsoids import WGS84_a, WGS84_f
from .backend import interpolate_azimuth_plunges, interpolate_scalar, \
    interpolate_azimuth

#
# Stress tensor magnitude computations assuming
# a critically stressed crust:
#
def compute_R_prime(mu: float) -> float:
    """
    Computes the (corrected) R' from Sibson (1974) as
    a function of friction coefficient `mu`.
    """
    return 1/(sqrt(1 + mu**2) - mu)**2

def compute_S3(
        b1: NDArray[np.double], db1: NDArray[np.double],
        b2: NDArray[np.double], db2: NDArray[np.double],
        sz: NDArray[np.double], l: NDArray[np.double] | float,
        R: NDArray[np.double], dR: NDArray[np.double],
        mu: float) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """
    Computes the smallest principal stress magnitude S₃
    using the assumption of the critically stressed crust.

    Parameters:
       b1     : Plunge β₁ of the largest principal stress magnitude
                given in degrees.
       db1    : Standard error δβ₁ of β₁ given in degrees.
       b2     : Plunge β₂ of the intermediary principal stress magnitude
                given in degrees.
       db2    : Standard error δβ₂ of β₂ given in degrees.
       sz     : Vertical overburden stress S, i.e. gravitational loading.
                The resulting stress magnitude S₃ will have the same unit
                as sz, so the stress tensor can also be expressed as its
                derivative by depth.
       l      : Relative magnitude λ of the pore pressure compared to the
                overburden stress. Must be in the interval [0,1].
       R      : Stress ratio (S₁-S₂)/(S₁-S₃) following Lund (2000).
       dR     : Standard error of the stress ratio.
       mu     : Friction coefficient µ.

    Returns:
       S₃     : Smallest principal stress magnitude.
       dS₃    : Linear error propagation of δβ₁ and δβ₂ assuming indepen-
                dence of β₁ and β₂.

    The independence of β₁ and β₂ is an oversimplification but there
    might be sufficient degree of freedoms to make it a good one (taking
    also into consideration that the linear error propagation might not
    be the best approximation for the large errors).
    """
    d2r = np.pi/180.0
    sb1 = np.sin(d2r*b1)
    sb2 = np.sin(d2r*b2)
    k = sb1**2 + (1 - R) * sb2**2
    dk = np.sqrt(  (2 * sb1 * np.cos(d2r*b1) * (d2r*db1))**2
                 + (2 * (1 - R) * sb2 * np.cos(d2r*b2) * (d2r*db2))**2
                 + (sb2**2 * dR)**2)
    C = (compute_R_prime(mu) - 1) * (1 - l)
    S3 = 1/(1+k*C) * sz
    dS3 = C / (1+k*C)**2 * sz * dk
    return S3, dS3


def compute_S1(
        b1: NDArray[np.double], db1: NDArray[np.double],
        b2: NDArray[np.double], db2: NDArray[np.double],
        sz: NDArray[np.double], l: NDArray[np.double] | float,
        R: NDArray[np.double], dR: NDArray[np.double],
        mu: float
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """
    Computes the largest principal stress magnitude S₁
    using the assumption of the critically stressed crust.

    Parameters:
       b1     : Plunge β₁ of the largest principal stress magnitude
                given in degrees.
       db1    : Standard error δβ₁ of β₁ given in degrees.
       b2     : Plunge β₂ of the intermediary principal stress magnitude
                given in degrees.
       db2    : Standard error δβ₂ of β₂ given in degrees.
       sz     : Vertical overburden stress S, i.e. gravitational loading.
                The resulting stress magnitude S₃ will have the same unit
                as sz, so the stress tensor can also be expressed as its
                derivative by depth.
       l      : Relative magnitude λ of the pore pressure compared to the
                overburden stress. Must be in the interval [0,1].
       R      : Stress ratio (S₁-S₂)/(S₁-S₃) following Lund (2000).
       dR     : Standard error of the stress ratio.
       mu     : Friction coefficient µ.

    Returns:
       S₁     : Largest principal stress magnitude.
       dS₁    : Linear error propagation of δβ₁ and δβ₂ assuming indepen-
                dence of β₁ and β₂.

    The independence of β₁ and β₂ is an oversimplification but there
    might be sufficient degree of freedoms to make it a good one (taking
    also into consideration that the linear error propagation might not
    be the best approximation for the large errors).
    """
    d2r = np.deg2rad(1)
    sb1 = np.sin(d2r*b1)
    sb2 = np.sin(d2r*b2)
    k = sb1**2 + (1 - R) * sb2**2
    dk = np.sqrt(  (2 * sb1 * np.cos(d2r*b1) * (d2r*db1))**2
                 + (2 * (1 - R) * sb2 * np.cos(d2r*b2) * (d2r*db2))**2
                 + (sb2**2 * dR)**2)
    C = (compute_R_prime(mu) - 1) * (1 - l)
    S1 = (1 + C)/(1 + k*C) * sz
    dS1 = (1 + C) * C / (1 + k*C)**2 * sz * dk
    return S1, dS1


def compute_S2(
        S1: NDArray[np.double], dS1: NDArray[np.double],
        S3: NDArray[np.double], dS3: NDArray[np.double],
        R: NDArray[np.double], dR: NDArray[np.double]
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """
    Computes the intermediary principal stress magnitude S₂
    from the other principal stress magnitudes and the stress
    ratio.
    """
    S2 = S1 - R * (S1 - S3)
    dS2 = np.sqrt(((1 - R)*dS1)**2 + (dR * (S1 - S3))**2 + (R*dS3)**2)
    return S2, dS2


def ziebarth_et_al_2020_S1_S1_S3(
        regime: NDArray[np.double],
        sz: NDArray[np.double],
        l: NDArray[np.double] | float,
        mu: float,
        kappa: NDArray[np.double] | float | None = 0.5,
        regime_mode: Literal['category','smooth'] = 'category'
    ) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    """
    Computes the principal stress magnitudes S₁, S₂, and S₃
    using the assumption of Ziebarth et al. (2020), that is, a critically
    stressed crust, S₂ being vertical, and a regime parameter switching
    between the three faulting-dependent stress tensor configurations
    discussed by Sibson (1974).

    Parameters:
       regime : Faulting regime.
       sz     : Vertical overburden stress S, i.e. gravitational loading.
                The resulting stress magnitude S₃ will have the same unit
                as sz, so the stress tensor can also be expressed as its
                derivative by depth.
       l      : Relative magnitude λ of the pore pressure compared to the
                overburden stress. Must be in the interval [0,1].
       mu     : Friction coefficient µ.
       kappa  : Scaling parameter between NF, SS, and TF. Must be in the
                interval [0,1].

    Returns:
       S₁     : Largest principal stress magnitude.
       S₂     : Intermediary principal stress magnitude.
       S₃     : Smallest principal stress magnitude.
    """
    nan_mask = np.isnan(regime)

    S1 = np.empty_like(regime)
    S2 = np.empty_like(regime)
    S3 = np.empty_like(regime)

    S1[nan_mask] = np.nan
    S2[nan_mask] = np.nan
    S3[nan_mask] = np.nan

    R = compute_R_prime(mu)

    # Now handle the three faulting regimes.
    #
    # Normal faulting:
    if regime_mode == 'category':
        if kappa is None:
            raise ValueError(
                "'kappa' must be given if regime_mode 'category' is used."
            )
        tf: NDArray[np.bool] = regime < 0.5
        ss: NDArray[np.bool] = (regime >= 0.5) & (regime < 1.5)
        nf: NDArray[np.bool] = regime >= 1.5
        S1[nf] = S2[nf] = sz[nf]
        S3[nf] = (1.0 - (R - 1)/R * (1.0 - l)) * sz[nf]

        # Strike-slip faulting:
        dS = (R-1.0)/(kappa*(R-1.0)) * (1.0 - l) * sz[ss]
        S2[ss] = sz[ss]
        S1[ss] = sz[ss] + 0.5 * dS
        S3[ss] = sz[ss] - 0.5 * dS

        # Thrust faulting:
        S3[tf] = S2[tf] = sz[tf]
        S1[tf] = (1.0 + (R - 1) * (1.0 - l)) * sz[tf]

    else:
        if kappa is not None:
            raise ValueError(
                "'kappa' must be None if regime_mode 'smooth' is used."
            )
        # Use the kappa parameter as a smooth interpolator.
        kappa = regime / 2
        C = (R-1) / (kappa * (R-1) + 1) * (1-l)
        S1 = sz * (1.0 + C*(1-kappa))
        S3 = sz * (1.0 - kappa * C)
        S2 = sz

    return S1, S2, S3






def _preprocess_interpolation_arguments(
        lon, lat, table, mu, Sz,
        marker=None,
        search_radii=np.geomspace(1.5e6, 1e2, 200),
        Nmin=10, critical_azimuth_std=15.0,
        failure_policy='smallest',
        lambda_pore_pressure=0.0,
        a=WGS84_a, f=WGS84_f
    ) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double],
               NDArray[np.double] | None, NDArray[np.double] | None,
               NDArray[np.double], NDArray[np.double], NDArray[np.double],
               NDArray[np.double], NDArray[np.double], NDArray[np.double],
               tuple[int, ...],
               NDArray[np.double], NDArray[np.double],
               int, float, float, float, float, Literal['smallest', 'nan'],
               NDArray[np.ushort], NDArray[np.ushort]
    ]:

    # Obtain all relevant data from the table:
    if not isinstance(table, StressTable):
        # Try to load:
        if isinstance(table,str):
            if 'wsm' in table and '2016' in table and table[-4:] == '.csv':
                table = StressTable.from_wsm2016(table)
            else:
                raise TypeError("`table` has to be a StressTable instance.")
        else:
            raise TypeError("`table` has to be a StressTable instance.")

    data_lon: NDArray[np.double] \
        = np.ascontiguousarray(table.lon, dtype=np.double)
    data_lat: NDArray[np.double] \
        = np.ascontiguousarray(table.lat, dtype=np.double)
    data_azimuth: NDArray[np.double] \
        = np.ascontiguousarray(table.azimuth, dtype=np.double)
    data_plunge1: NDArray[np.double] | None
    data_plunge2: NDArray[np.double] | None
    if table.plunge1 is None or table.plunge2 is None:
        data_plunge1 = None
        data_plunge2 = None
    else:
        data_plunge1 = np.ascontiguousarray(table.plunge1, dtype=np.double)
        data_plunge2 = np.ascontiguousarray(table.plunge2, dtype=np.double)
    data_weight = np.ascontiguousarray(table.weight, dtype=np.double)
    data_R = np.ascontiguousarray(table.stress_ratio, dtype=np.double)
    data_regime = np.ascontiguousarray(table.regime, dtype=np.double)

    # Sanity check all other data:
    search_radii: NDArray[np.double] \
        = np.ascontiguousarray(search_radii, dtype=np.double)
    lon: NDArray[np.double] \
        = np.ascontiguousarray(lon, dtype=np.double)
    lat: NDArray[np.double] \
        = np.ascontiguousarray(lat, dtype=np.double)
    shape: tuple[int, ...] = lon.shape
    long: NDArray[np.double] \
        = np.ascontiguousarray(lon.reshape(-1), dtype=np.double)
    latg: NDArray[np.double] \
        = np.ascontiguousarray(lat.reshape(-1), dtype=np.double)
    Nmin = int(Nmin)
    a = float(a)
    f = float(f)
    critical_azimuth_std = float(critical_azimuth_std)
    lambda_pore_pressure = float(lambda_pore_pressure)
    if failure_policy not in ('nan','smallest'):
        raise ValueError("`failure_policy` must be one of 'nan' or "
                            "'smallest'.")
    if marker is None:
        data_marker: NDArray[np.ushort] = np.empty((0,), dtype=np.ushort)
        grid_marker: NDArray[np.ushort] = np.empty((0,), dtype=np.ushort)
    else:
        data_marker: NDArray[np.ushort] \
            = np.ascontiguousarray(marker(data_lon, data_lat),
                                   dtype=np.ushort)
        grid_marker: NDArray[np.ushort] \
            = np.ascontiguousarray(marker(long, latg),
                                   dtype=np.ushort)

    return (data_lon, data_lat, data_azimuth, data_plunge1, data_plunge2,
        data_weight, data_R, data_regime, search_radii, lon, lat, shape, long,
        latg, Nmin, a, f, critical_azimuth_std, lambda_pore_pressure,
        failure_policy, data_marker, grid_marker
    )




#
# The main stress tensor class:
#

class StressTensor:
    """
    The crustal stress tensor, evaluated at multiple locations.
    """
    def __init__(self, lon, lat, azimuth, azimuth_std, plunge1, plunge1_std,
                 plunge2, plunge2_std, S1, S1_std, S2, S2_std, S3, S3_std):
        self.shape = lon.shape
        self.lola = np.stack((lon,lat), axis=len(self.shape))
        self.lon = self.lola[...,0]
        self.lat = self.lola[...,1]
        self.angles = np.stack((azimuth, azimuth_std,
                                plunge1, plunge1_std,
                                plunge2, plunge2_std),
                               axis=len(self.shape))
        self.azimuth = self.angles[...,0]
        self.plunge1 = self.angles[...,2]
        self.plunge2 = self.angles[...,4]
        self.S = np.stack((S1, S1_std,
                           S2, S2_std,
                           S3, S3_std),
                          axis=len(self.shape))


    @staticmethod
    def critical_stress_interpolated(lon, lat, table, mu, Sz,
                                     marker=None,
                                     search_radii=np.geomspace(1.5e6, 1e2, 200),
                                     kernel=GaussianKernel(), Nmin=10,
                                     Nmin_R=5, critical_azimuth_std=15.0,
                                     failure_policy='smallest',
                                     lambda_pore_pressure=0.0,
                                     a=WGS84_a, f=WGS84_f):
        """
        Interpolate a stress table assuming a critically stressed crust.
        """
        # Obtain all relevant data from the table:
        if not isinstance(table, StressTable):
            # Try to load:
            if isinstance(table,str):
                if 'wsm' in table and '2016' in table and table[-4:] == '.csv':
                    table = StressTable.from_wsm2016(table)
                else:
                    raise TypeError("`table` has to be a StressTable instance.")
            else:
                raise TypeError("`table` has to be a StressTable instance.")

        (
            data_lon, data_lat, data_azimuth, data_plunge1, data_plunge2,
            data_weight, data_R, data_regime, search_radii, lon, lat, shape,
            long, latg, Nmin, a, f, critical_azimuth_std, lambda_pore_pressure,
            failure_policy, data_marker, grid_marker
        ) = _preprocess_interpolation_arguments(
                lon, lat, table, mu, Sz,
                marker=marker, search_radii=search_radii,
                Nmin=Nmin, critical_azimuth_std=critical_azimuth_std,
                failure_policy=failure_policy,
                lambda_pore_pressure=0.0,
                a=a, f=f
            )

        if data_plunge1 is None or data_plunge2 is None:
            raise RuntimeError("Plunges are needed in function "
                                "`critical_stress_interpolated`. If no plunges "
                                "are available, consider using the "
                                "`ziebarth_et_al_2020` method.")

        # Perform the interpolation:
        azi_g, azi_std_g, pl1_g, pl1_std_g, pl2_g, pl2_std_g, r_g \
           = interpolate_azimuth_plunges(data_lon, data_lat, data_azimuth,
                                         data_plunge1, data_plunge2,
                                         data_weight, data_marker, search_radii,
                                         long, latg, grid_marker,
                                         critical_azimuth_std, Nmin,
                                         failure_policy, kernel, a, f)

        # Interpolate the stress ratio.
        has_ratio = ~np.isnan(data_R)
        R_g, R_std_g \
           = interpolate_scalar(data_lon[has_ratio], data_lat[has_ratio],
                                data_R[has_ratio], data_weight[has_ratio],
                                long, latg, r_g, Nmin_R, kernel, a, f)

        # Compute magnitudes of the principal stresses assuming
        # the critically stressed crust:
        S1, dS1 = compute_S1(pl1_g, pl1_std_g, pl2_g, pl2_std_g, Sz,
                             lambda_pore_pressure, R_g, R_std_g, mu)
        S3, dS3 = compute_S3(pl1_g, pl1_std_g, pl2_g, pl2_std_g, Sz,
                             lambda_pore_pressure, R_g, R_std_g, mu)
        S2, dS2 = compute_S2(S1, dS1, S3, dS3, R_g, R_std_g)

        # Return the result:
        return StressTensor(lon, lat, azi_g.reshape(shape),
                            azi_std_g.reshape(shape), pl1_g.reshape(shape),
                            pl1_std_g.reshape(shape), pl2_g.reshape(shape),
                            pl2_std_g.reshape(shape), S1.reshape(shape),
                            dS1.reshape(shape), S2.reshape(shape),
                            dS2.reshape(shape), S3.reshape(shape),
                            dS3.reshape(shape))


    @staticmethod
    def ziebarth_et_al_2020(
            lon,
            lat,
            table,
            mu,
            Sz: NDArray[np.double] | float,
            marker=None,
            search_radii=np.geomspace(1.5e6, 1e2, 200),
            kernel=GaussianKernel(),
            Nmin=10,
            critical_azimuth_std=15.0,
            failure_policy='smallest',
            lambda_pore_pressure=0.0,
            a=WGS84_a,
            f=WGS84_f,
            kappa: float | Literal['auto'] = 'auto',
            regime_mode: Literal['category','smooth'] = 'category'
        ):
        """
        Interpolate a stress table as done in Ziebarth et al. (2020).
        """
        # Sanity on kappa and regime mode:
        kappa_: float | None = None
        if regime_mode == 'category':
            if kappa == 'auto':
                kappa_ = 0.5
            else:
                kappa_ = kappa
        elif regime_mode == 'smooth':
            if kappa != 'auto':
                raise ValueError(
                    "If regime mode 'smooth' is used, 'kappa' has to be 'auto'."
                )

        else:
            raise ValueError(
                "regime_mode must be one of 'category' or 'smooth'."
            )


        # Obtain all relevant data from the table:
        if not isinstance(table, StressTable):
            # Try to load:
            if isinstance(table,str):
                if 'wsm' in table and '2016' in table and table[-4:] == '.csv':
                    table = StressTable.from_wsm2016(table)
                else:
                    raise TypeError("`table` has to be a StressTable instance.")
            else:
                raise TypeError("`table` has to be a StressTable instance.")

        (
            data_lon, data_lat, data_azimuth, data_plunge1, data_plunge2,
            data_weight, data_R, data_regime, search_radii, lon, lat, shape,
            long, latg, Nmin, a, f, critical_azimuth_std, lambda_pore_pressure,
            failure_policy, data_marker, grid_marker
        ) = _preprocess_interpolation_arguments(
                lon, lat, table, mu, Sz,
                marker=marker, search_radii=search_radii,
                Nmin=Nmin, critical_azimuth_std=critical_azimuth_std,
                failure_policy=failure_policy,
                lambda_pore_pressure=0.0,
                a=a, f=f
            )

        # Perform the interpolations:
        azi_g, azi_std_g, r_g \
           = interpolate_azimuth(
                data_lon, data_lat, data_azimuth, data_weight, data_marker,
                search_radii, long, latg, grid_marker, critical_azimuth_std,
                Nmin, failure_policy, kernel, a, f
             )

        regime_mask = ~np.isnan(data_regime)
        regime, regime_std = \
            interpolate_scalar(
                data_lon[regime_mask],
                data_lat[regime_mask],
                data_regime[regime_mask],
                data_weight[regime_mask],
                long, latg, r_g, Nmin,
                kernel, a, f
            )

        # Convert Sz to NDArray:
        sz = np.asarray(Sz)
        if sz.size == 1:
            sz = np.full_like(regime, sz.flat[0])
        else:
            assert sz.shape == regime.shape

        # Compute magnitudes of the principal stresses assuming
        # the critically stressed crust:
        S1, S2, S3 = ziebarth_et_al_2020_S1_S1_S3(
            regime, sz, lambda_pore_pressure, mu, kappa_, regime_mode
        )

        # Set nans:
        nan_dummy = np.full_like(azi_g, np.nan).reshape(shape)

        # Return the result:
        return StressTensor(lon, lat, azi_g.reshape(shape),
                            azi_std_g.reshape(shape), nan_dummy,
                            nan_dummy, nan_dummy,
                            nan_dummy, S1.reshape(shape),
                            nan_dummy, S2.reshape(shape),
                            nan_dummy, S3.reshape(shape),
                            nan_dummy)
