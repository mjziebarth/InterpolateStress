# The main stress tensor API.
#
# Author: Malte J. Ziebarth (mjz.science@fmvkb.de)
#
# Copyright (C) 2022 Malte J. Ziebarth
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
from math import sqrt
from .table import StressTable
from .kernel import GaussianKernel
from .ellipsoids import WGS84_a, WGS84_f
from .backend import interpolate_azimuth_plunges, interpolate_scalar

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

def compute_S3(b1: np.ndarray, db1: np.ndarray, b2: np.ndarray, db2:np.ndarray,
               sz: np.ndarray, l: np.ndarray, R: np.ndarray, dR: np.ndarray,
               mu: float) -> np.ndarray:
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


def compute_S1(b1: np.ndarray, db1: np.ndarray, b2: np.ndarray, db2:np.ndarray,
               sz: np.ndarray, l: np.ndarray, R: np.ndarray, dR: np.ndarray,
               mu: float) -> np.ndarray:
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


def compute_S2(S1: np.ndarray, dS1: np.ndarray, S3: np.ndarray, dS3: np.ndarray,
               R: np.ndarray, dR: np.ndarray):
    """
    Computes the intermediary principal stress magnitude S₂
    from the other principal stress magnitudes and the stress
    ratio.
    """
    S2 = S1 - R * (S1 - S3)
    dS2 = np.sqrt(((1 - R)*dS1)**2 + (dR * (S1 - S3))**2 + (R*dS3)**2)
    return S2, dS2



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

        data_lon = np.ascontiguousarray(table.lon)
        data_lat = np.ascontiguousarray(table.lat)
        data_azimuth = np.ascontiguousarray(table.azimuth)
        if table.plunge1 is None or table.plunge2 is None:
            raise RuntimeError("Plunges are needed in function "
                               "`critical_stress_interpolated`. If no plunges "
                               "are available, consider using the "
                               "`ziebarth_et_al_2020` method.")
        data_plunge1 = np.ascontiguousarray(table.plunge1)
        data_plunge2 = np.ascontiguousarray(table.plunge2)
        data_weight = np.ascontiguousarray(table.weight)
        data_R = np.ascontiguousarray(table.stress_ratio)

        # Sanity check all other data:
        search_radii = np.ascontiguousarray(search_radii)
        lon = np.array(lon,copy=False)
        lat = np.array(lat,copy=False)
        shape = lon.shape
        long = np.ascontiguousarray(lon.reshape(-1))
        latg = np.ascontiguousarray(lat.reshape(-1))
        Nmin = int(Nmin)
        a = float(a)
        f = float(f)
        critical_azimuth_std = float(critical_azimuth_std)
        if failure_policy not in ('nan','smallest'):
            raise ValueError("`failure_policy` must be one of 'nan' or "
                             "'smallest'.")

        # Perform the interpolation:
        azi_g, azi_std_g, pl1_g, pl1_std_g, pl2_g, pl2_std_g, r_g \
           = interpolate_azimuth_plunges(data_lon, data_lat, data_azimuth,
                                         data_plunge1, data_plunge2,
                                         data_weight, search_radii, long, latg,
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
    def ziebarth_et_al_2020(table, kernel, mu):
        """
        Interpolate a stress table as done in Ziebarth et al. (2020).
        """
        raise NotImplementedError()
