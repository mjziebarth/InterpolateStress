# Interface to C++ backend.
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

cdef extern from "api.hpp" namespace "interpolatestress":
    void interpolate_azimuth_uniform(size_t N, const double* lon,
             const double* lat, const double* azi, const double* w,
             size_t Nr, const double* r, size_t Ng, const double* lon_g,
             const double* lat_g, double* azi_g, double* azi_std_g,
             double* r_g, double critical_azi_std, size_t Nmin,
             double a, double f) except+

    void interpolate_azimuth_gauss(size_t N, const double* lon,
             const double* lat, const double* azi, const double* w,
             size_t Nr, const double* r, size_t Ng, const double* lon_g,
             const double* lat_g, double* azi_g, double* azi_std_g,
             double* r_g, double critical_azi_std, size_t Nmin,
             double kernel_bandwidth, double a, double f) except+


import numpy as np
from .kernel import UniformKernel, GaussianKernel



def interpolate_azimuth(const double[:] lon, const double[:] lat,
                        const double[:] azi, const double[:] weight,
                        const double[:] search_radii,
                        const double[:] lon_g, const double[:] lat_g,
                        double critical_azi_std, size_t Nmin,
                        kernel, double a, double f):
    """
    Interpolate the stress tensor.
    """
    # Sanity checks:
    cdef size_t N = lon.size
    if lat.size != N:
        raise RuntimeError("Shapes of `lon` and `lat` not equal.")
    if azi.size != N:
        raise RuntimeError("Shapes of `lon` and `azi` not equal.")
    if weight.size != N:
        raise RuntimeError("Shapes of `lon` and `weight` not equal.")

    cdef size_t Ng = lon_g.size
    if lat_g.size != Ng:
        raise RuntimeError("Shapes of `lon_g` and `lat_g` not equal.")

    cdef double[:] azi_g = np.empty(Ng)
    cdef double[:] azi_std_g = np.empty(Ng)
    cdef double[:] r_g = np.empty(Ng)

    cdef double kernel_bandwidth
    if isinstance(kernel, UniformKernel):
        interpolate_azimuth_uniform(N, &lon[0], &lat[0], &azi[0], &weight[0],
                                    search_radii.size, &search_radii[0],
                                    Ng, &lon_g[0], &lat_g[0], &azi_g[0],
                                    &azi_std_g[0], &r_g[0], critical_azi_std,
                                    Nmin, a, f)
    elif isinstance(kernel, GaussianKernel):
        kernel_bandwidth = kernel._bandwidth
        interpolate_azimuth_gauss(N, &lon[0], &lat[0], &azi[0], &weight[0],
                                  search_radii.size, &search_radii[0],
                                  Ng, &lon_g[0], &lat_g[0], &azi_g[0],
                                  &azi_std_g[0], &r_g[0], critical_azi_std,
                                  Nmin, kernel_bandwidth, a, f)
    else:
        raise TypeError("`kernel` must be a kernel from the "
                        "interpolatestress.kernel submodule.")


    return azi_g.base, azi_std_g.base, r_g.base