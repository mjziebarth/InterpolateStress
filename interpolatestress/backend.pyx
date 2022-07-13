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

cdef extern from "api.hpp" namespace "interpolatestress" nogil:
    const unsigned char FAILURE_POLICY_NAN
    const unsigned char FAILURE_POLICY_SMALLEST_R_WITH_NMIN

    void interpolate_azimuth_uniform(size_t N, const double* lon,
             const double* lat, const double* azi, const double* w,
             size_t Nr, const double* r, size_t Ng, const double* lon_g,
             const double* lat_g, double* azi_g, double* azi_std_g,
             double* r_g, double critical_azi_std, size_t Nmin,
             unsigned char failure_policy,
             double a, double f) except+

    void interpolate_azimuth_gauss(size_t N, const double* lon,
             const double* lat, const double* azi, const double* w,
             size_t Nr, const double* r, size_t Ng, const double* lon_g,
             const double* lat_g, double* azi_g, double* azi_std_g,
             double* r_g, double critical_azi_std, size_t Nmin,
             unsigned char failure_policy, double kernel_bandwidth,
             double a, double f) except+


import numpy as np
from .kernel import UniformKernel, GaussianKernel



def interpolate_azimuth(const double[::1] lon, const double[::1] lat,
                        const double[::1] azi, const double[::1] weight,
                        const double[::1] search_radii,
                        const double[::1] lon_g, const double[::1] lat_g,
                        double critical_azi_std, size_t Nmin,
                        str failure_policy,
                        kernel, double a, double f):
    """
    Interpolate the stress tensor.

    Call signature:
       interpolate_azimuth(lon, lat, azi, weight, search_radii,
                           lon_g, lat_g, critical_azi_std, Nmin,
                           kernel, a, f)

    Parameters:
       lon              : Data point longitudes of shape (N,)
       lat              : Data point latitudes of shape (N,)
       azi              : Azimuths at the data points, shape (N,)
       weight           : Weighting of the data points, shape (N,)
       search_radii     : Array of search radii, from large to small.
                          Shape (Nr,)
       lon_g            : Longitudes of the interpolated grid, flattened.
                          Shape (Ng,)
       lat_g            : Latitudes of the interpolated grid, flattened.
                          Shape (Ng,)
       critical_azi_std : Target standard deviation of the azimuth within
                          a search radius below which the search radius is
                          accepted.
                          Float.
       Nmin             : Minimum number of data points within a search
                          radius required to accept the radius.
                          int.
       failure_policy   : How to handle interpolations at destination points
                          where the search criteria cannot be fulfilled
                          simultaneously. Must be one of the following
                          options:
                            - "nan":
                               Fill result at grid cells with nan
                            - "smallest":
                               Take the smallest search radius with N>=Nmin
                               data points in range, and use the resulting
                               mean and standard deviation. If no search
                               radius has enough points, fill result with nan.
       kernel           : Spatial weighting kernel to use. Must be an instance
                          of one of the kernels defined in the
                          interpolatestress.kernel submodule.
       a                : Ellipsoid large half axis. Float.
       f                : Ellipsoid flattening. Float.

    Returns:
       azi_g, azi_std_g, r_g

       azi_g     : Mean azimuth evaluated at the (flattened) grid points.
                   Shape (Ng,)
       azi_std_g : Standard deviation of azimuth evalauted at the grid points.
                   Shape (Ng,)
       r_g       : Search radii at the grid points.
                   Shape (Ng,)

    If the search algorithm failed at a grid point because the termination
    conditions could not be satisfied (simultaneously), NaN is returned in
    each of the three arrays at the corresponding index.
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

    cdef double[::1] azi_g = np.empty(Ng)
    cdef double[::1] azi_std_g = np.empty(Ng)
    cdef double[::1] r_g = np.empty(Ng)

    cdef size_t Nr = search_radii.size

    cdef unsigned char failure_policy_cpp
    if failure_policy == "nan":
        failure_policy_cpp = FAILURE_POLICY_NAN
    elif failure_policy == "smallest":
        failure_policy_cpp = FAILURE_POLICY_SMALLEST_R_WITH_NMIN
    else:
        raise RuntimeError("`failure_policy` must be one of 'nan' or "
                           "'smallest'.")

    cdef double kernel_bandwidth
    if isinstance(kernel, UniformKernel):
        with nogil:
            interpolate_azimuth_uniform(N, &lon[0], &lat[0], &azi[0],
                                        &weight[0], Nr, &search_radii[0],
                                        Ng, &lon_g[0], &lat_g[0], &azi_g[0],
                                        &azi_std_g[0], &r_g[0],
                                        critical_azi_std, Nmin,
                                        failure_policy_cpp, a, f)
    elif isinstance(kernel, GaussianKernel):
        kernel_bandwidth = kernel._bandwidth
        with nogil:
            interpolate_azimuth_gauss(N, &lon[0], &lat[0], &azi[0], &weight[0],
                                      Nr, &search_radii[0],
                                      Ng, &lon_g[0], &lat_g[0], &azi_g[0],
                                      &azi_std_g[0], &r_g[0], critical_azi_std,
                                      Nmin, failure_policy_cpp,
                                      kernel_bandwidth, a, f)
    else:
        raise TypeError("`kernel` must be a kernel from the "
                        "interpolatestress.kernel submodule.")


    return azi_g.base, azi_std_g.base, r_g.base