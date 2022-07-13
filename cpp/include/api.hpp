/*
 * C++ API of InterpolateStress.
 *
 * Author: Malte J. Ziebarth (mjz.science@fmvkb.de)
 *
 * Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

#include <cstddef>

#ifndef INTERPOLATE_STRESS_API_HPP
#define INTERPOLATE_STRESS_API_HPP

namespace interpolatestress {

constexpr unsigned char FAILURE_POLICY_NAN = 0;
constexpr unsigned char FAILURE_POLICY_SMALLEST_R_WITH_NMIN = 1;


/*
 * Uniform kernel.
 */
void interpolate_azimuth_uniform(size_t N, const double* lon, const double* lat,
                                 const double* azi, const double* w,
                                 size_t Nr, const double* r,
                                 size_t Ng, const double* lon_g,
                                 const double* lat_g, double* azi_g,
                                 double* azi_std_g, double* r_g,
                                 double critical_azi_std, size_t Nmin,
                                 unsigned char failure_policy,
                                 double a, double f);


/*
 * Gaussian kernel.
 */
void interpolate_azimuth_gauss(size_t N, const double* lon, const double* lat,
                               const double* azi, const double* w,
                               size_t Nr, const double* r,
                               size_t Ng, const double* lon_g,
                               const double* lat_g, double* azi_g,
                               double* azi_std_g, double* r_g,
                               double critical_azi_std, size_t Nmin,
                               unsigned char failure_policy,
                               double kernel_bandwidth,
                               double a, double f);



}

#endif