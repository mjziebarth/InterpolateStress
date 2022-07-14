/*
 * The API callable functions.
 *
 * Author: Malte J. Ziebarth (mjz.science@fmvkb.de)
 *
 * Copyright (C) 2022 Malte J. Ziebarth
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

#include <../include/api.hpp>
#include <../include/kernel.hpp>
#include <../include/interpolate.hpp>
#include <../include/exitcondition.hpp>

using interpolatestress::point_t;
using interpolatestress::data_azi_t;
using interpolatestress::FAIL_NAN;
using interpolatestress::FAIL_SMALLEST_NMIN_R;


static std::vector<point_t>
fill_points(size_t N, const double* lon, const double* lat)
{
	std::vector<point_t> pts(N);
	for (size_t i=0; i<N; ++i){
		pts[i].lon = lon[i];
		pts[i].lat = lat[i];
	}
	return pts;
}

static std::vector<data_azi_t>
fill_data_azi(size_t N, const double* lon, const double* lat, const double* azi,
              const double* w)
{
	std::vector<data_azi_t> data(N);
	for (size_t i=0; i<N; ++i){
		data[i].pt.lon = lon[i];
		data[i].pt.lat = lat[i];
		data[i].w = w[i];
		data[i].azi = azi[i];
	}
	return data;

}

/*
 * Basic setup of interpolating the azimuth:
 */
namespace interpolatestress {

template<typename kernel_t>
void interpolate_azimuth_base(size_t N, const double* lon, const double* lat,
                         const double* azi, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* r_g,
                         double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double a, double f, kernel_t&& kernel)
{
	typedef data_azi_t data_t;
	typedef interpolated_t<typename data_t::result_t> interp_t;

	/* Initialize the data and vantage tree: */
	std::vector<data_t> data(fill_data_azi(N, lon, lat, azi, w));
	VantageTree<data_t> tree(data, a, f);

	/* Initialize the grid points: */
	std::vector<point_t> grid(fill_points(Ng, lon_g, lat_g));

	/* Search radii: */
	std::vector<double> search_radii(r, r+Nr);

	/* Exit condition: */
	ExitConditionAzimuthStd<data_t> exit_condition(Nmin, critical_azi_std);

	std::vector<interp_t> result(0);
	if (failure_policy == FAILURE_POLICY_NAN)
		result = interpolate<FAIL_NAN>(grid, data, tree, search_radii,
		                               exit_condition, kernel);
	else if (failure_policy == FAILURE_POLICY_SMALLEST_R_WITH_NMIN)
		result = interpolate<FAIL_SMALLEST_NMIN_R>(grid, data, tree,
		                                           search_radii, exit_condition,
		                                           kernel);

	/* Transfer results: */
	for (size_t i=0; i<Ng; ++i){
		azi_g[i] = result[i].res.azi;
	}
	for (size_t i=0; i<Ng; ++i){
		azi_std_g[i] = result[i].res.azi_std;
	}
	for (size_t i=0; i<Ng; ++i){
		r_g[i] = result[i].r;
	}
}

}


void interpolatestress::interpolate_azimuth_uniform(size_t N, const double* lon,
                         const double* lat,
                         const double* azi, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* r_g,
                         double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double a, double f)
{
	interpolate_azimuth_base(N, lon, lat, azi, w, Nr, r, Ng, lon_g, lat_g,
	                         azi_g, azi_std_g, r_g, critical_azi_std, Nmin,
	                         failure_policy, a, f, UniformKernel{});
}


void interpolatestress::interpolate_azimuth_linear(size_t N, const double* lon,
                         const double* lat,
                         const double* azi, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* r_g,
                         double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double a, double f)
{
	interpolate_azimuth_base(N, lon, lat, azi, w, Nr, r, Ng, lon_g, lat_g,
	                         azi_g, azi_std_g, r_g, critical_azi_std, Nmin,
	                         failure_policy, a, f, LinearKernel{});
}


void interpolatestress::interpolate_azimuth_gauss(size_t N, const double* lon,
                         const double* lat,
                         const double* azi, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* r_g,
                         double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double kernel_bandwidth, double a, double f)
{
	interpolate_azimuth_base(N, lon, lat, azi, w, Nr, r, Ng, lon_g, lat_g,
	                         azi_g, azi_std_g, r_g, critical_azi_std, Nmin,
	                         failure_policy, a, f,
	                         GaussianKernel{kernel_bandwidth});
}
