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
using interpolatestress::data_azi_2plunge_t;
using interpolatestress::data_scalar_t;
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

static std::vector<data_azi_2plunge_t>
fill_data_azi_plunges(size_t N, const double* lon, const double* lat,
                      const double* azi, const double* pl1, const double* pl2,
                      const double* w)
{
	std::vector<data_azi_2plunge_t> data(N);
	for (size_t i=0; i<N; ++i){
		data[i].pt.lon = lon[i];
		data[i].pt.lat = lat[i];
		data[i].w = w[i];
		data[i].azi = azi[i];
		data[i].plunge1 = pl1[i];
		data[i].plunge2 = pl2[i];
	}
	return data;
}

static std::vector<data_scalar_t>
fill_data_scalar(size_t N, const double* lon, const double* lat,
                 const double* z, const double* w)
{
	std::vector<data_scalar_t> data(N);
	for (size_t i=0; i<N; ++i){
		data[i].pt.lon = lon[i];
		data[i].pt.lat = lat[i];
		data[i].w = w[i];
		data[i].z = z[i];
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
		result = search_radius_interpolate<FAIL_NAN>(grid, data, tree,
		                                             search_radii,
		                                             exit_condition, kernel);
	else if (failure_policy == FAILURE_POLICY_SMALLEST_R_WITH_NMIN)
		result = search_radius_interpolate<FAIL_SMALLEST_NMIN_R>(grid, data,
		                                           tree, search_radii,
		                                           exit_condition, kernel);

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


/*
 * Basic setup of interpolating the azimuth and the plunges:
 */
namespace interpolatestress {

template<typename kernel_t>
void interpolate_azimuth_plunges_base(size_t N, const double* lon,
                         const double* lat, const double* azi,
                         const double* pl1, const double* pl2, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* pl1_g,
                         double* pl1_std_g, double* pl2_g, double* pl2_std_g,
                         double* r_g, double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double a, double f, kernel_t&& kernel)
{
	typedef data_azi_2plunge_t data_t;
	typedef interpolated_t<typename data_t::result_t> interp_t;

	/* Initialize the data and vantage tree: */
	std::vector<data_t> data(fill_data_azi_plunges(N, lon, lat, azi, pl1,
	                                               pl2, w));
	VantageTree<data_t> tree(data, a, f);

	/* Initialize the grid points: */
	std::vector<point_t> grid(fill_points(Ng, lon_g, lat_g));

	/* Search radii: */
	std::vector<double> search_radii(r, r+Nr);

	/* Exit condition: */
	ExitConditionAzimuthStdValidData<data_t>
	    exit_condition(Nmin, critical_azi_std);

	std::vector<interp_t> result(0);
	if (failure_policy == FAILURE_POLICY_NAN)
		result = search_radius_interpolate<FAIL_NAN>(grid, data, tree,
		                               search_radii, exit_condition, kernel);
	else if (failure_policy == FAILURE_POLICY_SMALLEST_R_WITH_NMIN)
		result = search_radius_interpolate<FAIL_SMALLEST_NMIN_R>(grid, data,
		                                           tree, search_radii,
		                                           exit_condition, kernel);

	/* Transfer results: */
	for (size_t i=0; i<Ng; ++i){
		azi_g[i] = result[i].res.azi;
	}
	for (size_t i=0; i<Ng; ++i){
		azi_std_g[i] = result[i].res.azi_std;
	}
	for (size_t i=0; i<Ng; ++i){
		pl1_g[i] = result[i].res.pl1;
	}
	for (size_t i=0; i<Ng; ++i){
		pl1_std_g[i] = result[i].res.pl1_std;
	}
	for (size_t i=0; i<Ng; ++i){
		pl2_g[i] = result[i].res.pl2;
	}
	for (size_t i=0; i<Ng; ++i){
		pl2_std_g[i] = result[i].res.pl2_std;
	}
	for (size_t i=0; i<Ng; ++i){
		r_g[i] = result[i].r;
	}
}

}


void interpolatestress::interpolate_azimuth_plunges_uniform(
                         size_t N, const double* lon, const double* lat,
                         const double* azi, const double* plunge1,
                         const double* plunge2, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* pl1_g,
                         double* pl1_std_g, double* pl2_g, double* pl2_std_g,
                         double* r_g, double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double a, double f)
{
	interpolate_azimuth_plunges_base(N, lon, lat, azi, plunge1, plunge2, w,
	                         Nr, r, Ng, lon_g, lat_g, azi_g, azi_std_g, pl1_g,
	                         pl1_std_g, pl2_g, pl2_std_g, r_g,
	                         critical_azi_std, Nmin, failure_policy, a, f,
	                         UniformKernel{});
}


void interpolatestress::interpolate_azimuth_plunges_linear(
                         size_t N, const double* lon, const double* lat,
                         const double* azi, const double* plunge1,
                         const double* plunge2, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* pl1_g,
                         double* pl1_std_g, double* pl2_g, double* pl2_std_g,
                         double* r_g, double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double a, double f)
{
	interpolate_azimuth_plunges_base(N, lon, lat, azi, plunge1, plunge2, w,
	                         Nr, r, Ng, lon_g, lat_g, azi_g, azi_std_g, pl1_g,
	                         pl1_std_g, pl2_g, pl2_std_g, r_g,
	                         critical_azi_std, Nmin, failure_policy, a, f,
	                         LinearKernel{});
}


void interpolatestress::interpolate_azimuth_plunges_gauss(
                         size_t N, const double* lon, const double* lat,
                         const double* azi, const double* plunge1,
                         const double* plunge2, const double* w,
                         size_t Nr, const double* r,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         double* azi_g, double* azi_std_g, double* pl1_g,
                         double* pl1_std_g, double* pl2_g, double* pl2_std_g,
                         double* r_g, double critical_azi_std, size_t Nmin,
                         unsigned char failure_policy,
                         double kernel_bandwidth, double a, double f)
{
	interpolate_azimuth_plunges_base(N, lon, lat, azi, plunge1, plunge2, w,
	                         Nr, r, Ng, lon_g, lat_g, azi_g, azi_std_g, pl1_g,
	                         pl1_std_g, pl2_g, pl2_std_g, r_g,
	                         critical_azi_std, Nmin, failure_policy, a, f,
	                         GaussianKernel{kernel_bandwidth});
}


/*
 * Basic setup for interpolating a scalar:
 */
namespace interpolatestress {

template<typename kernel_t>
void interpolate_scalar_base(size_t N, const double* lon, const double* lat,
                         const double* z, const double* w,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         const double* r_g, double* z_g, double* z_std_g,
                         size_t Nmin, double a, double f, kernel_t&& kernel)
{
	typedef data_scalar_t data_t;

	/* Initialize the data and vantage tree: */
	std::vector<data_t> data(fill_data_scalar(N, lon, lat, z, w));
	VantageTree<data_t> tree(data, a, f);

	/* Initialize the grid points: */
	std::vector<std::pair<point_t,double>> grid(Ng);
	for (size_t i=0; i<Ng; ++i){
		grid[i].first.lon = lon_g[i];
		grid[i].first.lat = lat_g[i];
		grid[i].second = r_g[i];
	}

	/* Interpolate: */
	std::vector<typename data_t::result_t>
	   result(interpolate(data, grid, tree, Nmin, kernel));

	/* Transfer results: */
	for (size_t i=0; i<Ng; ++i){
		z_g[i] = result[i].z;
	}
	for (size_t i=0; i<Ng; ++i){
		z_std_g[i] = result[i].z_std;
	}
}

}


void interpolatestress::interpolate_scalar_uniform(
                         size_t N, const double* lon, const double* lat,
                         const double* z, const double* w,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         const double* r_g, double* z_g, double* z_std_g,
                         size_t Nmin, double a, double f)
{
	interpolate_scalar_base(N, lon, lat, z, w, Ng, lon_g, lat_g, r_g, z_g,
	                        z_std_g, Nmin, a, f, UniformKernel{});
}


void interpolatestress::interpolate_scalar_linear(
                         size_t N, const double* lon, const double* lat,
                         const double* z, const double* w,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         const double* r_g, double* z_g, double* z_std_g,
                         size_t Nmin, double a, double f)
{
	interpolate_scalar_base(N, lon, lat, z, w, Ng, lon_g, lat_g, r_g, z_g,
	                        z_std_g, Nmin, a, f, LinearKernel{});
}


void interpolatestress::interpolate_scalar_gauss(
                         size_t N, const double* lon, const double* lat,
                         const double* z, const double* w,
                         size_t Ng, const double* lon_g, const double* lat_g,
                         const double* r_g, double* z_g, double* z_std_g,
                         size_t Nmin, double kernel_bandwidth, double a,
                         double f)
{
	interpolate_scalar_base(N, lon, lat, z, w, Ng, lon_g, lat_g, r_g, z_g,
	                        z_std_g, Nmin,a, f,
	                        GaussianKernel{kernel_bandwidth});
}
