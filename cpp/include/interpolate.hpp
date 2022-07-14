/*
 * Interpolate azimuth and plunges following mostly the algorithm
 * described by Ziegler & Heidbach (2017).
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
 *
 * References:
 * Ziegler, M. O. and Heidbach, O. (2017). Manual of the Matlab Script
 *    Stress2Grid. World Stress Map Technical Report 17-02, GFZ German Research
 *    Centre for Geosciences. DOI: https://doi.org/10.2312/wsm.2017.002
 */

#include <vector>
#include <data.hpp>
#include <algorithm>
#include <vantagetree.hpp>
#include <exitcondition.hpp>

#ifndef INTERPOLATE_STRESS_INTERPOLATE_HPP
#define INTERPOLATE_STRESS_INTERPOLATE_HPP

namespace interpolatestress {

template<typename result_t>
struct interpolated_t {
	double r;
	result_t res;
};


/*
 * This enum defines how to handle cases in which the exit condition
 * cannot be fulfilled.
 *   FAIL_NAN :
 */
enum failure_policy {
	FAIL_NAN, FAIL_SMALLEST_NMIN_R
};

template<typename data_t, failure_policy faipol>
struct smallest_res_t {
	smallest_res_t() = default;
	smallest_res_t(double r, const data_t& last)
	{};
	constexpr static double r = -1.0;
	data_t get_res() const {
		return data_t();
	}
};

template<typename data_t>
struct smallest_res_t<data_t,FAIL_SMALLEST_NMIN_R> {
	smallest_res_t() : r(-1.0) {};
	smallest_res_t(double r, const data_t& last) : r(r), dat(last) {};
	double r;
	data_t dat;
	data_t get_res() const {
		return dat;
	};
};



template<typename data_t, typename exit_condition_t, typename kernel_t,
         failure_policy failpol>
interpolated_t<typename data_t::result_t>
interpolate_point(const point_t& p,
                  const typename std::vector<data_t>& data,
                  const VantageTree<data_t>& tree,
                  const std::vector<double>& search_radii,
                  const exit_condition_t& exit_condition,
                  const kernel_t& kernel
                 )
{
	/* Determine all points within the longest radius: */
	std::vector<int> largest_neighbor_set(tree.within_range(p,search_radii[0],
	                                                        data));
	const size_t jmax = largest_neighbor_set.size();

	/* Compute the distances */
	struct ri_t {
		double r;
		size_t j;
		bool operator<(const ri_t& other) const {
			return r < other.r;
		};
	};
	std::vector<ri_t> distance_ordered(jmax);
	for (size_t j=0; j<jmax; ++j){
		int i = largest_neighbor_set[j];
		distance_ordered[j] = {.r=tree.distance(p, data[i].pt), .j=j};
	}

	/* Order by distance: */
	std::sort(distance_ordered.begin(), distance_ordered.end());

	typedef typename data_t::result_t result_t;
	/* */
	bool success = false;
	interpolated_t<result_t> res;
	smallest_res_t<result_t,failpol> last;
	for (double r : search_radii){
		/* Obtain all data records within range: */
		const auto end = std::upper_bound(distance_ordered.begin(),
		                                  distance_ordered.end(),
		                                  ri_t({.r=r, .j=0}));
		const size_t M = end - distance_ordered.begin();
		std::vector<int> neighbors(M, 0);
		auto itn = neighbors.begin();
		for (auto it = distance_ordered.begin(); it != end; ++it){
			*itn = largest_neighbor_set[it->j];
			++itn;
		}

		/* Check minimum size: */
		if (!exit_condition.size_ok(M))
			continue;

		/* Compute the weights: */
		std::vector<std::pair<double,data_t>> w_d(M);
		auto itw = w_d.begin();
		for (auto it = distance_ordered.begin(); it != end; ++it){
			int i = largest_neighbor_set[it->j];
			itw->first = kernel(it->r, r) * data[i].w;
			itw->second = data[i];
			++itw;
		}

		/* Compute the result: */
		res.res = data_t::compute_result(w_d);

		/* Save the last result (if needed): */
		if (failpol == FAIL_SMALLEST_NMIN_R)
			last = smallest_res_t<result_t,failpol>(r, res.res);

		/* Exit condition: */
		success = exit_condition.accept(res.res);
		if (success){
			res.r = r;
			break;
		}
	}

	/* If we did not succeed, return NaN: */
	if (!success){
		if (failpol == FAIL_NAN){
			res.res.set_nan();
			res.r = std::nan("");
		} else if (failpol == FAIL_SMALLEST_NMIN_R) {
			if (last.r > 0){
				res.r = last.r;
				res.res = last.get_res();
			} else {
				res.res.set_nan();
				res.r = std::nan("");
			}
		}
	}

	return res;
}

void sanity_check_radii(const std::vector<double>& search_radii);


template<failure_policy failpol, typename data_t, typename exit_condition_t,
         typename kernel_t>
std::vector<interpolated_t<typename data_t::result_t>>
interpolate(const std::vector<point_t>& pts,
            const typename std::vector<data_t>& data,
            const VantageTree<data_t>& tree,
            const std::vector<double>& search_radii,
            const exit_condition_t& exit_condition,
            const kernel_t& kernel
           )
{
	/* Assert that the radii are descending: */
	sanity_check_radii(search_radii);

	std::vector<interpolated_t<typename data_t::result_t>> res(pts.size());
	#pragma omp parallel for
	for (size_t i=0; i<pts.size(); ++i){
		try {
			res[i]
			   = interpolate_point<data_t, exit_condition_t,
			                       kernel_t, failpol>
			         (pts[i], data, tree, search_radii, exit_condition,
			          kernel);
		} catch (...) {
			res[i].res.set_nan();
			res[i].r = std::nan("");
		}
	}
	return res;
}




}
#endif