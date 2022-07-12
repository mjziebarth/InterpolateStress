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


template<typename data_t, typename exit_condition_t, typename data_weighting_t>
interpolated_t<typename data_t::result_t>
interpolate_point(const point_t& p,
                  const typename std::vector<data_t>& data,
                  const VantageTree<data_t>& tree,
                  const std::vector<double>& search_radii,
                  const exit_condition_t& exit_condition,
                  const data_weighting_t& data_weighting
                 )
{
	typedef typename data_t::result_t result_t;
	/* */
	bool success = false;
	interpolated_t<result_t> res;
	for (double r : search_radii){
		/* Obtain all data records within range: */
		std::vector<int> neighbors(tree.within_range(p,r,data));

		/* Check minimum size: */
		if (!exit_condition.size_ok(neighbors.size()))
			continue;

		/* Compute the weights: */
		std::vector<std::pair<double,data_t>> w_d(neighbors.size());
		auto it = w_d.begin();
		for (int i : neighbors){
			it->first = data_weighting(p, i, data);
			it->second = data[i];
			++it;
		}

		/* Compute the result: */
		res.res = data_t::compute_result(w_d);

		/* Exit condition: */
		success = exit_condition.accept(res.res);
		if (success){
			res.r = r;
			break;
		}
	}

	/* If we did not succeed, return NaN: */
	if (!success){
		res.res.set_nan();
		res.r = std::nan("");
	}

	return res;
}


template<typename data_t, typename exit_condition_t, typename data_weighting_t>
std::vector<interpolated_t<typename data_t::result_t>>
interpolate(const std::vector<point_t>& pts,
            const typename std::vector<data_t>& data,
            const VantageTree<data_t>& tree,
            const std::vector<double>& search_radii,
            const exit_condition_t& exit_condition,
            const data_weighting_t& data_weighting
           )
{
	std::vector<interpolated_t<typename data_t::result_t>> res(pts.size());
	#pragma omp parallel for
	for (size_t i=0; i<pts.size(); ++i){
		res[i] = interpolate_point(pts[i], data, tree, search_radii,
		                           exit_condition, data_weighting);
	}
	return res;
}




}
#endif