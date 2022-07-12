/*
 * Vantage tree wrapper of the GeographicLib NearestNeighbor class.
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

#include <functional>
#include <memory>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <GeographicLib/NearestNeighbor.hpp>
#include <GeographicLib/Geodesic.hpp>
#include <data.hpp>

#ifndef INTERPOLATE_STRESS_VANTAGETREE_HPP
#define INTERPOLATE_STRESS_VANTAGETREE_HPP

namespace interpolatestress {

template<typename data_t>
class VantageTree {
public:
	VantageTree(const std::vector<data_t>& pts, double a,
	            double f);

	std::vector<int> within_range(const point_t& p, double r,
	                              const std::vector<data_t>& data) const;

private:
	GeographicLib::Geodesic geod;
	typedef std::function<double(const data_t&, const data_t&)> distfun_t;
	GeographicLib::NearestNeighbor<double,data_t,distfun_t> tree;
};



/*
 * Template implementation.
 */

template<typename data_t>
VantageTree<data_t>::VantageTree(const std::vector<data_t>& pts,
                         double a, double f)
   : geod(a,f)
{
	if (pts.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
		throw std::runtime_error("Too many data points for VantageTree. "
		                         "Can only use `int` as indexing data type "
		                         "in GeographicLib. The given number of points "
		                         "would lead to overflow.");

	auto distfun = [&](const data_t& p0, const data_t& p1) -> double {
		double distance = 0;
		geod.Inverse(p0.pt.lat, p0.pt.lon, p1.pt.lat, p1.pt.lon, distance);
		return distance;
	};
	tree.Initialize(pts, distfun);
}

template<typename data_t>
std::vector<int>
VantageTree<data_t>::within_range(const point_t& p, double r,
                                  const std::vector<data_t>& data) const
{
	/*
	 * This function finds all nearest neighbors within range r
	 * of point p.
	 */
	auto distfun = [&](const data_t& p0, const data_t& p1) -> double {
		double distance = 0;
		geod.Inverse(p0.pt.lat, p0.pt.lon, p1.pt.lat, p1.pt.lon, distance);
		return distance;
	};

	/* Find indices as integers: */
	data_t p_dat;
	p_dat.pt = p;
	std::vector<int> ids;
	tree.Search(data, distfun, p_dat, ids, data.size(), r);

	/* Sanity check: */
	if (std::any_of(ids.begin(), ids.end(), [](int i)->bool {return i < 0;}))
		throw std::runtime_error("Returned negative int in tree.Search.");

	return ids;
}


}
#endif