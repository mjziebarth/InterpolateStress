/*
 * Class to perform data weighting, merging the user-supplied data weights
 * (e.g. as supplied by the World Stress Map) and the spatial kernel weights.
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

#include <vector>
#include <cstddef>
#include <data.hpp>

#ifndef INTERPOLATE_STRESS_WEIGHTING_HPP
#define INTERPOLATE_STRESS_WEIGHTING_HPP

namespace interpolatestress {


template<typename data_t, typename kernel_t>
class DataWeighting {
public:
	DataWeighting(kernel_t&& kernel) : kernel(std::move(kernel))
	{};

	double operator()(const point_t center, size_t i,
	                  const std::vector<data_t>& data) const
	{
		return kernel(center, data[i].pt) * data[i].w;
	};

private:
	const kernel_t kernel;
};




}

#endif