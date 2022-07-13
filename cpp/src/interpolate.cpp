/*
 * Sanity checking for the interpolation routines.
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

#include <../include/interpolate.hpp>
#include <stdexcept>


void interpolatestress::sanity_check_radii(
         const std::vector<double>& search_radii
)
{
	if (search_radii.size() < 1)
		throw std::runtime_error("Need at least one search radius.");
	double rlast = search_radii[0];
	for (auto it = search_radii.begin()+1; it != search_radii.end(); ++it){
		if (*it >= rlast)
			throw std::runtime_error("Search radii need to be descending.");
		rlast = *it;
	}
}
