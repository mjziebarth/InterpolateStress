/*
 * Data transfer structures: Coordinates and interpolation results.
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

#include <array>
#include <memory>
#include <vector>
#include <utility>

#ifndef INTERPOLATE_STRESS_DATA_HPP
#define INTERPOLATE_STRESS_DATA_HPP

namespace interpolatestress {

struct point_t {
	double lon;
	double lat;
};

struct data_azi_t {
	/* Data members: */
	point_t pt;
	double w;
	double azi;

	/* Return type of the stress interpolation: */
	struct result_t {
		double azi;
		double azi_std;

		void set_nan();
	};

	static result_t
	compute_result(const std::vector<std::pair<double, data_azi_t>>& data);
};

struct data_azi_2plunge_t {
	point_t pt;
	double w;
	double azi;
	double plunge1;
	double plunge2;

	/* Return type of the stress interpolation: */
	struct result_t {
		double azi;
		double azi_std;
		double pl1;
		double pl1_std;
		double pl2;
		double pl2_std;

		void set_nan();
	};

	static result_t
	compute_result(const std::vector<std::pair<double,data_azi_2plunge_t>>&
	                   data);
};


}

#endif