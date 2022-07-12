/*
 * Exit conditions for the radius search.
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
#include <cmath>
#include <vector>
#include <cstddef>
#include <data.hpp>
#include <constants.hpp>

#ifndef INTERPOLATE_STRESS_EXITCONDITION_HPP
#define INTERPOLATE_STRESS_EXITCONDITION_HPP

namespace interpolatestress {

/*
 * Exit condition that accepts a data set if the standard deviation of the
 * azimuths is below a threshold.
 */

template<typename data_t>
class ExitConditionAzimuthStd {
public:
	ExitConditionAzimuthStd(size_t nmin, double critical_std)
	   : nmin(nmin), critical_std(critical_std)
	{};

	bool size_ok(size_t n) const
	{
		return n >= nmin;
	};

	bool accept(typename data_t::result_t& res) const
	{
		return res.azi_std <= critical_std;
	};

private:
	const size_t nmin;
	const double critical_std;
};




}

#endif