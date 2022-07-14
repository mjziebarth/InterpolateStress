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

#include <cmath>
#include <../include/constants.hpp>
#include <../include/data.hpp>

using interpolatestress::PI;
using interpolatestress::data_azi_t;
using interpolatestress::data_azi_2plunge_t;
using interpolatestress::data_scalar_t;




void data_azi_t::result_t::set_nan() {
	azi = std::nan("");
	azi_std = std::nan("");
}

void data_azi_2plunge_t::result_t::set_nan() {
	azi = std::nan("");
	azi_std = std::nan("");
	pl1 = std::nan("");
	pl1_std = std::nan("");
	pl2 = std::nan("");
	pl2_std = std::nan("");
}

void data_scalar_t::result_t::set_nan() {
	z = nan;
	z_std = nan;
};

/*
 * Query routines:
 */
bool data_azi_t::any_nan() const {
	return std::isnan(azi);
}

bool data_azi_2plunge_t::any_nan() const {
	return std::isnan(azi) || std::isnan(plunge1) || std::isnan(plunge2);
}

bool data_scalar_t::any_nan() const {
	return std::isnan(z);
}

/*
 * Static common routines:
 */

static inline double compute_mean(const double S, const double C)
{
	const double mean = std::fmod(90.0/PI*std::atan2(S,C), 180.0);
	if (mean < 0)
		return mean + 180.0;
	return mean;
}

static inline double compute_std(const double S, const double C)
{
	return 90.0/PI * std::sqrt(-std::log(C*C + S*S));
}

data_azi_t::result_t
data_azi_t::compute_result(const std::vector<std::pair<double, data_azi_t>>&
                           data)
{
	/* Compute statistics of the data according to section
	 * "2.1 Statistics of bipolar data" of Ziegler & Heidbach (2017)
	 */
	double C = 0.0;
	double S = 0.0;
	double Z = 0.0;
	for (const auto& d : data){
		Z += d.first;
		const double azi = PI/180.0 * d.second.azi;
		C += d.first * std::cos(2*azi);
		S += d.first * std::sin(2*azi);
	}
	C /= Z;
	S /= Z;
	const double SHmean = compute_mean(S,C);
	const double s0 = compute_std(S,C);
	return {.azi = SHmean, .azi_std = s0};
}


data_azi_2plunge_t::result_t
data_azi_2plunge_t::compute_result(
    const std::vector<std::pair<double, data_azi_2plunge_t>>& data
)
{
	/* Compute statistics of the data according to section
	 * "2.1 Statistics of bipolar data" of Ziegler & Heidbach (2017)
	 */
	double Ca = 0.0, Sa = 0.0, Za = 0.0;
	double Cp1 = 0.0, Sp1 = 0.0, Zp1 = 0.0;
	double Cp2 = 0.0, Sp2 = 0.0, Zp2 = 0.0;
	size_t np1 = 0;
	size_t np2 = 0;
	for (const auto& d : data){
		Za += d.first;
		const double azi = PI/180.0 * d.second.azi;
		Ca += d.first * std::cos(2*azi);
		Sa += d.first * std::sin(2*azi);
		/* Use the same statistics also for plunges: */
		if (!std::isnan(d.second.plunge1)){
			const double pl1 = PI/180.0 * d.second.plunge1;
			++np1;
			Cp1 += d.first * std::cos(2*pl1);
			Sp1 += d.first * std::sin(2*pl1);
			Zp1 += d.first;
		}
		if (!std::isnan(d.second.plunge2)){
			const double pl2 = PI/180.0 * d.second.plunge2;
			++np2;
			Cp2 += d.first * std::cos(2*pl2);
			Sp2 += d.first * std::sin(2*pl2);
			Zp2 += d.first;
		}
	}
	Ca /= Za;
	Sa /= Za;
	result_t res;
	res.azi = compute_mean(Sa,Ca);
	res.azi_std = compute_std(Sa,Ca);
	if (np1 > 0){
		Sp1 /= Zp1;
		Cp1 /= Zp1;
		res.pl1 = compute_mean(Sp1,Cp1);
		res.pl1_std = compute_std(Sp1,Cp1);
	} else {
		res.pl1 = std::nan("");
		res.pl1_std = std::nan("");
	}
	if (np2 > 0){
		Sp2 /= Zp2;
		Cp2 /= Zp2;
		res.pl2 = compute_mean(Sp2,Cp2);
		res.pl2_std = compute_std(Sp2,Cp2);
	} else {
		res.pl2 = std::nan("");
		res.pl2_std = std::nan("");
	}
	return res;
}

data_scalar_t::result_t
data_scalar_t::weighted_average(
    const std::vector<std::pair<double,data_scalar_t>>& data
)
{
	if (data.size() == 0)
		return {.z=nan, .z_std=nan};
	double W = 0.0;
	double W2 = 0.0;
	double val = 0.0;
	for (auto d : data){
		if (std::isnan(d.second.z))
			continue;
		W += d.first;
		W2 += d.first * d.first;
		val += d.first * d.second.z;
	}
	if (W == 0 || std::isinf(W)){
		return {.z=nan, .z_std=nan};
	}
	val /= W;
	double var = 0.0;
	for (auto d : data){
		if (std::isnan(d.second.z))
			continue;
		const double dz = d.second.z - val;
		var += d.first * dz * dz;
	}
	var *= W / (W*W - W2);
	return {.z=val, .z_std=std::sqrt(var)};
}
