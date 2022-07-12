/*
 * Spatial kernels.
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

#include <../include/kernel.hpp>
#include <stdexcept>

using interpolatestress::UniformKernel;
using interpolatestress::GaussianKernel;

UniformKernel::UniformKernel() noexcept {
}


GaussianKernel::GaussianKernel(double bandwidth, double a, double f)
   : ibw2(1.0/(bandwidth*bandwidth)), geod(a,f)
{
	if (bandwidth <= 0){
		throw std::runtime_error("Negative or zero bandwidth not allowed for "
		                         "GaussianKernel.");
	}
}

double GaussianKernel::operator()(const point_t& p0, const point_t& p1) const
{
	double dist = 0.0;
	geod.Inverse(p0.lat, p0.lon, p1.lat, p1.lon, dist);
	return std::exp(-0.5 * dist * dist * ibw2);
}
