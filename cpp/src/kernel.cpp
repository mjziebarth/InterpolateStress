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
#include <cmath>
#include <stdexcept>

using interpolatestress::UniformKernel;
using interpolatestress::LinearKernel;
using interpolatestress::GaussianKernel;

UniformKernel::UniformKernel() noexcept {
}

LinearKernel::LinearKernel() noexcept {
}

double LinearKernel::operator()(double dist, double r) const
{
	return std::max(1.0 - dist / r, 0.0);
}


GaussianKernel::GaussianKernel(double relative_bandwidth)
   : irbw2(1.0/(relative_bandwidth*relative_bandwidth))
{
	if (relative_bandwidth <= 0){
		throw std::runtime_error("Negative or zero bandwidth not allowed for "
		                         "GaussianKernel.");
	}
}

double GaussianKernel::operator()(double dist, double r) const
{
	/* Use relative distance: */
	double rd = dist / r;
	return std::exp(-0.5 * rd * rd * irbw2);
}
