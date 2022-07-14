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

#include <data.hpp>

#ifndef INTERPOLATE_STRESS_KERNEL_HPP
#define INTERPOLATE_STRESS_KERNEL_HPP

namespace interpolatestress {

class UniformKernel {
public:
	UniformKernel() noexcept;

	constexpr double operator()(double d, double r) const
	{
		return 1.0;
	};
};

class LinearKernel {
public:
	LinearKernel() noexcept;

	double operator()(double d, double r) const;
private:
};

class GaussianKernel {
public:
	GaussianKernel(double bandwidth);

	double operator()(double d, double r) const;

private:
	double ibw2;

};


}

#endif