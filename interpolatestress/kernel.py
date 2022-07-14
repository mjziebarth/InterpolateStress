# Kernels.
#
# Author: Malte J. Ziebarth (mjz.science@fmvkb.de)
#
# Copyright (C) 2022 Malte J. Ziebarth
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.


class UniformKernel:
    """
    A uniform kernel.
    """
    pass

class LinearKernel:
    """
    A kernel decreasing linearly along radius.
    """
    pass


class GaussianKernel:
    """
    A Gaussian kernel.
    """
    def __init__(self, relative_bandwidth = 3.0):
        self._relative_bandwidth = float(relative_bandwidth)
