# Setup script.
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

from setuptools import setup, Extension
from mebuex import MesonExtension, build_ext

backend = MesonExtension('interpolatestress.backend')


setup(name='interpolatestress',
      version='0.1.0',
      author='Malte J. Ziebarth',
      description='Interpolate principal stress directions based on the method '
                  '`Stress2Grid` by Ziegler & Heidbach (2017)',
      packages = ['interpolatestress'],
      ext_modules=[backend],
      cmdclass={'build_ext' : build_ext}
      )
