# A stress record table containing all relevant (for this package)
# information from the World Stress Map.
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

import numpy as np
from .data import load_wsm_2016

#
#  Quality weighting:
#
def heidbach_et_al_2010_weighting(quality):
    """
    This function translates a list of World Stress Map quality
    indicators A-E into an array of weights.
    """
    w_Q = {
        'A' : 1 / 15.,
        'B' : 1 / 20.,
        'C' : 1 / 25.0,
        'D' : 0.0,
        'E' : 0.0
    }
    return np.array([w_Q[q] for q in quality])


#
# The stress table class:
#

class StressTable:
    """
    A table containing relevant stress record information.
    """
    def __init__(self, lon, lat, azimuth, plunge1, plunge2, regime, weight,
                 stress_ratio):
        self.lon = lon
        self.lat = lat
        self.azimuth = azimuth
        self.plunge1 = plunge1
        self.plunge2 = plunge2
        self.regime = regime
        self.weight = weight
        self.stress_ratio = stress_ratio

    @staticmethod
    def from_wsm2016(filename, quality_weighting=heidbach_et_al_2010_weighting):
        """
        Load a World Stress Map 2016 CSV table.
        
        Arguments:
           filename : Path to the wsm2016.csv
        
        Keyword arguments:
           quality_weighting : A function that translates WSM quality
                               ranking A to E into data weights.
                               Must be a callable that takes a list of
                               quality strings (consisting of 'A' to 'E')
                               and returns a NumPy array of weights.
                               Default: a function performing the weighting
                                        defined by Heidbach et al. (2010)
        """
        table = load_wsm_2016(filename)

        return StressTable(table["LON"], table["LAT"], table["AZI"],
                           table["S1PL"], table["S2PL"], table["REGIME"],
                           quality_weighting(table["QUALITY"]), table["RATIO"])
