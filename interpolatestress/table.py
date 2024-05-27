# A stress record table containing all relevant (for this package)
# information from the World Stress Map.
#
# Author: Malte J. Ziebarth (mjz.science@fmvkb.de)
#
# Copyright (C) 2022 Malte J. Ziebarth,
#               2024 Technical University of Munich
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
    def from_wsm2016(
            filename,
            quality_weighting=heidbach_et_al_2010_weighting,
            lon_column: str = "LON",
            lat_column: str = "LAT",
            azi_column: str = "AZI",
            depth_column: str = "DEPTH",
            date_column: str = "DATE",
            number_column: str = "NUMBER",
            s1az_column: str = "S1AZ",
            s1pl_column: str = "S1PL",
            s2az_column: str = "S2AZ",
            s2pl_column: str = "S2PL",
            s3az_column: str = "S3AZ",
            s3pl_column: str = "S3PL",
            mag_int_s1_column: str = "MAG_INT_S1",
            slopes1_column: str = "SLOPES1",
            mag_int_s2_column: str = "MAG_INT_S2",
            slopes2_column: str = "SLOPES2",
            mag_int_s3_column: str = "MAG_INT_S3",
            slopes3_column: str = "SLOPES3",
            pore_magin_column: str = "PORE_MAGIN",
            pore_slope_column: str = "PORE_SLOPE",
            ratio_column: str = "RATIO"
        ):
        """
        Load a World Stress Map 2016 CSV table.

        Parameters:
        -----------
        filename : str
            Path to the wsm2016.csv
        quality_weighting : typing.Callable[[list[]]]
            A function that translates WSM quality ranking A to E
            into data weights. Must be a callable that takes a list
            of quality strings (consisting of 'A' to 'E') and
            returns a NumPy array of weights.
            Default: a function performing the weighting defined
            by Heidbach et al. (2010).
        lon_column : str, optional
            Name of the longitude column if it differs from "LON".
        lat_column : str, optional
            Name of the latitude column if it differs from "LAT".
        azi_column : str, optional
            Name of the azimuth column if it differs from "AZI".
        depth_column : str, optional
            Name of the azimuth column if it differs from "DEPTH".
        date_column : str, optional
            Name of the azimuth column if it differs from "DATE".
        number_column : str, optional
            Name of the azimuth column if it differs from "NUMBER".
        s1az_column : str, optional
            Name of the azimuth column if it differs from "S1AZ".
        s1pl_column : str, optional
            Name of the azimuth column if it differs from "S1PL".
        s2az_column : str, optional
            Name of the azimuth column if it differs from "S2AZ".
        s2pl_column : str, optional
            Name of the azimuth column if it differs from "S2PL".
        s3az_column : str, optional
            Name of the azimuth column if it differs from "S3AZ".
        s3pl_column : str, optional
            Name of the azimuth column if it differs from "S3PL".
        mag_int_s1_column : str, optional
            Name of the azimuth column if it differs from
            "MAG_INT_S1".
        slopes1_column : str, optional
            Name of the azimuth column if it differs from
            "SLOPES1".
        mag_int_s2_column : str, optional
            Name of the azimuth column if it differs from
            "MAG_INT_S2".
        slopes2_column : str, optional
            Name of the azimuth column if it differs from
            "SLOPES2".
        mag_int_s3_column : str, optional
            Name of the azimuth column if it differs from
            "MAG_INT_S3".
        slopes3_column : str, optional
            Name of the azimuth column if it differs from
            "SLOPES3".
        pore_magin_column : str, optional
            Name of the azimuth column if it differs from
            "PORE_MAGIN".
        pore_slope_column : str, optional
            Name of the azimuth column if it differs from
            "PORE_SLOPE".
        ratio_column : str, optional
            Name of the azimuth column if it differs from "RATIO".
        """
        table = load_wsm_2016(filename,
            lon_column = lon_column,
            lat_column = lat_column,
            azi_column = azi_column,
            depth_column = depth_column,
            date_column = date_column,
            number_column = number_column,
            s1az_column = s1az_column,
            s1pl_column = s1pl_column,
            s2az_column = s2az_column,
            s2pl_column = s2pl_column,
            s3az_column = s3az_column,
            s3pl_column = s3pl_column,
            mag_int_s1_column = mag_int_s1_column,
            slopes1_column = slopes1_column,
            mag_int_s2_column = mag_int_s2_column,
            slopes2_column = slopes2_column,
            mag_int_s3_column = mag_int_s3_column,
            slopes3_column = slopes3_column,
            pore_magin_column = pore_magin_column,
            pore_slope_column = pore_slope_column,
            ratio_column = ratio_column
        )

        return StressTable(table["LON"], table["LAT"], table["AZI"],
                           table["S1PL"], table["S2PL"], table["REGIME"],
                           quality_weighting(table["QUALITY"]), table["RATIO"])
