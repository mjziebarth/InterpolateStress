# Load data from the World Stress Map 2016 release.
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
#
# Reference:
#
# Heidbach, O., Rajabi, M., Reiter, K., Ziegler, M., WSM Team (2016): World
#     Stress Map Database Release 2016. V. 1.1. GFZ Data Services.
#     https://doi.org/10.5880/WSM.2016.001.

import csv
import codecs
import numpy as np
from math import nan
from typing import Literal

def load_wsm_2016(
        filename,
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
        slopes3_column: str = "SLOPES1",
        pore_magin_column: str = "PORE_MAGIN",
        pore_slope_column: str = "PORE_SLOPE",
        ratio_column: str = "RATIO",
        regime_column: str = "REGIME"
    ):
    """

    """
    # Decode the CSV:
    table = []
    line_lengths = set()
    with codecs.open(filename, 'r', encoding='iso-8859-1') as f:
        reader = csv.reader(f)
        for line in reader:
            table.append(list(t.strip() for t in line))
            line_lengths.add(len(table[-1]))

    if len(line_lengths) != 1:
        raise ValueError("The number of columns is inconsistent. Found rows "
            "with the following number of columns: " + str(line_lengths)
        )

    # Header with all labels:
    header = [str(t) for t in table[0] if len(t) > 0]

    def regime_convert(r: Literal['NF','SS','TF','U']) -> float:
        if r == 'TF':
            return 0.0
        elif r == 'SS':
            return 1.0
        elif r == 'NF':
            return 2.0
        elif r == 'U':
            return nan

    # Conversion function for fields of the table:
    field_types \
      = {"LON"        : float,
         "LAT"        : float,
         "AZI"        : float,
         "DEPTH"      : float,
         "DATE"       : int,
         "NUMBER"     : int,
         "S1AZ"       : int,
         "S1PL"       : int,
         "S2AZ"       : int,
         "S2PL"       : int,
         "S3AZ"       : int,
         "S3PL"       : int,
         "MAG_INT_S1" : float,
         "SLOPES1"    : float,
         "MAG_INT_S2" : float,
         "SLOPES2"    : float,
         "MAG_INT_S3" : float,
         "SLOPES3"    : float,
         "PORE_MAGIN" : float,
         "PORE_SLOPE" : float,
         "RATIO"      : float,
         "REGIME"     : regime_convert
    }

    # Mapping potentially differently labeled columns to
    # standardized names:
    field2standard: dict[str,str]\
      = {
        lon_column        : "LON",
        lat_column        : "LAT",
        azi_column        : "AZI",
        depth_column      : "DEPTH",
        date_column       : "DATE",
        number_column     : "NUMBER",
        s1az_column       : "S1AZ",
        s1pl_column       : "S1PL",
        s2az_column       : "S2AZ",
        s2pl_column       : "S2PL",
        s3az_column       : "S3AZ",
        s3pl_column       : "S3PL",
        mag_int_s1_column : "MAG_INT_S1",
        slopes1_column    : "SLOPES1",
        mag_int_s2_column : "MAG_INT_S2",
        slopes2_column    : "SLOPES2",
        mag_int_s3_column : "MAG_INT_S3",
        slopes3_column    : "SLOPES3",
        pore_magin_column : "PORE_MAGIN",
        pore_slope_column : "PORE_SLOPE",
        ratio_column      : "RATIO",
        regime_column     : "REGIME"
    }

    # Ensure that now there is no non-uniqueness of the column names:
    standard_header = set(field2standard.values())
    for h in header:
        if h in standard_header and field2standard[h] != h:
            raise ValueError("If the standard header names are used, "
                "they shall refer to their standard meaning. Offending "
                "column header: " + str(h)
            )

    # Transform the header, potentially replacing CSV column names
    # with standard column names:
    header = [field2standard[h] if h in field2standard else h for h in header]

    def convert_field(field, col):
        if col in field_types:
            try:
                return field_types[col](field)
            except:
                return nan
        return field

    columns = {
        h : [convert_field(row[i],h) for row in table[1:]]
        for i,h in enumerate(header)
    }
    for h in header:
        if h in field_types:
            columns[h] = np.array(columns[h])

    return columns
