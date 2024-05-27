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

def load_wsm_2016(filename):
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

    # Conversion function for fields of the table:
    field_types \
      = {"LON"      : float,
         "LAT"      : float,
         "AZI"      : float,
         "DEPTH"    : float,
         "DATE"     : int,
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
         "RATIO"      : float
    }

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
