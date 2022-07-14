# Load polygons from a GeoJSON file.
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

import json
import numpy as np

def load_geojson_polygons(filename):
    """
    Loads polygons contained in a GeoJSON. De-multiplexes
    all Polygons and Multipolygons into a list of Polygon
    coordinates.

    Parameters:
       filename : Path to the GeoJSON file

    Returns:
       polygons : A list of shape-(Ni,2) numpy arrays,
                  each representing a polygon in geographic
                  coordinates.
    """
    with open(filename,'r') as f:
        geojson = json.load(f)

    polygons = []
    for feat in geojson["features"]:
        coords = feat["geometry"]["coordinates"]
        if feat["geometry"]["type"] == "MultiPolygon":
            polygons += (np.array(crds[0]) for crds in coords)
        elif feat["geometry"]["type"] == "Polygon":
            raise NotImplementedError()

    return polygons
