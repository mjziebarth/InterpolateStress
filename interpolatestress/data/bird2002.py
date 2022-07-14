# Plate boundaries from Bird (2002)
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
#
# Reference:
#
# Bird, P. (2003) An updated digital model of plate boundaries,
#        Geochemistry Geophysics Geosystems, 4(3), 1027,
#        doi:10.1029/2001GC000252.

import numpy as np

def load_plates_bird2002(filename):
    """
    Loads the plate boundaries from Bird (2002).

    Parameters:
        filename : Path to the 'PB2002_plates.dig' file.
                   (might contain additional '.txt' suffix)

    Bird, P. (2003) An updated digital model of plate boundaries,
        Geochemistry Geophysics Geosystems, 4(3), 1027,
        doi:10.1029/2001GC000252.
    """
    rings = []
    with open(filename,'r') as f:
        # states:
        #   0 : expect header
        #   1 : expect numeric
        state = 0
        ring = []
        for line in f:
            line = line.strip()
            if state == 0:
                state = 1
                continue
            elif state == 1:
                if line == "*** end of line segment ***":
                    rings.append(np.array(ring))
                    ring = []
                    state = 0
                    continue
                ring.append([float(x) for x in line.split(',')])
    return rings
