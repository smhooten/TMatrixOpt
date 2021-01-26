"""
TMatrixOpt/physical_constants.py

Simple physical constant definitions, also
define the solver precision here.
"""

import numpy as np
from math import pi

q = 1.60217662e-19;
h = 6.626070040e-34;
c = 299792458.0;
Kb = 1.38064852e-23;

##########################################
# Define precision of complex calculations
# in the TMatrix solver here
##########################################
PRECISION = np.complex128
##########################################
