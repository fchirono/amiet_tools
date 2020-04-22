"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


TestScript5_CreateMicArrayCsm.py

Test script 5: calculates and saves the microphone array CSM for a simulated 
    flat-plate Aerofoil-turbulence Interaction Noise (ATN) case. The aerofoil
    interacts with isotropic turbulence and radiates to the microphone array;
    the radiation contains the effects of a planar shear layer.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

import amiet_tools as AmT

plt.rc('text', usetex=True)
plt.close('all')

save_fig = False


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# load test setup from file
DARP2016Setup = AmT.loadTestSetup('../DARP2016_TestSetup.txt')

# export variables to current namespace
(c0, rho0, p_ref, Ux, turb_intensity, length_scale, z_sl, Mach, beta,
 flow_param, dipole_axis) = DARP2016Setup.export_values()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define airfoil points over the whole chord

# load airfoil geometry from file
DARP2016Airfoil = AmT.loadAirfoilGeom('../DARP2016_AirfoilGeom.txt')
(b, d, Nx, Ny, XYZ_airfoil, dx, dy) = DARP2016Airfoil.export_values()
XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# create DARP array

# DARP2016 spiral microphone array coordinates
XYZ_array, array_cal = AmT.DARP2016_MicArray()

# Number of mics
M = XYZ_array.shape[1]


# obtain propag time and shear layer crossing point for every source-mic pair
# (forward problem - frequency independent!)
T_sl_fwd, XYZ_sl_fwd = AmT.ShearLayer_matrix(XYZ_airfoil_calc, XYZ_array, z_sl,
                                             Ux, c0)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#

# loop over freqs
# calc surf pressure CSM
# propag to mic array CSM
# save to HDF5
