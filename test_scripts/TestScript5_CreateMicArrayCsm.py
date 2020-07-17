"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


TestScript5_CreateMicArrayCsm.py

Test script 5: calculates and saves the microphone array CSM for a simulated 
    flat-plate Aerofoil-turbulence Interaction Noise (ATIN) case. The aerofoil
    interacts with isotropic turbulence and radiates to the microphone array;
    the radiation includes the refraction effects at the planar shear layer.


***Note on memory usage:
    This computation takes a long time and uses a lot of memory, and often
    crashes when Python runs out of memory at every few frequencies.
    After a crash, simply restart the computer and run the code again. The code
    can be run multiple times, as it automatically loads the HDF5 file and
    verifies the last frequency successfully computed.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import gc

import numpy as np

from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.lines as mlines
import h5py

import amiet_tools as AmT

import MicArrayCsmHDF5 as CsmEssH5


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
print('Calculating shear layer correction - this should take a few minutes...')

T_shearLayer, XYZ_shearLayer = AmT.ShearLayer_matrix(XYZ_airfoil_calc,
                                                     XYZ_array, z_sl, Ux, c0)

print('Shear layer correction calculation done!')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define frequency variables

fs = 48e3
Ndft = 2**10

df = fs/Ndft
freq = np.linspace(0, fs-df, Ndft)[:Ndft//2+1]

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# ---->>> RUN ONLY WHEN CONTINUING THE CSM CALCULATIONS AFTER A CRASH <<<----

# create object to store CSM Essential info for HDF5 file, read data from
# existing file
CsmEss_DARP2016 = CsmEssH5.MicArrayCsmEss()
CsmEss_DARP2016.caseID = 'DARP2016_FlatPlate_Analytical'

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# ---->>> RUN ONLY WHEN CREATING THE HDF5 FILE FOR THE FIRST TIME <<<----

# Measurement data from experimental session (check lab notes)
temperatureDegC = 13.5
relativeHumidityPct = 43.

# std atmospheric pressure
atmPressurePa = 101325

# domain bounds, in metres [min, max]
xBounds = np.array([-0.25, 0.25])
yBounds = np.array([-0.25, 0.25])
zBounds = np.array([-0.075, 0.075])
domainBounds = np.concatenate((xBounds[:, np.newaxis],
                                yBounds[:, np.newaxis],
                                zBounds[:, np.newaxis]), axis=1)

# create object to store CSM Essential info for HDF5 file, populate with
# initial data 
CsmEss_DARP2016 = CsmEssH5.MicArrayCsmEss()
CsmEss_DARP2016.caseID = 'DARP2016_FlatPlate_Analytical'

CsmEss_DARP2016.binCenterFrequenciesHz = (freq+df/2).reshape((1, freq.shape[0]))
CsmEss_DARP2016.frequencyBinCount = freq.shape[0]

CsmEss_DARP2016.CsmUnits = 'Pa^2/Hz'
CsmEss_DARP2016.fftSign = -1
CsmEss_DARP2016.spectrumType = 'psd'

CsmEss_DARP2016.machNumber = np.array([Mach, 0, 0], dtype='f8')
CsmEss_DARP2016.relativeHumidityPct = relativeHumidityPct
CsmEss_DARP2016.speedOfSoundMPerS = CsmEssH5.speed_of_sound(temperatureDegC)
CsmEss_DARP2016.staticPressurePa = atmPressurePa
CsmEss_DARP2016.staticTemperatureK = temperatureDegC + 273.15

CsmEss_DARP2016.revisionNumberMajor = np.array([2], dtype='i4')
CsmEss_DARP2016.revisionNumberMinor = np.array([4], dtype='i4')

CsmEss_DARP2016.microphonePositionsM = XYZ_array.T
CsmEss_DARP2016.microphoneCount = M

CsmEss_DARP2016.coordinateReference = 'Aerofoil center'

CsmEss_DARP2016.domainBoundsM = domainBounds

CsmEss_DARP2016.flowType = 'Uniform flow in +x dir inside open jet (rectangular nozzle 0.15 x 0.45 m [height x width]), no flow outside'
CsmEss_DARP2016.testDescription = 'Analytical model of flat plate-turbulence interaction noise, rectangular open jet, planar mic array parallel to aerofoil'

# write data to HDF5 file
CsmEss_DARP2016.writeToHDF5File()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Create and store CSM data per frequency, automatically closes H5 file if
# code crashes or is interrupted

# Open HDF5 file and create handles for CsmReal, CsmImaginary datasets
with h5py.File(CsmEss_DARP2016.caseID +'CsmEss.h5', 'a') as CsmEssH5File:
    
    CsmReal = CsmEssH5File['CsmData/CsmReal']
    CsmImaginary = CsmEssH5File['CsmData/CsmImaginary']
    
    # set f=0 data to zeros
    CsmReal[:, :, 0] = np.zeros((M, M))
    CsmImaginary[:, :, 0] = np.zeros((M, M))
    
    # iterate over freqs, skip f=0 Hz
    
    # identify last successfully calculated index
    nonZeroCsm = np.nonzero(CsmReal[0, 0, 1:])[0]
    if nonZeroCsm.size == 0:
        # Csm hasn't been calculated yet; start from scratch
        i_last_success = 0
    else:
        # Csm has been partially calculated; start from last attempted freq
        i_last_success = nonZeroCsm[-1] + 1
    
    for i, f in enumerate(freq[1+i_last_success:]):
        print('Calculating CSM at f = {:.1f} Hz...'.format(f))
        
        # account for skipping zero and previous runs
        i += 1+i_last_success
        
        # frequency-related variables
        FreqVars = AmT.FrequencyVars(f, DARP2016Setup)
        (k0, Kx, Ky_crit) = FreqVars.export_values()
        
        # vector of spanwise hydrodynamic gust wavenumbers
        Ky_vec = AmT.ky_vector(b, d, k0, Mach, beta)
        
        # gust energy spectrum (von Karman)
        Phi = AmT.Phi_2D(Kx, Ky_vec, Ux, turb_intensity, length_scale)[0]
        
        # convected dipole transfer function - includes shear layer refraction
        Gdip = AmT.dipole_shear(XYZ_airfoil_calc, XYZ_array, XYZ_shearLayer,
                                T_shearLayer, k0, c0, Mach)
        
        # CSM of mic array pressures
        MicArrayCsm = AmT.calc_radiated_Spp(DARP2016Setup, DARP2016Airfoil,
                                            FreqVars, Ky_vec, Phi, Gdip)
        
        # write real/imag components to HDF5 file, one freq/chunk at a time
        CsmReal[:, :, i] = MicArrayCsm.real
        # force diagonal of imaginary component to zero
        CsmImaginary[:, :, i] = MicArrayCsm.imag - np.diag(np.diag(MicArrayCsm.imag))
        
        # use garbage collector to recover some memory
        gc.collect()

# %%

# # Manually open H5 file
# CsmEssH5File = h5py.File(CsmEss_DARP2016.caseID +'CsmEss.h5', 'a')


# # Manually close H5 file
# CsmEssH5File.close()
