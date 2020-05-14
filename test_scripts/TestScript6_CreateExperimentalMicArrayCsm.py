"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


TestScript6_CreateExperimentalMicArrayCsm.py

Test script 6: calculates and saves the microphone array CSM for a measured
    flat-plate Aerofoil-turbulence Interaction Noise (ATIN) case.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np

from scipy.io import wavfile
import scipy.signal as signal
from matplotlib import pyplot as plt

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
# create DARP array

# DARP2016 spiral microphone array coordinates
XYZ_array, array_cal = AmT.DARP2016_MicArray()

# Number of mics
M = XYZ_array.shape[1]

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define frequency variables

fs = 48e3
Ndft = 2**10

N_overlap = Ndft//2

df = fs/Ndft
freq = np.linspace(0, fs-df, Ndft)[:Ndft//2+1]

window = signal.hann(Ndft)

# recording length [s]
rec_length = 30

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# create object to store CSM Essential info for HDF5 file, populate with
# initial data
CsmEss_ExpDARP2016 = CsmEssH5.MicArrayCsmEss()
CsmEss_ExpDARP2016.caseID = 'DARP2016_FlatPlate_Experimental'


# Measurement data from experimental session (from lab notes)
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

CsmEss_ExpDARP2016.binCenterFrequenciesHz = (freq+df/2).reshape((1, freq.shape[0]))
CsmEss_ExpDARP2016.frequencyBinCount = freq.shape[0]

CsmEss_ExpDARP2016.CsmUnits = 'Pa^2/Hz'
CsmEss_ExpDARP2016.fftSign = -1
CsmEss_ExpDARP2016.spectrumType = 'psd'

CsmEss_ExpDARP2016.machNumber = np.array([Mach, 0, 0], dtype='f8')
CsmEss_ExpDARP2016.relativeHumidityPct = relativeHumidityPct
CsmEss_ExpDARP2016.speedOfSoundMPerS = CsmEssH5.speed_of_sound(temperatureDegC)
CsmEss_ExpDARP2016.staticPressurePa = atmPressurePa
CsmEss_ExpDARP2016.staticTemperatureK = temperatureDegC + 273.15

CsmEss_ExpDARP2016.revisionNumberMajor = np.array([2], dtype='i4')
CsmEss_ExpDARP2016.revisionNumberMinor = np.array([4], dtype='i4')

CsmEss_ExpDARP2016.microphonePositionsM = XYZ_array.T
CsmEss_ExpDARP2016.microphoneCount = M

CsmEss_ExpDARP2016.coordinateReference = 'Aerofoil center'

CsmEss_ExpDARP2016.domainBoundsM = domainBounds

CsmEss_ExpDARP2016.flowType = 'Uniform flow in +x dir inside open jet (rectangular nozzle 0.15 x 0.45 m [height x width]), no flow outside'
CsmEss_ExpDARP2016.testDescription = 'Experimental measurement of flat plate-turbulence interaction noise, rectangular open jet, planar mic array parallel to aerofoil'

# write data to HDF5 file
CsmEss_ExpDARP2016.writeToHDF5File()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# create and store CSM data per frequency

# relative path to .WAV files
path_to_wavs = '../../../DARP Experiments 2016/Case7 - FlatPlate/Case7_Recorded/'

# read wav files for all mics and recordings
N_samples = int(rec_length*fs)           # Recording length (in samples)
signals = np.zeros((M, N_samples))

for m in range(M):
    # read wav file (length might not be exactly 30 s)
    _, wav = wavfile.read(path_to_wavs + 'Track {}_004.wav'.format(m+1))

    # apply calibration and store wav samples on 'signals' variable
    signals[m, :wav.shape[0]] = wav*array_cal[m]


# Calculate the mic array CSM and store in dict
CSM = CsmEssH5.CSM(signals, Ndft, fs, N_overlap, window)

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Open HDF5 file, automatically closes at end of loop
with h5py.File(CsmEss_ExpDARP2016.caseID +'CsmEss.h5', 'a') as CsmEssH5:

    # Create handles for CsmReal, CsmImaginary datasets
    CsmReal = CsmEssH5['CsmData/CsmReal']
    CsmImaginary = CsmEssH5['CsmData/CsmImaginary']

    # set f=0 data to zeros
    CsmReal[:, :, 0] = np.zeros((M, M))
    CsmImaginary[:, :, 0] = np.zeros((M, M))

    # loop over freqs, write each as CSM chunk
    for i in range(1, Ndft//2+1):
        # write real/imag components to HDF5 file, one freq/chunk at a time
        CsmReal[:, :, i] = CSM[:, :, i].real
        CsmImaginary[:, :, i] = CSM[:, :, i].imag


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Test reading experimental CSM file: plot coherence-squared function across mics

with h5py.File(CsmEss_ExpDARP2016.caseID +'CsmEss.h5', 'a') as CsmEssH5:

    # Create handles for CsmReal, CsmImaginary datasets
    CsmReal = CsmEssH5['CsmData/CsmReal']
    CsmImaginary = CsmEssH5['CsmData/CsmImaginary']

    CsmComplex = CsmReal[:] + 1j*CsmImaginary[:]

    line_styles=['-', '--', '-.', ':', '-']

    plt.figure()

    for m in range(31, M):
        cohere_m = (np.abs(CsmComplex[0, m, :])**2
                    / np.real(CsmComplex[0, 0, :]*CsmComplex[m, m, :]))
        plt.semilogx(freq, cohere_m, label='Mic {}'.format(m+1), linestyle=line_styles[m-31])

    plt.legend()
    plt.xlim(100, 20e3)
    plt.ylim(-0.1, 1.1)
