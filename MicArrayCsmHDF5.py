"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


MicArrayCsmHDF5.py

Class and functions to store microphone array CSM, read from and save to HDF5
file, as per definitions in:

    Array Methods HDF5 File Definitions
    Revision 2.4 Release
    Sep 2016
    https://www.b-tu.de/fg-akustik/lehre/aktuelles/arraybenchmark


Author:
Fabio Casagrande Hirono
fchirono@gmail.com


*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    --->>> CSM Essential file structure - Revision 2.4 <<<---
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

File name: <caseID>CsmEss.h5

File structure:
    -MetaData
        -revisionNumberMajor
            Attribute, int

        -revisionNumberMinor
            Attribute, int

        -dataLayout
            Dataset, int
            Shape: (2, 3, 4) - (rows, columns, pages)
            Example data structure for validating data read orientation
            Contains integers 1-24 in increasing, column-major order:
                page 1: [1 3 5,
                         2 4 6]
                page 2: [7  9 11,
                         8 10 12]
                etc.

        -ArrayAttributes
            -microphonePositionsM
                Dataset, float
                Shape: (microphoneCount, 3)
                Each row contains a single mic (x, y, z) coordinate; in meters

            -microphoneCount
                Attribute, int
                Number of microphones in array


        -TestAttributes
            -coordinateReference
                Attribute, string
                String describing coordinate reference frame and/or origin

            -domainBoundsM
                Dataset, float
                Shape: (2, 3)
                Bounds of the volume containing the source region of interest
                Rows are domain min,max; columns are x, y, z bounds; in meters

            -flowType
                Attribute, string
                String describing flow field
                Ex.: 'no flow', 'uniform flow', 'open jet'

            -testDescription
                Attribute, string
                String outlining details of simulation/measurement


    -MeasurementData
        -machNumber
            Dataset, float
            Mach number of flow field in x, y and z directions
            Shape: (1, 3)

        -relativeHumidityPct
            Dataset, float

        -speedOfSoundMPerS
            Dataset, float

        -staticPressurePa
            Dataset, float

        -staticTemperatureK
            Dataset, float

    -CsmData
        -binCenterFrequenciesHz
            Dataset, float
            Shape: (1, frequencyBinCount)

            -frequencyBinCount
                Attribute, 32bit signed int [np.dtype('i4')]

        -CsmImaginary
            Dataset, float
            Shape: (microphoneCount, microphoneCount, frequencyBinCount)
            Chunk size: (microphoneCount, microphoneCount, 1)

        -CsmReal
            Dataset, float
            Shape: (microphoneCount, microphoneCount, frequencyBinCount)
            Chunk size: (microphoneCount, microphoneCount, 1)

        -CsmUnits
            Attribute, string
            - 'Pa^2' for spectrumType 'narrowband' and 'octave-n';
            - 'Pa^2/Hz' for spectrumType 'psd';

        -fftSign
            Attribute, int
            - sign of the exponent used to Fourier Transform the data (j*omega*t vs. -j*omega*t)
            - +1 or -1

        -spectrumType
            Attribute, string:
            - type of spectrum in CSM data
            - ex: 'narrowband', 'octave-n', 'psd'
"""


import numpy as np
import scipy.signal as signal
import numpy.fft as FFT
import h5py

import sys


RevisionNumberMajor = 2
RevisionNumberMinor = 4


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

class MicArrayCsmEss:
    """
    Class to store and write microphone array CSM essential information, as per
    'Array Methods HDF5 File Definitions' document, revision 2.4, Sep 2016
    """

    def __init__(self):

        # Attributes with fixed shape initialize with zeros,
        # attributes with setup-varying shapes (i.e. number of mics) initialize empty

        self.caseID = ''
        self.binCenterFrequenciesHz = np.array([], dtype='f8')
        self.frequencyBinCount = np.array([], dtype='i4')

        # self.CsmImaginary = np.array([], dtype='f8')
        # self.CsmReal = np.array([], dtype='f8')

        self.CsmUnits = ''
        self.fftSign = np.array([0], dtype='i4')
        self.spectrumType = ''

        self.machNumber = np.zeros((1, 3), dtype='f8')
        self.relativeHumidityPct = np.array([0.], dtype='f8')
        self.speedOfSoundMPerS = np.array([0.], dtype='f8')
        self.staticPressurePa = np.array([0.], dtype='f8')
        self.staticTemperatureK = np.array([0.], dtype='f8')

        self.revisionNumberMajor = np.array([2], dtype='i4')
        self.revisionNumberMinor = np.array([4], dtype='i4')

        self.microphonePositionsM = np.array([], dtype='f8')
        self.microphoneCount = np.array([0], dtype='i4')

        self.coordinateReference = ''

        self.domainBoundsM = np.zeros((2, 3), dtype='f8')

        self.flowType = ''
        self.testDescription = ''

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    def assertContents(self):
        """
        Test whether instance has been populated with data or if it's empty
        (e.g. recently created but not used yet). This is a precondition for
        writing/saving to a HDF5 file.
        """
        hasData = True

        for attr, value in self.__dict__.items():

            if isinstance(value, str):
                # if attribute is string, test whether has letters (is not empty)
                stringHasLetters = bool(len(self.__dict__[attr]))
                hasData = (hasData and stringHasLetters)

            elif isinstance(value, np.ndarray):
                # if attr is numpy array, test whether has numbers
                npArrayHasNumbers = bool((self.__dict__[attr]).size)
                hasData = (hasData and npArrayHasNumbers)

        return hasData

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    def writeToHDF5File(self):
        """
        Writes current instance contents to <caseID>CsmEss.h5 file
        """
        # Assert whether object has been filled with data, stop writing .h5
        # file otherwise
        assert(self.assertContents(
        )), 'MicArrayCsmEss instance has empty attributes - cannot write <caseID>CsmEss.h5 file!'

        # Open file in the “append” mode - Read/write if exists, create otherwise
        with h5py.File(self.caseID + 'CsmEss.h5', 'w') as CsmEssFile:

            # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
            CsmData = CsmEssFile.create_group('CsmData')

            binCenterFrequenciesHz = CsmData.create_dataset('binCenterFrequenciesHz',
                                                            shape=(
                                                                1, self.frequencyBinCount),
                                                            data=self.binCenterFrequenciesHz,
                                                            dtype='f8')
            binCenterFrequenciesHz.attrs['frequencyBinCount'] = self.frequencyBinCount

            # CSM is first written as zeros, to be populated later in a
            # per-freq/per-chunk basis
            CsmData.create_dataset('CsmImaginary',
                                   shape=(
                                       self.microphoneCount, self.microphoneCount, self.frequencyBinCount),
                                   chunks=(self.microphoneCount,
                                           self.microphoneCount, 1),
                                   data=np.zeros(
                                       (self.microphoneCount, self.microphoneCount, self.frequencyBinCount)),
                                   dtype='f8')

            CsmData.create_dataset('CsmReal',
                                   shape=(
                                       self.microphoneCount, self.microphoneCount, self.frequencyBinCount),
                                   chunks=(self.microphoneCount,
                                           self.microphoneCount, 1),
                                   data=np.zeros(
                                       (self.microphoneCount, self.microphoneCount, self.frequencyBinCount)),
                                   dtype='f8')

            CsmData.attrs['CsmUnits'] = 'Pa^2/Hz'
            CsmData.attrs['fftSign'] = np.array([-1], dtype='i4')
            CsmData.attrs['spectrumType'] = 'psd'

            # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
            MeasurementData = CsmEssFile.create_group('MeasurementData')

            MeasurementData.create_dataset('machNumber',
                                           shape=(1, 3),
                                           data=self.machNumber,
                                           dtype='f8')
            MeasurementData.create_dataset('relativeHumidityPct',
                                           shape=(1,),
                                           data=self.relativeHumidityPct,
                                           dtype='f8')
            MeasurementData.create_dataset('speedOfSoundMPerS',
                                           shape=(1,),
                                           data=self.speedOfSoundMPerS,
                                           dtype='f8')
            MeasurementData.create_dataset('staticPressurePa',
                                           shape=(1,),
                                           data=self.staticPressurePa,
                                           dtype='f8')
            MeasurementData.create_dataset('staticTemperatureK',
                                           shape=(1,),
                                           data=self.staticTemperatureK,
                                           dtype='f8')

            # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
            MetaData = CsmEssFile.create_group('MetaData')

            MetaData.attrs['revisionNumberMajor'] = np.array([2], dtype='i4')
            MetaData.attrs['revisionNumberMinor'] = np.array([4], dtype='i4')

            MetaData.create_dataset("dataLayout",
                                    shape=(2, 3, 4),
                                    data=np.arange(1, 25).reshape(
                                        (2, 3, 4), order='F'),
                                    dtype='i4')

            ArrayAttributes = MetaData.create_group("ArrayAttributes")
            ArrayAttributes.create_dataset("microphonePositionsM",
                                           shape=(self.microphoneCount, 3),
                                           data=self.microphonePositionsM,
                                           dtype='f8')
            ArrayAttributes.create_dataset("microphoneCount",
                                           shape=(1,),
                                           data=self.microphoneCount,
                                           dtype='i4')

            TestAttributes = MetaData.create_group('TestAttributes')
            TestAttributes.attrs['coordinateReference'] = self.coordinateReference

            TestAttributes.create_dataset('domainBoundsM',
                                          shape=(2, 3),
                                          data=self.domainBoundsM,
                                          dtype='f8')

            TestAttributes.attrs['flowType'] = self.flowType
            TestAttributes.attrs['testDescription'] = self.testDescription

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    def readFromHDF5File(self, h5Filename):
        """
        Reads HDF5 Csm Essential file and writes data to current instance.

        Some .h5 files have data misplaced; this code tries to read them from
        the correct location and type (as per current revision), but prints a
        message(s) if data has not been found.

        If a message appears, use function 'print_hdf5_file_structure' to
        manually search for missing data and read it manually. These might be
        in wrong location, and/or saved as wrong type (dataset <-> attribute).
        """
        missingDataFlag = 0

        h5file = h5py.File(h5Filename, 'r')

        self.caseID = h5Filename[:-
                                 9] if 'CsmEss.h5' in h5Filename else h5Filename[:-3]
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # MetaData group

        self.dataLayout = h5file['MetaData/dataLayout'][:]

        self.revisionNumberMajor = h5file['MetaData'].attrs['revisionNumberMajor']
        self.revisionNumberMinor = h5file['MetaData'].attrs['revisionNumberMinor']

        # Check if file revision number is current (2.4)
        if (self.revisionNumberMajor != RevisionNumberMajor
                or self.revisionNumberMinor != RevisionNumberMinor):
            print('File revision No. ('
                  + str(self.revisionNumberMajor) + '.'
                  + str(self.revisionNumberMinor)
                  + ') differs from current revision No. ('
                  + str(RevisionNumberMajor) + '.'
                  + str(RevisionNumberMinor) + ')!\n')

            # set flag to print message
            missingDataFlag = 1

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # Array attributes
        self.microphonePositionsM = h5file['MetaData/ArrayAttributes/microphonePositionsM'][:]
        self.microphoneCount = h5file['MetaData/ArrayAttributes'].attrs['microphoneCount']

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # Test attributes
        # (some of these datasets are sometimes stored as attributes instead;
        #  if so, user must read manually)

        self.coordinateReference = h5file['MetaData/TestAttributes'].attrs['coordinateReference']

        if 'domainBoundsM' in list(h5file['MetaData/TestAttributes'].keys()):
            self.domainBoundsM = h5file['MetaData/TestAttributes/domainBoundsM'][:]
        else:
            print(
                "'domainBoundsM' not found as dataset under '/MetaData/TestAttributes' !")
            missingDataFlag = 1

        if 'flowType' in h5file['MetaData/TestAttributes'].attrs.keys():
            self.flowType = h5file['MetaData/TestAttributes'].attrs['flowType']
        else:
            print("'flowType' not found as attribute under '/MetaData/TestAttributes' !")
            missingDataFlag = 1

        self.testDescription = h5file['MetaData/TestAttributes'].attrs['testDescription']

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # CsmData group

        self.binCenterFrequenciesHz = h5file['CsmData/binCenterFrequenciesHz'][:]
        self.frequencyBinCount = h5file['CsmData/binCenterFrequenciesHz'].attrs['frequencyBinCount']

        # CSM datasets
        # --->>> DEPRECATED: READS TOO MUCH DATA AT ONCE!
        # --->>> CSM SHOULD BE READ BY CHUNKS AS REQUIRED (I.E. PER FREQ)
        # self.CsmImaginary = h5file['CsmData']['CsmImaginary'][:]
        # self.CsmReal = h5file['CsmData']['CsmReal'][:]

        # CSM attributes
        self.CsmUnits = h5file['CsmData'].attrs['csmUnits']
        self.fftSign = h5file['CsmData'].attrs['fftSign']
        self.spectrumType = h5file['CsmData'].attrs['spectrumType']

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # MeasurementData group

        # search for info as dataset in '/MeasurementData':
        #   (some files have these as '/MetaData' attributes, others as
        #    '/MetaData/TestAttributes' attributes; if so, user must read by hand)
        if 'machNumber' in h5file['MeasurementData'].keys():
            self.machNumber = h5file['MeasurementData/machNumber'][:]
        else:
            print("'machNumber' not found as dataset under '/MeasurementData' !")
            missingDataFlag = 1

        if 'relativeHumidityPct' in h5file['MeasurementData'].keys():
            self.relativeHumidityPct = h5file['MeasurementData/relativeHumidityPct'][:]
        else:
            print("'relativeHumidityPct' not found as dataset under '/MeasurementData' !")
            missingDataFlag = 1

        if 'speedOfSoundMPerS' in h5file['MeasurementData'].keys():
            self.speedOfSoundMPerS = h5file['MeasurementData/speedOfSoundMPerS'][:]
        else:
            print("'speedOfSoundMPerS' not found as dataset under '/MeasurementData' !")
            missingDataFlag = 1

        if 'staticPressurePa' in h5file['MeasurementData'].keys():
            self.staticPressurePa = h5file['MeasurementData/staticPressurePa'][:]
        else:
            print("'staticPressurePa' not found as dataset under '/MeasurementData' !")
            missingDataFlag = 1

        if 'staticTemperatureK' in h5file['MeasurementData'].keys():
            self.staticTemperatureK = h5file['MeasurementData/staticTemperatureK'][:]
        else:
            print("'staticTemperatureK' not found as dataset under '/MeasurementData' !")
            missingDataFlag = 1

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # check missing data flag; print message if 1
        if missingDataFlag:
            print(
                '\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
            print(
                '  --->>> Check whether .h5 file data has been interpreted correctly! <<<---   ')
            print(
                '*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # close file
        h5file.close()
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Functions to analyse and print unknown HDF5 file structure

def print_hdf5_file_structure(file_name):
    """
    Prints the HDF5 file structure

    Functions modified from:
        https://confluence.slac.stanford.edu/display/PSDM/HDF5
    """
    file = h5py.File(file_name, 'r')    # open read-only
    item = file                         # ['/Configure:0000/Run:0000']

    print_hdf5_item_structure(item, offset='')
    file.close()


def print_hdf5_item_structure(g, offset='    '):
    """
    Prints the input file/group/dataset (g) name and begin iterations on its
    content

    Functions modified from:
        https://confluence.slac.stanford.edu/display/PSDM/HDF5
    """
    if isinstance(g, h5py.File):
        print('\n')
        print(g.file)
        print('[File]', g.name)
        print('File attributes:')
        for f_attr in g.attrs.keys():
            print('    ', f_attr, ': ', g.attrs[f_attr])

    elif isinstance(g, h5py.Dataset):
        print('\n')
        print(offset + '[Dataset]', g.name)
        print(offset + 'shape =', g.shape)
        print(offset + 'dtype =', g.dtype.name)
        print(offset + 'Dataset attributes:')
        for d_attr in g.attrs.keys():
            print(offset + '    ', d_attr, ': ', g.attrs[d_attr])

    elif isinstance(g, h5py.Group):
        print('\n')
        print(offset + '[Group]', g.name)
        print(offset + 'Group Members:')
        for g_members in g.keys():
            print(offset + '    ', g_members)

        print(offset + 'Group Attributes:')
        for g_attrs in g.attrs.keys():
            print(offset + '    ', g_attrs)

    else:
        print('WARNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
        sys.exit('EXECUTION IS TERMINATED')

    if isinstance(g, (h5py.File, h5py.Group)):
        for key in g.keys():
            subg = g[key]
            # print(offset, key,)
            print_hdf5_item_structure(subg, offset + '    ')


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#

def CSM(mic_signals, N_dft, fs, N_overlap=None, window=None):
    """
    Calculates the cross-spectral matrix (CSM) from an array of time-domain
    microphone recordings using Welch's method.

    Parameters
    ----------
    mic_signals : (M, signal_length) array_like
        Array containing the 'M' time-domain microphone signals, with
        'signal_length' samples.

    N_dft : int
        Number of points used in DFT.

    fs : int
        Sampling frequency in Hz.

    N_overlap : int, optional
        Number of overlapping points when using Welch's method. Defaults to
        None, which is internally converted to 'N_dft//2'.

    window : (N_dft,) array_like, optional
        1-D array containing coefficients of window function used in Welch's
        method. Defaults to None, which is internally converted to
        'scipy.signal.hann(N_dft)'.

    Returns
    -------

    CSM : (M, M, N_dft) array_like
        Matrix containing the (M, M) mic array CSM for each frequency.

    Notes
    -----
    The CSM is calculated so the 'm'-th element of its diagonal contains the
    'm'-th microphone PSD.
    """
    # define default 'N_overlap'
    if N_overlap is None:
        N_overlap = N_dft//2

    # define default window as a 'N_dft'-long Hann
    if window is None:
        window = signal.hann(N_dft)

    # window normalisation
    W = (window**2).sum()/N_dft

    # Number of mics (array elements) and length of recording (in samples)
    M, signal_length = mic_signals.shape

    # define number of frames to be used on CSM computation (round down)
    N_frames = int(np.floor((signal_length - N_overlap)/(N_dft - N_overlap)))

    # Mic signals in the freq domain
    signals_freq = np.zeros((M, N_dft), 'complex')

    # Cross-spectral matrix in the freq domain (single-sided)
    CSM = np.zeros((M, M, N_dft//2 + 1), 'complex')

    # for every frame...
    for n in range(N_frames):

        # ... take the FFT of the mic signals at the current frame...
        signals_freq = FFT.fft(window*mic_signals[:, (N_dft-N_overlap)*(n):
                                                  (N_dft*(n+1)-N_overlap*n)],
                               N_dft, axis=1)

        # ...and calculate and add the CSM at every frequency (single sided)
        # [DC is counted only once]
        CSM[:, :, 0] += np.outer(signals_freq[:, 0],
                                 signals_freq[:, 0].conj())/(fs*N_dft*W)

        for i in range(1, N_dft//2):
            CSM[:, :, i] += 2*np.outer(signals_freq[:, i],
                                       signals_freq[:, i].conj())/(fs*N_dft*W)

        # [Nyquist freq is counted only once]
        CSM[:, :, N_dft//2] += (np.outer(signals_freq[:, N_dft//2],
                                         signals_freq[:, N_dft//2].conj())
                                / (fs*N_dft*W))

    # Average over number of frames and return the result
    return CSM/N_frames


def speed_of_sound(Tc):
    """
    Returns the estimated speed of sound 'c' in air as a function of the
    ambient temperature 'temp' (in degrees Celsius).

    Uses Equation 5.6.6, p. 121, from Kinsler et al, "Fundamentals of
    Acoustics", 4th Ed., 2000.
    """
    # speed of sound in air at 0 deg Celsius and 1 atm pressure
    c0 = 331.5

    # temperature (in Kelvin)
    Tk = Tc + 273.15

    return c0*np.sqrt(Tk/273.15)
