"""Author: Fabio Casagrande Hirono"""
import numpy as np
from .xml_utils import save_XML, XML_calib, XML_mic


def DARP2016_MicArray():
    """
    Returns the microphone coordinates for the 'near-field' planar microphone
    array used in the 2016 experiments with the DARP open-jet wind tunnel at
    the ISVR, Univ. of Southampton, UK [Casagrande Hirono, 2018].

    The array is a Underbrink multi-arm spiral array with 36 electret
    microphones, with 7 spiral arms containing 5 mics each and one central mic.
    The array has a circular aperture (diameter) of 0.5 m.

    The calibration factors were obtained using a 1 kHz, 1 Pa RMS calibrator.
    Multiply the raw mic data by its corresponding factor to obtain a
    calibrated signal in Pascals.
    """

    M = 36                      # number of mics
    array_height = -0.49        # [m] (ref. to airfoil height at z=0)

    # mic coordinates (corrected for DARP2016 configuration)
    XYZ_array = np.array([[0.,  0.025,  0.08477,  0.12044,  0.18311,  0.19394,
                           0.01559,  0.08549,  0.16173,  0.19659,  0.24426, -0.00556,
                           0.02184,  0.08124,  0.06203,  0.11065, -0.02252, -0.05825,
                           -0.06043, -0.11924, -0.10628, -0.02252, -0.09449, -0.15659,
                           -0.21072, -0.24318, -0.00556, -0.05957, -0.13484, -0.14352,
                           -0.19696,  0.01559,  0.02021, -0.01155,  0.03174, -0.00242],
                          [-0., -0.,  0.04175,  0.11082,  0.10542,  0.15776,
                           -0.01955, -0.04024, -0.02507, -0.07743, -0.05327, -0.02437,
                           -0.09193, -0.14208, -0.20198, -0.22418, -0.01085, -0.0744,
                           -0.1521, -0.17443, -0.22628,  0.01085, -0.00084, -0.04759,
                           -0.01553, -0.05799,  0.02437,  0.07335,  0.09276,  0.15506,
                           0.15397,  0.01955,  0.09231,  0.16326,  0.20889,  0.24999],
                          array_height*np.ones(M)])

    # calibration factors
    array_cal = np.array([73.92182641429085,    96.84446743391487,  85.48777846463159,
                          85.24410968090712,    83.63917149322562,  68.94090765134432,
                          79.2385037527723,     112.77357210746612, 84.8483307868491,
                          87.18956628936178,    97.75046920293282,  89.2829545690508,
                          79.51644155562396,    90.39403884030057,  80.71754629014218,
                          89.4418210091059,     98.33634233056068,  79.2212022850229,
                          91.25543447201031,    89.55040012572815,  85.77495667666254,
                          82.74418222820202,    84.63061055646973,  77.01568014644964,
                          95.52764533324982,    92.16734812591154,  95.27123074600838,
                          87.93335310521428,    96.65066131188675,  93.58564782091074,
                          78.1446818728945,     101.3047738767648,  83.68569643491034,
                          84.7981031520437,     94.40796508430756,  83.52266614867919])

    return XYZ_array, array_cal


def DARP2016_Acoular_XML():
    """Saves the DARP2016_MicArray into a .xml file, compatile with Acoular.
    """
    XYZ_array, array_cal = DARP2016_MicArray()
    array_name = "DARP2016"
    array_pos = XML_mic(XYZ_array)
    array_file = '<?xml version="1.0" encoding="utf-8"?>\n<MicArray name="{name}">\n{pos}</MicArray>'.format(
        name=array_name, pos=array_pos)
    calib_data = XML_calib(array_cal, 100)
    calib_file = '<?xml version="1.0" encoding="utf-8"?>\n<Calib name="{name}">\n{data}</Calib>'.format(
        name=20160101, data=calib_data)

    save_XML(array_name, array_file)
    save_XML(array_name+"_calib", calib_file)
