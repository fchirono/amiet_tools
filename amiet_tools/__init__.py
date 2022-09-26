"""
Amiet_tools
===========
Author:
    Fabio Casagrande Hirono - fchirono@gmail.com

Amiet_tools:

    The 'amiet_tools' (AmT) Python package provides a reference implementation of
    Amiet's [JSV 41, 1975] model for turbulence-aerofoil interaction noise with
    extensions. These functions allow the calculation of the surface pressure jump
    developed over the aerofoil surface (i.e. the acoustic source distribution) in
    response to incoming turbulence, and of the acoustic field radiated by the
    interaction.

    Incoming turbulence can be a single sinusoidal gust, or a sum of incoherent
    gusts with amplitudes given by a prescribed energy spectrum.

Main Technical References:

    Amiet, R. K., "Acoustic radiation from an airfoil in a turbulent stream",
    Journal of Sound and Vibration, Vol. 41, No. 4:407–420, 1975.

    Blandeau, V., "Aerodynamic Broadband Noise from Contra-Rotating Open
    Rotors", PhD Thesis, Institute of Sound and Vibration Research, University
    of Southampton, Southampton - UK, 2011.

    Casagrande Hirono, F., "Far-Field Microphone Array Techniques for Acoustic
    Characterisation of Aerofoils", PhD Thesis, Institute of Sound and
    Vibration Research, University of Southampton, Southampton - UK, 2018.

    Reboul, G., "Modélisation du bruit à large bande de soufflante de
    turboréacteur", PhD Thesis, Laboratoire de Mécanique des Fluides et
    d’Acoustique - École Centrale de Lyon, Lyon - France, 2010.

    Roger, M., "Broadband noise from lifting surfaces: Analytical modeling and
    experimental validation". In Roberto Camussi, editor, "Noise Sources in
    Turbulent Shear Flows: Fundamentals and Applications". Springer-Verlag,
    2013.

    de Santana, L., "Semi-analytical methodologies for airfoil noise
    prediction", PhD Thesis, Faculty of Engineering Sciences - Katholieke
    Universiteit Leuven, Leuven, Belgium, 2015.

Dependencies:
    - numpy: array processing for numbers, strings, records, and objects;
    - scipy: scientific library.


    All dependencies are already included in the Anaconda Python Distribution, a
    free and open source distribution of Python. Anaconda 4.8.2 (with Python 3.7)
    was used to develop and test AmT, and is recommended for using AmT.

    You can find out everything available reading the submodules documentation.
    For further information, check the specific module, class, method or function
    documentation.
"""
from amiet_tools.classes import *
from amiet_tools.functions import *

# package submodules and scripts to be called as amiet_tools.something
__all__ = [
    # Functions
    'create_airf_mesh',
    'chord_sampling',
    'loadTestSetup',
    'loadAirfoilGeom',
    'DARP2016_MicArray',
    'DARP2016_Acoular_XML',
    'calc_airfoil_Sqq',
    'calc_radiated_Spp',
    'fr_int', 'fr_int_cc',
    'fr_integrand_im',
    'fr_integrand_re',
    'r', 'r_bar', 'sigma_',
    't_convect', 't_sound',
    't_total', 'constr_xl',
    'constr_yl', 'ShearLayer_X',
    'ShearLayer_matrix',
    'ShearLayer_Corr',
    'dipole_shear', 'dipole3D',
    'delta_p', 'g_LE',
    'g_LE_sub', 'g_LE_super',
    'L_LE', 'L_LE_sub',
    'L_LE_super', 'ky_vector',
    'ky_att', 'Phi_2D',
    'rect_grid',
    'read_ffarray_lvm',
    # Classes
    'TestSetup',
    'AirfoilGeom',
    'FrequencyVars']

__author__ = "Fabio Casagrande Hirono"
__date__ = "13 May 2022"
__version__ = "0.0.2"
