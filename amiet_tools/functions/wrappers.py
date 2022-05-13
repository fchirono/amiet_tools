"""Author: Fabio Casagrande Hirono"""
import numpy as np
from .flat_plate_response import delta_p


def calc_airfoil_Sqq(testSetup, airfoilGeom, frequencyVars, Ky_vec, Phi):
    """
    Calculates the aerofoil surface pressure jump cross-spectral density matrix
    (CSM).

    Parameters
    ----------
    testSetup : instance of TestSetup class
        Instance containing test setup data.

    airfoilGeom : instance of AirfoilGeom class
        Instance containing airfoil geometry data.

    frequencyVars : instance of FrequencyVars class
        Instance containing frequency data.

    Ky_vec : (N_ky,) array_like
        1D array containing range of negative and positive spanwise gust
        wavenumber values.

    Phi : (N_ky,) array_like
        1D array containing the turbulent wavenumber energy spectrum for values
        of 'Ky_vec'.

    Returns
    -------
    Sqq : (Nx*Ny, Nx*Ny) array_like
        2D array containing the cross-spectral density of the airfoil surface
        pressure.

    Sqq_dxy : (Nx*Ny, Nx*Ny) array_like
        2D array containing the surface area weights "(dx*dy)*(dx' * dy')" for
        all pairs (x, y), (x', y') of airfoil points.

    Notes
    -----
    'Ky_vec' can be calculated using 'amiet_tools.ky_vec' function.

    'Phi' can be calculated with 'amiet_tools.Phi_2D' function.

    For surface pressure analyses, such as cross-spectrum phase or coherence
    lengths, use 'Sqq' only. For acoustic radiation analysis, the function
    'amiet_tools.calc_radiated_Spp' applies the equivalent areas 'Sqq_dxy' to
    numerically calculate the integration over the airfoil surface.
    """
    # Surface area weighting matrix for applying to Sqq
    dxy = np.ones((airfoilGeom.Ny, airfoilGeom.Nx)) * \
        airfoilGeom.dx[np.newaxis, :]*airfoilGeom.dy
    Sqq_dxy = np.outer(dxy, dxy)

    # gust spanwise wavenumber interval
    dky = Ky_vec[1]-Ky_vec[0]

    # source CSM
    nxny = airfoilGeom.Nx*airfoilGeom.Ny
    Sqq = np.zeros((nxny, nxny), 'complex')

    for kyi in range(Ky_vec.shape[0]):

        # sinusoidal gust peak value
        w0 = np.sqrt(Phi[kyi])

        # Pressure 'jump' over the airfoil (for single gust)
        delta_p1 = delta_p(testSetup.rho0, airfoilGeom.b, w0,
                           frequencyVars.Kx, Ky_vec[kyi],
                           airfoilGeom.XYZ[0:2], testSetup.Mach)

        # reshape for vector calculation
        delta_p1_calc = delta_p1.reshape(nxny)

        Sqq[:, :] += np.outer(delta_p1_calc, delta_p1_calc.conj())

    Sqq *= (testSetup.Ux*dky)

    return Sqq, Sqq_dxy


def calc_radiated_Spp(testSetup, airfoilGeom, frequencyVars, Ky_vec, Phi, G):
    """
    Calculates the cross-spectral density matrix (CSM) of the acoustic field
    radiated by the airfoil.

    Parameters
    ----------
    testSetup : instance of TestSetup class
        Instance containing test setup data.

    airfoilGeom : instance of AirfoilGeom class
        Instance containing airfoil geometry data.

    frequencyVars : instance of FrequencyVars class
        Instance containing frequency data.

    Ky_vec : (N_ky,) array_like
        1D array containing range of negative and positive spanwise gust
        wavenumber values. Calculate with 'amiet_tools.ky_vec' function.

    Phi : (N_ky,) array_like
        1D array containing the turbulent wavenumber energy spectrum for values
        of 'Ky_vec'. Calculate with 'amiet_tools.Phi_2D' function.

    G : (M, Nx*Ny) array_like
        2D matrix of complex transfer function between 'M' observer locations
        and 'Nx*Ny' points over the airfoil surface.

    Returns
    -------
    Spp : (M, M) array_like
        2D matrix of cross-spectral density of the acoustic field seen at 'M'
        observers/microphones.

    Notes
    -----
    'G' can be calculated using 'amiet_tools.dipole3D' for dipole sources in
    steady or moving medium, or 'amiet_tools.dipole_shear' for dipole sources
    in a shear layer medium.
    """
    Sqq, Sqq_dxy = calc_airfoil_Sqq(
        testSetup, airfoilGeom, frequencyVars, Ky_vec, Phi)

    # apply weights for surface area
    Sqq *= Sqq_dxy

    return (G @ Sqq @ G.conj().T)*4*np.pi
