"""Author: Fabio Casagrande Hirono"""
import numpy as np


def ky_vector(b, d, k0, Mach, beta, method='AcRad', xs_ref=None):
    """
    Returns a vector of spanwise gust wavenumbers for acoustic calculations

    Parameters
    ----------
    b : float
        Airfoil semi chord.

    d : float
        Airfoil semi span.

    k0 : float
        Acoustic wavenumber 'k0'. Can be obtained from the
        temporal frequency 'f' [in Hz] and the speed of sound 'c0' [in m/s]
        as 'k0 = 2*pi*f/c0'.'

    Mach : float
        Mean flow Mach number.

    beta : float
        Prandtl–Glauert parameter (=sqrt(1-M**2))

    method : {'AcRad', 'SurfPressure'}, optional
        Calculation method to use. Defaults to 'AcRad'.

    xs_ref : float, optional
        Chordwise coordinate of reference point, defined in interval (-b, +b].
        Used in 'SurfPressure' mode, not required for 'AcRad' mode. Defaults to
        None.

    Returns
    -------
    Ky : (Nk,) array_like
        1D array containing spanwise gust wavenumbers in range [-ky_max, +ky_max],
        with center sample at ky=0

    Notes
    -----
    Returns a vector of equally-spaced spanwise hydrodynamic (gust) wavenumber
    values for calculations of airfoil response, either for calculating
    the airfoil acoustic radiation (method = 'AcRad') or for calculating the
    airfoil surface pressure cross-spectrum (method = 'SurfPressure').

    'AcRad' mode returns a shorter range of gusts for acoustic radiation
    calculations, and 'SurfPressure' returns a larger range for unsteady
    surface pressure calculations (excessive for acoustic radiation
    calculations).
    """
    if method not in ['AcRad', 'SurfPressure']:
        # Assert 'method' string for valid inputs
        method_error = "'method' not recognized; please use either 'AcRad' or 'SurfPressure'"
        raise ValueError(method_error)

    # critical hydrodynamic spanwise wavenumber
    ky_crit = k0/beta

    # width of ky sinc function
    sinc_width = 2*np.pi/(2*d)

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # for acoustic radiation calculations:
    if method == 'AcRad':

        if ky_crit < 2*np.pi/d:
            # 'low freq' - include some subcritical gusts (up to 1st sidelobe
            # of sinc function)
            N_ky = 41           # value obtained empirically
            ky_T = 2*np.pi/d    # main lobe + 1st sidelobes in sinc function
            Ky = np.linspace(-ky_T, ky_T, N_ky)

        else:
            # 'high freq' - restrict to supercritical gusts only

            # get ky with spacing equal to approx. 1/8 width of sinc function
            N_ky = int(np.ceil(2*ky_crit/sinc_width)*8)+1
            Ky = np.linspace(-ky_crit, ky_crit, N_ky)

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # for surface pressure cross-spectra calculations
    elif method == 'SurfPressure':

        # find ky that is at -20 dB at reference chord point
        ky_20dBAtt = ky_att(xs_ref, b, Mach, k0, Att=-20)

        # largest ky under consideration (25% above ky_20dBAtt, for safety)
        ky_max = 1.25*ky_20dBAtt

        # get ky with spacing equal to approx. 1/8 width of sinc function
        N_ky = int(np.ceil(2*ky_max/sinc_width)*8)+1

        Ky = np.linspace(-ky_max, ky_max, N_ky)

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    return Ky


def ky_att(xs, b, Mach, k0, Att=-20):
    """
    Returns the spanwise gust wavenumber 'ky_att' with response at 'xs' attenuated by 'Att' decibels

    Parameters
    ----------
    xs : float
        Chordwise coordinate of reference point, defined in interval (-b, +b].

    b : float
        Airfoil semi chord.

    Mach : float
        Mean flow Mach number.

    k0 : float
        Acoustic wavenumber 'k0'. Can be obtained from the
        temporal frequency 'f' [in Hz] and the speed of sound 'c0' [in m/s]
        as 'k0 = 2*pi*f/c0'.

    Att : float, optional
        Level of attenuation of the surface pressure at point 'xs', in decibels.
        Defaults to -20 dB.

    Returns
    -------
    ky_att : float
        Subcritical gust spanwise wavenumber 'ky_att' such that the aerofoil
        response at point 'xs' is 'Att' dB reduced.
    """
    beta = np.sqrt(1-Mach**2)

    # critical gust spanwise wavenumber
    ky_crit = k0/beta

    term1 = -(beta**2)*np.log(10**(Att/20))/(k0*(xs + b))

    return ky_crit*np.sqrt(term1**2 + 1)
