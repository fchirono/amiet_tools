"""Author: Fabio Casagrande Hirono"""
import numpy as np
from .fresnel import fr_int, fr_int_cc
import scipy.special as ss


def delta_p(rho0, b, w0, Kx, ky, xy, Mach):
    """
    Calculates the pressure jump response 'delta_p' for a single turbulent gust.

    Parameters
    ----------
    rho0 : float
        Density of air.

    b : float
        Airfoil semichord.

    w0 : float
        Gust amplitude.

    Kx : float
        Chordwise turbulent gust wavenumber.

    ky : float
        Spanwise turbulent gust wavenumber.

    xy : ({2, 3}, Ny, Nx) array_like
        2D array containing (x, y) coordinates of airfoil surface mesh.

    Mach : float
        Mean flow Mach number.

    Returns
    -------
    delta_p : (Ny, Nx) array_like
        Surface pressure jump over airfoil surface mesh in response to a single
        turbulent gust with wavenumbers (Kx, ky) and amplitude 'w0'.
    """
    # pressure difference over the whole airfoil surface
    delta_p = np.zeros(xy[0].shape, 'complex')

    if xy.ndim == 3:
        # unsteady lift over the chord line (mid-span)
        g_x = np.zeros(xy[0][0].shape, 'complex')

        # calculates the unsteady lift over the chord
        g_x = g_LE(xy[0][0], Kx, ky, Mach, b)

        # broadcasts a copy of 'g_x' to 'delta_p'
        delta_p = g_x[np.newaxis, :]

    elif xy.ndim == 2:
        # unsteady lift over the chord line (mid-span)
        g_x = np.zeros(xy[0].shape, 'complex')

        # calculates the unsteady lift over the chord
        delta_p = g_LE(xy[0], Kx, ky, Mach, b)

    # adds the constants and the 'k_y' oscillating component
    delta_p = 2*np.pi*rho0*w0*delta_p*np.exp(-1j*ky*xy[1])

    return delta_p


def g_LE(xs, Kx, ky, Mach, b):
    """
    Airfoil non-dimensional chordwise pressure jump in response to a single gust.

    Parameters
    ----------
    xs : (Ny, Nx) or (Nx,) array_like
        Airfoil surface mesh chordwise coordinates.

    Kx : float
        Chordwise turbulent gust wavenumber.

    ky : float
        Spanwise turbulent gust wavenumber.

    Mach : float
        Mean flow Mach number.

    b : float
        Airfoil semichord.

    Returns
    -------
    g_LE : (Ny, Nx) array_like
        Non-dimensional chordwise surface pressure jump over airfoil surface
        mesh in response to a single turbulent gust with wavenumbers (Kx, ky)
        and amplitude 'w0'.

    Notes
    -----
    This function provides the airfoil responses for either subcritical or
    supercritical gusts. For critical gusts, the airfoil response is
    interpolated from slightly sub- and slightly supercritical responses.
    """
    beta = np.sqrt(1-Mach**2)
    ky_critical = Kx*Mach/beta

    # p_diff < 0: supercritical
    # p_diff > 0: subcritical
    p_diff = (np.abs(ky) - ky_critical)/ky_critical

    # supercritical gusts
    if p_diff < -1e-3:
        return g_LE_super(xs, Kx, ky, Mach, b)

    elif p_diff > 1e-3:
        return g_LE_sub(xs, Kx, ky, Mach, b)

    else:

        # get gusts 1% above and below critical ky
        ky_sp = ky*0.99
        ky_sb = ky*1.01

        g_sp = g_LE_super(xs, Kx, ky_sp, Mach, b)
        g_sb = g_LE_sub(xs, Kx, ky_sb, Mach, b)

        return (g_sp + g_sb)/2.


def g_LE_super(xs, Kx, ky, Mach, b):
    """
    Returns airfoil non-dimensional pressure jump for supercritical gusts.

    Parameters
    ----------
    xs : (Ny, Nx) or (Nx,) array_like
        Airfoil surface mesh chordwise coordinates.

    Kx : float
        Chordwise turbulent gust wavenumber.

    ky : float
        Spanwise turbulent gust wavenumber.

    Mach : float
        Mean flow Mach number.

    b : float
        Airfoil semichord.

    Returns
    -------
    g_LE_super : (Ny, Nx) array_like
        Non-dimensional chordwise surface pressure jump over airfoil surface
        mesh in response to a single supercritical turbulent gust with
        wavenumbers (Kx, ky)

    Notes
    -----
    This function includes two terms of the Schwarzchild technique; the first
    term contains the solution for a infinite-chord airfoil with a leading edge
    but no trailing edge, while the second term contains a correction factor
    for a infinite-chord airfoil with a trailing edge but no leading edge.
    """
    beta = np.sqrt(1-Mach**2)
    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*Mach

    kappa = np.sqrt(mu_a**2 - (ky*b/beta)**2)

    g1_sp = (np.exp(-1j*((kappa - mu_a*Mach)*((xs/b) + 1) + np.pi/4))
             / (np.pi*np.sqrt(np.pi*((xs/b) + 1)*(Kx*b + (beta**2)*kappa))))

    g2_sp = -(np.exp(-1j*((kappa - mu_a*Mach)*((xs/b) + 1) + np.pi/4))
              * (1-(1+1j)*fr_int_cc(2*kappa*(1-xs/b)))
              / (np.pi*np.sqrt(2*np.pi*(Kx*b + (beta**2)*kappa))))

    return g1_sp + g2_sp


def g_LE_sub(xs, Kx, ky, Mach, b):
    """
    Returns airfoil non-dimensional pressure jump for subcritical gusts.

    Parameters
    ----------
    xs : (Ny, Nx) or (Nx,) array_like
        Airfoil surface mesh chordwise coordinates.

    Kx : float
        Chordwise turbulent gust wavenumber.

    ky : float
        Spanwise turbulent gust wavenumber.

    Mach : float
        Mean flow Mach number.

    b : float
        Airfoil semichord.

    Returns
    -------
    g_LE_sub : (Ny, Nx) array_like
        Non-dimensional chordwise surface pressure jump over airfoil surface
        mesh in response to a single subcritical turbulent gust with
        wavenumbers (Kx, ky)

    Notes
    -----
    This function includes two terms of the Schwarzchild technique; the first
    term contains the solution for a infinite-chord airfoil with a leading edge
    but no trailing edge, while the second term contains a correction factor
    for a infinite-chord airfoil with a trailing edge but no leading edge.
    """
    beta = np.sqrt(1-Mach**2)
    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*Mach

    kappa1 = np.sqrt(((ky*b/beta)**2) - mu_a**2)

    g1_sb = (np.exp((-kappa1 + 1j*mu_a*Mach)*((xs/b) + 1))*np.exp(-1j*np.pi/4)
             / (np.pi*np.sqrt(np.pi*((xs/b) + 1)*(Kx*b - 1j*(beta**2)*kappa1))))

    g2_sb = -(np.exp((-kappa1 + 1j*mu_a*Mach)*((xs/b) + 1))
              * np.exp(-1j*np.pi/4)*(1 - ss.erf(2*kappa1*(1-xs/b)))
              / (np.pi*np.sqrt(2*np.pi*(Kx*b - 1j*(beta**2)*kappa1))))

    return g1_sb + g2_sb


def L_LE(x, sigma, Kx, ky, Mach, b):
    """
    Returns the effective lift functions - i.e. chordwise integrated surface pressures

    Parameters
    ----------
    x : (M,) array_like
        1D array of observer locations 'x'-coordinates

    sigma : (M,) array_like
        1D array of observer locations flow-corrected distances

    Kx : float
        Chordwise turbulent gust wavenumber.

    ky : float
        Spanwise turbulent gust wavenumber.

    Mach : float
        Mean flow Mach number.

    b : float
        Airfoil semichord.


    Returns
    -------
    L_LE : (M,) array_like
        Effective lift function for all observer locations.

    Notes
    -----
    These functions are the chordwise integrated surface pressures, and are
    parts of the far-field-approximated model for airfoil-turbulente noise.
    """
    beta = np.sqrt(1-Mach**2)
    ky_critical = Kx*Mach/beta

    # percentage difference in ky
    # p_diff < 0: supercritical / p_diff > 0: subcritical
    p_diff = (np.abs(ky) - ky_critical)/ky_critical

    # supercritical gusts
    if p_diff < -1e-3:
        return L_LE_super(x, sigma, Kx, ky, Mach, b)

    elif p_diff > 1e-3:
        return L_LE_sub(x, sigma, Kx, ky, Mach, b)

    else:
        # get gusts 1% above and below critical ky
        ky_sp = ky*0.99
        ky_sb = ky*1.01

        L_sp = L_LE_super(x, sigma, Kx, ky_sp, Mach, b)
        L_sb = L_LE_sub(x, sigma, Kx, ky_sb, Mach, b)

        return (L_sp + L_sb)/2.


def L_LE_super(x, sigma, Kx, Ky, Mach, b):
    """
    Returns the effective lift functions for supercritical gusts

    Parameters
    ----------

    x : (M,) array_like
        1D array of observer locations 'x'-coordinates

    sigma : (M,) array_like
        1D array of observer locations flow-corrected distances

    Kx : float
        Chordwise turbulent gust wavenumber.

    ky : float
        Spanwise turbulent gust wavenumber.

    Mach : float
        Mean flow Mach number.

    b : float
        Airfoil semichord.


    Returns
    -------

    Notes
    -----
    These functions are the chordwise integrated surface pressures, and are
    parts of the far-field-approximated model for airfoil-turbulente noise.
    """
    beta = np.sqrt(1-Mach**2)
    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*Mach

    kappa = np.sqrt(mu_a**2 - (Ky*b/beta)**2)
    H1 = kappa - mu_a*x/sigma
    H2 = mu_a*(Mach - x*sigma) - np.pi/4

    L1 = ((1/np.pi)*np.sqrt(2/((Kx*b + (beta**2)*kappa)*H1))
          * fr_int_cc(2*H1)*np.exp(1j*H2))

    L2 = ((np.exp(1j*H2)
          / (np.pi*H1*np.sqrt(2*np.pi*(Kx*b + (beta**2)*kappa))))
          * (1j*(1 - np.exp(-2j*H1))
             + (1 - 1j)*(fr_int_cc(4*kappa)
                         - np.sqrt(2*kappa/(kappa + mu_a*x/sigma))
                         * np.exp(-2j*H1)
                         * fr_int_cc(2*(kappa + mu_a*x/sigma)))))

    return L1+L2


def L_LE_sub(x, sigma, Kx, Ky, Mach, b):
    """
    Returns the effective lift functions for subcritical gusts

    Parameters
    ----------

    x : (M,) array_like
        1D array of observer locations 'x'-coordinates

    sigma : (M,) array_like
        1D array of observer locations flow-corrected distances

    Kx : float
        Chordwise turbulent gust wavenumber.

    ky : float
        Spanwise turbulent gust wavenumber.

    Mach : float
        Mean flow Mach number.

    b : float
        Airfoil semichord.

    Returns
    -------

    Notes
    -----
    These functions are the chordwise integrated surface pressures, and are
    parts of the far-field-approximated model for airfoil-turbulente noise.
    """
    beta = np.sqrt(1-Mach**2)
    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*Mach

    kappa1 = np.sqrt((Ky*b/beta)**2 - (mu_a**2))
    H2 = mu_a*(Mach - x*sigma) - np.pi/4
    H3 = kappa1 - 1j*mu_a*x/sigma

    L1 = ((1/np.pi)*np.sqrt(2/((Kx*b - 1j*(beta**2)*kappa1)
                               * (1j*kappa1 - mu_a*x/sigma)))
          * fr_int(2*(1j*kappa1 - mu_a*x/sigma))*np.exp(1j*H2))

    L2 = ((1j*np.exp(1j*H2)
           / (np.pi*H3*np.sqrt(2*np.pi*(Kx*b - 1j*(beta**2)*kappa1))))
          * (1 - np.exp(-2*H3) - ss.erf(np.sqrt(4*kappa1))
              + 2*np.exp(-2*H3)*np.sqrt(kappa1/(1j*kappa1 + mu_a*x/sigma))
              * fr_int(2*(1j*kappa1 - mu_a*x/sigma))))

    return L1+L2
