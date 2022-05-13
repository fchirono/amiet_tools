"""Author: Fabio Casagrande Hirono"""
import numpy as np
import scipy.special as ss


def Phi_2D(Kx, ky_vec, Ux, turb_intensity, length_scale, model='K'):
    """
    Returns 2D isotropic turbulence energy spectrum in wavenumber domain.

    Parameters
    ----------
    Kx : (N_kx,) array_like or float
        1D array of chordwise gust wavenumbers.

    ky_vec : (N_ky,) array_like
        1D array of spanwise gust wavenumbers.

    Ux : float
        Mean flow velocity, assumed in '+x' direction.

    turb_intensity : float
        Turbulence intensity: sqrt(w_meanSquared/(Ux**2))

    length_scale : float
        Turbulence integral length scale, in meters

    model : {'K', 'L'}
        Type of spectrum: 'K' for von Karman spectrum, or 'L' for Liepmann spectrum.

    Returns
    -------
    Phi : (N_kx, N_ky) or (N_ky,) array_like
        2D or 1D array containing the values of two-dimensional turbulence
        energy for each wavenumber, according to von Karman or Liepmann
        spectrum.
    """
    u_mean2 = (Ux*turb_intensity)**2

    if type(Kx) is not np.ndarray:
        Kx = np.asarray([Kx])

    # von Karman model (Amiet 1975)
    if model == 'K':
        ke = (np.sqrt(np.pi)/length_scale)*(ss.gamma(5./6)/ss.gamma(1./3))

        kxe2_ye2 = (Kx[:, np.newaxis]/ke)**2 + (ky_vec/ke)**2

        return (4./(9*np.pi))*(u_mean2/(ke**2))*kxe2_ye2/((1+kxe2_ye2)**(7./3))

    # 2D Liepmann turbulence spectrum
    elif model == 'L':

        ls2 = length_scale**2

        return ((u_mean2*ls2/(4*np.pi))
                * ((1+ls2*(4*Kx[:, np.newaxis]**2 + ky_vec**2)))
                / (1+ls2*(Kx[:, np.newaxis]**2 + ky_vec**2))**(5./2))
    return None
