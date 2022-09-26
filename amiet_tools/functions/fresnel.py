"""Author: Fabio Casagrande Hirono"""
import numpy as np
import scipy.integrate as integrate


def fr_integrand_re(x):
    """Creates the argument to the Fresnel integral."""
    return (np.exp(1j*x)/np.sqrt(x)).real


def fr_integrand_im(x):
    """Creates the argument to the complex conjugate Fresnel integral."""
    return (np.exp(1j*x)/np.sqrt(x)).imag


def fr_int(zeta):
    """
    Calculates the Fresnel integral of 'zeta'

    Parameters
    ----------
    zeta : (Nz,) array_like
        1D array of parameter 'zeta' for integration.

    Returns
    -------
    E : (Nz,) array_like
        1D array with results of Fresnel integral of each value of 'zeta'

    Notes
    -----
    Its complex-conjugate version can be obtained from the
    'amiet_tools.fr_int_cc' function.
    """
    # Check if zeta is array or float
    if type(zeta) is np.ndarray:
        E = np.zeros(zeta.shape, 'complex')

        # Calculate Fresnel integral for all non-zero values of zeta
        for i in range(zeta.size):
            if zeta[i] != 0:
                E[i] = (integrate.quad(fr_integrand_re, 0, zeta[i])[0]
                        + 1j*integrate.quad(fr_integrand_im, 0, zeta[i])[0])

    elif zeta != 0:
        E = (integrate.quad(fr_integrand_re, 0, zeta)[0]
             + 1j*integrate.quad(fr_integrand_im, 0, zeta)[0])

    return (1/np.sqrt(2*np.pi))*E


def fr_int_cc(zeta):
    """
    Calculates the complex-conjugate Fresnel integral of 'zeta'

    Parameters
    ----------
    zeta : (Nz,) array_like
        1D array of parameter 'zeta' for integration.

    Returns
    -------
    E_conj : (Nz,) array_like
        1D array with results of complex-conjugate Fresnel integral of each
        value of 'zeta'

    Notes
    -----
    Its non-complex-conjugate version can be obtained from the
    'amiet_tools.fr_int' function.
    """
    # Check if zeta is array or float
    if type(zeta) is np.ndarray:
        E_conj = np.zeros(zeta.shape, 'complex')

        # Calculate complex-conjugate Fresnel integral for all non-zero values
        # of zeta
        for i in range(zeta.size):
            if zeta[i] != 0:
                E_conj[i] = (integrate.quad(fr_integrand_re, 0, zeta[i])[0]
                             - 1j*integrate.quad(fr_integrand_im, 0, zeta[i])[0])

    elif zeta != 0:
        E_conj = (integrate.quad(fr_integrand_re, 0, zeta)[0]
                  - 1j*integrate.quad(fr_integrand_im, 0, zeta)[0])

    return (1/np.sqrt(2*np.pi))*E_conj
