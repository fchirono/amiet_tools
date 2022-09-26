"""Author: Fabio Casagrande Hirono"""
import numpy as np


def chord_sampling(b, Nx=200, exp_length=2):
    """
    Calculates 'Nx' points non-uniformly sampled over the half-open
    interval (-b, b], with the leading edge at '-b'.

    Parameters
    ----------
    b : float
        Airfoil semichord, in meters.

    Nx : int, optional
        Number of points used in sampling.

    exp_length : float, optional
        Length of exponential interval to be sampled.

    Returns
    -------
    x : (Nx,) array_like
        1D array of non-uniformly sampled points in half-open interval (-b, +b]

    dx : (Nx,) array_like
        1D vector of non-uniform sample intervals.

    Notes
    -----
    The function samples an exponential function over the interval
    [-exp_length/2, exp_length/2] uniformly at 'Nx' points, which are remapped
    to the actual chord interval [-b, b] and the leading edge then removed.
    This type of sampling assigns more points around the leading edge.

    Higher values of 'exp_length' provide more non-uniform sampling, while
    lower values provide more uniform sampling.
    """
    # calculates exponential curve
    x = np.exp(np.linspace(-exp_length/2, exp_length/2, Nx+1))

    # normalise to [0, 1] interval
    x = x-x.min()
    x = x/x.max()

    # normalise to [-b, b] interval
    x = (x*2*b)-b

    # calculate dx (for numerically integrating chord functions)
    dx = np.diff(x)

    # remove leading edge singularity
    x = x[1:]

    return x, dx


def create_airf_mesh(b, d, Nx=100, Ny=101):
    """
    Creates a (3, Ny, Nx) mesh containing the airfoil surface points coordinates.

    Parameters
    ----------
    b : float
        Airfoil semichord, in meters.

    d : float
        Airfoil semispan, in meters.

    Nx : int, optional
        Number of points used in sampling the chord (non-uniformly).

    Ny : int, optional
        Number of points used in sampling the span (uniformly).

    Returns
    -------
    XYZ_airfoil : (3, Ny, Nx) array_like
        3D array containing the coordinates of each point on the sampled
        airfoil surface.

    dx : (Nx,) array_like
        1D vector of non-uniform chord sample intervals.

    dy : float
       Span sample interval.

    Notes
    -----
    The airfoil 'z' coordinate is always set to zero.
    """
    x_airfoil, dx = chord_sampling(b, Nx)

    y_airfoil, dy = np.linspace(-d, d, Ny, retstep=True)
    #dy = y_airfoil[1] - y_airfoil[0]

    XY_airfoil = np.meshgrid(x_airfoil, y_airfoil)
    Z_airfoil = np.zeros(XY_airfoil[0].shape)
    XYZ_airfoil = np.array([XY_airfoil[0], XY_airfoil[1], Z_airfoil])

    return XYZ_airfoil, dx, dy
