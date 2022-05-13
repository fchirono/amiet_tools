"""Author: Fabio Casagrande Hirono"""
import numpy as np
import scipy.optimize as so     # for shear layer correction functions


def r(x):
    """
    Cartesian radius of a point 'x' in 3D space

    Parameters
    ----------
    x : (3,) array_like
        1D vector containing the (x, y, z) coordinates of a point.

    Returns
    -------
    r : float
        Radius of point 'x' relative to origin of coordinate system
    """
    return np.sqrt((x[0]**2) + (x[1]**2) + (x[2]**2))


def r_bar(x, Mach):
    """
    Flow-transformed cartesian radius of a point 'x' in 3D space, with mean
    flow in the '+x' direction.

    Parameters
    ----------
    x : (3,) array_like
        1D vector containing the (x, y, z) coordinates of a point.

    Mach : float
        Mach number of the mean flow (assumed in '+x' direction).

    Returns
    -------
    r_bar : float
        Flow-transformed radius of point 'x' relative to origin of coordinate
        system.

    Notes
    -----
    This function considers full flow-transformed coordinates [Chapman, JSV
    233, 2000] to calculate the output.
    """
    beta = np.sqrt(1-Mach**2)
    return np.sqrt(((x[0]/beta**2)**2) + ((x[1]/beta)**2) + ((x[2]/beta)**2))


def sigma_(x, Mach):
    """
    Flow-corrected cartesian radius of a point 'x' in 3D space, with mean flow
    in the '+x' direction.

    Parameters
    ----------
    x : (3,) array_like
        1D vector containing the (x, y, z) coordinates of a point.

    Mach : float
        Mach number of the mean flow (assumed in '+x' direction).

    Returns
    -------
    sigma : float
        Flow-corrected radius of point 'x' relative to origin of coordinate
        system

    Notes
    -----
    This function does NOT considers full flow-transformed coordinates
    [Chapman, JSV 233, 2000].
    """
    beta2 = 1-(Mach**2)
    return np.sqrt((x[0]**2) + beta2*((x[1]**2) + (x[2]**2)))


def t_sound(x1, x2, c0):
    """
    Calculates the time taken for sound to move from points 'x1' to 'x2' at
    speed 'c0'.

    Parameters
    ----------
    x1 : (3,) array_like
        1D vector containing the (x, y, z) coordinates of initial point.

    x2 : (3,) array_like
        1D vector containing the (x, y, z) coordinates of final point.

    c0 : float
        Speed of sound.

    Returns
    -------
    t : float
        Time taken for sound to travel from point 'x1' to point 'x2' at speed
        'c0'.
    """
    return r(x1-x2)/c0


def t_convect(x1, x2, Ux, c0):
    """
    Propagation time for sound in convected medium, with mean flow in the '+x'
    direction.

    Parameters
    -----
    x1 : (3,) array_like
        1D vector containing the (x, y, z) coordinates of initial point.

    x2 : (3,) array_like
        1D vector containing the (x, y, z) coordinates of final point.

    Ux : float
        Mean flow velocity, assumed in '+x' direction.

    c0 : float
        Speed of sound.

    Returns
    -------
    t_convect : float
        Time taken for sound to travel from point 'x1' to point 'x2' at speed
        'c0' while subject to convection effects in the '+x' direction.
    """
    Mach = Ux/c0
    beta2 = 1-Mach**2  # beta squared

    return (-(x2[0]-x1[0])*Mach + sigma_(x2-x1, Mach))/(c0*beta2)


def t_total(x_layer, x_source, x_mic, Ux, c0):
    """
    Total propagation time for sound to move from source to mic and through a shear layer.

    Parameters
    ----------
    x_layer : (3,) array_like
        1D vector containing the (x, y, z) coordinates of shear layer crossing
        point.

    x_source : (3,) array_like
        1D vector containing the (x, y, z) coordinates of final point.

    x_mic : (3,) array_like
        1D vector containing the (x, y, z) coordinates of final point.

    Ux : float
        Mean flow velocity, assumed in '+x' direction.

    c0 : float
        Speed of sound.

    Returns
    -------
    t_total : float
        Time taken for sound to travel from source, through shear layer, to mic.

    Notes
    -----
    The shear layer crossing point 'x_layer' can be obtained from
    'amiet_tools.ShearLayer_X' function.
    """
    return t_convect(x_source, x_layer, Ux, c0) + t_sound(x_layer, x_mic, c0)


def constr_xl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0):
    """
    Shear layer solver constraint in 'xl'
    """
    Mach = Ux/c0
    beta2 = 1-Mach**2

    return np.abs((XYZ_sl[0]-XYZ_s[0])/sigma_(XYZ_sl-XYZ_s, Mach) - Mach
                  - beta2*(XYZ_m[0]-XYZ_sl[0])/r(XYZ_m-XYZ_sl))


def constr_yl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0):
    """
    Shear layer solver constraint in 'yl'
    """
    Mach = Ux/c0

    return np.abs((XYZ_sl[1]-XYZ_s[1])/sigma_(XYZ_sl-XYZ_s, Mach)
                  - (XYZ_m[1]-XYZ_sl[1])/r(XYZ_m-XYZ_sl))


def ShearLayer_X(XYZ_s, XYZ_m, Ux, c0, z_sl):
    """
    Calculates the shear layer crossing point of an acoustic ray emitted from a
    source at 'xs' to a microphone at 'xm'.

    Parameters
    ----------
    XYZ_s : (3,) array_like
        1D vector containing the (x, y, z) coordinates of source (within mean flow)

    XYZ_m : (3,) array_like
        1D vector containing the (x, y, z) coordinates of microphone (outside mean flow)

    Ux : float
        Mean flow velocity, assumed in '+x' direction.

    c0 : float
        Speed of sound.

    z_sl : float
        Shear layer height.

    Returns
    -------
    XYZ_sl : (3,) array_like
        1D vector containing the (x, y, z) coordinates of the shear layer
        crossing point.

    Notes
    -----
    This code uses a numerical minimization routine to calculate the shear
    layer crossing point 'XYZ_sl' that minimizes the total travel time that an
    acoustic ray takes to propagate from a source point 'XYZ_s' within the
    mean flow to a microphone 'XYZ_m' outside the mean flow.
    """
    # optimization constraints
    cons = ({'type': 'eq', 'fun': lambda XYZ_sl: XYZ_sl[2]-z_sl},
            {'type': 'eq', 'fun':
             lambda XYZ_sl: constr_xl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0)},
            {'type': 'eq', 'fun':
             lambda XYZ_sl: constr_yl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0)})

    # initial guess (straight line between source and mic)
    XYZ_0 = ((XYZ_m + XYZ_s)*((z_sl-XYZ_s[2])/(XYZ_m[2] - XYZ_s[2])))

    # optimize and get result
    XYZ_sl_opt = so.minimize(t_total, XYZ_0, args=(XYZ_s, XYZ_m, Ux, c0),
                             method='SLSQP', constraints=cons)
    return XYZ_sl_opt.x


def ShearLayer_matrix(XYZ_s, XYZ_o, z_sl, Ux, c0):
    """
    Returns two matrices containing the propagation times and the shear
    layer crossing points for each source-observer pair.

    Parameters
    ----------
    XYZ_s : (3, N) array_like
        2D vector containing the (x, y, z) coordinates of 'N' source points
        (within mean flow)

    XYZ_o : (3, M) array_like
        2D vector containing the (x, y, z) coordinates of 'M' observer points
        (outside the mean flow)

    z_sl : float
        Shear layer height.

    Ux : float
        Mean flow velocity, assumed in '+x' direction.

    c0 : float
        Speed of sound.

    Returns
    -------
    T : (M, N) array_like
        2D array containing the total propagation times of the acoustic signal
        from each source 'n' to each observer 'm'.

    XYZ_sl : (3, M, N) array_like
        3D array containing the (x, y, z) coordinates of the shear layer
        crossing point for each source 'n' to each observer 'm'.

    Notes
    -----
    This is a convenience function, essentially two nested 'for'-loops around
    the 'ShearLayer_X' and 'ShearLayer_t' functions.
    """
    # ensure source/observer coordinates have appropriate shape
    # ( i.e. [3, N_points])
    if XYZ_s.ndim == 1:
        XYZ_s = np.array([XYZ_s]).transpose()

    if XYZ_o.ndim == 1:
        XYZ_o = np.array([XYZ_o]).transpose()

    if np.prod(np.sign(XYZ_o[2] - z_sl)) != np.prod(np.sign(z_sl - XYZ_s[2])):
        # check if shear layer is located between sources and obs
        sl_height_error = "Shear layer is not located between all sources and observers"
        raise ValueError(sl_height_error)

    # number of obs and sources
    M = XYZ_o.shape[1]
    N = XYZ_s.shape[1]

    XYZ_sl = np.zeros((3, M, N))     # shear layer crossing point
    T = np.zeros((M, N))            # propag time

    for n in range(N):
        for m in range(M):
            # shear layer crossing point
            XYZ_sl[:, m, n] = ShearLayer_X(XYZ_s[:, n], XYZ_o[:, m], Ux, c0,
                                           z_sl)
            # total propag time
            T[m, n] = t_total(XYZ_sl[:, m, n], XYZ_s[:, n],
                              XYZ_o[:, m], Ux, c0)

    return T, XYZ_sl


def ShearLayer_Corr(XYZ_s, XYZ_sl, XYZ_m, Ux, c0):
    """
    Calculates a corrected position for a microphone measuring sound through
    a shear layer using Amiet's method [JSV 58, 1978].

    Parameters
    ----------
    x_layer : (3,) array_like
        1D vector containing the (x, y, z) coordinates of shear layer crossing
        point.

    x_source : (3,) array_like
        1D vector containing the (x, y, z) coordinates of final point.

    x_mic : (3,) array_like
        1D vector containing the (x, y, z) coordinates of final point.

    Ux : float
        Mean flow velocity, assumed in '+x' direction.

    c0 : float
        Speed of sound.

    Returns
    -------
    t_total : float
        Time taken for sound to travel from source, through shear layer, to mic.

    Notes
    -----
    The corrected position is calculated with Amiet's shear layer correction
    method [JSV 58, 1978], and is defined as the point the acoustic ray would
    have reached if there was no shear layer (i.e. flow everywhere). The
    distance is corrected so that 'r(xc-xr) = r(xm-xs)'.
    """
    # calculate travel time inside flow (source to shear layer)
    tf = t_convect(XYZ_s, XYZ_sl, Ux, c0)

    # determine ray phase velocity in flow (direction and magnitude)
    cp_ray = (XYZ_sl-XYZ_s)/tf

    # travel time for corrected position
    tc = r(XYZ_m-XYZ_s)/c0

    # corrected position
    XYZ_c = XYZ_s + cp_ray*tc

    # retarded source position
    XYZ_r = XYZ_s + np.array([Ux, 0, 0])*tc

    return XYZ_c, XYZ_r
