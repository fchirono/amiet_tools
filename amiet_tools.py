"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono

The 'amiet_tools' (AmT) Python package provides a reference implementation of
Amiet's [JSV 41, 1975] model for turbulence-aerofoil interaction noise with
extensions. These functions allow the calculation of the surface pressure jump
developed over the aerofoil surface (i.e. the acoustic source distribution) in
response to incoming turbulence, and of the acoustic field radiated by the
interaction.

Incoming turbulence can be a single sinusoidal gust, or a sum of incoherent
gusts with amplitudes given by a prescribed energy spectrum.


Dependencies:
    - numpy: array processing for numbers, strings, records, and objects;
    - scipy: scientific library;
    - mpmath: library for arbitrary-precision floating-point arithmetic.


All dependencies are already included in the Anaconda Python Distribution, a
free and open source distribution of Python. Anaconda 4.8.2 (with Python 3.7)
was used to develop and test AmT, and is recommended for using AmT.


Author:
    Fabio Casagrande Hirono - fchirono@gmail.com


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
"""

import numpy as np
import scipy.special as ss
import scipy.optimize as so     # for shear layer correction functions
import mpmath as mp


def H(A):
    """ Calculate the Hermitian conjugate transpose of a matrix 'A' """
    return A.conj().T


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> "Convenience" functions for vectorizing calculations with mpmath

mpexp = np.vectorize(mp.exp)
mpsqrt = np.vectorize(mp.sqrt)
mperf = np.vectorize(mp.erf)


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> Sound propagation functions

def dipole3D(xyz_source, xyz_obs, k0, dipole_axis='z', flow_param=None,
             far_field=False):
    """
    Calculates a (M, N)-shaped matrix of dipole transfer functions
    between 'N' sources and 'M' observers/microphones at a single frequency.

    Parameters
    ----------
    xyz_source : (3, N) array_like
        Array containing the (x, y, z) coordinates of the 'N' sources.

    xyz_obs : (3, M) array_like
        Array containing the (x, y, z) coordinates of the 'M' observers /
        microphones.

    k0 : float
        Acoustic wavenumber / spatial frequency. Can be obtained from the
        temporal frequency 'f' [in Hz] and the speed of sound 'c0' [in m/s]
        as 'k0 = 2*pi*f/c0'.

    dipole_axis : {'x', 'y', 'z'}
        String indicating the direction aligned with the dipole axis; default
        is 'z'.

    flow_param : (flow_dir = {'x', 'y', 'z'}, Mach = float), optional
        Tuple containing the flow parameters: 'flow_dir' is the flow direction
        and can assume the strings 'x', 'y' or 'z'; 'Mach' is the Mach number
        of the flow, calculated by 'Mach = U/c0', where 'U' is the flow
        velocity [in m/s].

    far_field : boolean, optional
        Boolean variable determining if returns far-field approximation;
        default is 'False'

    Returns
    -------
    G : (M, N) array_like
        Matrix of complex transfer functions between sources and observers.

    Notes
    -----
    To calculate a vector of acoustic pressures 'p' at the observer locations
    as induced by a vector of source strengths 'q' at the source locations,
    simply do

    >>> p = G @ q

    """

    # check if source/observer coordinates have appropriate shape
    # ( i.e. [3, N_points])
    if (xyz_source.ndim == 1):
        xyz_source = np.array([xyz_source]).transpose()

    if (xyz_obs.ndim == 1):
        xyz_obs = np.array([xyz_obs]).transpose()

    # M = xyz_obs.shape[1]
    # N = xyz_source.shape[1]

    # if calculating Greens function for a steady medium (no flow):
    if not flow_param:

        # matrix of mic-source distances (Euclidean distance)
        r = np.sqrt(((xyz_obs[:, :, np.newaxis]
                      - xyz_source[:, np.newaxis, :])**2).sum(0))

        # dictionary with direction-to-axis mapping
        dir_keys = {'x': 0, 'y': 1, 'z': 2}

        # read dictionary entry for 'flow_dir'
        i_dip = dir_keys[dipole_axis]

        # dipole directivity term (cosine)
        dip_cos = (xyz_obs[i_dip, :, np.newaxis]-xyz_source[i_dip])/r

        # matrix of Green's functions
        G_dipole = (1j*k0 + 1/r)*dip_cos*np.exp(-1j*k0*r)/(4*np.pi*r)

        return G_dipole

    # if calculating Greens function for a convected medium :
    else:

        # parse 'flow_param' tuple
        flow_dir, Mach = flow_param

        # flow correction factor
        beta = np.sqrt(1.-Mach**2)

        # dictionary with direction-to-axis mapping
        dir_keys = {'x': 0, 'y': 1, 'z': 2}

        # read dictionary entry for 'flow_dir'
        i_flow = dir_keys[flow_dir]

        # apply '1/beta' factor to all coordinates
        xyz_obsB = xyz_obs/beta
        xyz_sourceB = xyz_source/beta

        # apply extra '1/beta' factor to flow direction
        xyz_obsB[i_flow] = xyz_obsB[i_flow]/beta
        xyz_sourceB[i_flow] = xyz_sourceB[i_flow]/beta

        # matrix of delta-x in flow direction (uses broadcasting)
        xB = xyz_obsB[i_flow, :, np.newaxis] - xyz_sourceB[i_flow, :]

        # matrix of transformed mic-source distances (Euclidean distance)
        rB = np.sqrt(((xyz_obsB[:, :, np.newaxis]
                       - xyz_sourceB[:, np.newaxis, :])**2).sum(0))

        # read dictionary entry for 'flow_dir'
        i_dip = dir_keys[dipole_axis]

        # dipole directivity term (cosine)
        dip_cos = (xyz_obsB[i_dip, :, np.newaxis]-xyz_sourceB[i_dip])/(rB*beta)

        # if using far-field approximation...
        if far_field:
            # matrix of convected far-field greens functions
            sigma = np.sqrt(xyz_obs[0, :, np.newaxis]**2
                            + (beta**2)*(xyz_obs[1, :, np.newaxis]**2
                                         + xyz_obs[2, :, np.newaxis]**2))

            G_dipole = ((1j*k0*xyz_obs[2, :, np.newaxis]/(4*np.pi*(sigma**2)))
                        * np.exp(1j*k0*(Mach*xyz_obs[0, :, np.newaxis]-sigma)
                                 / (beta**2))
                        * np.exp(1j*k0*(xyz_obs[0, :, np.newaxis]-Mach*sigma)
                                 * xyz_source[0]/(sigma*(beta**2)))
                        * np.exp(1j*k0*(xyz_obs[1, :, np.newaxis]/sigma)
                                 *xyz_source[1]))

        else:
            # Matrix of convected Green's functions
            G_dipole = ((1j*k0 + 1./rB)*dip_cos*np.exp(1j*k0*Mach*xB)
                        * np.exp(-1j*k0*rB)/(4*np.pi*(beta**2)*rB))

        return G_dipole


def dipole_shear(XYZ_source, XYZ_obs, XYZ_sl, T_sl, k0, c0, Mach):
    """
    Calculates a (M, N)-shaped matrix of dipole transfer functions
    between 'N' sources and 'M' observers/microphones at a single frequency
    including shear layer refraction effects. The mean flow is assumed to be
    in the '+x' direction with velocity 'Ux = Mach*c0', and the dipoles are
    assumed to be located with their axes in the '+z' direction.


    Parameters
    ----------
    XYZ_source : (3, N) array_like
        Array containing the (x, y, z) coordinates of the 'N' sources.

    XYZ_obs : (3, M) array_like
        Array containing the (x, y, z) coordinates of the 'M' observers /
        microphones.

    XYZ_sl : (3, M, N) array_like
        Array containing the (x, y, z) coordinates of the shear-layer crossing
        point for an acoustic ray leaving the 'n'-th source and reaching
        the 'm'-th observer. Can be obtained from 'ShearLayer_matrix' function.

    T_sl : (3, M, N) array_like
        Array containing the total propagation time for an acoustic ray leaving
        the 'n'-th source and reaching the 'm'-th observer. Can be obtained
        from 'ShearLayer_matrix' function.

    k0 : float
        Acoustic wavenumber / spatial frequency. Can be obtained from the speed
        of sound 'c0' [m/s] and the angular frequency 'omega0' [rad/s] (or
        temporal frequency 'f0' [in Hz]) as 'k0 = 2*np.pi*f0/c0 = omega0/c0'.

    c0 : float
        Speed of sound in free air [m/s].

    Mach : float
        Mean flow Mach number; the flow velocity is 'Ux = Mach*c0'.

    Returns
    -------
    G : (M, N) array_like
        Matrix of complex transfer functions between sources and observers,
        including shear layer refraction effects.

    Notes
    -----
    This code uses a ray-acoustics approximation to predict the refraction
    effects at the shear layer crossing: for every source-observer pair, it
    shoots an acoustic ray that leaves the source, gets refracted at the shear
    layer and reaches the observer. The ray obeys the convected wave equation
    within the convecting region (i.e. before the shear layer), and the
    standard wave equation outside the convecting region.

    To calculate a vector of acoustic pressures 'p' at the observer locations
    as induced by a vector of source strengths 'q' at the source locations,
    simply do

    >>> p = G @ q

    """

    # check if source/observer coordinates have appropriate shape
    # ( i.e. [3, N_points])
    if (XYZ_source.ndim == 1):
        XYZ_source = np.array([XYZ_source]).transpose()

    if (XYZ_obs.ndim == 1):
        XYZ_obs = np.array([XYZ_obs]).transpose()

    # flow-corrected source-to-shear-layer propag distance
    sigma_sl = _sigma(XYZ_sl - XYZ_source[:, np.newaxis, :], Mach)

    # dipole in-flow (cosine) directivity2
    dip_cos = (XYZ_sl[2]-XYZ_source[2, np.newaxis, :])/sigma_sl

    beta2 = 1-Mach**2
    rbar_sl = sigma_sl/beta2

    # geometric shear-layer-to-mic distance
    r_lm = r(XYZ_sl - XYZ_obs[:, :, np.newaxis])

    omega0 = k0*c0

    G_dip_sl = ((1j*k0 + 1/(rbar_sl + r_lm))*dip_cos
                * np.exp(-1j*omega0*T_sl)/(4*np.pi*(sigma_sl + r_lm)))

    return G_dip_sl


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> Shear Layer Correction Functions

def r(x):
    """ cartesian radius """
    return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)


def r_bar(x, Mach):
    """ flow-corrected cartesian radius """
    beta = np.sqrt(1-Mach**2)
    return np.sqrt((x[0]/beta**2)**2 + (x[1]/beta)**2 + (x[2]/beta)**2)


def _sigma(x, Mach):
    """ flow-corrected radius """
    beta2 = 1-Mach**2
    return np.sqrt(x[0]**2 + beta2*(x[1]**2 + x[2]**2))


def t_sound(x1, x2, c0):
    """ propagation time for sound in steady medium """
    return r(x1-x2)/c0


def t_convect(x1, x2, Ux, c0):
    """
    Propagation time for sound in convected medium, flow in 'x' direction
    """
    Mach = Ux/c0
    beta2 = 1-Mach**2  # beta squared
    return (-(x2[0]-x1[0])*Mach + _sigma(x2-x1, Mach))/(c0*beta2)


def t_total(x_layer, x_source, x_mic, Ux, c0):
    """ total propagation time through shear layer """
    return t_convect(x_source, x_layer, Ux, c0) + t_sound(x_layer, x_mic, c0)


def zeta(x, Mach):
    return np.sqrt((1-Mach*x[0]/r(x))**2 - (x[0]**2 + x[1]**2)/(r(x)**2))


def constr_xl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0):
    """
    Shear layer solver constraint in 'xl'
    """
    Mach = Ux/c0
    beta2 = 1-Mach**2

    return np.abs((XYZ_sl[0]-XYZ_s[0])/_sigma(XYZ_sl-XYZ_s, Mach) - Mach
                  - beta2*(XYZ_m[0]-XYZ_sl[0])/r(XYZ_m-XYZ_sl))


def constr_yl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0):
    """
    Shear layer solver constraint in 'yl'
    """
    Mach = Ux/c0

    return np.abs((XYZ_sl[1]-XYZ_s[1])/_sigma(XYZ_sl-XYZ_s, Mach)
                  - (XYZ_m[1]-XYZ_sl[1])/r(XYZ_m-XYZ_sl))


def ShearLayer_X(XYZ_s, XYZ_m, Ux, c0, z_sl):
    """
    Calculates the propagation time of an acoustic ray emitted from a source
    at 'xs' to a microphone at 'xm' in two stages:

        - through a mean flow in the '+x' direction, with Mach number
        'Mach = Ux/c0', at height 'z_sl', up to the shear layer crossing point
        'xL';

        - and through a steady medium from the shear layer to the microphone
        position.

    Returns the shear layer crossing point 'xL' that minimises the total
    propagation time.
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
    XYZ_sl = XYZ_sl_opt.x

    return XYZ_sl


def ShearLayer_Corr(XYZ_s, XYZ_sl, XYZ_m, Ux, c0):
    """
    Calculates a corrected position for a microphone measuring sound through
    a shear layer using ray acoustics.

    The corrected position is defined as that the acoustic ray would have
    reached if there was no shear layer (i.e. flow everywhere).

    The distance is corrected to that 'r(xc-xr) = r(xm-xs)' - see Amiet for
    details.
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


def ShearLayer_matrix(XYZ_s, XYZ_o, z_sl, Ux, c0):
    """ Returns two matrices containing the propagation times and the shear
    layer crossing points for each source-observer pair.

    This is a convenience function, essentially two nested 'for'-loops around
    the 'ShearLayer_X' and 'ShearLayer_t' functions."""

    # ensure source/observer coordinates have appropriate shape
    # ( i.e. [3, N_points])
    if (XYZ_s.ndim == 1):
        XYZ_s = np.array([XYZ_s]).transpose()

    if (XYZ_o.ndim == 1):
        XYZ_o = np.array([XYZ_o]).transpose()

    # check if shear layer is located between sources and obs
    sl_height_error = "Shear layer is not located between all sources and observers"
    assert (np.prod(np.sign(XYZ_o[2] - z_sl))
            == np.prod(np.sign(z_sl - XYZ_s[2]))), sl_height_error

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


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>>

def chord_sampling(b, N=200, exp_length=2):
    """
    Calculates 'N' points non-uniformly sampled over the half-open
    interval (-b, b], with the leading edge being at '-b'.

    The code calculates the exponential function over an interval
    [-exp_length/2, exp_length/2] with 'N' points, which is remapped to the
    actual chord interval [-b, b] and the leading edge then removed. This type
    of sampling assigns more points around the leading edge.

    Lower values of 'exp_length' provide more uniform sampling.
    """

    # calculates exponential curve
    x = np.exp(np.linspace(-exp_length/2, exp_length/2, N+1))

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


def create_airf_mesh(b, d, Nx, Ny):
    """ Creates the mesh containing the airfoil surface points. The 'z'
    coordinate is set to always be zero.

    The final array has shape (3, Ny, Nx). """

    x_airfoil, dx = chord_sampling(b, Nx)

    y_airfoil = np.linspace(-d, d, Ny)
    dy = y_airfoil[1] - y_airfoil[0]

    XY_airfoil = np.meshgrid(x_airfoil, y_airfoil)
    Z_airfoil = np.zeros(XY_airfoil[0].shape)
    XYZ_airfoil = np.array([XY_airfoil[0], XY_airfoil[1], Z_airfoil])

    return XYZ_airfoil, dx, dy


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> Other assorted functions

def read_ffarray_lvm(filename):
    """
    Reads .lvm file from LabView: the first column contains the time samples,
    and the remaining columns contain each mic signal; also removes the mean.
    """

    # number of columns: time vector + 12 mic vectors
    n_columns = 13

    # open file as text
    lvm_file = open(filename, 'r')

    # count the number of lines (has to read the whole file,
    # hence closing and opening again)
    n_lines = len(list(lvm_file))
    lvm_file.close()

    t = np.zeros(n_lines)
    mics = np.zeros((n_lines, n_columns-1))

    # read line ignoring the '\r\n' at the end,
    # and using '\t' as column separators
    lvm_file = open(filename, 'r')
    for line in range(n_lines):
        current_line = lvm_file.readline().split('\r\n')[0].split('\t')
        t[line] = float(current_line[0])
        mics[line, :] = [float(x) for x in current_line][1:]

    # close file
    lvm_file.close()

    return t, mics.T


def index_log(index_init, index_final, N):
    """
    Returns a 1D Numpy array containing (approximately) 'N' indices for
    accessing a linearly-spaced vector in a (roughly) logarithmically-spaced
    fashion.

    Given the necessary rounding of the resulting values to integers, some
    lower indices may appear more than once; however, only unique indices are
    returned, which in turn often does not result in exactly 'N' indices.
    """

    log10_init = np.log10(index_init)
    log10_final = np.log10(index_final)

    index_all = (np.logspace(log10_init, log10_final, N)).round().astype('int')

    return np.unique(index_all)


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> wrappers to create surface pressure cross-spectra matrix

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#           >>> DO NOT USE - UNDER DEVELOPMENT! <<<
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
"""
def calc_Sqq(airfoil_tuple, kx, Mach, rho0, turb_tuple, how_many_ky):

    #airfoil_tuple = (XYZ_airfoil, dx, dy)
    #turb_tuple = (Ux, turb_u_mean2, turb_length_scale, turb_model = {'K', 'L'})
    #how_many_ky = {'many', 'few'}


    beta = np.sqrt(1-Mach**2)

    # untuple turbulence properties
    Ux, turb_u_mean2, turb_length_scale, turb_model = turb_tuple

    # untuple airfoil geometry
    XYZ_airfoil, dx, dy = airfoil_tuple
    Ny, Nx = XYZ_airfoil.shape[1:]
    d = XYZ_airfoil[1, -1, 0]   # airfoil semi span
    b = XYZ_airfoil[0, 0, -1]   # airfoil semi chord

    # critical gusts
    ky_crit = kx*Mach/beta

    # period of sin in sinc
    ky_T = 2*np.pi/d

    # integrating ky with many points
    if how_many_ky is 'many':

        # 'low freq'
        if ky_crit < 2*np.pi/d:
            N_ky = 41
            Ky = np.linspace(-ky_T, ky_T, N_ky)

        # 'high freq'
        else:
            # count how many sin(ky*d) periods in Ky range
            N_T = 2*ky_crit/ky_T
            N_ky = np.int(np.ceil(N_T*20)) + 1      # use 20 points per period
            Ky = np.linspace(-ky_crit, ky_crit, N_ky)

    # integrating ky with few points
    elif how_many_ky is 'few':
        # 'low freq'
        if ky_crit < 2*np.pi/d:
            N_ky = 5
            Ky = np.linspace(-ky_T, ky_T, N_ky)

        # ' high freq'
        else:
            # count how many sin(ky*d) periods in Ky range
            N_T = 2*ky_crit/ky_T
            N_ky = np.int(np.ceil(N_T*2))          # use 2 points per period
            Ky = np.linspace(-ky_crit, ky_crit, N_ky)

    dky = Ky[1]-Ky[0]

    # create turbulence wavenumber spectrum
    Phi2 = Phi_2D(kx, Ky, turb_u_mean2, turb_length_scale, turb_model)[0]

    Sqq = np.zeros((Nx*Ny, Nx*Ny), 'complex')

    # for every gust...
    for kyi in range(Ky.shape[0]):
        # sinusoidal gust peak value
        w0 = np.sqrt(Phi2[kyi])

        # Pressure 'jump' over the airfoil
        delta_p1 = delta_p(rho0, b, w0, Ux, kx, Ky[kyi], XYZ_airfoil[0:2],
                           Mach)

        # reshape and reweight for vector calculation
        delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

        # add cross-product to source amplitude CSM
        Sqq += np.outer(delta_p1_calc, delta_p1_calc.conj())*(Ux)*dky

    return Sqq


def calc_Spp(airfoil_tuple, kx, Mach, rho0, turb_tuple, how_many_ky, G):

    Sqq = calc_Sqq(airfoil_tuple, kx, Mach, rho0, turb_tuple, how_many_ky)

    return (G @ Sqq @ H(G))*4*np.pi
"""

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> Aeroacoustic Functions

def fr_integrand(x):
    """ Creates the argument to the Fresnel integral."""

    return mp.exp(1j*x)/mp.sqrt(x)


def fr_integrand_cc(x):
    """ Creates the argument to the complex conjugate Fresnel integral."""

    return mp.exp(-1j*x)/mp.sqrt(x)


def fr_int(zeta):
    """ Calculates the Fresnel integral, as defined in Blandeau (2011) -
    eq. 2.84. """

    # Check if zeta is array or float
    if type(zeta) is np.ndarray:
        E_conj = np.zeros(zeta.shape, 'complex')

        for i in range(zeta.size):
            E_conj[i] = ((1/np.sqrt(2*np.pi))
                         * (mp.quad(fr_integrand, [0, zeta[i]])))
    else:
        E_conj = ((1/np.sqrt(2*np.pi))
                  * (mp.quad(fr_integrand, [0, zeta])))

    return E_conj


def fr_int_cc(zeta):
    """ Calculates the complex conjugate of the Fresnel integral, as defined in
    Blandeau (2011) - eq. 2.84. """

    # Check if zeta is array or float
    if type(zeta) is np.ndarray:
        E_conj = np.zeros(zeta.shape, 'complex')

        for i in range(zeta.size):
            E_conj[i] = ((1/np.sqrt(2*np.pi))
                         * (mp.quad(fr_integrand_cc, [0, zeta[i]])))
    else:
        E_conj = ((1/np.sqrt(2*np.pi))
                  * (mp.quad(fr_integrand_cc, [0, zeta])))

    return E_conj


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def delta_p(rho0, b, w0, Ux, Kx, ky, xy, M):
    """ Calculates the pressure 'jump' across the airfoil from the airfoil
    response function - see Blandeau (2011), eq. 2.19. """

    # pressure difference over the whole airfoil surface
    delta_p = np.zeros(xy[0].shape, 'complex')

    if xy.ndim == 3:
        # unsteady lift over the chord line (mid-span)
        g_x = np.zeros(xy[0][0].shape, 'complex')

        # calculates the unsteady lift over the chord
        g_x = g_LE(xy[0][0], Kx, ky, M, b)

        # broadcasts a copy of 'g_x' to 'delta_p'
        delta_p = g_x[np.newaxis, :]

    elif xy.ndim == 2:
        # unsteady lift over the chord line (mid-span)
        g_x = np.zeros(xy[0].shape, 'complex')

        # calculates the unsteady lift over the chord
        delta_p = g_LE(xy[0], Kx, ky, M, b)

    # adds the constants and the 'k_y' oscillating component
    delta_p = 2*np.pi*rho0*w0*delta_p*np.exp(-1j*ky*xy[1])

    return delta_p


def g_LE(xs, Kx, ky, M, b):
    """
    Implements Amiet leading edge airfoil response function for an incoming
    turbulent gust (G Reboul's thesis - 2010)

    Checks for percentage difference between 'abs(ky)' and 'ky_critical' to
    define a gust as either subcritical (%diff < 1e-6), supercritical
    """

    beta = np.sqrt(1-M**2)
    ky_critical = Kx*M/beta

    # p_diff < 0: supercritical
    # p_diff > 0: subcritical
    p_diff = (np.abs(ky) - ky_critical)/ky_critical

    # supercritical gusts
    if p_diff < -1e-3:
        g = g_LE_super(xs, Kx, ky, M, beta, b)

    # subcritical gusts
    elif p_diff > 1e-3:
        g = g_LE_sub(xs, Kx, ky, M, beta, b)

    # critical gusts (interpolate between super- and subcritical)
    else:

        # get gusts 1% above and below critical ky
        ky_sp = ky*0.99
        ky_sb = ky*1.01

        g_sp = g_LE_super(xs, Kx, ky_sp, M, beta, b)
        g_sb = g_LE_sub(xs, Kx, ky_sb, M, beta, b)

        g = (g_sp + g_sb)/2.

    # if single mp_complex, convert to float
    if type(g) is mp.ctx_mp_python.mpc:
        # convert to single float
        return float(g.real) + 1j*float(g.imag)

    # if single float, return as is
    elif type(g) is np.complex128:
        return g

    # if array of mp_complex, convert to np.ndarray
    elif type(g) is np.ndarray and g.dtype is np.dtype(mp.ctx_mp_python.mpc):
        return np.array([float(x.real) + 1j*float(x.imag) for x in g])

    # if array of complex floats, just return as is
    elif type(g) is np.ndarray and g.dtype is np.dtype(np.complex128):
        return g


def g_LE_super(xs, Kx, ky, M, beta, b):
    """
    Implements Amiet leading edge airfoil response for supercritical gusts.

    For a gust to be supercritical, it must obey that

        abs(ky) < Kx*M/beta

    """

    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*M

    kappa = np.sqrt(mu_a**2 - (ky*b/beta)**2)

    g1_sp = (np.exp(-1j*((kappa - mu_a*M)*((xs/b) + 1) + np.pi/4))
             / (np.pi*np.sqrt(np.pi*((xs/b) + 1)*(Kx*b + (beta**2)*kappa))))

    g2_sp = -(np.exp(-1j*((kappa - mu_a*M)*((xs/b) + 1) + np.pi/4))
              * (1-(1+1j)*fr_int_cc(2*kappa*(1-xs/b)))
              / (np.pi*np.sqrt(2*np.pi*(Kx*b + (beta**2)*kappa))))

    return g1_sp + g2_sp


def g_LE_sub(xs, Kx, ky, M, beta, b):
    """
    Implements Amiet leading edge airfoil response for subcritical gusts.

    For a gust to be supercritical, it must obey that

        abs(ky) > Kx*M/beta

    """

    beta = np.sqrt(1-M**2)
    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*M

    kappa1 = np.sqrt(((ky*b/beta)**2) - mu_a**2)

    g1_sb = (np.exp((-kappa1 + 1j*mu_a*M)*((xs/b) + 1))*np.exp(-1j*np.pi/4)
             / (np.pi*np.sqrt(np.pi*((xs/b) + 1)
                              * (Kx*b - 1j*(beta**2)*kappa1))))

    g2_sb = -(np.exp((-kappa1 + 1j*mu_a*M)*((xs/b) + 1))
              * np.exp(-1j*np.pi/4)*(1 - ss.erf(2*kappa1*(1-xs/b)))
              / (np.pi*np.sqrt(2*np.pi*(Kx*b - 1j*(beta**2)*kappa1))))

    return g1_sb + g2_sb


def ky_att(xs, b, M, k0, Att=-20):
    """
    For a given chord point 'xs', Mach number 'M', ac wavenumber 'k0' and
    attenuation 'Att' [in dB], calculates the subcritical gust spanwise
    wavenumber 'ky_att' (> 'ky_crit' by definition) such that the aerofoil
    response at that point is 'Att' dB reduced.
    """

    beta = np.sqrt(1-M**2)

    # critical gust spanwise wavenumber
    ky_crit = k0/beta

    term1 = -(beta**2)*np.log(10**(Att/20))/(k0*(xs + b))

    return ky_crit*np.sqrt(term1**2 + 1)


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def L_LE(x, sigma, Kx, ky, M, b):
    """Effective lift functions (G Reboul's thesis - 2010)"""

    beta = np.sqrt(1-M**2)
    ky_critical = Kx*M/beta

    # percentage difference in ky
    # p_diff < 0: supercritical / p_diff > 0: subcritical
    p_diff = (np.abs(ky) - ky_critical)/ky_critical

    # supercritical gusts
    if p_diff < -1e-3:
        L = L_LE_super(x, sigma, Kx, ky, M, b)

    # subcritical gusts
    elif p_diff > 1e-3:
        L = L_LE_sub(x, sigma, Kx, ky, M, b)

    # critical gusts (interpolate between super- and subcritical)
    else:
        # get gusts 1% above and below critical ky
        ky_sp = ky*0.99
        ky_sb = ky*1.01

        L_sp = L_LE_super(x, sigma, Kx, ky_sp, M, b)
        L_sb = L_LE_sub(x, sigma, Kx, ky_sb, M, b)

        L = (L_sp + L_sb)/2.

    # if single mp_complex, convert to float
    if type(L) is mp.ctx_mp_python.mpc:
        # convert to single float
        return float(L.real) + 1j*float(L.imag)

    # if single float, return as is
    elif type(L) is np.complex128:
        return L

    # if array of mp_complex, convert to np.ndarray
    elif type(L) is np.ndarray and L.dtype is np.dtype(mp.ctx_mp_python.mpc):
        return np.array([float(x.real) + 1j*float(x.imag) for x in L])

    # if array of complex floats, just return as is
    elif type(L) is np.ndarray and L.dtype is np.dtype(np.complex128):
        return L


def L_LE_super(x, sigma, Kx, Ky, M, b):
    """
    Effective lift integral - supercritical gust
    """

    beta = np.sqrt(1-M**2)
    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*M

    kappa = np.sqrt(mu_a**2 - (Ky*b/beta)**2)
    H1 = kappa - mu_a*x/sigma
    H2 = mu_a*(M - x*sigma) - np.pi/4

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


def L_LE_sub(x, sigma, Kx, Ky, M, b):
    """
    Effective lift integral - subcritical gust
    """

    beta = np.sqrt(1-M**2)
    mu_h = Kx*b/(beta**2)
    mu_a = mu_h*M

    kappa1 = np.sqrt((Ky*b/beta)**2 - (mu_a**2))
    H2 = mu_a*(M - x*sigma) - np.pi/4
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


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# turbulent velocity spectra

def Phi_2D(kx, ky, u_mean2, length_scale, model='K'):
    """2D isotropic turbulence spectrum.

    'model' can be 'K' for von Karman spectrum, or 'L' for Liepmann
    spectrum."""

    if type(kx) is not np.ndarray:
        kx = np.asarray([kx])

    # von Karman model (Amiet 1975)
    if model == 'K':
        ke = (np.sqrt(np.pi)/length_scale)*(ss.gamma(5./6)/ss.gamma(1./3))

        kxe2_ye2 = (kx[:, np.newaxis]/ke)**2 + (ky/ke)**2

        return (4./(9*np.pi))*(u_mean2/(ke**2))*kxe2_ye2/((1+kxe2_ye2)**(7./3))

    # 2D Liepmann turbulence spectrum (Chaitanya's Upgrade)
    elif model == 'L':

        ls2 = length_scale**2

        return ((u_mean2*ls2/(4*np.pi))
                * ((1+ls2*(4*kx[:, np.newaxis]**2 + ky**2)))
                / (1+ls2*(kx[:, np.newaxis]**2 + ky**2))**(5./2))


def Phi_1D(kx, u_mean2, length_scale):
    """1D von Karman isotropic turbulence model (Paterson & Amiet 1976)"""

    ke = (np.sqrt(np.pi)/length_scale)*(ss.gamma(5./6)/ss.gamma(1./3))
    kxe2 = (kx/ke)**2

    return ((u_mean2*length_scale/(2*np.pi))*(1+8.*kxe2/3)
            / ((1+kxe2)**(11./6)))


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Reproduces Figure 6 from M Roger book chapter (unsteady lift)
"""
import matplotlib.pyplot as plt

N = 201


b = 1
M = 0.3
Kx = np.pi/(b*M)
xs, _ = chord_sampling(b, N, exp_length=2)

beta = np.sqrt(1-M**2)
mu = Kx*b*M/(beta**2)

plt.figure()
for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.95, 1.05, 1.5, 2., 5]:
    ky = k*beta*mu/b
    g = g_LE(xs, Kx, ky, M, b)
    plt.plot(xs, np.abs(g))
"""

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# show non-uniform chordwise sampling

"""
import matplotlib.pyplot as plt
b = 0.075       # airfoil half chord [m]
d = 0.225       # airfoil half span [m]

Nx = 50         # number of points sampling the chord (non-uniformly)
Ny = 101

# create airfoil mesh coordinates, and reshape for calculations
XYZ_airfoil, dx, dy = create_airf_mesh(b, d, Nx, Ny)


plt.figure(figsize=(6, 6))
for nx in range(Nx):
    plt.plot((XYZ_airfoil[0, 0, nx], XYZ_airfoil[0, 0, nx]),
            (XYZ_airfoil[1, 0, nx], XYZ_airfoil[1, -1, nx]),
            linestyle='-', color='k', linewidth='1')
for ny in range(Ny//2, Ny):
    plt.plot((XYZ_airfoil[0, ny, 0], XYZ_airfoil[0, ny, -1]),
            (XYZ_airfoil[1, ny, 0], XYZ_airfoil[1, ny, -1]), 'k-')
plt.axis('equal')
plt.xlim(-1.2*b, 1.2*b)
plt.ylim(-2.1*b + d, 0.5*b+d)
plt.xticks([-b, 0, b], [r'$-b$', r'$0$', r'$b$'], fontsize=20)
plt.yticks([0.5*d, d], [r'$0.5d$', r'$d$'], fontsize=20)
#plt.savefig('Aerofoil_mesh.eps')
"""
