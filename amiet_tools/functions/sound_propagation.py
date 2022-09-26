"""Author: Fabio Casagrande Hirono"""
import numpy as np
from .shear_layer import r, sigma_


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
    if xyz_source.ndim == 1:
        xyz_source = np.array([xyz_source]).transpose()

    if xyz_obs.ndim == 1:
        xyz_obs = np.array([xyz_obs]).transpose()

    # dictionary with direction-to-axis mapping
    dir_keys = {'x': 0, 'y': 1, 'z': 2}

    # M = xyz_obs.shape[1]
    # N = xyz_source.shape[1]

    # if calculating Greens function for a steady medium (no flow):
    if not flow_param:

        # matrix of mic-source distances (Euclidean distance)
        r = np.sqrt(((xyz_obs[:, :, np.newaxis]
                      - xyz_source[:, np.newaxis, :])**2).sum(0))

        # read dictionary entry for 'flow_dir'
        i_dip = dir_keys[dipole_axis]

        # dipole directivity term (cosine)
        dip_cos = (xyz_obs[i_dip, :, np.newaxis]-xyz_source[i_dip])/r

        # matrix of Green's functions
        G_dipole = (1j*k0 + 1/r)*dip_cos*np.exp(-1j*k0*r)/(4*np.pi*r)

    else:

        # parse 'flow_param' tuple
        flow_dir, Mach = flow_param

        # flow correction factor
        beta = np.sqrt(1.-Mach**2)

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
                                 * xyz_source[1]))

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
        the 'm'-th observer.

    T_sl : (3, M, N) array_like
        Array containing the total propagation time for an acoustic ray leaving
        the 'n'-th source and reaching the 'm'-th observer.

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

    The 'XYZ_sl' and the 'T_sl' variables can be obtained from the
    'amiet_tools.ShearLayer_matrix' function.

    To calculate a vector of acoustic pressures 'p' at the observer locations
    as induced by a vector of source strengths 'q' at the source locations,
    simply do

    >>> p = G @ q
    """
    # check if source/observer coordinates have appropriate shape
    # ( i.e. [3, N_points])
    if XYZ_source.ndim == 1:
        XYZ_source = np.array([XYZ_source]).transpose()

    if XYZ_obs.ndim == 1:
        XYZ_obs = np.array([XYZ_obs]).transpose()

    # flow-corrected source-to-shear-layer propag distance
    sigma_sl = sigma_(XYZ_sl - XYZ_source[:, np.newaxis, :], Mach)

    # dipole in-flow (cosine) directivity2
    dip_cos = (XYZ_sl[2]-XYZ_source[2, np.newaxis, :])/sigma_sl

    beta2 = 1-Mach**2
    rbar_sl = sigma_sl/beta2

    # geometric shear-layer-to-mic distance
    r_lm = r(XYZ_sl - XYZ_obs[:, :, np.newaxis])

    omega0 = k0*c0

    return ((1j*k0 + 1/(rbar_sl + r_lm))*dip_cos
            * np.exp(-1j*omega0*T_sl)/(4*np.pi*(sigma_sl + r_lm)))
