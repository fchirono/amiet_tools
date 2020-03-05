b"""
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
Implementing Amiet's model for airfoil noise sources (normal incidence gust)
and a model for monopole and dipole source radiation in a uniform flow based on
Chapman's similarity variables.

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
>>> Sources:

--- C Chapman   -   Similarity Variables for Sound Radiation in a Uniform Flow
                    [JSV 233(1), p.157-164, 2000]

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
Author:
Fabio Casagrande Hirono
fch1g10@soton.ac.uk
April 2015
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
"""

import numpy as np
import scipy.special as ss
import mpmath as mp


def H(A):
    """ Calculate the Hermitian conjugate transpose of a matrix 'A' """
    return A.conj().T


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> "Convenience" functions

mpexp = np.vectorize(mp.exp)
mpsqrt = np.vectorize(mp.sqrt)
mperf = np.vectorize(mp.erf)


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


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# --->>> wrappers to create surface pressure cross-spectra matrix

def calc_Sqq(airfoil_tuple, kx, Mach, rho0, turb_tuple, how_many_ky):
    """
    airfoil_tuple = (XYZ_airfoil, dx, dy)
    turb_tuple = (Ux, turb_u_mean2, turb_length_scale, turb_model = {'K', 'L'})
    how_may_ky = {'many', 'few'}
    """

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

    if xy.ndim is 3:
        # unsteady lift over the chord line (mid-span)
        g_x = np.zeros(xy[0][0].shape, 'complex')

        # calculates the unsteady lift over the chord
        g_x = g_LE(xy[0][0], Kx, ky, M, b)

        # broadcasts a copy of 'g_x' to 'delta_p'
        delta_p = g_x[np.newaxis, :]

    elif xy.ndim is 2:
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
    """Analytical solutions for the unsteady lift (G Reboul's thesis - 2010)"""

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
    Analytical solution for unsteady lift integral - supercritical gust
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
    Analytical solution for unsteady lift integral - subcritical gust
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
    if model is 'K':
        ke = (np.sqrt(np.pi)/length_scale)*(ss.gamma(5./6)/ss.gamma(1./3))

        kxe2_ye2 = (kx[:, np.newaxis]/ke)**2 + (ky/ke)**2

        return (4./(9*np.pi))*(u_mean2/(ke**2))*kxe2_ye2/((1+kxe2_ye2)**(7./3))

    # 2D Liepmann turbulence spectrum (Chaitanya's Upgrade)
    elif model is 'L':

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
xs, _ = AmT.chord_sampling(b, N, exp_length=2)

beta = np.sqrt(1-M**2)
mu = Kx*b*M/(beta**2)

plt.figure()
for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.95, 1.05, 1.5, 2., 5]:
    ky = k*beta*mu/b
    g = AmT.g_LE(xs, Kx, ky, M, b)
    plt.plot(xs, np.abs(g))
"""
# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# show non-uniform chordwise sampling
"""
import matplotlib.pyplot as plt

b = 0.075       # airfoil half chord [m]
d = 0.225       # airfoil half span [m]

Nx = 50
length = 2

xn = np.linspace(-length, length, Nx+1)
x = np.exp(xn)

# normalise to [0, 1] interval
x = x-x.min()
x = x/x.max()

# normalise to [-b, b] interval
x = (x*2*b)-b

xn = xn[1:]
xs = x[1:]

# plot sampling
plt.figure()
plt.plot(xn, np.zeros(Nx), 'bs')
plt.plot(xn, xs)
plt.plot(np.zeros(Nx), xs, 'ro')
plt.title('Nx = {}, length = {}'.format(Nx, length))

# plot flat plate grid
Ny = 31
ys = np.linspace(-d, d, Ny)

plt.figure(figsize=(5, 11))
for nx in range(Nx):
    plt.plot((xs[nx], xs[nx]), (ys[0], ys[-1]), 'k')

for ny in range(Ny):
    plt.plot((xs[0], xs[-1]), (ys[ny], ys[ny]), 'k')
plt.axis('equal')
plt.title('Flat plate mesh')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.tight_layout()
"""
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
## plot aerofoil grid
#b = 0.075       # airfoil half chord [m]
#d = 0.225       # airfoil half span [m]
#
#Nx = 50         # number of points sampling the chord (non-uniformly)
#Ny = 101
#
## create airfoil mesh coordinates, and reshape for calculations
#XYZ_airfoil, dx, dy = AmT.create_airf_mesh(b, d, Nx, Ny)
#
#
#plt.figure(figsize=(6, 6))
#for nx in range(Nx):
#    plt.plot((XYZ_airfoil[0, 0, nx], XYZ_airfoil[0, 0, nx]),
#             (XYZ_airfoil[1, 0, nx], XYZ_airfoil[1, -1, nx]),
#             linestyle='-', color='k', linewidth='1')
#for ny in range(Ny//2, Ny):
#    plt.plot((XYZ_airfoil[0, ny, 0], XYZ_airfoil[0, ny, -1]),
#             (XYZ_airfoil[1, ny, 0], XYZ_airfoil[1, ny, -1]), 'k-')
#plt.axis('equal')
#plt.xlim(-1.2*b, 1.2*b)
#plt.ylim(-2.1*b + d, 0.5*b+d)
#plt.xticks([-b, 0, b], [r'$-b$', r'$0$', r'$b$'], fontsize=20)
#plt.yticks([0.5*d, d], [r'$0.5d$', r'$d$'], fontsize=20)
#plt.savefig('Aerofoil_mesh.eps')

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# testing L integral
"""
from matplotlib import pylab


b = 0.075
#xs = np.linspace(-b, b, 200)
Nx = 100
xs, dx = AmT.chord_sampling(b, Nx, 1)

M = 0.3

Kx = np.pi/(b*M)

k0 = Kx*M
ky = 0.

beta = np.sqrt(1-M**2)
mu = Kx*b*M/(beta**2)

Nx = 1001

x = np.linspace(-10, 10, Nx)
y = np.zeros(Nx)
z = np.ones(Nx)*10.

sigma = np.sqrt(x**2 + (beta**2)*(y**2 + z**2))

L_integrand = lambda xs: AmT.g_LE(xs, Kx, ky, M, b)*np.exp(-1j*k0*xs*(M-x/sigma)/(beta**2))
#L_int = (1/b)*mp.quad(L_integrand, [-b, b])

L_int = np.zeros(Nx, 'complex')
for xi in range(xs.shape[0]):
    L_int += (1/b)*L_integrand(xs[xi])*dx[xi]

L_analytical = AmT.L_LE(x, sigma, Kx, ky, M, b)

plt.figure()
plt.plot(x, 10*np.log10(np.abs(L_int)), label='L_int')
plt.plot(x, 10*np.log10(np.abs(L_analytical)), label='L_analytical')
"""
