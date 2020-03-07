"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


Test script 2: calculate radiated acoustic pressure for an observer at 90deg
above aerofoil, at two distances (1.2m and 15m), and compare results for:

    a) near-field, multiple gusts model (pre-calculated);
    b) near-field, single gust (ky=0) model;
    c) far-field, single gust model.


This test case requires an external file 'Spp_SumKy_NearFar.mat', containing
the microphone PSD pre-calculated over multiple gusts (see file
'Create_Spp_SumKy.py').


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np
from scipy.io import loadmat

import amiet_tools as AmT

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.close('all')


def H(A):
    """ Calculate the Hermitian conjugate transpose of a matrix 'A' """
    return A.conj().T


save_fig = False

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# DFT parameters
N_dft = 2**10
fs = 50e3
df = fs/N_dft
freq = np.linspace(0, fs-df, N_dft)[:N_dft//2+1]

# mic arc radius
R = 1.18

# mic arc angles
angles = np.array([130, 120, 110, 100, 95, 90, 85, 80, 70, 60, 50,
                    40])*np.pi/180
m_FF_90deg = 5


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define airfoil geometry

b = 0.075       # airfoil half chord [m]
d = 0.225       # airfoil half span [m]


Nx = 100         # number of points sampling the chord (non-uniformly)
x_length = 1
x_airfoil, dx = AmT.chord_sampling(b, Nx, exp_length=x_length)

Ny = 101
y_airfoil = np.linspace(-d, d, Ny)
dy = y_airfoil[1]-y_airfoil[0]

XY_airfoil = np.meshgrid(x_airfoil, y_airfoil)
Z_airfoil = np.zeros(XY_airfoil[0].shape)
XYZ_airfoil = np.array([XY_airfoil[0], XY_airfoil[1], Z_airfoil])

XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)

# dipole sources are perpendicular to airfoil surface
dipole_axis = 'z'

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Acoustic characteristics
c0 = 343.                   # Speed of sound [m/s]
rho0 = 1.2                  # Air density [kg/m**3]
omega0 = 2*np.pi*freq
k0 = omega0/c0
p_ref = 20e-6               # Reference pressure [Pa]

# Aeroacoustic characteristics
Ux = 60.     # flow velocity [m/s]
Mach = Ux/c0   # Mach number
beta = np.sqrt(1-Mach**2)

# flow direction
flow_dir = 'x'
flow_param = (flow_dir, Mach)

Kx = 2*np.pi*freq/Ux    # turbulence/gust wavenumber
dKx = Kx[1]-Kx[0]

turb_intensity = 0.025   # turb intensity = u_rms/U  [m/s]
length_scale = 0.007     # turb length scale [m]

u_mean2 = (Ux*turb_intensity)**2

# single gust turbulence spectrum
ky = 0.
dky = 2*np.pi/(2*d)

Phi2 = AmT.Phi_2D(Kx, ky, u_mean2, length_scale, model='K')

kx_50 = np.where(freq > 48)[0][0]
kx_20k = np.where(freq > 20e3)[0][0]

# logarithmically-spaced indexes (rounded)
N_freq = 200         # approx. number of indices
index_log = AmT.index_log(kx_50, kx_20k, N_freq)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Place a mic at 1.2 m and 15m
M = 2
R_mic = np.array([1.2, 15])

X_mic = np.array([0., 0.])
Y_mic = np.array([0., 0.])
XYZ_mic = np.array([X_mic, Y_mic, R_mic])

# shear layer height
sl_z = 0.075

# airfoil centre location
xs_centre = np.array([0., 0., 0.])

XYZ_c = np.zeros(XYZ_mic.shape)
sigma_c = np.zeros(M)

for m in range(M):
    # shear layer crossing point
    XYZ_L = AmT.ShearLayer_X(xs_centre, XYZ_mic[:, m], Ux, c0, sl_z)

    # shear-layer-corrected microphone position
    XYZ_c[:, m], _ = AmT.ShearLayer_Corr(xs_centre, XYZ_L, XYZ_mic[:, m], Ux,
                                         c0)

    # flow-corrected, shear-layer-corrected mic position
    sigma_c[m] = np.sqrt(XYZ_c[0, m]**2
                         + (beta**2)*(XYZ_c[1, m]**2+ XYZ_c[2, m]**2))


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculate far-field PSD

# 'Classical' integrand for the unsteady loading at 90 deg
L_90deg_an = np.zeros((M, N_dft//2 + 1), 'complex')

# Green's function matrix
G = np.zeros((M, Nx*Ny), 'complex')

# source cross-spectral matrix
Sqq = np.zeros((Nx*Ny, Nx*Ny), 'complex')

# microphone PSD
Spp = np.zeros((M, N_dft//2 + 1), 'complex')

# for every freq...
for kxi in index_log:

    # analytical solution of L at 90deg
    L_90deg_an[:, kxi] = AmT.L_LE(XYZ_c[0], sigma_c, Kx[kxi], ky, Mach, b)

    # sinusoidal gust peak value
    w0 = np.sqrt(Phi2[kxi, 0])

    # Pressure 'jump' over the airfoil (for single gust)
    delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx[kxi], ky, XYZ_airfoil[0:2],
                           Mach)

    # reshape and reweight for vector calculation
    delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

    # source CSM (add cross-spectrum extra parameters)
    Sqq = np.outer(delta_p1_calc, delta_p1_calc.conj())*Ux*dky

    # Calculate the matrices of Greens functions for the airfoil-mic radiation
    G = AmT.dipole3D(XYZ_airfoil_calc, XYZ_c, k0[kxi], dipole_axis, flow_param)

    # Calculate microphone PSDs (using '@' for matrix multiplication)
    Spp[:, kxi] = np.real(np.diag(G @ Sqq @ H(G)))*(4*np.pi)


# Calculate PSD - far field dipole, infinite span
Spp_is = np.zeros((M, N_dft//2+1))
for m in range(M):
    Spp_is[m, :] = (((rho0*k0*b*XYZ_c[2, m]/(sigma_c[m]**2))**2)
                    * (Ux*np.pi*d)*(Phi2[:, 0])*(np.abs(L_90deg_an[m, :])**2)
                    * (4*np.pi))

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# load full-ky spectrum from 'Spp_SumKy_NearFar.mat' file

ky_integ_dict = loadmat('Spp_SumKy_NearFar')
Spp_SumKy = ky_integ_dict['Spp_SumKy']


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plot PSD vs. k0*c (chordwise normalised frequency)
plt.figure()

Ax = plt.subplot(111)

# 1st mic (1.2 m)
plt.semilogx(k0[index_log]*(2*b),
             10*np.log10(Spp_SumKy[0, index_log]/(p_ref**2)),
             color='C0', linestyle='-', linewidth=2,
             label=r'Near-Field, Multiple Gusts')
plt.semilogx(k0[index_log]*(2*b),
             10*np.log10(np.real(Spp[0, index_log])/(p_ref**2)),
             color='C1', linestyle='--', linewidth=2,
             label='Near-Field, $k_\psi=0$ only')
plt.semilogx(k0[index_log]*(2*b), 10*np.log10(Spp_is[0, index_log]/(p_ref**2)),
             color='C3', linestyle=':', linewidth=2,
             label='Far-Field, Single Gust')


# 2nd mic (15 m)
plt.semilogx(k0[index_log]*(2*b),
             10*np.log10(Spp_SumKy[1, index_log]/(p_ref**2)),
             color='C0', linestyle='-', linewidth=2)
plt.semilogx(k0[index_log]*(2*b),
             10*np.log10(np.real(Spp[1, index_log])/(p_ref**2)),
             color='C1', linestyle='--', linewidth=2)
plt.semilogx(k0[index_log]*(2*b), 10*np.log10(Spp_is[1, index_log]/(p_ref**2)),
             color='C3', linestyle=':', linewidth=2)

plt.legend(loc='lower center', fontsize=15)
plt.xlim([k0[kx_50]*(2*b), k0[kx_20k]*(2*b)])
plt.ylim([-20, 70])
plt.grid()

Ax.tick_params(labelsize=15)

# write observer distances
plt.annotate(r'$R=8c$', xy=(5, 53), xytext=(10, 60), fontsize=20, color='w',
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate(r'$R=8c$', xy=(6., 45), xytext=(10, 60), fontsize=20,
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate(r'$R=100c$', xy=(5, 27), xytext=(2, 12), fontsize=20,
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.xlabel(r'$k_0 c$', fontsize=20)
plt.ylabel(r'PSD [dB re 20 $\mu$Pa/Hz]', fontsize=18)
plt.tight_layout()


if save_fig:
    plt.savefig('PSD_SingleGust_MultGusts.eps')
