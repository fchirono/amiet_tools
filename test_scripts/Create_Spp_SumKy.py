"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


Auxiliary file for Test Script 2: pre-calculates radiated acoustic pressure for
an observer at 90deg above aerofoil, at two distances (1.2m and 15m), using the
near-field, multiple gusts model. The results are stored in a '.mat' file for
reading by Test Script 2.

Every iteration result is saved on the .mat file, to allow recovery in case of
crashes.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np
import amiet_tools as AmT

from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

plt.close('all')
plt.ion()


def H(A):
    """ Calculate the Hermitian conjugate transpose of a matrix 'A' """
    return A.conj().T


print_flag = False

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Acoustic parameters
p_ref = 20e-6

# DFT parameters
N_dft = 2**10
fs = 50e3
df = fs/N_dft
freq = np.linspace(0, fs-df, N_dft)[:N_dft//2+1]

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define airfoil geometry

b = 0.075       # airfoil half chord [m]
d = 0.225       # airfoil half span [m]

Nx = 100         # number of points sampling the chord (non-uniformly)
Ny = 101
#Ny = 61

# create airfoil mesh coordinates, and reshape for calculations
XYZ_airfoil, dx, dy = AmT.create_airf_mesh(b, d, Nx, Ny)
XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)

# airfoil dipole sources are perpendicular to airfoil surface
dipole_axis = 'z'


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Acoustic characteristics
c0 = 343.                   # Speed of sound [m/s]
rho0 = 1.2                  # Air density [kg/m**3]
k0 = 2*np.pi*freq/c0        # Acoustic wavenumber
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

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 1 mic at FF

# spherical coordinates
R_mic = np.array([1.2, 15])     # radius / height

# cartesian coordinates
X_mic = np.array([0., 0.])
Y_mic = np.array([0., 0.])

XYZ_mic = np.array([X_mic, Y_mic, R_mic])

M = 2

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

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Compare the induced PSD for every gust

kx_50 = np.where(freq > 48)[0][0]
kx_20k = np.where(freq > 20e3)[0][0]

# logarithmically-spaced indexes (rounded)
N_freq = 200         # approx. number of indices
index_log = AmT.index_log(kx_50, kx_20k, N_freq)

Spp_SumKy = np.zeros((M, N_dft//2+1))
Sqq = np.zeros((Nx*Ny, Nx*Ny), 'complex')

# %%

## create dict to save data - run only when 1st creating the data
#ky_integ_dict = {}
#ky_integ_dict['XYZ_mic'] = XYZ_mic
#ky_integ_dict['sl_z'] = sl_z
#ky_integ_dict['XYZ_c'] = XYZ_c
#ky_integ_dict['fs'] = fs
#ky_integ_dict['N_dft'] = N_dft
#ky_integ_dict['freq'] = freq
#ky_integ_dict['index_log'] = index_log

# %%
# if code crashes, reload from mat file

ky_integ_dict = loadmat('Spp_SumKy_NearFar')
last_kxi = ky_integ_dict['last_kxi'][0][0]
Spp_SumKy = ky_integ_dict['Spp_SumKy']

current_kxi = np.where(index_log > last_kxi)[0][0]

for kxi in index_log[current_kxi:]:
    #for kxi in index_log:

    # critical gusts
    ky_crit = Kx[kxi]*Mach/beta

    # period of sin in sinc
    ky_T = 2*np.pi/d

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # integrating ky with many points
    if ky_crit < 2*np.pi/d:
        # 'low freq'
        N_ky = 41       # many points
        Ky = np.linspace(-ky_T, ky_T, N_ky)

    else:
        # 'high freq' - count how many sin(ky*d) periods in Ky range
        N_T = 2*ky_crit/ky_T
        N_ky = np.int(np.ceil(N_T*20)) + 1      # 20 points per period
        Ky = np.linspace(-ky_crit, ky_crit, N_ky)

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    #    # integrating ky with few points
    #    if ky_crit < 2*np.pi/d:
    #        # 'low freq'
    #        N_ky = 5       # few points
    #        Ky = np.linspace(-ky_T, ky_T, N_ky)
    #
    #    else:
    #        # 'high freq' - count how many sin(ky*d) periods in Ky range
    #        N_T = 2*ky_crit/ky_T
    #        N_ky = np.int(np.ceil(N_T*2))          # 2 points per period
    #        Ky = np.linspace(-ky_crit, ky_crit, N_ky)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    dky = Ky[1]-Ky[0]

    # mic PSD
    Spp_ky = np.zeros((M, N_ky))

    Phi2 = AmT.Phi_2D(Kx[kxi], Ky, u_mean2, length_scale, model='K')[0]

    for kyi in range(Ky.shape[0]):

        # sinusoidal gust peak value
        w0 = np.sqrt(Phi2[kyi])

        # Pressure 'jump' over the airfoil (for single gust)
        delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx[kxi], Ky[kyi],
                               XYZ_airfoil[0:2], Mach)

        # reshape and reweight for vector calculation
        delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

        Sqq[:, :] = np.outer(delta_p1_calc, delta_p1_calc.conj())*(Ux)*dky

        # Calculate the matrices of Greens functions
        G_Xdir = AmT.dipole3D(XYZ_airfoil_calc, XYZ_c, k0[kxi], dipole_axis,
                              flow_param)

        Spp_ky[:, kyi] = np.real(np.diag(G_Xdir @ Sqq @ H(G_Xdir)))*4*np.pi

    # sum the ky contributions
    Spp_SumKy[:, kxi] = np.sum(Spp_ky, axis=1)

    # add data to mat dict
    ky_integ_dict['Spp_SumKy'] = Spp_SumKy
    ky_integ_dict['last_kxi'] = kxi
    savemat('Spp_SumKy_NearFar', ky_integ_dict)

# %%

plt.figure()
for m in range(M):
    plt.semilogx(freq[index_log],
                 10*np.log10(Spp_SumKy[m, index_log]/(p_ref**2)),
                 '-', label=r'$\Sigma \ k_y$')

plt.legend(loc='lower left', fontsize=15)

plt.xlim([freq[kx_50], freq[kx_20k]])
plt.ylim([20, 60])
plt.grid()

plt.xlabel(r'Frequency [Hz]', fontsize=15)
plt.ylabel(r'PSD [dB re 20 uPa]', fontsize=15)
plt.tight_layout()
#plt.savefig('PSD_ManyGusts.png')
