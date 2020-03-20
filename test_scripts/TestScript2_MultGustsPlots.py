"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


Test script 2: calculates airfoil chordwise (y=0) and spanwise (x=0) far-field
    directivities (in dB) for the multiple-gusts, near-field model.

    The code calculates the airfoil response only to gusts that are
    significant
    This script may take a few minutes to run, due to the sum of the many
    gusts' contributions.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np

import amiet_tools as AmT

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.close('all')


def H(A):
    """ Calculate the Hermitian conjugate transpose of a matrix 'A' """
    return A.conj().T


save_fig = False


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# load test setup from file
DARP2016Setup = AmT.loadTestSetup('../DARP2016_setup.txt')

# export variables to current namespace
(c0, rho0, p_ref, b, d, Nx, Ny, Ux, turb_intensity, length_scale, z_sl, Mach,
 beta,flow_param, dipole_axis) = DARP2016Setup.export_values()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# frequency of operation
# kc = 0.5    # approx 180 Hz
kc = 5      # approx 1.8 kHz
# kc = 20     # approx 7.2 kHz

f0 = kc*c0/(2*np.pi*(2*b))      # Hz

# Acoustic wavelength
ac_wavelength = c0/f0           # [m/rad]

# Acoustic wavenumber
k0 = 2*np.pi/ac_wavelength      # [rad/m]

Kx = 2*np.pi*f0/Ux              # turbulence/gust wavenumber

ky_crit = Kx*Mach/beta          # critical spanwise wavenumber

mu_h = Kx*b/(beta**2)   # hydrodynamic reduced frequency
mu_a = mu_h*Mach        # chord-based acoustic reduced frequency


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# create airfoil mesh coordinates, and reshape for calculations
XYZ_airfoil, dx, dy = AmT.create_airf_mesh(b, d, Nx, Ny)
XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Create arc of far field points for directivity measurements

R_farfield = 50     # far-field mic radius [m]
M_farfield = 181    # number of far-field mics in arc

theta_farfield = np.linspace(-np.pi/2, np.pi/2, M_farfield)
x_farfield = R_farfield*np.sin(theta_farfield)
z_farfield = -R_farfield*np.cos(theta_farfield)

XZ_farfield = np.array([x_farfield, np.zeros(x_farfield.shape), z_farfield])
YZ_farfield = np.array([np.zeros(x_farfield.shape), x_farfield, z_farfield])


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# obtain vector of spanwise hydrodynamic gusts 'Ky' for acoustic radiation
# (vector is in interval [0, ky_max])
Ky = AmT.ky_vector(b, d, k0, Mach, beta, method='AcRad')
dky = Ky[1]-Ky[0]

# source CSM
Sqq = np.zeros((Nx*Ny, Nx*Ny), 'complex')

# chordwise and spanwise directivities (PSDs)
Spp_Xdir = np.zeros((M_farfield,), 'complex')
Spp_Ydir = np.zeros((M_farfield,), 'complex')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculate the matrices of Greens functions
G_Xdir = AmT.dipole3D(XYZ_airfoil_calc, XZ_farfield, k0, dipole_axis,
                      flow_param)
G_Ydir = AmT.dipole3D(XYZ_airfoil_calc, YZ_farfield, k0, dipole_axis,
                      flow_param)

# turbulent velocity spectrum
Phi2 = AmT.Phi_2D(Kx, Ky, Ux, turb_intensity, length_scale, model='K')[0]

for kyi in range(Ky.shape[0]):

    # sinusoidal gust peak value
    w0 = np.sqrt(Phi2[kyi])

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # positive gusts (ky >= 0)

    # Pressure 'jump' over the airfoil (for single gust)
    delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx, Ky[kyi], XYZ_airfoil[0:2],
                           Mach)

    # reshape and reweight for vector calculation
    delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

    Sqq[:, :] += np.outer(delta_p1_calc, delta_p1_calc.conj())*(Ux)*dky

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # negative gusts (ky < 0)

    # Pressure 'jump' over the airfoil (for single gust)
    delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx, -Ky[kyi], XYZ_airfoil[0:2],
                           Mach)

    # reshape and reweight for vector calculation
    delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

    # add negative gusts' radiated pressure to source CSD
    Sqq[:, :] += np.outer(delta_p1_calc, delta_p1_calc.conj())*(Ux)*dky

# calculates chordwise and spanwise PSDs (diag of mic CSMs)
Spp_Xdir += np.real(np.diag(G_Xdir @ Sqq @ H(G_Xdir)))*4*np.pi
Spp_Ydir += np.real(np.diag(G_Ydir @ Sqq @ H(G_Ydir)))*4*np.pi


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plot the far field directivities

# normalise with respect to maximum ff pressure for parallel gust
Spp_max = np.max((Spp_Xdir.max(), Spp_Ydir.max()))

Spp_max = 4.296e-8

Spp_Xnorm = Spp_Xdir/Spp_max
Spp_Ynorm = Spp_Ydir/Spp_max

fig_dir_XZ = plt.figure(figsize=(6, 4))
ax_dir_XZ = fig_dir_XZ.add_subplot(111, polar=True)
plot_dir_XZ = ax_dir_XZ.plot(theta_farfield, 10*np.log10(np.abs(Spp_Xnorm)))
ax_dir_XZ.set_thetamin(-90)
ax_dir_XZ.set_thetamax(90)
ax_dir_XZ.set_ylim([-40, 0])
ax_dir_XZ.set_theta_zero_location('N')
ax_dir_XZ.set_theta_direction('clockwise')
ax_dir_XZ.set_thetagrids([-90, -45, 0, 45, 90],
                         labels=[r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$',
                                 r'$\theta = 0$', r'$+\frac{\pi}{4}$',
                                 r'$+\frac{\pi}{2}$'], size=18)
ax_dir_XZ.set_rgrids([0., -10, -20, -30, -40],
                     labels=['0 dB', '-10', '-20', '-30', '-40'],
                     fontsize=12)

# compensate axes position for half-circle plot
ax_dir_XZ.set_position([0.1, -0.55, 0.8, 2])

title_dir_XZ = ax_dir_XZ.set_title('Normalised Directivity on $y=0$ plane ($\phi=0$)',
                                   fontsize=18, pad=-55)



fig_dir_YZ = plt.figure(figsize=(6, 4))
ax_dir_YZ = fig_dir_YZ.add_subplot(111, polar=True)
plot_dir_YZ = ax_dir_YZ.plot(theta_farfield, 10*np.log10(np.abs(Spp_Ynorm)))
ax_dir_YZ.set_thetamin(-90)
ax_dir_YZ.set_thetamax(90)
ax_dir_YZ.set_ylim([-40, 0])
ax_dir_YZ.set_theta_zero_location('N')
ax_dir_YZ.set_theta_direction('clockwise')
ax_dir_YZ.set_thetagrids([-90, -45, 0, 45, 90],
                         labels=[r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$',
                                 r'$\theta = 0$', r'$+\frac{\pi}{4}$',
                                 r'$+\frac{\pi}{2}$'], size=18)
ax_dir_YZ.set_rgrids([0., -10, -20, -30, -40],
                     labels=['0 dB', '-10', '-20', '-30', '-40'],
                     fontsize=12)

# compensate axes position for half-circle plot
ax_dir_YZ.set_position([0.1, -0.55, 0.8, 2])

title_dir_YZ = ax_dir_YZ.set_title('Normalised Directivity on $x=0$ plane ($\phi=\pi/2$)',
                                   fontsize=18, pad=-55)

if save_fig:

    if kc == 0.5:
        fig_dir_XZ.savefig('MultGust_Xdir_kc05.png'.format(kc))
        fig_dir_YZ.savefig('MultGust_Ydir_kc05.png'.format(kc))

    elif kc == 5:
        fig_dir_XZ.savefig('MultGust_Xdir_kc5.png'.format(kc))
        fig_dir_YZ.savefig('MultGust_Ydir_kc5.png'.format(kc))

    elif kc == 20:
        fig_dir_XZ.savefig('MultGust_Xdir_kc20.png'.format(kc))
        fig_dir_YZ.savefig('MultGust_Ydir_kc20.png'.format(kc))

