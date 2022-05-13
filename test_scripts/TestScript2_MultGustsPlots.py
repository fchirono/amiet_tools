"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


TestScript2_MultGustsPlots.py

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
    """Calculate the Hermitian conjugate transpose of a matrix 'A'"""
    return A.conj().T


save_fig = False


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# load test setup from file
DARP2016Setup = AmT.loadTestSetup('../DARP2016_TestSetup.txt')

# export variables to current namespace
(c0, rho0, p_ref, Ux, turb_intensity, length_scale, z_sl, Mach, beta,
 flow_param, dipole_axis) = DARP2016Setup.export_values()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define airfoil points over the whole chord

# load airfoil geometry from file
DARP2016Airfoil = AmT.loadAirfoilGeom('../DARP2016_AirfoilGeom.txt')
(b, d, Nx, Ny, XYZ_airfoil, dx, dy) = DARP2016Airfoil.export_values()
XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Frequency of analysis

# # Chordwise normalised frequency = k0*(2*b)
# kc = 0.5    # approx 180 Hz
# kc = 5      # approx 1.8 kHz
kc = 20     # approx 7.2 kHz

# frequency [Hz]
f0 = kc*c0/(2*np.pi*(2*b))

FreqVars = AmT.FrequencyVars(f0, DARP2016Setup)
(k0, Kx, Ky_crit) = FreqVars.export_values()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Create arc of far field points for directivity measurements

R_farfield = 50     # far-field mic radius [m]
M_farfield = 181    # number of far-field mics in arc

theta_farfield = np.linspace(-np.pi/2, np.pi/2, M_farfield)
x_farfield = R_farfield*np.sin(theta_farfield)
z_farfield = -R_farfield*np.cos(theta_farfield)

XZ_farfield = np.array([x_farfield, np.zeros(x_farfield.shape), z_farfield])
YZ_farfield = np.array([np.zeros(x_farfield.shape), x_farfield, z_farfield])


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# vector of spanwise hydrodynamic gusts 'Ky' (for acoustic radiation)
Ky = AmT.ky_vector(b, d, k0, Mach, beta, method='AcRad')

# turbulent velocity spectrum
Phi2 = AmT.Phi_2D(Kx, Ky, Ux, turb_intensity, length_scale, model='K')[0]

# calculate source CSM
Sqq, Sqq_dxy = AmT.calc_airfoil_Sqq(
    DARP2016Setup, DARP2016Airfoil, FreqVars, Ky, Phi2)

# apply airfoil grid area weighting to source CSM
Sqq *= Sqq_dxy


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# chordwise and spanwise directivities (PSDs)
Spp_Xdir = np.zeros((M_farfield,), 'complex')
Spp_Ydir = np.zeros((M_farfield,), 'complex')

# Matrices of Greens functions for directivities
G_Xdir = AmT.dipole3D(XYZ_airfoil_calc, XZ_farfield, k0, dipole_axis,
                      flow_param)
G_Ydir = AmT.dipole3D(XYZ_airfoil_calc, YZ_farfield, k0, dipole_axis,
                      flow_param)

# calculates chordwise and spanwise PSDs (diag of mic CSMs)
Spp_Xdir += np.real(np.diag(G_Xdir @ Sqq @ H(G_Xdir)))*4*np.pi
Spp_Ydir += np.real(np.diag(G_Ydir @ Sqq @ H(G_Ydir)))*4*np.pi


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plot the far field directivities

# normalise with respect to maximum ff pressure for parallel gust
Spp_max = np.max((Spp_Xdir.max(), Spp_Ydir.max()))

# Spp_max = 4.296e-8

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
    fig_dir_XZ.savefig('MultGust_Xdir_kc_{}.png'.format(kc))
    fig_dir_YZ.savefig('MultGust_Ydir_kc_{}.png'.format(kc))
