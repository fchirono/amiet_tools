"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


Test script 3: emulates a beamforming measurement at the ISVR open-jet wind
    tunnel. First, calculates aerofoil interaction noise as seen by a planar
    microphone array, positioned outsite of the mean flow (includes shear layer
    refraction effects). Then, computes the array cross-spectral matrix (CSM)
    and obtain the beamforming map of the source distribution over the
    aerofoil.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

import amiet_tools as AmT

plt.rc('text', usetex=True)
plt.close('all')

save_fig = False


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# load test setup from file
DARP2016Setup = AmT.loadTestSetup('../DARP2016_setup.txt')

# export variables to current namespace
(c0, rho0, p_ref, b, d, Nx, Ny, Ux, turb_intensity, length_scale, z_sl, Mach,
 beta,flow_param, dipole_axis) = DARP2016Setup.export_values()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define airfoil points over the whole chord

# # create airfoil mesh coordinates, and reshape for calculations
# XYZ_airfoil, dx, dy = AmT.create_airf_mesh(b, d, Nx, Ny)
# XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)

# define airfoil points over the whole chord
DARP2016Airfoil = AmT.airfoilGeom()
(b, d, Nx, Ny, XYZ_airfoil, dx, dy) = DARP2016Airfoil.export_values()

XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# create DARP array

# DARP2016 spiral microphone array coordinates
XYZ_array, array_cal = AmT.DARP2016_MicArray()

# Number of mics
M = XYZ_array.shape[1]


# obtain propag time and shear layer crossing point for every source-mic pair
# (Frequency independent!)
T_sl_fwd, XYZ_sl_fwd = AmT.ShearLayer_matrix(XYZ_airfoil_calc, XYZ_array, z_sl,
                                             Ux, c0)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Define frequency of analysis
f0 = 5000

# Hydrodynamic (gust) chordwise wavenumber
Kx = 2*np.pi*f0/Ux

# critical gusts
ky_crit = Kx*Mach/beta

# acoustic wavenumber
k0 = 2*np.pi*f0/c0


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculate airfoil acoustic source strength CSM

# period of sin in sinc
ky_T = 2*np.pi/d

# vector of spanwise gust wavenumbers
Ky =  AmT.ky_vector(b, d, k0, Mach, beta)
dky = Ky[1]-Ky[0]

# Turbulence spectrum (von Karman)
Phi2 = AmT.Phi_2D(Kx, Ky, Ux, turb_intensity, length_scale, model='K')[0]

# Cross-spectral matrix (CSM) of source strengths
Sqq = np.zeros((Nx*Ny, Nx*Ny), 'complex')

for kyi in range(Ky.shape[0]):

    # sinusoidal gust peak value
    w0 = np.sqrt(Phi2[kyi])

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # positive spanwise wavenumbers (ky < 0)

    # Pressure 'jump' over the airfoil (for single gust)
    delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx, Ky[kyi], XYZ_airfoil[0:2],
                           Mach)

    # reshape and reweight for vector calculation
    delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

    Sqq[:, :] += np.outer(delta_p1_calc, delta_p1_calc.conj())*(Ux)*dky

    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # negative spanwise wavenumbers (ky < 0)

    # Pressure 'jump' over the airfoil (for single gust)
    delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx, -Ky[kyi], XYZ_airfoil[0:2],
                           Mach)

    # reshape and reweight for vector calculation
    delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

    # add negative gusts' radiated pressure to source CSD
    Sqq[:, :] += np.outer(delta_p1_calc, delta_p1_calc.conj())*(Ux)*dky


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Create mic array CSM

# create fwd transfer function
G_fwd = AmT.dipole_shear(XYZ_airfoil_calc, XYZ_array, XYZ_sl_fwd, T_sl_fwd, k0,
                         c0, Mach)

# calculate mic array CSM
CSM = (G_fwd @ Sqq @ G_fwd.conj().T)*4*np.pi

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Create grid of scan points

scan_sides = np.array([.65, .65])       # scan plane side length
scan_spacings = np.array([0.01, 0.01])  # scan points spacing

scan_xy = AmT.rect_grid(scan_sides, scan_spacings)

# Reshape grid points for 2D plotting
plotting_shape = (scan_sides/scan_spacings+1)[::-1].astype(int)
scan_x = scan_xy[0, :].reshape(plotting_shape)
scan_y = scan_xy[1, :].reshape(plotting_shape)

# Number of grid points
N = scan_xy.shape[1]

# create array with (x, y, z) coordinates of the scan points
scan_xyz = np.concatenate((scan_xy, np.zeros((1, N))))


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Plot the mics and grid points as 3D scatter plot

fig_grid = plt.figure()
ascan_x = fig_grid.add_subplot(111, projection='3d')
plot_grid1 = ascan_x.scatter(XYZ_array[0], XYZ_array[1], XYZ_array[2], c='r',
                             marker='o')
plot_grid2 = ascan_x.scatter(scan_xyz[0], scan_xyz[1], scan_xyz[2], c='b',
                             marker='^')
ascan_x.set_xlabel('x [m]')
ascan_x.set_ylabel('y [m]')
ascan_x.set_zlabel('z [m]')

# Create proxy artist to add legend
# --> numpoints = 1 to get only one dot in the legend
# --> linestyle= "none" So there is no line drawn in the legend
scatter1_proxy = mlines.Line2D([0], [0], linestyle="none", c='r', marker='o')
scatter2_proxy = mlines.Line2D([0], [0], linestyle="none", c='b', marker='^')
ascan_x.legend([scatter1_proxy, scatter2_proxy], ['Mic Array', 'Grid Points'],
               numpoints=1)


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Preparations for beamforming calculations

# Dynamic range for plotting
dynamic_range = 15      # [dB]

# obtain propag time and shear layer crossing point for every scan-mic pair
T_sl, XYZ_sl = AmT.ShearLayer_matrix(scan_xyz, XYZ_array, z_sl, Ux, c0)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Apply classical beamforming algorithm

# Creates steering vector and beamforming filters
G_grid = np.zeros((M, N), 'complex')
W = np.zeros((M, N), 'complex')

# monopole grid without flow
#G_grid = ArT.monopole3D(scan_xyz, XYZ_array, k0)

# dipole grid with shear layer correction
G_grid = AmT.dipole_shear(scan_xyz, XYZ_array, XYZ_sl, T_sl, k0, c0, Mach)

# calculate beamforming filters
for n in range(N):
    W[:, n] = G_grid[:, n]/(np.linalg.norm(G_grid[:, n], ord=2)**2)

# vector of source powers
A = np.zeros(N)

# apply the beamforming algorithm
for n in range(N):
    A[n] = (W[:, n].conj().T @ CSM @ W[:, n]).real

# Reshape grid points for 2D plotting
A_grid = np.zeros(plotting_shape, 'complex')
A_grid = A.reshape(plotting_shape)

# Good colormaps: viridis, inferno, plasma, magma
colormap = 'plasma'

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Plot the beamforming map

fig1 = plt.figure(figsize=(6., 4.8))
map1 = plt.pcolormesh(scan_x, scan_y, 10*np.log10(A_grid/A_grid.max()),
                      cmap=colormap, vmax=0, vmin=-dynamic_range)
map1.cmap.set_under('w')
plt.title(r'Conventional Beamforming', fontsize=15)
plt.axis('equal')
plt.xlim([-scan_sides[0]/2, scan_sides[0]/2])
plt.ylim([-scan_sides[1]/2, scan_sides[1]/2])
plt.xlabel(r'$x$ [m]', fontsize=15)
plt.ylabel(r'$y$ [m]', fontsize=15)

cbar1 = fig1.colorbar(map1)
cbar1.set_label('Normalised dB', fontsize=15)
cbar1.ax.tick_params(labelsize=12)


# Indicate the leading edge, trailing edge and sideplates on beamforming plot
plt.vlines(-b, -d, d, color='k')
plt.vlines(b, -d, d, color='k')
plt.hlines(-d, -scan_sides[0]/2, scan_sides[0]/2, color='k')
plt.hlines(d, -scan_sides[0]/2, scan_sides[0]/2, color='k')
plt.text(-b+0.01, d-0.05, r'\textbf{LE}', fontsize='18', color='k')
plt.text(b+0.01, d-0.05, r'\textbf{TE}', fontsize='18', color='k')

if save_fig:
    plt.savefig('AirfoilBeamf_' + f0 +'Hz.png', dpi=200)

