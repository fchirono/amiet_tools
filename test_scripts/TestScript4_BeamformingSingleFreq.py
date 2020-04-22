"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


TestScript4_BeamformingSingleFreq.py

Test script 4: emulates a beamforming measurement at the ISVR open-jet wind
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
# create DARP array

# DARP2016 spiral microphone array coordinates
XYZ_array, array_cal = AmT.DARP2016_MicArray()

# Number of mics
M = XYZ_array.shape[1]


# obtain propag time and shear layer crossing point for every source-mic pair
# (forward problem - frequency independent!)
T_sl_fwd, XYZ_sl_fwd = AmT.ShearLayer_matrix(XYZ_airfoil_calc, XYZ_array, z_sl,
                                             Ux, c0)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Define frequency of analysis

# # Chordwise normalised frequency = k0*(2*b)
# kc = 5      # approx 1.8 kHz
kc = 10      # approx 3.6 kHz
# kc = 20     # approx 7.2 kHz

# frequency [Hz]
f0 = kc*c0/(2*np.pi*(2*b))

FreqVars = AmT.FrequencyVars(f0, DARP2016Setup)
(k0, Kx, Ky_crit) = FreqVars.export_values()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculate airfoil acoustic source strength CSM

# vector of spanwise gust wavenumbers
Ky =  AmT.ky_vector(b, d, k0, Mach, beta)

# Turbulence spectrum (von Karman)
Phi2 = AmT.Phi_2D(Kx, Ky, Ux, turb_intensity, length_scale, model='K')[0]

# calculate source CSM
Sqq, Sqq_dxy = AmT.calc_airfoil_Sqq(DARP2016Setup, DARP2016Airfoil, FreqVars, Ky, Phi2)

# apply weighting for airfoil grid areas
Sqq *= Sqq_dxy

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

