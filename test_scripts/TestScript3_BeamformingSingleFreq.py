"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


Test script 3: calculates aerofoil interaction noise at a single frequency as
    seen by a planar microphone array (multi-arm spiral), computes the array
    cross-spectral matrix (CSM) and obtain the beamforming map of the source
    distribution over the aerofoil.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np

from matplotlib import pyplot as plt

import amiet_tools as AmT
import array_tools as ArT

plt.rc('text', usetex=True)
plt.close('all')

save_fig = False


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# # load test setup from file (DARP2016 configuration by default)
# data = AmT.loadTestSetup()

# alternative: load from setup file
DARP2016Setup = AmT.loadTestSetup('../DARP2016_setup.txt')

# export variables to current namespace
(c0, rho0, p_ref, b, d, Nx, Ny, Ux, turb_intensity, length_scale, z_sl, Mach,
 beta,flow_param, dipole_axis) = DARP2016Setup.export_values()

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# create DARP array

# Microphone Spiral Array coordinates
XYZ_array, array_cal = AmT.DARP2016_MicArray()

# Number of mics
M = XYZ_array.shape[1]


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
# create airfoil mesh coordinates, and reshape for calculations
XYZ_airfoil, dx, dy = AmT.create_airf_mesh(b, d, Nx, Ny)
XYZ_airfoil_calc = XYZ_airfoil.reshape(3, Nx*Ny)

# period of sin in sinc
ky_T = 2*np.pi/d

# vector of spanwise gust wavenumbers
Ky =  AmT.ky_vector(b, d, k0, Mach, beta)
dky = Ky[1]-Ky[0]

# Turbulent spectrum
Phi2 = AmT.Phi_2D(Kx, Ky, Ux, turb_intensity, length_scale, model='K')[0]

Sqq = np.zeros((Nx*Ny, Nx*Ny), 'complex')

for kyi in range(Ky.shape[0]):

    # sinusoidal gust peak value
    w0 = np.sqrt(Phi2[kyi])

    # Pressure 'jump' over the airfoil (for single gust)
    delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx, Ky[kyi], XYZ_airfoil[0:2],
                           Mach)

    # reshape and reweight for vector calculation
    delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

    Sqq[:, :] += np.outer(delta_p1_calc, delta_p1_calc.conj())*(Ux)*dky


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Create mic CSM

# shear layer height (below airfoil)
sl_z = -0.075

# obtain propag time and shear layer crossing point for every airfoil-mic pair
T_sl_fwd, XYZ_sl_fwd = AmT.ShearLayer_matrix(XYZ_airfoil_calc, XYZ_array, sl_z,
                                             Ux, c0)

# create fwd transfer function
G_fwd = AmT.dipole_shear(XYZ_airfoil_calc, XYZ_array, XYZ_sl_fwd, T_sl_fwd, k0,
                         c0, Mach)

# calculate mic array CSM
CSM = (G_fwd @ Sqq @ G_fwd.conj().T)*4*np.pi

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# scan plane side length
scan_sides = np.array([.55, .55])

# scan points spacing
scan_spacings = np.array([0.01, 0.01])
scan_xy = ArT.rect_array(scan_sides, scan_spacings, save_txt='False')

# Reshape grid points for 2D plotting
plotting_shape = (scan_sides/scan_spacings+1)[::-1].astype(int)
scan_x = scan_xy[0, :].reshape(plotting_shape)
scan_y = scan_xy[1, :].reshape(plotting_shape)

# Number of grid points
N = scan_xy.shape[1]

# create array with (x, y, z) coordinates of the scan points
scan_xyz = np.concatenate((scan_xy, np.zeros((1, N))))

"""
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
"""


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Preparations for beamforming calculations

# Dynamic range for plotting
dynamic_range = 15      # [dB]

# obtain propag time and shear layer crossing point for every airfoil-mic pair
T_sl, XYZ_sl = AmT.ShearLayer_matrix(scan_xyz, XYZ_array, sl_z, Ux, c0)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Apply classical beamforming algorithm

# Creates steering vector and beamforming filters
G_grid = np.zeros((M, N), 'complex')
W = np.zeros((M, N), 'complex')

# monopole grid without flow
#G_grid = ArT.monopole3D(scan_xyz, XYZ_array, k0)

# dipole grid with shear layer correction
G_grid = AmT.dipole_shear(scan_xyz, XYZ_array, XYZ_sl, T_sl, k0, c0, Mach)

for n in range(N):
    W[:, n] = G_grid[:, n]/(np.linalg.norm(G_grid[:, n], ord=2)**2)

# source power estimation
A = np.zeros(N)

# apply the beamforming algorithm
for n in range(N):
    A[n] = (W[:, n].conj().T @ CSM @ W[:, n]).real

# Reshape grid points for 2D plotting
A_grid = np.zeros(plotting_shape, 'complex')
A_grid = A.reshape(plotting_shape)

# Good colormaps: viridis, inferno, plasma, magma
colormap = 'plasma'

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
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


# Marks the leading edge, trailing edge and sideplates on beamforming plot
plt.vlines(-b, -d, d, color='k')
plt.vlines(b, -d, d, color='k')
plt.hlines(-d, -scan_sides[0]/2, scan_sides[0]/2, color='k')
plt.hlines(d, -scan_sides[0]/2, scan_sides[0]/2, color='k')
plt.text(-b+0.01, d-0.05, r'\textbf{LE}', fontsize='18', color='k')
plt.text(b+0.01, d-0.05, r'\textbf{TE}', fontsize='18', color='k')

if save_fig:
    plt.savefig('AirfoilBeamf_' + f0 +'Hz.png', dpi=200)

"""
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(scan_x, scan_y, 10*np.log10(np.abs(A_grid)),
                       rstride=1, cstride=1, cmap=colormap)
ax.set_zlim(10*np.log10(np.abs(A_grid.max()))-dynamic_range,
            10*np.log10(np.abs(A_grid.max())))
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('Magnitude [dB]')
"""

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Apply CLEAN-SC

# Runs CLEAN-SC from array_tools
clean_sc_results = ArT.clean_sc(CSM, W, loop_gain=0.95)
location_sc, power_sc, component_sc, final_CSM_sc = clean_sc_results

# Rebuilt the CSM for the 1st source component for CLEAN-SC
gcs1_CSM = power_sc[0]*np.outer(component_sc[0], component_sc[0].conj().T)
gcs2_CSM = power_sc[1]*np.outer(component_sc[1], component_sc[1].conj().T)
gcs3_CSM = power_sc[2]*np.outer(component_sc[2], component_sc[2].conj().T)

# Reconstruct the source map using clean beams (initialize using the final CSM)
A_cleanpsf = np.zeros(N)
A_cleansc = np.zeros(N)
gc1_cleansc = np.zeros(N)
gc2_cleansc = np.zeros(N)
gc3_cleansc = np.zeros(N)

for n in range(N):
    A_cleansc[n] = (W[:, n].conj().T @ final_CSM_sc @ W[:, n]).real
    gc1_cleansc[n] = (W[:, n].conj().T @ gcs1_CSM @ W[:, n]).real
    gc2_cleansc[n] = (W[:, n].conj().T @ gcs2_CSM @ W[:, n]).real
    gc3_cleansc[n] = (W[:, n].conj().T @ gcs3_CSM @ W[:, n]).real


# Not use the 'dirty' CSM from CLEAN-SC
#A_cleansc = np.zeros(N, 'complex')

# Add the clean beams corresponding to each source
clean_beam_width = 0.05      # [m]
for sc in range(location_sc.__len__()):
    A_cleansc += (power_sc[sc] *
                  ArT.clean_beam(scan_xy, scan_xy[:, location_sc[sc]],
                                 clean_beam_width))


# Reshape grid points for 2D plotting
A_cleansc_grid = np.zeros(plotting_shape)
A_cleansc_grid = A_cleansc.reshape(plotting_shape)

gc1_cleansc_grid = np.zeros(plotting_shape)
gc1_cleansc_grid = gc1_cleansc.reshape(plotting_shape)

gc2_cleansc_grid = np.zeros(plotting_shape)
gc2_cleansc_grid = gc2_cleansc.reshape(plotting_shape)

gc3_cleansc_grid = np.zeros(plotting_shape)
gc3_cleansc_grid = gc3_cleansc.reshape(plotting_shape)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# CLEAN-SC grid

fig2 = plt.figure(figsize=(6, 4.8))
clean1 = plt.pcolormesh(scan_x, scan_y,
                        10*np.log10(A_cleansc_grid/A_cleansc_grid.max()),
                        cmap=colormap, vmax=0, vmin=-dynamic_range)
plt.plot(scan_xy[0, location_sc[:10]], scan_xy[1, location_sc[:10]],
         marker='*', linestyle='', markerfacecolor='r', markeredgecolor='k',
         alpha=0.65, markersize=18, label='Estimated Locations')
plt.axis('equal')
plt.title('CLEAN-SC', fontsize=15)
plt.legend(loc='lower right')
plt.xlim([-scan_sides[0]/2, scan_sides[0]/2])
plt.ylim([-scan_sides[1]/2, scan_sides[1]/2])
plt.xlabel(r'$x$ [m]', fontsize=15)
plt.ylabel(r'$y$ [m]', fontsize=15)

cbar2 = fig2.colorbar(clean1)
cbar2.set_label('Normalised dB', fontsize=15)
cbar2.ax.tick_params(labelsize=12)

# Marks the leading edge, trailing edge and sideplates on beamforming plot
plt.vlines(-b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.vlines(b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.hlines(-d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.hlines(d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.text(-b+0.01, d-0.05, r'\textbf{LE}', fontsize='18', color='0.35')
plt.text(b+0.01, d-0.05, r'\textbf{TE}', fontsize='18', color='0.35')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# CLEAN-SC 1st source component
fig3 = plt.figure(figsize=(6., 4.8))
component1 = plt.pcolormesh(scan_x, scan_y,
                            10*np.log10(gc1_cleansc_grid/A_cleansc_grid.max()),
                            vmax=0, vmin=-dynamic_range, cmap=colormap)
component1.cmap.set_under('w')
plt.plot(scan_xy[0, location_sc[0]], scan_xy[1, location_sc[0]],
         marker='*', linestyle='', markerfacecolor='r', markeredgecolor='k',
         alpha=0.65, markersize=18, label='Estimated Location')
plt.axis('equal')
plt.title(r'CLEAN-SC Component 1', fontsize=15)
plt.legend(loc='lower right')
plt.xlim([-scan_sides[0]/2, scan_sides[0]/2])
plt.ylim([-scan_sides[1]/2, scan_sides[1]/2])
plt.xlabel(r'$x$ [m]', fontsize=15)
plt.ylabel(r'$y$ [m]', fontsize=15)

cbar_c1 = fig3.colorbar(component1)
cbar_c1.set_label('Normalised dB', fontsize=15)
cbar_c1.ax.tick_params(labelsize=12)

# Marks the leading edge, trailing edge and sideplates on beamforming plot
# Marks the leading edge, trailing edge and sideplates on beamforming plot
plt.vlines(-b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.vlines(b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.hlines(-d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.hlines(d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.text(-b+0.01, d-0.05, r'\textbf{LE}', fontsize='18', color='0.35')
plt.text(b+0.01, d-0.05, r'\textbf{TE}', fontsize='18', color='0.35')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# CLEAN-SC 2nd source component
fig4 = plt.figure(figsize=(6., 4.8))
component2 = plt.pcolormesh(scan_x, scan_y,
                            10*np.log10(gc2_cleansc_grid/A_cleansc_grid.max()),
                            vmax=0, vmin=-dynamic_range, cmap=colormap)
component2.cmap.set_under('w')
plt.plot(scan_xy[0, location_sc[1]], scan_xy[1, location_sc[1]],
         marker='*', linestyle='', markerfacecolor='r', markeredgecolor='k',
         alpha=0.65, markersize=18, label='Estimated Location')
plt.axis('equal')
plt.title(r'CLEAN-SC Component 2', fontsize=15)
plt.legend(loc='lower right')
plt.xlim([-scan_sides[0]/2, scan_sides[0]/2])
plt.ylim([-scan_sides[1]/2, scan_sides[1]/2])
plt.xlabel(r'$x$ [m]', fontsize=15)
plt.ylabel(r'$y$ [m]', fontsize=15)

cbar_c2 = fig4.colorbar(component1)
cbar_c2.set_label('Normalised dB', fontsize=15)
cbar_c2.ax.tick_params(labelsize=12)

# Marks the leading edge, trailing edge and sideplates on beamforming plot
# Marks the leading edge, trailing edge and sideplates on beamforming plot
plt.vlines(-b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.vlines(b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.hlines(-d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.hlines(d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.text(-b+0.01, d-0.05, r'\textbf{LE}', fontsize='18', color='0.35')
plt.text(b+0.01, d-0.05, r'\textbf{TE}', fontsize='18', color='0.35')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# CLEAN-SC 3rd source component
fig5 = plt.figure(figsize=(6., 4.8))
component3 = plt.pcolormesh(scan_x, scan_y,
                            10*np.log10(gc3_cleansc_grid/A_cleansc_grid.max()),
                            vmax=0, vmin=-dynamic_range, cmap=colormap)
component3.cmap.set_under('w')
plt.plot(scan_xy[0, location_sc[2]], scan_xy[1, location_sc[2]],
         marker='*', linestyle='', markerfacecolor='r', markeredgecolor='k',
         alpha=0.65, markersize=18, label='Estimated Location')
plt.axis('equal')
plt.title(r'CLEAN-SC Component 3', fontsize=15)
plt.legend(loc='lower right')
plt.xlim([-scan_sides[0]/2, scan_sides[0]/2])
plt.ylim([-scan_sides[1]/2, scan_sides[1]/2])
plt.xlabel(r'$x$ [m]', fontsize=15)
plt.ylabel(r'$y$ [m]', fontsize=15)

cbar_c3 = fig5.colorbar(component1)
cbar_c3.set_label('Normalised dB', fontsize=15)
cbar_c3.ax.tick_params(labelsize=12)

# Marks the leading edge, trailing edge and sideplates on beamforming plot
plt.vlines(-b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.vlines(b, -d, d, linestyle='--', linewidth=3, color='0.35')
plt.hlines(-d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.hlines(d, -scan_sides[0]/2, scan_sides[0]/2, linestyle='--', linewidth=3,
           color='0.35')
plt.text(-b+0.01, d-0.05, r'\textbf{LE}', fontsize='18', color='0.35')
plt.text(b+0.01, d-0.05, r'\textbf{TE}', fontsize='18', color='0.35')
