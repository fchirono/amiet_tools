"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


Test script 4: calculate the surface pressure jump cross-spectral density
matrix and show the power spectrum magnitude, cross-spectrum phase and
coherence vs. a reference point (xs, ys).


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import array_tools as ArT
import amiet_tools as AmT


def find_index(array, points):
    """
    Routine to find indices where elements of 'array' are closest to the
    coordinates given by 'point'.

    First dimension should match (number of spatial coordinates)
    """

    # if looking for one point only...
    if points.ndim == 1:
        matrix_dist = np.linalg.norm(array - points[:, np.newaxis], axis=0)

        return matrix_dist.argmin()

    # if looking for multiple points...
    else:
        # create matrix of distances
        matrix_dist = np.linalg.norm(array[:, :, np.newaxis]
                                     - points[:, np.newaxis, :], axis=0)

        return matrix_dist.argmin(0)


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

dxy = dx[np.newaxis, :]*(np.ones(Ny)*dy)[:, np.newaxis]

X_plane = XYZ_airfoil[0]
Y_plane = XYZ_airfoil[1]

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# frequency of operation
# kx = [0.5, 5, 20], f0 ~= [180 Hz, 1.8 kHz, 7.2 kHz]

kc = 20                          # chordwise normalised frequency = k0*(2*b)
f0 = kc*c0/(2*np.pi*(2*b))

FreqVars = AmT.FrequencyVars(f0, DARP2016Setup)
(k0, Kx, Ky_crit) = FreqVars.export_values()

ac_wavelength = 2*np.pi/k0

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define coherence phase ref

# # (almost at) leading edge, mid span (approx 1% chord)
# xy_phase_ref = np.array([XYZ_airfoil[0, 0, 2], 0])
# ref_chord = '001c'

## just before leading edge, mid span (10% chord)
#xy_phase_ref = np.array([0.8*(-b), 0])
#ref_chord = '01c'

# mid chord, mid span (50% chord)
xy_phase_ref = np.array([0, 0])
ref_chord = '05c'

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

ref_index = find_index(XYZ_airfoil_calc[:2], xy_phase_ref)


# %%
# find ky that is at -20 dB at every chord point
ky_20dBAtt = AmT.ky_att(xy_phase_ref[0], b, Mach, k0, Att=-20)

# critical gust spanwise wavenumber
ky_crit = k0/beta

ky_max = 1.5*ky_20dBAtt

sinc_width = 2*np.pi/(2*d)

# get ky with spacing equal to 1/4 width of sinc function
N_ky = np.int(np.ceil(ky_max/(sinc_width/4)))

Ky, dKy = np.linspace(-ky_max, ky_max, (2*N_ky)+1, retstep=True)
Phi2 = AmT.Phi_2D(Kx, Ky, Ux, turb_intensity, length_scale, model='K')[0]

# Calculate CSM for airfoil surface
Sqq, Sqq_dxy = AmT.calc_airfoil_Sqq(DARP2016Setup, DARP2016Airfoil, FreqVars, Ky, Phi2)

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# display cross-spectrum magnitude, phase and coherence on aerofoil surface

fig_XSpec = plt.figure(figsize=(9, 5))

# create rectangles for subplots
left = 0.05
bottom = 0.1
width = 0.149
height = 0.8

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# cross-spectrum magnitude
rect_ax1 = [left, bottom, width, height]
ax_XSpec_dB = plt.axes(rect_ax1)

# original source distr
xspec_ref = Sqq[ref_index, :]

xspec_dB = 10*np.log10(np.abs(xspec_ref)).reshape(X_plane.shape)

XSpec_dB = ax_XSpec_dB.pcolormesh(X_plane, Y_plane, xspec_dB,
                                  vmax=xspec_dB.max(), vmin=xspec_dB.max()-30,
                                  cmap='inferno')
ax_XSpec_dB.axis('equal')

ax_XSpec_dB.set_xticks([])
ax_XSpec_dB.set_yticks([])
ax_XSpec_dB.set_title('$|S_{\Delta p \Delta p\'}(\mathbf{r}_s, \mathbf{r}_{ref})|$ [dB]',
                      fontsize=18)

# axes for colorbar
rect_cb1 = [left+width+0.02, bottom, 0.02, height]
ax_cb1 = plt.axes(rect_cb1)
cb_XSpec_dB = fig_XSpec.colorbar(XSpec_dB, cax=ax_cb1)
cb_XSpec_dB.ax.tick_params(labelsize=11)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# cross-spectrum phase
rect_ax2 = [0.34 + left, bottom, width, height]
ax_XSpec_ph = plt.axes(rect_ax2)

XSpec_ph = ax_XSpec_ph.pcolormesh(X_plane, Y_plane,
                                  np.angle(xspec_ref.reshape(X_plane.shape)),
                                  vmin=-np.pi, vmax=np.pi, cmap='seismic')
ax_XSpec_ph.axis('equal')
ax_XSpec_ph.set_xticks([])
ax_XSpec_ph.set_yticks([])

ax_XSpec_ph.set_title('$\\angle S_{\Delta p \Delta p\'}(\mathbf{r}_s, \mathbf{r}_{ref})$ [rad]',
                      fontsize=18)

# axes for colorbar
rect_cb2 = [0.34+left+width+0.02, bottom, 0.02, height]
ax_cb2 = plt.axes(rect_cb2)
cb_XSpec_ph = fig_XSpec.colorbar(XSpec_ph, cax=ax_cb2)
cb_XSpec_ph.ax.tick_params(labelsize=11)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# coherence
cohere_ref = np.real((np.abs(xspec_ref)**2)
                     / (Sqq[ref_index, ref_index]
                        * np.diag(Sqq))).reshape(X_plane.shape)

rect_ax3 = [0.66 + left, bottom, width, height]
ax_XSpec_Co = plt.axes(rect_ax3)

XSpec_Co = ax_XSpec_Co.pcolormesh(X_plane, Y_plane, cohere_ref, vmin=0, vmax=1)
ax_XSpec_Co.axis('equal')
ax_XSpec_Co.set_xticks([])
ax_XSpec_Co.set_yticks([])
ax_XSpec_Co.set_title('$\gamma^2(\mathbf{r}_s, \mathbf{r}_{ref})$', fontsize=15)

rect_cb3 = [0.66+left+width+0.02, bottom, 0.02, height]
ax_cb3 = plt.axes(rect_cb3)
cb_XSpec_Co = fig_XSpec.colorbar(XSpec_Co, cax=ax_cb3)
cb_XSpec_Co.ax.tick_params(labelsize=11)


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# calculate wavanumber transform for surface pressure

Nkx = 101
kx_vec = np.linspace(-1.5*k0, 1.5*k0, Nkx)

Nky = 101
ky_vec = np.linspace(-1.5*k0, 1.5*k0, Nky)

Kx_mesh, Ky_mesh = np.meshgrid(kx_vec, ky_vec)

Kxy = np.array([Kx_mesh.reshape(Nkx*Nky), Ky_mesh.reshape(Nkx*Nky)])

CSM_k = (ArT.wavenumber_spectrum2(Sqq*np.outer(dxy, dxy),
                                  XYZ_airfoil_calc[0:2], Kxy) / ((2*np.pi)**4))

CSM_k_dBmax = 10*np.log10(np.abs(np.diag(CSM_k))).max()
CSM_k_dBmin = CSM_k_dBmax-30


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plot wavenumber autospectrum
fig_Sqqkx = plt.figure(figsize=(7, 6))
ax_Sqqkx = plt.axes([0.11, 0.1, 0.685, 0.8])

Sqqkx_plot = ax_Sqqkx.pcolormesh(Kx_mesh/k0, Ky_mesh/k0,
                                 10*np.log10(np.abs(np.diag(CSM_k))
                                             .reshape(Nky, Nkx)),
                                 vmax=CSM_k_dBmax, vmin=CSM_k_dBmin)

ax_Sqqkx.set_title('$k_0 c = {:.1f}$'.format(kc), fontsize=20)

ax_Sqqkx.set_xlabel('$k_x/k_0$', fontsize=18)
ax_Sqqkx.set_ylabel('$k_y/k_0$', fontsize=18)

ax_Sqqkx.axis('equal')

# add radiation ellipse
kx1 = k0*Mach/(beta**2)
r1 = k0/(1-Mach**2)
r2 = k0/np.sqrt(1-Mach**2)

rad_ellipse = patches.Ellipse(xy=(-kx1/k0, 0), width=2*r1/k0,
                              height=2*r2/k0, fill=False, edgecolor='w',
                              linestyle='-.', linewidth=2)
ax_Sqqkx.add_artist(rad_ellipse)

# add colorbar
rect_cbar = [0.85, 0.1, 0.04, 0.8]
ax_Sqqkx_cbar = plt.axes(rect_cbar)
cb_Sqqkx = fig_Sqqkx.colorbar(Sqqkx_plot, cax=ax_Sqqkx_cbar)
cb_Sqqkx.ax.tick_params(labelsize=11)
cb_Sqqkx.set_label('dB', fontsize=12)
