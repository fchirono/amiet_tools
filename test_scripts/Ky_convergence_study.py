"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


Study on Ky integration convergence

Ky intervals equal to approx. 1/8 of a sinc function width appears to provide
a good convergence of cross-spectrum magnitude and coherence.


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt

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

# bbox keyword arguments (for semitransparent box around text in plots)
bbox_text = {'facecolor': 'w', 'edgecolor': 'w', 'alpha': 0.5}

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
# kx = [0.5, 5, 20]
# f0 ~= [180 Hz, 1.8 kHz, 7.2 kHz]

kc = 50                          # chordwise normalised frequency = k0*(2*b)
f0 = kc*c0/(2*np.pi*(2*b))

FreqVars = AmT.FrequencyVars(f0, DARP2016Setup)
(k0, Kx, Ky_crit) = FreqVars.export_values()

ac_wavelength = 2*np.pi/k0

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# define coherence phase ref

# # (almost at) leading edge, mid span (approx 1% chord)
# xy_phase_ref = np.array([XYZ_airfoil[0, 0, 2], 0])
# ref_chord = '001c'

# just before leading edge, mid span (10% chord)
xy_phase_ref = np.array([0.8*(-b), 0])
ref_chord = '01c'

# # mid chord, mid span (50% chord)
# xy_phase_ref = np.array([0, 0])
# ref_chord = '05c'

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

## percentage of chord used as reference
#(xy_phase_ref[0]+b)/(2*b)

ref_index = find_index(XYZ_airfoil_calc[:2], xy_phase_ref)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# find span and chord reference points

span_points = np.array([[0, 0, 0, 0],
                        [0., -0.02, -0.04, -0.06]])
span_points += xy_phase_ref[:, np.newaxis]

# find indices for span points
span_indices = find_index(XYZ_airfoil_calc[:2], span_points)

## show span points
#for si in span_indices:
#    print('(x, y) = ({}, {})'.format(XYZ_airfoil_calc[0, si],
#                                     XYZ_airfoil_calc[1, si]))

# CSM for chord points
chord_points = np.array([[0, 0.02, 0.04, 0.06],
                        [0., 0., 0., 0.]])
chord_points += xy_phase_ref[:, np.newaxis]

# find indices for span points
chord_indices = find_index(XYZ_airfoil_calc[:2], chord_points)

## show chord points
#for ci in chord_indices:
#    print('(x, y) = ({}, {})'.format(XYZ_airfoil_calc[0, ci],
#                                     XYZ_airfoil_calc[1, ci]))

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plot surface reference points
fig_ref = plt.figure(figsize=(3, 6))
ax_ref = plt.axes([0.1, 0.05, 0.8, 0.9])

ax_ref.plot(np.array([-b, b, b, -b, -b]), np.array([-d, -d, d, d, -d]),
            color='0.7')


# mark LE mid span (ref)
ax_ref.plot(XYZ_airfoil_calc[0, span_indices[0]],
            XYZ_airfoil_calc[1, span_indices[0]], 'x', color='C0',
            markersize=12)
ax_ref.text(XYZ_airfoil_calc[0, span_indices[0]]-0.04,
            XYZ_airfoil_calc[1, span_indices[0]], 'ref', color='C0',
            fontsize=18, bbox=bbox_text)

for i in range(1, span_indices.shape[0]):
    ax_ref.plot(XYZ_airfoil_calc[0, span_indices[i]],
                XYZ_airfoil_calc[1, span_indices[i]], 'o',
                color='C{}'.format(i))
    ax_ref.text(XYZ_airfoil_calc[0, span_indices[i]]-0.025,
                XYZ_airfoil_calc[1, span_indices[i]]-0.01, '$s_{}$'.format(i),
                color='C{}'.format(i), fontsize=15, bbox=bbox_text)

for i in range(1, chord_indices.shape[0]):
    ax_ref.plot(XYZ_airfoil_calc[0, chord_indices[i]],
                XYZ_airfoil_calc[1, chord_indices[i]], 's',
                color='C{}'.format(i+3))
    ax_ref.text(XYZ_airfoil_calc[0, chord_indices[i]]-0.005,
                XYZ_airfoil_calc[1, chord_indices[i]]+0.02, '$c_{}$'.format(i),
                color='C{}'.format(i+3), fontsize=15, bbox=bbox_text)


ax_ref.set_xticks([])
ax_ref.set_yticks([])

ax_ref.spines['top'].set_visible(False)
ax_ref.spines['bottom'].set_visible(False)
ax_ref.spines['left'].set_visible(False)
ax_ref.spines['right'].set_visible(False)

ax_ref.set_aspect('equal')


# %%
# find ky that is at -20 dB at every chord point
ky_20dBAtt = AmT.ky_att(xy_phase_ref[0], b, Mach, k0, Att=-20)

# critical gust spanwise wavenumber
ky_crit = k0/beta

ky_max = 1.5*ky_20dBAtt

# sinc function main lobe width
sinc_width = 2*np.pi/d

# # get ky with spacing equal to 1/8 width of sinc function main lobe
# N_ky = np.int(np.ceil(ky_max/(sinc_width/8)))

# Ky, dKy = np.linspace(-ky_max, ky_max, (2*N_ky)+1, retstep=True)
# Phi2 = AmT.Phi_2D(Kx, Ky, Ux, turb_intensity, length_scale, model='K')[0]

# Calculate CSM for airfoil surface
# CSM_q, CSMq_dxy = AmT.calc_airfoil_Sqq(DARP2016Setup, DARP2016Airfoil, FreqVars, Ky, Phi2)


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# test ky integratal convergence w/ ky_max

# very fine ky sampling - 1/16th of sinc function ML width (start from ky=0)
N_ky_conv = int(np.ceil(ky_max/sinc_width)*16)

Ky_conv, dKy_conv = np.linspace(0, ky_max, N_ky_conv, retstep=True)
Phi_conv = AmT.Phi_2D(Kx, Ky_conv, Ux, turb_intensity, length_scale, model='K')[0]


# CSM for entire surface (separate ky contributions)
CSM_temp = np.zeros((Nx*Ny, Nx*Ny), 'complex')

# CSM vs. ky for span points (separate ky contributions)
CSM_span = np.zeros((span_points.shape[1], span_points.shape[1], N_ky_conv),
                    'complex')

# CSM vs. ky for chord points (separate ky contributions)
CSM_chord = np.zeros((chord_points.shape[1], chord_points.shape[1], N_ky_conv),
                    'complex')

# full CSM (integrated ky contributions)
CSM_q = np.zeros((Nx*Ny, Nx*Ny), 'complex')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# ky=0
q = AmT.delta_p(rho0, b, np.sqrt(Phi_conv[0]), Ux, Kx, Ky_conv[0],
                XYZ_airfoil[0:2], Mach)
CSM_temp = np.outer(q.reshape(Nx*Ny), q.reshape(Nx*Ny).conj())*(Ux)

# read CSM info for span points
for si in range(span_indices.size):
    CSM_span[si, :, 0] = CSM_temp[span_indices[si], span_indices]

# read CSM info for chord points
for ci in range(chord_indices.size):
    CSM_chord[ci, :, 0] = CSM_temp[chord_indices[ci], chord_indices]

CSM_q += CSM_temp*dKy_conv

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# sum range of ky (both positive and negative ky)
for kyi in range(1, N_ky_conv):

    # positive ky
    q = AmT.delta_p(rho0, b, np.sqrt(Phi_conv[kyi]), Ux, Kx, Ky_conv[kyi],
                    XYZ_airfoil[0:2], Mach)
    CSM_temp = np.outer(q.reshape(Nx*Ny), q.reshape(Nx*Ny).conj())*(Ux)

    # negative ky
    q = AmT.delta_p(rho0, b, np.sqrt(Phi_conv[kyi]), Ux, Kx, -Ky_conv[kyi],
                    XYZ_airfoil[0:2], Mach)
    CSM_temp += np.outer(q.reshape(Nx*Ny), q.reshape(Nx*Ny).conj())*(Ux)

    # read CSM info for span points
    for si in range(span_indices.size):
        CSM_span[si, :, kyi] = CSM_temp[span_indices[si], span_indices]

    # read CSM info for chord points
    for ci in range(chord_indices.size):
        CSM_chord[ci, :, kyi] = CSM_temp[chord_indices[ci], chord_indices]
    
    CSM_q += CSM_temp*dKy_conv


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 1/8 sinc width sampling appears to provide a good convergence of cross-
# spectrum magnitude and coherence

# *-*-*-*-*-*-*-*- chord convergence *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

line_styles = ['-', '--', '-.', ':']
line_iter = iter(line_styles)

fig_conv, ax_conv = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 5))

# sampling = [1/16, 1/8, 1/4, 1/2]*sinc_width
for n in [1, 2, 4, 8]:
    CSM_cumsum = np.cumsum(CSM_chord[:, :, ::n], axis=2)*(dKy_conv*n)
    cohere_cumsum = np.real(np.abs(CSM_cumsum[0, :])**2
                      / (CSM_cumsum[0, 0]*np.diagonal(CSM_cumsum).T))
    
    # change line style per ky step size
    line_i = next(line_iter)
    
    colors_iter = iter(['C0', 'C1', 'C2', 'C3', 'C4'])
    for i in range(4):
        # change color per chord point
        color_n = next(colors_iter)
        
        ax_conv[0].plot(Ky_conv[::n]/ky_crit, 10*np.log10(np.abs(CSM_cumsum[0, i])),
                        color=color_n, linestyle=line_i)
        ax_conv[1].plot(Ky_conv[::n]/ky_crit, cohere_cumsum[i, :],
                        color=color_n, linestyle=line_i)

ax_conv[0].set_ylabel('CSD magnitude [dB]')
ax_conv[0].grid()

# mark ky_20dB
ax_conv[0].vlines(ky_20dBAtt/ky_crit, -100, 0, linestyles='-.', color='C8', linewidth=1)
ax_conv[0].set_title('Convergence of Ky integration - chord points')

ax_conv[1].set_ylabel('Coherence squared')
ax_conv[1].set_xlabel('Ky/ky\_crit')
ax_conv[1].grid()

# mark ky_20dB
ax_conv[1].vlines(ky_20dBAtt/ky_crit, 0, 1, linestyles='-.', color='C8', linewidth=1)




# *-*-*-*-*-*-*-*- span convergence *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

line_styles = ['-', '--', '-.', ':']
line_iter = iter(line_styles)

fig_conv2, ax_conv2 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 5))

# sampling = [1/16, 1/8, 1/4, 1/2]*sinc_width
for n in [1, 2, 4, 8]:
    CSM_cumsum2 = np.cumsum(CSM_span[:, :, ::n], axis=2)*(dKy_conv*n)
    cohere_cumsum2 = np.real(np.abs(CSM_cumsum2[0, :])**2
                      / (CSM_cumsum2[0, 0]*np.diagonal(CSM_cumsum2).T))
    
    # change line style per ky step size
    line_i = next(line_iter)
    
    colors_iter = iter(['C0', 'C4', 'C5', 'C6'])
    for i in range(4):
        # change color per chord point
        color_n = next(colors_iter)
        
        ax_conv2[0].plot(Ky_conv[::n]/ky_crit, 10*np.log10(np.abs(CSM_cumsum2[0, i])),
                        color=color_n, linestyle=line_i)
        ax_conv2[1].plot(Ky_conv[::n]/ky_crit, cohere_cumsum2[i, :],
                        color=color_n, linestyle=line_i)

ax_conv2[0].set_ylabel('CSD magnitude [dB]')
ax_conv2[0].grid()

# mark ky_20dB
ax_conv2[0].vlines(ky_20dBAtt/ky_crit, -100, 0, linestyles='-.', color='C8', linewidth=1)
ax_conv2[0].set_title('Convergence of Ky integration - span points')

ax_conv2[1].set_ylabel('Coherence squared')
ax_conv2[1].set_xlabel('Ky/ky\_crit')
ax_conv2[1].grid()

# mark ky_20dB
ax_conv2[1].vlines(ky_20dBAtt/ky_crit, 0, 1, linestyles='-.', color='C8', linewidth=1)

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plot chord/span points CSD convergence

line_styles = ['-', '--', '-.', ':']

fig_CSD, ax_CSD = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 5))

# find min/max in chordwise PSD data
C_CSD_data = 10*np.log10(np.abs(CSM_chord[0, 0, :]))
Cdata_min = C_CSD_data.min()
Cdata_max = C_CSD_data.max()

# plot chord points PSD
ax_CSD[0].plot(Ky_conv/ky_crit, C_CSD_data, color='C0', linewidth=2,
              label='(ref, ref)')

for i in range(1, CSM_span.shape[0]):
    # find min/max in CSD data
    C_CSD_data = 10*np.log10(np.abs(CSM_chord[0, i, :]))
    Cdata_min = np.min((Cdata_min, C_CSD_data.min()))
    Cdata_max = np.max((Cdata_max, C_CSD_data.max()))

    # plot data
    ax_CSD[0].plot(Ky_conv/ky_crit, C_CSD_data, color='C{}'.format(i+3),
                  linewidth=2, label='(ref, $c_{}$)'.format(i),
                  linestyle=line_styles[i])

ax_CSD[0].set_ylabel('CSD magnitude [dB]', fontsize=15)
ax_CSD[0].legend(loc='lower right', fontsize=13)

ax_CSD[0].set_ylim([Cdata_min-5, Cdata_max+5])

# height to write text: 10% of axis height
Ctext_y = (Cdata_max-Cdata_min + 10)*0.10 + (Cdata_min-5)

text_dx = (Ky_conv[-1]/ky_crit)/50

# mark ky_crit
ax_CSD[0].vlines(1, Cdata_min-5, Cdata_max+5, linestyles='--', color='0.4',
                linewidth=1)

#if freq_i == 0:
#    # write at 10% height
#    ax_CSD[0].text(1. + text_dx, Ctext_y, '$k_\psi^{crit}$', fontsize=18,
#                   color='0.4', bbox=bbox_text)
#elif freq_i == 2:
#    # write outside axes
#    ax_CSD[0].text(1. + text_dx, Ctext_y, '$k_\psi^{crit}$', fontsize=18,
#                   color='0.4', bbox=bbox_text)

# mark ky_20dB
ax_CSD[0].vlines(ky_20dBAtt/ky_crit, Cdata_min-5, Cdata_max+5, linestyles='-.',
                color='0.4', linewidth=1)

#if freq_i == 0:
#    # write at 10% height
#    ax_CSD[0].text(ky_20dBAtt/ky_crit + text_dx, Ctext_y, '$k_\psi^{[-20]}$',
#                   fontsize=18, color='0.4', bbox=bbox_text)
#elif freq_i == 2:
#    # write outside axes
#    ax_CSD[0].text(ky_20dBAtt/ky_crit + text_dx, Ctext_y, '$k_\psi^{[-20]}$',
#                   fontsize=18, color='0.4', bbox=bbox_text)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# find min/max in PSD data (span)
S_CSD_data = 10*np.log10(np.abs(CSM_span[0, 0, :]))
Sdata_min = S_CSD_data.min()
Sdata_max = S_CSD_data.max()

# plot span PSD data
ax_CSD[1].plot(Ky_conv/ky_crit, S_CSD_data, color='C0', label='(ref, ref)')


for i in range(1, CSM_span.shape[0]):
    # find min/max in CSD data
    S_CSD_data = 10*np.log10(np.abs(CSM_span[0, i, :]))
    Sdata_min = np.min((Sdata_min, S_CSD_data.min()))
    Sdata_max = np.max((Sdata_max, S_CSD_data.max()))

    # plot data
    ax_CSD[1].plot(Ky_conv/ky_crit, S_CSD_data, color='C{}'.format(i), linewidth=2,
                  label='(ref, $s_{}$)'.format(i),
                  linestyle=line_styles[i])

ax_CSD[1].set_ylabel('CSD magnitude [dB]', fontsize=15)
ax_CSD[1].set_xlabel('Max $k_\psi / k_\psi^{crit}$', fontsize=15)
ax_CSD[1].legend(loc='lower right', fontsize=13)

ax_CSD[1].set_ylim([Sdata_min-5, Sdata_max+5])

# height to write text: 10% of axis height
Stext_y = (Sdata_max-Sdata_min + 10)*0.1 + (Sdata_min-5)

# mark ky_crit
ax_CSD[1].vlines(1, Sdata_min-5, Sdata_max+5, linestyles='--', color='0.4',
                linewidth=1)

# # write outside axes
# ax_CSD[1].text(1-text_dx, Sdata_min-10.5, '$k_\psi^{crit}$', fontsize=18,
#                color='0.4', bbox=bbox_text)

# mark ky_20dB
ax_CSD[1].vlines(ky_20dBAtt/ky_crit, Sdata_min-5, Sdata_max+5, linestyles='-.',
                color='0.4', linewidth=1)

# write outside axes
ax_CSD[1].text(ky_20dBAtt/ky_crit - text_dx, Sdata_min-10.5,
              '$k_\psi^{[-20]}$', fontsize=18, color='0.4',
              bbox=bbox_text)

fig_CSD.set_tight_layout(True)


# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# calculate chord points coherence
cohere_chord = np.real(np.abs(CSM_chord[0, :])**2
                      / (CSM_chord[0, 0]*np.diagonal(CSM_chord,
                                                      axis1=0, axis2=1).T))

# calculate span points coherence
cohere_span = np.real(np.abs(CSM_span[0, :])**2
                      / (CSM_span[0, 0]*np.diagonal(CSM_span,
                                                    axis1=0, axis2=1).T))

fig_Cohere, ax_Cohere = plt.subplots(nrows=2, ncols=1, sharex=True,
                                    figsize=(8, 5))

# plot span points coherence
for i in range(1, cohere_chord.shape[0]):
    ax_Cohere[0].plot(Ky_conv/ky_crit, cohere_chord[i, :], color='C{}'.format(i+3),
                      linestyle=line_styles[i],
                      label='(ref, $c_{}$)'.format(i))

ax_Cohere[0].set_ylabel('$\gamma^2$', fontsize=15)
ax_Cohere[0].legend(loc='lower right', fontsize=13)

ax_Cohere[0].set_ylim([-0.1, 1.1])

# mark ky_crit
ax_Cohere[0].vlines(1, -0.1, 1.1, linestyles='--', color='0.4', linewidth=1)
#if freq_i == 0:
#    ax_Cohere[0].text(1. + text_dx, 0.1, '$k_\psi^{crit}$', fontsize=18,
#                      color='0.4', bbox=bbox_text)

# mark ky_20dB
ax_Cohere[0].vlines(ky_20dBAtt/ky_crit, -0.1, 1.1, linestyles='-.',
                    color='0.4', linewidth=1)
#if freq_i == 0:
#    ax_Cohere[0].text(ky_20dBAtt/ky_crit + text_dx, 0.1, '$k_\psi^{[-20]}$',
#                      fontsize=18, color='0.4', bbox=bbox_text)


# plot span points coherence
for i in range(1, CSM_span.shape[0]):
    ax_Cohere[1].plot(Ky_conv/ky_crit, cohere_span[i, :], color='C{}'.format(i),
                      linestyle=line_styles[i],
                      label='(ref, $s_{}$)'.format(i))

ax_Cohere[1].set_xlabel('Max $k_\psi / k_\psi^{crit}$', fontsize=15)
ax_Cohere[1].set_ylabel('$\gamma^2$', fontsize=15)
ax_Cohere[1].legend(loc='upper right', fontsize=13)

ax_Cohere[1].set_ylim([-0.1, 1.1])

# mark ky_crit
ax_Cohere[1].vlines(1, -0.1, 1.1, linestyles='--', color='0.4', linewidth=1)

#if freq_i == 0:
#    ax_Cohere[1].text(1. + text_dx, 0.1, '$k_\psi^{crit}$', fontsize=18,
#                      color='0.4', bbox=bbox_text)
## write outside axes
#ax_Cohere[1].text(1. - text_dx, -0.45, '$k_\psi^{crit}$', fontsize=18,
#                  color='0.4', bbox=bbox_text)

# mark ky_20dB
ax_Cohere[1].vlines(ky_20dBAtt/ky_crit, -0.1, 1.1, linestyles='-.',
                    color='0.4', linewidth=1)
#if freq_i == 0:
#    ax_Cohere[1].text(ky_20dBAtt/ky_crit + text_dx, 0.1, '$k_\psi^{[-20]}$',
#                      fontsize=18, color='0.4', bbox=bbox_text)
##ax_Cohere[1].text(ky_20dBAtt/ky_crit - text_dx, -0.45, '$k_\psi^{[-20]}$',
##                  fontsize=18, color='0.4', bbox=bbox_text)

fig_Cohere.set_tight_layout(True)

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
xspec_ref = CSM_q[ref_index, :]

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
                     / (CSM_q[ref_index, ref_index]
                        * np.diag(CSM_q))).reshape(X_plane.shape)

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

