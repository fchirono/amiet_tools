"""
amiet_tools - a Python package for turbulence-aerofoil noise prediction.
https://github.com/fchirono/amiet_tools
Copyright (c) 2020, Fabio Casagrande Hirono


TestScript1_SingleGustPlots.py

Test script 1: calculate interaction of a single turbulent gust with a flat
plate aerofoil, and plot:
    a) the surface pressure jump (real part and magnitude);
    b) the radiated acoustic field (real part) near the aerofoil over the
    x=0 and y=0 planes;
    c) the chordwise (y=0) and spanwise (x=0) far-field directivities (in dB).

The flow speed is assumed constant through all space (including far-field) -
i.e. there are no shear layer refraction effects.


This code was used to generate the Figures in Chap. 4, Section 4.1 of the
Authors' PhD thesis [Casagrande Hirono, 2018].


Author:
Fabio Casagrande Hirono
fchirono@gmail.com

"""


import numpy as np

import amiet_tools as AmT

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.close('all')


# flag for saving figures
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
# frequency of operation
kc = 5                          # chordwise normalised frequency = k0*(2*b)
f0 = kc*c0/(2*np.pi*(2*b))      # approx 1.8 kHz

FreqVars = AmT.FrequencyVars(f0, DARP2016Setup)
(k0, Kx, Ky_crit) = FreqVars.export_values()

ac_wavelength = 2*np.pi/k0

# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# turbulence/gust amplitude
w0 = 1

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# parallel incidence wavenumber component - select one case

test_case = 2       # int between 1 and 4

if test_case == 1:
    # supercritical, normal incidence gust
    ky = 0
    fig_title = 'Kphi_000'

elif test_case == 2:
    # supercritical gust, oblique incidence
    ky = 0.35*Ky_crit
    fig_title = 'Kphi_035'

elif test_case == 3:
    # supercritical gust, oblique incidence / close to critical
    ky = 0.75*Ky_crit
    fig_title = 'Kphi_075'

elif test_case == 4:
    # subcritical gust
    ky = 1.25*Ky_crit
    fig_title = 'Kphi_125'

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# Calculate the pressure 'jump' over the airfoil
delta_p1 = AmT.delta_p(rho0, b, w0, Ux, Kx, ky, XYZ_airfoil[0:2], Mach)

# reshape airfoil ac. source strengths, apply weights by grid area
delta_p1_calc = (delta_p1*dx).reshape(Nx*Ny)*dy

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Plot the airfoil source strength distribution
re_p1_max = np.max(np.real(np.abs(delta_p1)))
abs_p1_max = np.max(np.abs(delta_p1))

fig_airfoil = plt.figure(figsize=(6.4, 5))
ax1_airfoil = plt.subplot(121)
re_p = ax1_airfoil.pcolormesh(XYZ_airfoil[0], XYZ_airfoil[1],
                              np.real(delta_p1)/re_p1_max, cmap='seismic',
                              shading='nearest', vmin=-1, vmax=+1)
ax1_airfoil.axis('equal')
ax1_airfoil.set_title(r'$Re\{\Delta p(x_s, y_s)\}$ [a.u.]', fontsize=18)

ax1_airfoil.set_yticks([-d, 0, +d])
ax1_airfoil.set_yticklabels([r'$-d$', r'$0$', r'$+d$'], fontsize=18)
ax1_airfoil.set_xticks([-b, 0, b])
ax1_airfoil.set_xticklabels([r'$-b$', r'$0$', r'$+b$'], fontsize=18)

ax1_airfoil.set_ylim(-1.1*d, 1.1*d)
ax1_airfoil.set_xlabel(r'$x_s$', fontsize=18)
ax1_airfoil.set_ylabel(r'$y_s$', fontsize=18)
plt.colorbar(re_p)


ax2_airfoil = plt.subplot(122)
abs_p = ax2_airfoil.pcolormesh(XYZ_airfoil[0], XYZ_airfoil[1],
                               20*np.log10(np.abs(delta_p1)/abs_p1_max),
                               cmap='inferno', shading='nearest',
                               vmax=0, vmin=-30)
ax2_airfoil.axis('equal')
ax2_airfoil.set_title(r'$|\Delta p(x_s, y_s)|$ [dB]', fontsize=18)

ax2_airfoil.set_yticks([-d, 0, +d])
ax2_airfoil.set_yticklabels([r'$-d$', r'$0$', r'$+d$'], fontsize=18)
ax2_airfoil.set_xticks([-b, 0, b])
ax2_airfoil.set_xticklabels([r'$-b$', r'$0$', r'$+b$'], fontsize=18)

ax2_airfoil.set_ylim(-1.1*d, 1.1*d)
ax2_airfoil.set_xlabel(r'$x_s$', fontsize=18)
plt.colorbar(abs_p)
fig_airfoil.set_tight_layout(True)

if save_fig:
    plt.savefig('DeltaP_' + fig_title + '.png')

# %%*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Plot airfoil directivity in x-plane and y-plane

# Create far field points for directivity
R_farfield = 50     # [m]
theta_farfield = np.linspace(-np.pi/2, np.pi/2, 181)
x_farfield = R_farfield*np.sin(theta_farfield)
z_farfield = -R_farfield*np.cos(theta_farfield)

XZ_farfield = np.array([x_farfield, np.zeros(x_farfield.shape), z_farfield])
YZ_farfield = np.array([np.zeros(x_farfield.shape), x_farfield, z_farfield])

# Pressure field generated by the airfoil at the mesh
p_XZ_farfield = np.zeros(x_farfield.shape, 'complex')
p_YZ_farfield = np.zeros(x_farfield.shape, 'complex')


# calculate matrices of convected dipole Greens functions for each source
# to observers
G_ffXZ = AmT.dipole3D(XYZ_airfoil_calc, XZ_farfield, k0, dipole_axis,
                      flow_param)

G_ffYZ = AmT.dipole3D(XYZ_airfoil_calc, YZ_farfield, k0, dipole_axis,
                      flow_param)

# Calculate the pressure in the far field
p_XZ_farfield = G_ffXZ @ delta_p1_calc
p_YZ_farfield = G_ffYZ @ delta_p1_calc

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# plot the far field directivities [in dB]

# normalise with respect to maximum FF pressure for parallel gust
# (obtained from previous analysis)
p_ff_max = 0.001139

p_XZ_ff_norm = p_XZ_farfield/p_ff_max
p_YZ_ff_norm = p_YZ_farfield/p_ff_max

fig_dir_XZ = plt.figure(figsize=(6, 4))
ax_dir_XZ = fig_dir_XZ.add_subplot(111, polar=True)
plot_dir_XZ = ax_dir_XZ.plot(theta_farfield, 20*np.log10(np.abs(p_XZ_ff_norm)))
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

if save_fig:
    fig_dir_XZ.savefig('dir_XZ_' + fig_title + '.png')


fig_dir_YZ = plt.figure(figsize=(6, 4))
ax_dir_YZ = fig_dir_YZ.add_subplot(111, polar=True)
plot_dir_YZ = ax_dir_YZ.plot(theta_farfield, 20*np.log10(np.abs(p_YZ_ff_norm)))
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
    fig_dir_YZ.savefig('dir_YZ_' + fig_title + '.png')


# %% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculate the near-field radiation over the 2D cuts
"""
# create mesh for acoustic field 2D cuts
mesh_side = 6*ac_wavelength  # [m]
N_mesh = 201

coord_vector = np.linspace(-mesh_side/2., mesh_side/2., N_mesh)
X_mesh1, Z_mesh1 = np.meshgrid(coord_vector, coord_vector)
Y_mesh1 = np.zeros(X_mesh1.shape)
XZ_mesh1 = np.array([X_mesh1, Y_mesh1, Z_mesh1])

Y_mesh2, Z_mesh2 = np.meshgrid(coord_vector, coord_vector)
X_mesh2 = np.zeros(Y_mesh2.shape)
YZ_mesh2 = np.array([X_mesh2, Y_mesh2, Z_mesh2])

# Pressure field generated by the airfoil at the 2D mesh
pressure_XZ_calc = np.zeros(X_mesh1.shape[0]*X_mesh1.shape[1], 'complex')
pressure_YZ_calc = np.zeros(X_mesh2.shape[0]*X_mesh2.shape[1], 'complex')

XZ_mesh1_calc = XZ_mesh1.reshape(3, XZ_mesh1.shape[1]*XZ_mesh1.shape[2])
YZ_mesh2_calc = YZ_mesh2.reshape(3, YZ_mesh2.shape[1]*YZ_mesh2.shape[2])


# Too many points in 2D cuts to generate entire G matrices; calculate the ac.
# field of source grid point individually
for s in range(delta_p1_calc.shape[0]):
    G_pXZ = AmT.dipole3D(XYZ_airfoil_calc[:, s, np.newaxis], XZ_mesh1_calc, k0,
                          dipole_axis, flow_param)

    G_pYZ = AmT.dipole3D(XYZ_airfoil_calc[:, s, np.newaxis], YZ_mesh2_calc, k0,
                          dipole_axis, flow_param)

    # Calculate the pressure in the near field
    pressure_XZ_calc += delta_p1_calc[s]*G_pXZ[:, 0]
    pressure_YZ_calc += delta_p1_calc[s]*G_pYZ[:, 0]


# reshape nearfield meshes
pressure_XZ = pressure_XZ_calc.reshape(XZ_mesh1[0].shape)
pressure_YZ = pressure_YZ_calc.reshape(YZ_mesh2[0].shape)

# find max pressure values for setting up color scale
p_XZ_max = np.max(np.abs(np.real(pressure_XZ)))
p_YZ_max = np.max(np.abs(np.real(pressure_YZ)))
p_max = np.max((p_XZ_max, p_YZ_max))

plt.figure(figsize=(6.3, 5.))
plt.pcolormesh(X_mesh1/ac_wavelength, Z_mesh1/ac_wavelength,
               np.real(pressure_XZ)/p_max, cmap='seismic',
               shading='nearest', vmin=-1.5, vmax=1.5)
plt.plot((-b/ac_wavelength, b/ac_wavelength), (0, 0), 'k', linewidth=6)
plt.xlabel(r"$x/\lambda_0$", fontsize=18)
plt.ylabel(r"$z/\lambda_0$", fontsize=18)
plt.axis('equal')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
cbar1 = plt.colorbar()
cbar1.set_label('Ac. Pressure [a.u.]', fontsize=15)
plt.title(r"Acoustic Field on $y=0$ plane", fontsize=18)

if save_fig:
    plt.savefig('p_XZ_' + fig_title + '.png')


plt.figure(figsize=(6.3, 5.))
plt.pcolormesh(Y_mesh2/ac_wavelength, Z_mesh2/ac_wavelength,
               np.real(pressure_YZ)/p_max, cmap='seismic',
               shading='nearest', vmin=-1.5, vmax=1.5)
plt.plot((-d/ac_wavelength, d/ac_wavelength), (0, 0), 'k', linewidth=6)
plt.xlabel(r"$y/\lambda_0$", fontsize=18)
plt.ylabel(r"$z/\lambda_0$", fontsize=18)
plt.axis('equal')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

cbar2 = plt.colorbar()
cbar2.set_label('Ac. Pressure [a.u.]', fontsize=15)
plt.title(r"Acoustic Field on $x=0$ plane", fontsize=18)

if save_fig:
    plt.savefig('p_YZ_' + fig_title + '.png')
"""