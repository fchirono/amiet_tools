"""
Author: Fabio Casagrande Hirono

Available functions:
---------------------
    >>> create_airf_mesh(b, Nx, exp_length)
    >>> chord_sampling(b, d, Nx, Ny)
    >>> loadTestSetup(filename)
    >>> loadAirfoilGeom(filename)
    >>> DARP2016_MicArray()
    >>> DARP2016_Acoular_XML()
    >>> calc_airfoil_Sqq(testSetup, airfoilGeom, frequencyVars, Ky_vec, Phi)
    >>> calc_radiated_Spp(testSetup, airfoilGeom, frequencyVars, Ky_vec, Phi, G)
    >>> fr_int(x)
    >>> fr_int_cc(x)
    >>> fr_integrand_im(zeta)
    >>> fr_integrand_re(zeta)
    >>> r(x)
    >>> r_bar(x, Mach)
    >>> sigma_(x, Mach)
    >>> t_convect(x1, x2, Ux, c0)
    >>> t_sound(x1, x2, c0)
    >>> t_total(x_layer, x_source, x_mic, Ux, c0)
    >>> constr_xl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0)
    >>> constr_yl(XYZ_sl, XYZ_s, XYZ_m, Ux, c0)
    >>> ShearLayer_X(XYZ_s, XYZ_m, Ux, c0, z_sl)
    >>> ShearLayer_matrix(XYZ_s, XYZ_o, z_sl, Ux, c0)
    >>> ShearLayer_corr(XYZ_s, XYZ_sl, XYZ_m, Ux, c0)
    >>> dipole_shear(XYZ_source, XYZ_obs, XYZ_sl, T_sl, k0, c0, Mach)
    >>> dipole3D(xyz_source, xyz_obs, k0, dipole_axis, flow_param, far_field)
    >>> delta_p(rho0, b, w0, Kx, ky, xy, Mach)
    >>> g_LE(xs, Kx, ky, Mach, b)
    >>> g_LE_sub(xs, Kx, ky, Mach, b)
    >>> g_LE_super(xs, Kx, ky, Mach, b)
    >>> L_LE(x, sigma, Kx, ky, Mach, b)
    >>> L_LE_sub(x, sigma, Kx, ky, Mach, b)
    >>> L_LE_super(x, sigma, Kx, ky, Mach, b)
    >>> ky_vector(b, d, k0, Mach, beta, xs_ref)
    >>> ky_att(xs, b, Mach, k0, Att)
    >>> Phi_2D(Kx, ky_vec, Ux, turb_intensity, length_scale, model)
    >>> rect_grid(grid_sides, point_spacings)
    >>> read_ffarray_lvm(filename, n_columns)
For further information, check the function specific documentation.
"""
from .discretizers import create_airf_mesh, chord_sampling
from .loaders import loadTestSetup, loadAirfoilGeom
from .DARP2016_MicArray import DARP2016_MicArray, DARP2016_Acoular_XML
from .wrappers import calc_airfoil_Sqq, calc_radiated_Spp
from .fresnel import fr_int, fr_int_cc, fr_integrand_im, fr_integrand_re
from .shear_layer import r, r_bar, sigma_, t_convect, t_sound, t_total, \
    constr_xl, constr_yl, ShearLayer_X, ShearLayer_matrix, ShearLayer_Corr
from .sound_propagation import dipole_shear, dipole3D
from .flat_plate_response import delta_p, g_LE, g_LE_sub, g_LE_super, L_LE, L_LE_sub, L_LE_super
from .hydronamics import ky_att, ky_vector
from .turbulent_velocity import Phi_2D
from .others import read_ffarray_lvm, rect_grid

__all__ = [
    # Functions
    'create_airf_mesh',
    'chord_sampling',
    'loadTestSetup',
    'loadAirfoilGeom',
    'DARP2016_MicArray',
    'DARP2016_Acoular_XML',
    'calc_airfoil_Sqq',
    'calc_radiated_Spp',
    'fr_int', 'fr_int_cc',
    'fr_integrand_im',
    'fr_integrand_re',
    'r', 'r_bar', 'sigma_',
    't_convect', 't_sound',
    't_total', 'constr_xl',
    'constr_yl', 'ShearLayer_X',
    'ShearLayer_matrix',
    'ShearLayer_Corr',
    'dipole_shear', 'dipole3D',
    'delta_p', 'g_LE',
    'g_LE_sub', 'g_LE_super',
    'L_LE', 'L_LE_sub',
    'L_LE_super', 'ky_vector',
    'ky_att', 'Phi_2D',
    'rect_grid',
    'read_ffarray_lvm']
