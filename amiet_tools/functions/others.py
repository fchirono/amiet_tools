"""Author: Fabio Casagrande Hirono"""
import numpy as np


def rect_grid(grid_sides, point_spacings):
    """
    Returns the 2D coordinates for a uniformly spaced rectangular grid.

    Parameters
    ----------
    grid_sides : (2,) array_like
        1D array containing the lengths 'Lx' and 'Ly' of the grid in the 'x'
        and 'y' directions, respectively.

    point_spacings : (2,) array_like
        1D array containing the spacings 'dx' and 'dy' between the points in
        the 'x' and 'y' directions, respectively.

    Returns
    -------
    XY_grid : (2, Nx*Ny) array_like
        2D array containing the (x, y) coordinates of all points in the grid.
        The grid contains 'Nx = Lx/dx+1' points in the 'x' direction, and
        'Ny = Ly/dy+1' points in the 'y' direction.
    """
    # number of points on each side = Dx/dx + 1
    N_points = np.array([round(grid_sides[0]/point_spacings[0] + 1),
                         round(grid_sides[1]/point_spacings[1] + 1)],
                        dtype='int')

    x_points = np.linspace(-grid_sides[0]/2., grid_sides[0]/2., N_points[0])
    y_points = np.linspace(-grid_sides[1]/2., grid_sides[1]/2., N_points[1])

    X_points, Y_points = np.meshgrid(x_points, y_points)

    return np.array([X_points.flatten(), Y_points.flatten()])


def read_ffarray_lvm(filename, n_columns=13):
    """
    Reads a .lvm file containing time-domain data acquired from LabView.

    Parameters
    ----------
    filename : string
        Name of the '.lvm' file to be read.

    N_columns : int, optional
        Number of columns to read (time samples + (N-1) signals). Defaults to 13.

    Returns
    -------
    t : (N,) array_like
        1D array containing the time samples.

    mics : (M, N) array_like
        2D array containing the 'M' microphone signals.

    Notes
    -----
    The .lvm file is assuemd to contain the time samples in the first column
    and each microphone signal in the remaining columns. Default is time signal
    plus 12 microphones.
    """
    with open(filename, 'r') as lvm_file:
        # count the number of lines (has to read the whole file,
        # hence closing and opening again)
        n_lines = len(list(lvm_file))
    t = np.zeros(n_lines)
    mics = np.zeros((n_lines, n_columns-1))

    with open(filename, 'r') as lvm_file:
        for line in range(n_lines):
            current_line = lvm_file.readline().split('\r\n')[0].split('\t')
            t[line] = float(current_line[0])
            mics[line, :] = [float(x) for x in current_line][1:]

    return t, mics.T
