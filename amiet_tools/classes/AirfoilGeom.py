import amiet_tools as AmT
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AirfoilGeom:
    """
    Class to store aerofoil geometry
    """

    # Aerofoil geometry
    b: float  # airfoil half chord [m]
    d: float  # airfoil half span [m]
    Nx: float  # number of chordwise points (non-uniform sampl)
    Ny: float  # number of spanwise points (uniform sampl)

    XYZ: Any = field(init=False)
    dx: Any = field(init=False)
    dy: Any = field(init=False)

    def __post_init__(self):
        self.XYZ, self.dx, self.dy = AmT.create_airf_mesh(
            self.b, self.d, self.Nx, self.Ny)

    def export_values(self):
        """Returns all class attributes inside a tuple."""
        return (self.b, self.d, self.Nx, self.Ny, self.XYZ, self.dx,
                self.dy)
