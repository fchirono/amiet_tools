import numpy as np
from dataclasses import dataclass, field


@dataclass
class TestSetup:
    """
    Class to store test setup variables. Initializes to DARP2016 configuration
    by default.
    """
    # Acoustic characteristics
    c0: float  # Speed of sound (m/s)
    rho0: float  # Air density (kg/m**3)
    p_ref: float  # Ref acoustic pressure (Pa RMS)

    # turbulent flow properties
    Ux: float			# flow velocity (m/s)
    turb_intensity: float  # turbulence intensity = u_rms/U
    length_scale: float  # turb length scale (m)

    z_sl: float  # shear layer height (m)

    flow_dir: str = 'x'              # mean flow in the +x dir
    # airfoil dipoles are pointing 'up' (+z dir)
    dipole_axis: str = 'z'

    Mach: float = field(init=False)
    beta: float = field(init=False)
    flow_param: tuple() = field(init=False)

    def __post_init__(self):
        self.Mach = self.Ux/self.c0
        self.beta = np.sqrt(1 - self.Mach**2)
        self.flow_param = (self.flow_dir, self.Mach)

    def export_values(self):
        """Returns all class attributes inside a tuple."""
        return (self.c0, self.rho0, self.p_ref, self.Ux, self.turb_intensity,
                self.length_scale, self.z_sl, self.Mach, self.beta,
                self.flow_param, self.dipole_axis)
