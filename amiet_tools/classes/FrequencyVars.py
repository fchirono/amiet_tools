import numpy as np
from dataclasses import dataclass


@dataclass
class FrequencyVars:
    """
    Class to store frequency-related variables
    """

    f0: float  # frequency [Hz]
    testSetup: object

    def __post_init__(self):
        self.c0 = self.testSetup.c0           # speed of sound [m/s]
        self.Mach = self.testSetup.Mach       # Mach number
        self.beta = self.testSetup.beta

        self.k0 = 2*np.pi*self.f0/self.c0     # acoustic wavenumber
        self.Kx = self.k0/self.Mach       # gust/hydrodynamic chordwise wavenumber

        # gust/hydrodynamic spanwise critical wavenumber
        self.Ky_crit = self.Kx*self.Mach/self.beta

    def export_values(self):
        """Returns all class attributes inside a tuple."""
        return (self.k0, self.Kx, self.Ky_crit)
