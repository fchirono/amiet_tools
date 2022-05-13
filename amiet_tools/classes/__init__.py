"""
Author: Fabio Casagrande Hirono

Available classes:
---------------------
    >>> TestSetup()
    >>> AirfoilGeom()
    >>> FrequencyVars(f0, testSetup)
For further information, check the class specific documentation.
"""

from .TestSetup import TestSetup
from .AirfoilGeom import AirfoilGeom
from .FrequencyVars import FrequencyVars

__all__ = [  # Classes
    'TestSetup',
    'AirfoilGeom',
    'FrequencyVars']
