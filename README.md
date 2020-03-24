# amiet_tools

A Python package for turbulence-aerofoil noise prediction.

https://github.com/fchirono/amiet_tools

Copyright (c) 2020, Fabio Casagrande Hirono

The 'amiet_tools' (AmT) Python package provides a reference implementation of
Amiet's [JSV 41, 1975] model for turbulence-aerofoil interaction noise with
extensions. These functions allow the calculation of the surface pressure jump
developed over the aerofoil surface (i.e. the acoustic source distribution) in
response to incoming turbulence, and of the acoustic field radiated by the
interaction. Incoming turbulence can be a single sinusoidal gust, or a sum of
incoherent gusts with amplitudes given by a prescribed energy spectrum.

Other capabilities include the modeling of convection and
refraction effects on sound propagation to model acoustic measurements
performed in a closed- or open-jet wind tunnel.

These codes were originally written by Fabio Casagrande Hirono between 2014 and
2018, while studying for a PhD degree in Acoustical Engineering at the Institute
of Sound and Vibration Research (ISVR), University of Southampton, Southampton, UK.


## Dependencies:
* numpy: array processing for numbers, strings, records, and objects;
* scipy: scientific library;
* mpmath: library for arbitrary-precision floating-point arithmetic.


All dependencies are already included in the Anaconda Python Distribution, a
free and open source distribution of Python. Anaconda 4.8.2 (with Python 3.7)
was used to develop and test AmT, and is recommended for using AmT.


## Python Tutorials
Here are some recommended tutorials on Python programming for scientists and engineers. All are of excellent quality, and discuss not only the Python language itself but also good programming practices in general:
* Hans Fangohr, *Python for Computational Science and Engineering*, 2018, DOI: 10.5281/zenodo.1411868, https://github.com/fangohr/introduction-to-python-for-computational-science-and-engineering (available in PDF, HTML, Jupyter Notebook files. A translated version in Portuguese is also available)
* Software Carpentry, *Programming with Python*, http://swcarpentry.github.io/python-novice-inflammation/

## Author:
Fabio Casagrande Hirono - fchirono [at] gmail.com


## Main Technical References:

Amiet, R. K., "Acoustic radiation from an airfoil in a turbulent stream",
Journal of Sound and Vibration, Vol. 41, No. 4:407–420, 1975.

Blandeau, V., "Aerodynamic Broadband Noise from Contra-Rotating Open
Rotors", PhD Thesis, Institute of Sound and Vibration Research, University
of Southampton, Southampton - UK, 2011.

Casagrande Hirono, F., "Far-Field Microphone Array Techniques for Acoustic
Characterisation of Aerofoils", PhD Thesis, Institute of Sound and
Vibration Research, University of Southampton, Southampton - UK, 2018.

Reboul, G., "Modélisation du bruit à large bande de soufflante de
turboréacteur", PhD Thesis, Laboratoire de Mécanique des Fluides et
d’Acoustique - École Centrale de Lyon, Lyon - France, 2010.

Roger, M., "Broadband noise from lifting surfaces: Analytical modeling and
experimental validation". In Roberto Camussi, editor, "Noise Sources in
Turbulent Shear Flows: Fundamentals and Applications". Springer-Verlag,
2013.

de Santana, L., "Semi-analytical methodologies for airfoil noise
prediction", PhD Thesis, Faculty of Engineering Sciences - Katholieke
Universiteit Leuven, Leuven, Belgium, 2015.
