Disclaimer: the following code is written by a master student physics as part of their thesis, changes can happen at any time.

This code does Monte Carlo simulations of the hedgehog supressed NLSM model according to the paper of A. Vishwanath, O.I. Motrunich (2004). 
- a lattice geometry is initialized using dictionaries
- gauge potentials and fluxes are calculated according to the paper's formulas
- only isolated monopole/anti-monopoles are allowed

The Metropolis Step Algorithm is applied to check wether single spin-flips respect the constraints and calculate the magnetization and energy of the systems given a certain dimension and exchange coupling value.


