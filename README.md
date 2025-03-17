Disclaimer: the following code is written by a master student physics as part of their thesis, changes can happen at any time.

This code does Monte Carlo simulations of the hedgehog supressed NLSM model according to the paper of A. Vishwanath, O.I. Motrunich (2004). 
- a lattice geometry is initialized using dictionaries
- gauge potentials and fluxes are calculated according to the paper's formulas
- only isolated monopole/anti-monopoles are allowed

The Metropolis Step Algorithm is applied to check wether single spin-flips respect the constraints and calculate the magnetization and energy of the systems given a certain dimension and exchange coupling value.

Overview of files:
- 1monopole_1cube : we check the flux (monopole number) of 1 monopole in a single-cube system. All spinvalues and coordinations are given by hand
-1monopole_outside : Checks that the neighbouring cube of a monopole-cube has zero total flux created inside
- 1monopole_2cubes : Combination of the two above
- random_2cubes : Checks that the common side and gauge potentials for a random 2cube system is the equal and opposite in sign 
- func_flips : Introduces all necessary functions and initializations used in the MC simulations with spin flips
- func_rot : Introduce all necessary functions and initializations used in the MC simulations with spin rotations
- MC_simulation : Imports functions as a package and perfoms + plots our MC simulations 
 


