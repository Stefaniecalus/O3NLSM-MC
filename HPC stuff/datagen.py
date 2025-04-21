import numpy as np
from sys import getsizeof 
# number of spins in lattice
Ls = [(6,1225),(8,2673),(10,4961),(12,8281),(14,12825),(16,18785)]
Js = np.linspace(0.0, 1.4, 15, endpoint=True, dtype=float)
steps_per_spin = 100
n = 100 # split calculating averages and such into n parts

data = []
for (L, nspins) in Ls:
    nth = int(1 * nspins*steps_per_spin) #for simulations before thermalization we don't need any data
    #nth = int(0) #for simulations after thermalization we want the data
    for J in Js:
        data.append((L, J, nspins*steps_per_spin, nth, nspins, n, np.nan)) #first run is without a lattice as one needs to be initialized
        #data.append((L, J, nspins*steps_per_spin, nth, nspins, n, "L{L}_J{J}.txt".format(L=L,J=J))) #after that we get the lattice from our previous simulations

data
np.savetxt("HPC stuff/mcvalues.csv", data, delimiter=",", header="L,J,nsteps,nth,nspin,n,file", fmt="%d,%0.1f,%d,%d,%d,%d,%f",comments="") #%s as last for next sims