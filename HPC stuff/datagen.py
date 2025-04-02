import numpy as np

# number of spins in lattice
Ls = [(6,1225),(8,2673),(10,4961),(12,8281),(14,12825),(16,18785)]
Js = np.linspace(0.0, 1.4, 15, endpoint=True, dtype=float)
steps_per_spin = 20000

data = []
for (L, nspins) in Ls:
    for J in Js:
        data.append((L, J, nspins*steps_per_spin))

data
np.savetxt("HPC stuff/mcvalues.csv", data, delimiter=",", header="L,J,nsteps", fmt="%d,%0.1f,%d")