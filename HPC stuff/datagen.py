import numpy as np

# number of spins in lattice
Ls = [(6,1225),(8,2673),(10,4961),(12,8281),(14,12825),(16,18785)]
Js = np.linspace(0.0, 1.4, 15, endpoint=True, dtype=float)
steps_per_spin = 5000

data = []
for (L, nspins) in Ls:
    nth = int(0.5 * nspins*steps_per_spin)
    for J in Js:
        data.append((L, J, nspins*steps_per_spin, nth, nspins))

data
np.savetxt("HPC stuff/mcvalues.csv", data, delimiter=",", header="L,J,nsteps,nth,nspins", fmt="%d,%0.1f,%d,%d,%d")