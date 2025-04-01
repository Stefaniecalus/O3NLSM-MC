import numpy as np

# number of spins in lattice
# 1225 2673 4961 8281 12825 18785
Ls = [(6,1225),(8,2673),(10,4961),(12,8281),(14,12825),(16,18785)]


Js = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
steps_per_spin = 20000

data = []
for (L, nspins) in Ls:
    for J in Js:
        data.append((L, J, nspins*steps_per_spin))

data
np.savetxt("HPC stuff/mcvalues.csv", data, delimiter=",", header="L,J,nsteps", fmt="%d,%0.1f,%d")


from func_flips import *
import time 
def time_estim(steps=1000, L=6, J=0.5):
    acceptance = [0,0,0]
    nref = vec()
    lattice = initial_lattice(L)
    e = energy(lattice, J)
    start = time.time()
    for _ in range(steps):
        e = metropolis_step(lattice, nref, J, acceptance, e)
    end = time.time()

    return (end-start)/steps

times = []
for L in [14, 16]:
    print(L)
    times.append(time_estim(steps=1000, L=L, J=0.5))
times
np.array(times)*1000