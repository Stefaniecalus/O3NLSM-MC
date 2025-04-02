
from func_flips import *
import time 
steps = 1000
L = 16
nref = vec()
J = 0.3
acceptance = [0,0,0]
lattice = initial_lattice(L)
lat_dic, lat_coords, spinvalues = lattice
len(lat_coords)
20000*8281*0.07053/3600
start = time.time()
e = energy(lattice, J)
end = time.time()
print(end-start)

start = time.time()
for i in range(steps):
    e = metropolis_step(lattice, nref, J, acceptance, e)
end = time.time()
print((end-start)/steps)
print(end-start)

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

def metropolis_steps(L, nref, J, steps):
    acceptance = []
    lattice = initial_lattice(L, nref)
    for i in range(steps):
        print("Step {i} of {n_steps}".format(i=i, n_steps=steps))
        metropolis_step(lattice, nref, J, acceptance)

    # E = energy(lattice, J)
    # m = magnetization(lattice)
    return acceptance

steps = 100
L = 6
nref = vec()
J = 0.3
acceptance = []

start = time.time()
acceptance = metropolis_steps(L, nref, J, steps)
end = time.time()
print((end-start)/steps)

start = time.time()
e,m,acc = MCS(L, nref, J, steps)
end = time.time()
print((end-start)/steps)


def newenergy(lattice, J):
    lat_dic, lat_coords, spinvalues = lattice
    energy = 0

    for coordinate in lat_coords:
        neighbors = spin_neighbours(coordinate, len(lat_dic)**(1/3))

        for neighbor in neighbors:
            neighbor = tuple([ceil_half_int(x) for x in neighbor])
            energy += np.dot(spinvalues[lat_coords.index(coordinate)], spinvalues[lat_coords.index(neighbor)])

    return -J * energy / 2 

def oldenergy(lattice, J):
    lat_dic, lat_coords, spinvalues = lattice
    energy = 0

    for coordinate in lat_coords:
        neighbors = spin_neighbours(coordinate, len(lat_dic)**(1/3))

        for neighbor in neighbors:
            energy += np.dot(spinvalues[lat_coords.index(coordinate)], spinvalues[lat_coords.index(neighbor)])

    return -J * energy / 2 

from math import ceil
def ceil_half_int(n):
    return ceil(2 * n) / 2

ceil_half_int(5.999999999999999)

steps = 200
L = 5
lattice = initial_lattice(L, nref)
start = time.time()
for i in range(steps):
    e = energy(lattice, J)
end = time.time()
print((end-start)/steps)

start = time.time()
for i in range(steps):
    e = newenergy(lattice, J)
end = time.time()
print((end-start)/steps)

start = time.time()
for i in range(steps):
    e = oldenergy(lattice, J)
end = time.time()
print((end-start)/steps)

for coordinate in lat_coords:
    neighbors = spin_neighbours(coordinate, len(lat_dic)**(1/3))
    print(neighbors)


# testing change_energy

L=6
nref = vec()
J = 0.3
steps = 10000
lattice = initial_lattice(L, nref)
lat_dic, lat_coords, spinvalues = lattice
start = time.time()
for i in range(steps):
    flipcoord = random.choice(lat_coords)
    e=change_energy(lattice, flipcoord, J)
    cone = get_cone(lat_coords, spinvalues, flipcoord, nref, len(lat_dic)**(1/3))
    flip_values(lat_coords, spinvalues, flipcoord, cone)
    E_added = change_energy(lattice, flipcoord, J)
end = time.time()
print((end-start)/steps)