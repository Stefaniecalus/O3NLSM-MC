
from func_flips import *
import time 
steps = 100
L = 6
nref = vec()
J = 0.3
acceptance = [0,0,0]
lattice = initial_lattice(L, nref)
lat_dic, lat_coords, spinvalues = lattice
e = energy(lattice, J)

start = time.time()
for i in range(steps):
    e = metropolis_step(lattice, nref, J, acceptance, e)
    # lat_dic, lat_coords, spinvalues = lattice
    # old_energy = energy(lattice, J)

    # #Define cone to give the flipped value a little nudge to avoid singularities
    # dx, dy = random.uniform(-np.pi/4, np.pi/4), random.uniform(-np.pi/4, np.pi/4)
    
    # #now pick random coord from lat_coords to flip 
    # flipcoord = random.choice(lat_coords)
    # OG = spinvalues[lat_coords.index(flipcoord)]
    # flip_values(lat_coords, spinvalues, flipcoord, [dx, dy, 0])

    # #look to which cube this spin belongs to
    # cubes = get_cubes(lat_dic, flipcoord)
    
    # #first look if this new configuration respects the hedgehog constraint
    # checks = []
    # for cube in cubes:
    #     update_flux(lattice, cube, nref)
    #     checks += [check_isolation(lat_dic, cube, nref)]
      
    # if np.sum(checks)==len(checks):
    #     #if the first contraint is respected now calculate the probability of acception
    #     new_energy = energy(lattice, J)
    #     dE = new_energy - old_energy

    #     if dE < 0 or np.random.rand() > np.exp(-dE):
    #         #If the Metropolis step is accepted we only still need to flip the dictionary value
    #         flip_dic(lat_dic, flipcoord, [dx, dy, 0])
    #         acceptance += [2]
        
    #     else:
    #         #If the Metropolis step is not accepted we need to flip the spinvalue and flux back to the old values
    #         spinvalues[lat_coords.index(flipcoord)] = OG
    #         for cube in cubes: update_flux(lattice, cube, nref)
    #         acceptance += [1]
            
            
    # else:
    #     #If the hedgehog constrained is not accepted we need to flip the spinvalue and flux back to the old values
    #     spinvalues[lat_coords.index(flipcoord)] = OG
    #     for cube in cubes: update_flux(lattice, cube, nref)
    #     acceptance += [0] 

end = time.time()
print((end-start)/steps)

# for L = 6
# time for initial_lattice = 0.4643947696685791s
# time for metropolis_step = 1.3-1.7s
# energy function: 0.42-0.45s -> done twice in metropolis_step
# everything except energy calcs: 0.01-0.02s


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