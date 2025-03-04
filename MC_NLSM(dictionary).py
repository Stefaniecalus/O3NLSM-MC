#import packages
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime 
from itertools import product



#Some functions that help initialize our lattice
def vec():
    """
    Make a random normalized three-component vector
    """
    theta = np.pi * random.random()
    phi = 2* np.pi * random.random()
    return (np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)) 

def make_coords(indices):
    """
    Determine the spin coordinates based on cube indices
    """
    i,j,k = indices

    return [(0+i,0+j,0+k), (0.5+i,0+j,0+k), (1+i,0+j,0+k), (0+i,0.5+j,0+k), (0+i,1+j,0+k),
            (0+i,0+j,0.5+k), (0+i,0+j,1+k), (1+i,0.5+j,0+k), (1+i,1+j,0+k), (0.5+i,1+j,0+k),
            (1+i,0+j,0.5+k), (0+i,1+j,0.5+k), (1+i,1+j,0.5+k), (0.5+i,0+j,1+k), (1+i,0+j,1+k),
            (0+i,0.5+j,1+k), (0+i,1+j,1+k), (1+i,0.5+j,1+k), (0.5+i,1+j,1+k), (1+i,1+j,1+k)]


def make_spins(N):
    """
    Put random three-component unit spin vectors on every lattice coordinate per cube
    """
    spins = []
    for i in range(N):
        spins += [vec()]
    return spins


def get_neighbors(indices,L):
    """
    Give all neighbours of a cube based on the cube indices
    """
    i,j,k = indices

    neighbors = [
        (i - 1, j, k), (i + 1, j, k),  # x neighbors
        (i, j - 1, k), (i, j + 1, k),  # y neighbors
        (i, j, k - 1), (i, j, k + 1)   # z neighbors
        ]
    return [(ni, nj, nk) for ni, nj, nk in neighbors if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L]


#Now we build up all useful functions to set up our MC criteria 
def spin_neighbors(lat_coords, coord):
    x,y,z = coord
    
    if z != float:
        neighbors = [(x,y,z+0.5), (x,y,z-0.5)]
    if y != float:
        neighbors = [(x,y+0.5,z), (x,y-0.5,z)]
    if x != float:
        neighbors = [(x+0.5,y,z), (x-0.5,y,z)]
    
    neighbors = [(x+0.5,y,z), (x-0.5,y,z), (x,y+0.5,z), 
                 (x,y-0.5,z), (x,y,z-0.5), (x,y,z+0.5)]
    
    for n in neighbors:
        if n not in lat_coords:
            neighbors.remove(n)

    return neighbors


def gauge_pot(lat_coords, spinvalues, coordi, coordj):
    ni, nj = np.array(spinvalues[lat_coords.index(coordi)]), np.array(spinvalues[lat_coords.index(coordj)])
    nref = np.array(vec())

    dot_product = np.dot(nref, ni) * np.dot(nref, nj) * np.dot(ni, nj)
    cross_product = np.dot(nref, np.cross(ni, nj))
    
    A_ij = np.angle((1 + dot_product + 1j * cross_product)/ 
                    np.sqrt((1+np.dot(nref, ni)) * (1 + np.dot(nref, nj))* (1+np.dot(ni,nj))))
    
    return A_ij


def get_sides(indices):
    i,j,k = indices
    sides = {}
    sides_coords = []
    
    sides_coords += [(0+i, 0+j, 0+k), (0+i, 0.5+j, 0+k), (0+i, 1+j, 0+k), (0.5+i, 0+j, 0+k), 
                (1+i, 0+k, 0+k), (1+i, 0.5+j, 0+k), (1+i, 1+j, 0+k), (0.5+i, 1+j, 0+k)]
    
    sides_coords += [(0+i, 0+j, 1+k), (0.5+i, 0+j, 1+k), (1+i, 0+j, 1+k), (1+i, 0.5+j, 1+k),
                 (1+i, 1+j, 1+k), (0.5+i, 1+j, 1+k), (0+i, 1+j, 1+k), (0+i, 0.5+j, 1+k)]
    
    sides_coords += [(0+i, 0+j, 0+k), (0+i, 0+j, 0.5+k), (0+i, 0+j, 1+k), (0+i, 0.5+j, 1+k), 
                 (0+i, 1+j, 1+k), (0+i, 1+j, 0.5+k), (0+i, 1+j, 0+k), (0+i, 0.5+j, 0+k)]
    
    sides_coords += [(1+i, 0+j, 0+k), (1+i, 0.5+j, 0+k), (1+i, 1+j, 0+k), (1+i, 1+j, 0.5+k), 
                 (1+i, 1+j, 1+k), (1+i, 0.5+j, 1+k), (1+i, 0+j, 1+k), (1+i, 0+j, 0.5+k)]
    
    sides_coords += [(0+i, 0+j, 0+k), (0.5+i, 0+j, 0+k), (1+i, 0+j, 0+k), (1+i, 0+j, 0.5+k), 
                (1+i, 0+j, 1+k), (0.5+i, 0+j, 1+k), (0+i, 0+j, 1+k), (0+i, 0+j, 0.5+k)]
    
    sides_coords += [(0+i, 1+j, 0+k), (0+i, 1+j, 0.5+k), (0+i, 1+j, 1+k), (0.5+i, 1+j, 1+k),
                 (1+i, 1+j, 1+k), (1+i, 1+j, 0.5+k), (1+i, 1+j, 0+k), (0.5+i, 1+j, 0+k)]

    for s in range(6):
        sides.update({s+1: sides_coords[0+s:7+s]})

    return sides 
    

def flux_side(lat_coords, spinvalues, side):
    flux = gauge_pot(lat_coords, spinvalues, side[-1], side[0])

    for spin in range(len(side)-1):
        flux += gauge_pot(lat_coords, spinvalues, side[spin], side[spin+1])

    return flux


def flux_cube(lat_coords, spinvalues, indices):
    sides = get_sides(indices)
    flux = 0

    for side in sides.values():
        flux += flux_side(lat_coords, spinvalues, side)

    return flux


#form lattice geometry using dictionaries
def initial_lattice(L):
    """
    Form a dictionary with keys of lattice coordinates and 3-component unit spin vectors as values

    There are 20 coordinates in one cube which is measured by length L, however for L > 2 we create
    spins that are lying in multiple cubes at the same time.
    """
    lattice = {}
    lat_coords = []
    
    for i,j,k in product(range(L), range(L), range(L)):
        
        #Make cube indices, coordinates
        indices = (i,j,k)
        coordinates = make_coords(indices)
        spins = [] 
        flux = 0

        lattice.update({indices: [coordinates, spins, flux]})
        
        #Fill in all coordinates in a single list for later use
        lat_coords += coordinates
    
    lat_coords = list(set(lat_coords))
    spinvalues = make_spins(len(lat_coords))
    
    for i,j,k in product(range(L), range(L), range(L)):
        indices = (i,j,k)
        coordinates, spins, _ = lattice[indices]
        for i in coordinates:
            index = lat_coords.index(i)
            spins += [spinvalues[index]]
    
    for i,j,k in product(range(L), range(L), range(L)):
        indices = (i,j,k)
        _, _, flux = lattice[indices]
        flux += flux_cube(lat_coords, spinvalues, indices)

    return lattice, lat_coords, spinvalues


#For a given lattice we can calculate certain parameters of interest
def energy(lattice, J):
    lat_dic, lat_coords, spinvalues = lattice
    energy = 0

    for coordinate in lat_coords:
        neighbors = spin_neighbors(lat_coords, coordinate)

        for neighbor in neighbors:
            energy += np.dot(coordinate, neighbor)

    #Each pair is counted twice
    return -J * energy / 2 


def magnetization(lattice):
    lat_dic, lat_coords, spinvalues = lattice
    V = len(lat_dic.keys())
    M = np.zeros(3)
    
    for spin in spinvalues:
        M += np.array(spin)
    
    return np.linalg.norm(M)/len(spinvalues)


def check_isolation(lattice, indices):
    lat_dic, lat_coords, spinvalues = lattice
    n_flux = 0
    checks = []
    
    flux = flux_cube(lat_coords, spinvalues, indices)
    if flux != 0:
        neighbors = get_neighbors(indices, len(lat_dic)**(1/3))

        for neighbor in neighbors:
            n_flux += flux_cube(lat_coords, spinvalues, neighbor)
            checks += [flux-n_flux]

        return  checks, neighbors
    return None 


#Now we set up the Metropolis step algorithm for our MCS
def metropolis_step(lattice, J):
    lat_dic, lat_coords, spinvalues = lattice
    old_energy = energy(lattice, J)

    #now pick random coord from lat_coords to flip 
    flip_coords = random.choice(lat_coords)
    index = lat_coords.index(flip_coords)
    spinvalues[index] = tuple([-1*x for x in spinvalues[index]])

    #look to which cube this spin belongs to
    cubes = []
    for keys in lat_dic.keys():
        coords, spins, _ = lat_dic[keys]
        if flip_coords in coords:
            spins[coords.index(flip_coords)] *= -1
            cubes += [keys]
    
    #first look if this new configuration respects the hedgehog constraint
    fluxes = []
    neighbors = []
    for cube in cubes:
        fluxes += [flux_cube(lat_coords, spinvalues, cube)]
        neighbors.append([get_neighbors(cube, len(lat_dic)**(1/3))])
    
    for i in range(len(fluxes)):
        if fluxes[i] != np.sum(neighbors[i]):
            spinvalues[index] = tuple([-1*x for x in spinvalues[index]])
            for keys in lat_dic.keys():
                coords, spins, _ = lat_dic[keys]
                if flip_coords in coords:
                    spins[coords.index(flip_coords)] *= -1


    #if the first contraint is respected now calculate the probability of acception
    new_energy = energy(lattice, J)
    dE = new_energy - old_energy

    if dE > 0 and np.random.rand() > np.exp(-dE):
        spinvalues[index] = tuple([-1*x for x in spinvalues[index]])
        for keys in lat_dic.keys():
            coords, spins, _ = lat_dic[keys]
            if flip_coords in coords:
                spins[coords.index(flip_coords)] *= -1


def MCS(L, J, n_steps):
    lattice = initial_lattice(L)
    for _ in range(n_steps):
        metropolis_step(lattice, J)

    E = energy(lattice, J)
    m = magnetization(lattice)
    return E,m


#Do simulations
start_time = datetime.now()
L = 2
J_values = [0, 0.5, 1, 1.5, 2]
n_steps = 500

magnetizations = []
energies = []

for J in J_values:
    E, m = MCS(L, J, n_steps)
    energies += [E]
    magnetizations += [m]

end_time = datetime.now()
print('Duration: {}'.format(end_time-start_time))


# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(J_values, magnetizations, marker='o')
plt.xlabel("Exchange Interaction J")
plt.ylabel("Magnetization per Spin")
plt.show()