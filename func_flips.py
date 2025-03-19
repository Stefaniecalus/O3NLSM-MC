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

def center(x):
    """
    This function return x between (-pi, pi]
    """
    return (x % (2*np.pi)) - (2*np.pi)*((x % (2*np.pi)) // (((2*np.pi) + 1)//2))


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
    Make a desired amount (N) of random oriented 3-component unit spin vectors
    """
    spins = []
    for i in range(N):
        spins += [vec()]
    return spins


def get_neighbors(indices,L):
    """
    Give all neighbours of a cube based on the cube indices, with PBC
    """
    i,j,k = indices

    neighbors = [
        (i - 1, j, k), (i + 1, j, k),  # x neighbors
        (i, j - 1, k), (i, j + 1, k),  # y neighbors
        (i, j, k - 1), (i, j, k + 1)   # z neighbors
        ]
    return list(set([(int(ni%L), int(nj%L), int(nk%L)) for ni, nj, nk in neighbors]))


#Now we build up all useful functions to set up our MC criteria 
def spin_neighbours(coord, L):
    """
    Calculate the neighboring spins of a given coordinate with PBC
    """
    x,y,z = coord
    neighbors = []
    
    # spins on the cube vertices only have two neighbors
    if int(z) != z:
        return [(x,y,z+0.5), (x,y,z-0.5)]
    if int(y) != y:
        return [(x,y+0.5,z), (x,y-0.5,z)]
    if int(x) != x:
        return [(x+0.5,y,z), (x-0.5,y,z)]
    
    # spins on the cube edges have six neighbors
    neighbors += ([(x + 0.5,y,z), (x - 0.5,y,z)] if 0 < x < L else [(0,y,z), (x - 0.5,y,z)] if x == L else [(x + 0.5,y,z), (L,y,z)] if x == 0 else [])

    neighbors += ([(x,y + 0.5,z), (x,y - 0.5,z)] if 0 < y < L else [(x,0,z), (x,y - 0.5,z)] if y == L else [(x,y + 0.5,z), (x,L,z)] if y == 0 else [])

    neighbors += ([(x,y,z + 0.5), (x,y,z - 0.5)] if 0 < z < L else [(x,y,0), (x,y,z - 0.5)] if z == L else [(x,y,z + 0.5), (x,y,L)] if z == 0 else [])

    return neighbors
    


def gauge_pot(lat_coords, spinvalues, coordi, coordj, nref):
    """
    Calculate the gauge potential between two spins, this formula was taken from A. Vishwanath, O.I. Motrunich (2004)
    The reference vector is just a randomized vector since it doesn't matter which one is used as long as it doesn't 
    have the same spinvalues as a spinvector in our lattice
    """
    ni, nj = np.array(spinvalues[lat_coords.index(coordi)]), np.array(spinvalues[lat_coords.index(coordj)])
    
    dot_product = np.dot(nref, ni) + np.dot(nref, nj) + np.dot(ni, nj)
    cross_product = np.dot(nref, np.cross(ni, nj))
    
    A_ij = np.angle((1 + dot_product + 1j *  cross_product) 
                    / np.sqrt(2 * (1+np.dot(nref, ni)) * (1+np.dot(nref, nj)) * (1+np.dot(ni,nj)) ) )
    
    return A_ij


def get_sides(indices):
    """
    Return a dictionary of all six sides of a cube with the respective coordinates as values
    The coordinates are choosen such that the flux through the given side is pointing outwards
    """
    i,j,k = indices
    sides = {}
    sides_coords = []
    
    #xy(z=0) vlak
    sides_coords += [(0+i, 0+j, 0+k), (0+i, 0.5+j, 0+k), (0+i, 1+j, 0+k), (0.5+i, 1+j, 0+k),
                     (1+i, 1+j, 0+k), (1+i, 0.5+j, 0+k), (1+i, 0+j, 0+k), (0.5+i, 0+j, 0+k)] 
    
    #xy(z=1) vlak
    sides_coords += [(0+i, 0+j, 1+k), (0.5+i, 0+j, 1+k), (1+i, 0+j, 1+k), (1+i, 0.5+j, 1+k),
                     (1+i, 1+j, 1+k), (0.5+i, 1+j, 1+k), (0+i, 1+j, 1+k), (0+i, 0.5+j, 1+k)]

    #yz(x=0) vlak
    sides_coords += [(0+i, 0+j, 0+k), (0+i, 0+j, 0.5+k), (0+i, 0+j, 1+k), (0+i, 0.5+j, 1+k), 
                     (0+i, 1+j, 1+k), (0+i, 1+j, 0.5+k), (0+i, 1+j, 0+k), (0+i, 0.5+j, 0+k)]
    
    #yz(x=1) vlak
    sides_coords += [(1+i, 0+j, 0+k), (1+i, 0.5+j, 0+k), (1+i, 1+j, 0+k), (1+i, 1+j, 0.5+k), 
                     (1+i, 1+j, 1+k), (1+i, 0.5+j, 1+k), (1+i, 0+j, 1+k), (1+i, 0+j, 0.5+k)]
    
    #xz(y=0) vlak
    sides_coords += [(0+i, 0+j, 0+k), (0.5+i, 0+j, 0+k), (1+i, 0+j, 0+k), (1+i, 0+j, 0.5+k), 
                     (1+i, 0+j, 1+k), (0.5+i, 0+j, 1+k), (0+i, 0+j, 1+k), (0+i, 0+j, 0.5+k)]
    
    #xz(y=1) vlak
    sides_coords += [(0+i, 1+j, 0+k), (0+i, 1+j, 0.5+k), (0+i, 1+j, 1+k), (0.5+i, 1+j, 1+k),
                     (1+i, 1+j, 1+k), (1+i, 1+j, 0.5+k), (1+i, 1+j, 0+k), (0.5+i, 1+j, 0+k)]

    for s in range(6):
        sides.update({s+1: sides_coords[0+(8*s):8+(8*s)]})

    return sides 
    

def flux_side(lat_coords, spinvalues, side, nref):
    """
    Calculate the flux through a given cube side
    """
    flux = 0
    for i in range(8):
        flux += gauge_pot(lat_coords, spinvalues, side[i], side[(i+1)%8], nref) 

    return center(flux)


def flux_cube(lat_coords, spinvalues, indices, nref):
    """
    Calculate the total flux through all six sides of a cube
    """
    sides = get_sides(indices)
    flux = 0

    for side in sides.values():
        flux += flux_side(lat_coords, spinvalues, side, nref)

    return flux


#form lattice geometry using dictionaries
def initial_lattice(L, nref):
    """
    L represents the amount of cubes in each direction of our system (LxLxL)
    From this we build a dictionary which gives indices (i,j,k) to each cube and belonging to each cube we build up the coordinates, spins and flux
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
        lattice[indices][2] += flux_cube(lat_coords, spinvalues, indices, nref)

    return lattice, lat_coords, spinvalues


#For a given lattice we can calculate certain parameters of interest
def energy(lattice, J):
    """
    Calculate the energy of the spin system by just adding the ferromagnetic coupling energy between all pairs
    """
    lat_dic, lat_coords, spinvalues = lattice
    energy = 0

    for coordinate in lat_coords:
        neighbors = spin_neighbours(coordinate, len(lat_dic)**(1/3))

        for neighbor in neighbors:
            energy += np.dot(coordinate, neighbor)

    #Each pair is counted twice
    return -J * energy / 2 


def magnetization(lattice):
    """
    Check the total magnetization of the spin system
    """
    lat_dic, lat_coords, spinvalues = lattice
    V = len(lat_dic.keys())
    M = np.zeros(3)
    
    for spin in spinvalues:
        M += np.array(spin)
    
    return np.linalg.norm(M)/len(spinvalues)


def check_isolation(lattice, indices, nref):
    """
    Check wether every monopole is accompanied by an equally strong anti-monopole
    """
    lat_dic, lat_coords, spinvalues = lattice
    flux = flux_cube(lat_coords, spinvalues, indices, nref)
    n_flux = []
    
    if flux > 1e-6:
        neighbors = get_neighbors(indices, len(lat_dic)**(1/3))
        for neighbor in neighbors:
            n_flux += [flux_cube(lat_coords, spinvalues, neighbor, nref)] 
     
    
    return -flux in n_flux and np.sum(n_flux)==-flux


def flip_dic(lat_dic, flipcoord):
    for keys in lat_dic.keys():
            coords, spins, _ = lat_dic[keys]
            if flipcoord in coords:
                spins[coords.index(flipcoord)] *= -1
    

def flip_values(lat_coords, spinvalues, flipcoord):
    index = lat_coords.index(flipcoord)
    spinvalues[index] = tuple([-1*x for x in spinvalues[index]])

    
def get_cubes(lat_dic, flipcoord):
    cubes =[]
    for keys in lat_dic.keys():
        coords, spins, _ = lat_dic[keys]
        if flipcoord in coords:
            cubes += [keys]
    return cubes


#Now we set up the Metropolis step algorithm for our MCS
def metropolis_step(lattice, nref, J):
    lat_dic, lat_coords, spinvalues = lattice
    old_energy = energy(lattice, J)

    #now pick random coord from lat_coords to flip 
    flipcoord = random.choice(lat_coords)
    flip_values(lat_coords, spinvalues, flipcoord)

    #look to which cube this spin belongs to
    cubes = get_cubes(lat_dic, flipcoord)
    
    #first look if this new configuration respects the hedgehog constraint
    checks = []
    for cube in cubes:
        checks += [check_isolation(lattice, cube, nref)]
            
    if np.sum(checks)==len(checks):
        #if the first contraint is respected now calculate the probability of acception
        new_energy = energy(lattice, J)
        dE = new_energy - old_energy

        if dE < 0 or np.random.rand() > np.exp(-dE):
            #If the Metropolis step is accepted we only still need to flip the dictionary value
            flip_dic(lat_dic, flipcoord)
            print(1)
            

    else:
        #If the hedgehog constrained is not accepted we need to flip the spinvalue back to the old value
        flip_values(lat_coords, spinvalues, flipcoord)
        print(0)
    


def MCS(L, nref, J, n_steps):
    lattice = initial_lattice(L, nref)
    for _ in range(n_steps):
        metropolis_step(lattice, nref, J)

    E = energy(lattice, J)
    m = magnetization(lattice)
    return E,m