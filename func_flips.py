#import packages
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime 
from itertools import product
from math import ceil

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
        #spins += [vec()]
        #spins += [(1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3))]
        spins += [(0, 0, 1)]
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
    
    #Uniquely identify a spinvalue with all coordinates
    lat_coords = list(set(lat_coords))
    spinvalues = make_spins(len(lat_coords))
    
    #Fill in spin- and fluxvalues for every cube in dictionary 
    for i,j,k in product(range(L), range(L), range(L)):
        indices = (i,j,k)
        coordinates, spins, _ = lattice[indices]
        for i in coordinates:
            index = lat_coords.index(i)
            spins += [spinvalues[index]]
        lattice[indices][2] += flux_cube(lat_coords, spinvalues, indices, nref)

    return lattice, lat_coords, spinvalues


# required for floating point representation errors
def ceil_half_int(n):
    return ceil(2 * n) / 2

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
            neighbor = tuple([ceil_half_int(x) for x in neighbor])
            energy += np.dot(spinvalues[lat_coords.index(coordinate)], spinvalues[lat_coords.index(neighbor)])

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


def check_isolation(lat_dic, indices):
    """
    Check wether every monopole is accompanied by an equally strong anti-monopole
    """
    flux = lat_dic[indices][2]
    n_flux = []
     
    neighbors = get_neighbors(indices, len(lat_dic)**(1/3))
    for neighbor in neighbors:
        n_flux += [lat_dic[neighbor][2]] 
    
    return -flux in n_flux and abs(np.sum(n_flux)+flux) < 1e-6


def flip_dic(lat_dic, flipcoord, cone):
    """
    Flip the spinvalues in the dictionary of our lattice system
    """
    for keys in lat_dic.keys():
        coords, spins, flux = lat_dic[keys]
        if flipcoord in coords:
            spins[coords.index(flipcoord)] = tuple([-1*spins[coords.index(flipcoord)][x] + cone[x] for x in range(3)])
            spins[coords.index(flipcoord)] = spins[coords.index(flipcoord)]/np.linalg.norm(spins[coords.index(flipcoord)])
    
    
def flip_values(lat_coords, spinvalues, flipcoord, cone):
    """
    Flip the spinvalues in the list of spinvalues of our lattice system
    """
    index = lat_coords.index(flipcoord)
    spinvalues[index] = tuple([-1*spinvalues[index][x] + cone[x] for x in range(3)])
    spinvalues[index] = spinvalues[index]/np.linalg.norm(spinvalues[index])
    
     
def update_flux(lattice, indices, nref):
    """
    Update the flux in the lattice dictionary of our system after a spinflip 
    """
    lat_dic, lat_coords, spinvalues = lattice
    lat_dic[indices][2] = flux_cube(lat_coords, spinvalues, indices, nref)


def get_cubes(lat_dic, flipcoord):
    """
    Calculate the cubes the flipped spincoordinate belongs to
    """
    cubes =[]
    for keys in lat_dic.keys():
        coords, spins, _ = lat_dic[keys]
        if flipcoord in coords:
            cubes += [keys]
    return cubes


def get_cone(lat_coords, spinvalues, flipcoord, nref, L, eps=0.01):
    """
    If the flipping of flipcoord creates two anti-parallel spinvalues, then the gauge_pot() function will return some singularities
    We check wether a random small nudge along the x- and/or y-axis is necessary or not 
    """
    flip_values(lat_coords, spinvalues, flipcoord, [0, 0, 0])
    flip_nn = spin_neighbours(flipcoord, L)
    A = []

    for nn in flip_nn:
        nn = tuple([ceil_half_int(x) for x in nn])
    A += [gauge_pot(lat_coords, spinvalues, flipcoord, nn, nref)]
    
    #If any value in A is 'nan' then we will need to construct a cone around our spin-flip
    if np.isnan(A).any():
        dx = random.choice([-eps, 0, eps])
        dy = random.choice([-eps, 0, eps])
        flip_values(lat_coords, spinvalues, flipcoord, [0, 0, 0])
        return [dx, dy, 0]
    
    else: 
        flip_values(lat_coords, spinvalues, flipcoord, [0, 0, 0])
        return [0, 0, 0]


def change_energy(lattice, flipcoord, J):
    """
    Calculate the energy of the bond between a flipped coordinate and its neighbours
    """
    lat_dic, lat_coords, spinvalues = lattice
    energy = 0
    
    flip_nn = spin_neighbours(flipcoord, len(lat_dic)**(1/3))
    for nn in flip_nn:
        nn = tuple([ceil_half_int(x) for x in nn])
        energy += np.dot(spinvalues[lat_coords.index(flipcoord)], spinvalues[lat_coords.index(nn)])
        
    return -J * energy 


#Now we set up the Metropolis step algorithm for our MCS
def metropolis_step(lattice, nref, J, acceptance, E):
    lat_dic, lat_coords, spinvalues = lattice
    
    #now pick random coord from lat_coords to flip 
    flipcoord = random.choice(lat_coords)
    OG = spinvalues[lat_coords.index(flipcoord)]
    E_removed = change_energy(lattice, flipcoord, J)

    #Define cone to give the flipped value a little nudge to avoid singularities
    cone = get_cone(lat_coords, spinvalues, flipcoord, nref, len(lat_dic)**(1/3))
    flip_values(lat_coords, spinvalues, flipcoord, cone)
    E_added = change_energy(lattice, flipcoord, J)

    #look to which cube this spin belongs to
    cubes = get_cubes(lat_dic, flipcoord)
    
    #first look if this new configuration respects the hedgehog constraint
    checks = []
    for cube in cubes:
        update_flux(lattice, cube, nref)
        checks += [check_isolation(lat_dic, cube)]
      
    if np.sum(checks)==len(checks):
        #if the first contraint is respected now calculate the probability of acception
        dE = E_added-E_removed

        if dE < 0 or np.random.rand() > np.exp(-dE):
            #If the Metropolis step is accepted we only still need to flip the dictionary value
            flip_dic(lat_dic, flipcoord, cone)
            acceptance[2] += 1
        
        else:
            #If the Metropolis step is not accepted we need to flip the spinvalue and flux back to the old values
            spinvalues[lat_coords.index(flipcoord)] = OG
            for cube in cubes: update_flux(lattice, cube, nref)
            acceptance[1] += 1
            
        return E-E_removed+E_added  
      
    else:
        #If the hedgehog constrained is not accepted we need to flip the spinvalue and flux back to the old values
        spinvalues[lat_coords.index(flipcoord)] = OG
        for cube in cubes: update_flux(lattice, cube, nref)
        acceptance[0] += 1
        return E

    
def MCS(L, nref, J, n_steps):
    acceptance = [0,0,0] # hedgehog constraint denied, energy constraint denied, energy constraint accepted
    lattice = initial_lattice(L, nref)
    E = energy(lattice, J)
    for i in range(n_steps):
        print("Step {i} of {n_steps}".format(i=i, n_steps=n_steps))
        E = metropolis_step(lattice, nref, J, acceptance, E)

    E = energy(lattice, J)
    m = magnetization(lattice)
    return E,m, acceptance