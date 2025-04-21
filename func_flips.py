#import packages
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime 
from itertools import product
from math import ceil
import re
import ast


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
    This function returns x between (-pi, pi]
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
def initial_lattice(L):
    """
    L represents the amount of cubes in each direction of our system (LxLxL)
    From this we build a dictionary which gives indices (i,j,k) to each cube and belonging to each cube we build up the coordinates, spins and flux
    """
    lattice = {}
    lat_coords = []
    
    for i,j,k in product(range(L), range(L), range(L)):
        
        #Make cube indices, coordinates, spinvalues and fluxes for a polarized state
        indices = (i,j,k)
        coordinates = make_coords(indices)
        spins = [(0, 0, 1)]*20
        flux = 0

        lattice.update({indices: [coordinates, spins, flux]})
        
        #Fill in all coordinates in a single list for later use
        lat_coords += coordinates
    
    #Make sure you only keep the unique coordinates and related spinvalues
    lat_coords = list(set(lat_coords))
    spinvalues = [(0, 0, 1)]*len(lat_coords)
    
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

    # Create a dictionary for fast index lookup
    coord_to_index = {coord: idx for idx, coord in enumerate(lat_coords)}

    # Loop through each coordinate and its neighbors
    for coordinate in lat_coords:
        neighbors = spin_neighbours(coordinate, len(lat_dic)**(1/3))

        for neighbor in neighbors:
            # Convert neighbor coordinates to integer form only once
            neighbor = tuple(ceil_half_int(x) for x in neighbor)

            # Ensure the pair (coordinate, neighbor) is only computed once (avoid double counting)
            if coord_to_index[coordinate] < coord_to_index[neighbor]:
                # Add energy contribution for this pair
                energy += np.dot(spinvalues[coord_to_index[coordinate]], spinvalues[coord_to_index[neighbor]])

    return -J * energy


def magnetization(lattice):
    """
    Check the total magnetization of the spin system by adding all x-, -y and z-components respectively
    and then taking the norm of that vector
    """
    lat_dic, lat_coords, spinvalues = lattice
    M = np.zeros(3)
    
    for spin in spinvalues:
        M += np.array(spin)
    
    return np.linalg.norm(M)


def check_isolation(lat_dic, indices):
    """
    Check wether every monopole is accompanied by one equally strong anti-monopole
    """
    flux = lat_dic[indices][2]
    n_flux = []
     
    neighbors = get_neighbors(indices, len(lat_dic)**(1/3))
    for neighbor in neighbors:
        n_flux += [lat_dic[neighbor][2]] 
    
    return -flux in n_flux and abs(np.sum(n_flux)+flux) < 1e-6


def flip_dic(lat_dic, flipcoord, cone):
    """
    Flip a spinvalue in the dictionary of our lattice system based on a chosen coordinate.
    """
    for coords, spins, flux in lat_dic.values():
        if flipcoord in coords:
            idx = coords.index(flipcoord)  # Find the index once
            # Flip the spin value with the cone adjustment and normalize it
            spins[idx] = tuple([-1*spins[idx][x] + cone[x] for x in range(3)])
            spins[idx] /= np.linalg.norm(spins[idx])  # Normalize the spin
    

def flip_values(lat_coords, spinvalues, flipcoord, cone):
    """
    Flip a spinvalues in the list of spinvalues of our lattice system based on a chosen coordinate
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
    Calculate the cubes the flipped spincoordinate belongs to.
    """
    return [key for key, (coords, _, _) in lat_dic.items() if flipcoord in coords]


def get_cone(lat_coords, spinvalues, flipcoord, nref, L, eps=0.01):
    """
    If the flipping of flipcoord creates two anti-parallel spinvalues, then the gauge_pot() function will return some singularities
    We check wether a random small nudge eps along the x- and/or y-axis is necessary or not 
    """
    #First flip the chosen coordinate 
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
        flip_values(lat_coords, spinvalues, flipcoord, [0, 0, 0]) #We return the coneless spinflip to its orignal value 
        return [dx, dy, 0]
    
    else: 
        flip_values(lat_coords, spinvalues, flipcoord, [0, 0, 0]) #We return the coneless spinflip to its orignal value
        return [0, 0, 0]


def change_energy(lattice, flipcoord, J):
    """
    Calculate the energy of the bond between a flipped coordinate and its neighbors
    """
    lat_dic, lat_coords, spinvalues = lattice
    energy = 0

    # Get the neighbors of the flipped coordinate
    flip_nn = spin_neighbours(flipcoord, len(lat_dic)**(1/3))
    
    # Precompute the index of flipcoord for efficiency
    flip_index = lat_coords.index(flipcoord)

    # Calculate the energy
    for nn in flip_nn:
        # Convert neighbor coordinates and get their index once
        nn_ceil = tuple([ceil_half_int(x) for x in nn])
        nn_index = lat_coords.index(nn_ceil)
        
        # Access the spin values directly
        energy += np.dot(spinvalues[flip_index], spinvalues[nn_index])

    return -J * energy

def chirality_z(lattice, L, nref):
    lat_dict, latcoords, spinvalues = lattice
    #Czz(r) =  [Czz(1), Czz(2), ... ,Czz(L)]
    Czz = np.zeros(L-1)
 
    #First choose a z=1 cube as reference Fz(0)
    for i, j in product(range(L), range(L)):
        zero = get_sides((i, j, 0))
        Fv = flux_side(latcoords, spinvalues, zero[4], nref)
        #Now calculate Fz(r) for all r along the z-axis for this reference cube
        for k in range(1, L):
            r = get_sides((i,j,k))
            Fmu = flux_side(latcoords, spinvalues, r[4], nref)
            #Add the chirality-chirality correlation to the Czz(r) vector based on r
            Czz[k-1] += np.sin(Fv) * np.sin(Fmu)
    
    #return the mean value of all correlations per r
    return Czz


def chirality_y(lattice, L, nref):
    lat_dict, latcoords, spinvalues = lattice
    #Cyy(r) =  [Cyy(1), Cyy(2), ... ,Cyy(L)]
    Cyy = np.zeros(L-1)
 
    #First choose a y=1 cube as reference Fy(0)
    for i, j in product(range(L), range(L)):
        zero = get_sides((i, j, 0))
        Fv = flux_side(latcoords, spinvalues, zero[4], nref)
        #Now calculate Fz(r) for all r along the z-axis for this reference cube
        for k in range(1, L):
            r = get_sides((i,j,k))
            Fmu = flux_side(latcoords, spinvalues, r[4], nref)
            #Add the chirality-chirality correlation to the Czz(r) vector based on r
            Cyy[k-1] += np.sin(Fv) * np.sin(Fmu)
    
    #return the mean value of all correlations per r
    return Cyy


#Now we set up the Metropolis step algorithm for our MCS
def metropolis_step(lattice, nref, J, acceptance, E):
    lat_dic, lat_coords, spinvalues = lattice
    
    #Pick random coord from lat_coords to flip 
    flipcoord = random.choice(lat_coords)
    OG = spinvalues[lat_coords.index(flipcoord)]
    E_removed = change_energy(lattice, flipcoord, J)
    

    #Define cone to give the flipped value a little nudge to avoid singularities if necessary 
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

        if dE < 0 or np.random.rand() <= np.exp(-dE):
            #If the Metropolis step is accepted we only still need to flip the dictionary value
            flip_dic(lat_dic, flipcoord, cone)
            acceptance[2] += 1
            return E-E_removed+E_added 
        
        else:
            #If the Metropolis step is not accepted we need to flip the spinvalue and flux back to the old values
            spinvalues[lat_coords.index(flipcoord)] = OG
            for cube in cubes: update_flux(lattice, cube, nref)
            acceptance[1] += 1
            return E
      
    else:
        #If the hedgehog constrained is not accepted we need to flip the spinvalue and flux back to the old values
        spinvalues[lat_coords.index(flipcoord)] = OG
        for cube in cubes: update_flux(lattice, cube, nref)
        acceptance[0] += 1
        return E


def MCS(L, nref, J, n_steps, n_th, n, file=np.nan):
    acceptance = [0,0,0] # hedgehog constraint denied, energy constraint denied, energy constraint accepted
    Nmem = n_th//n
    E, M = np.zeros(n), np.zeros(n)
    if file==np.nan:
        lattice = initial_lattice(L)
    e = energy(lattice, J)
    for i in range(n_steps):
        e = metropolis_step(lattice, nref, J, acceptance, e)
        if i > n_th:
            E[(i-n_th)//Nmem] += e
            M[(i-n_th)//Nmem] += magnetization(lattice)
    return E, M, acceptance, lattice


# The following functions help us read out and write out our data
def write_to_file(lattice_out, n_last, filename):
    lattice, latcoords, spinvalues = lattice_out
    with open(filename, 'w') as input:
        input.write(f'{n_last} \n')
        for key in lattice:
            input.write(f'{key}?{lattice[key][0]}?{lattice[key][1]}?{lattice[key][2]}\n')
        input.write(f'{latcoords} \n')
        input.write(f'{spinvalues}')


def get_from_file(filename):
    with open(filename, 'r') as input:
        lines = input.readlines()

    new_n = 0  
    new_lattice = {}
    new_spins = []
    new_latcoords = []  
    for e, line in enumerate(lines):
        if e == 0:
            new_n = int(line)
        elif e == len(lines) - 1:
            new_spins = ast.literal_eval(line.split(' \n')[0])
        elif e == len(lines) - 2:
            new_latcoords = ast.literal_eval(line.split(' \n')[0])
        else:
            parts = line.split(' \n')[0].split('?')
            indice = ast.literal_eval(parts[0])
            coords = ast.literal_eval(parts[1])
            spins = ast.literal_eval(parts[2])
            flux = float(ast.literal_eval(parts[3]))
            new_lattice.update({indice: [coords, spins, flux]})
    
    return new_lattice, new_latcoords, new_spins, new_n