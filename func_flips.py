#import packages
import numpy as np
import random
from itertools import product
from math import ceil
from collections import deque
import pickle
import matplotlib.pyplot as plt
from datetime import datetime


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
        flip_values(lat_coords, spinvalues, flipcoord, [0, 0, 0]) #We return the coneless spinflip to its orignal value 
        theta = 2* np.pi * random.random()
        return [eps*np.cos(theta), eps*np.sin(theta), 0]
    
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


def loop_nn_strong(cube, L):
    i,j,k = cube

    neighbors = [
        (i - 1, j, k), (i + 1, j, k),  # x neighbors
        (i, j - 1, k), (i, j + 1, k),  # y neighbors
        (i, j, k - 1), (i, j, k + 1),  # z neighbors
        (i - 1, j - 1, k), (i + 1, j - 1, k), (i - 1, j + 1, k), (i + 1, j + 1, k), #xy neighbours 
        (i - 1, j, k -1), (i + 1, j, k - 1), (i - 1, j, k + 1), (i + 1, j, k + 1), #xz neighbours 
        (i, j - 1, k - 1), (i, j - 1, k + 1), (i, j + 1, k - 1), (i, j + 1, k + 1), #yz neighbours
        (i + 1, j + 1, k + 1), (i + 1, j + 1, k - 1), (i + 1, j - 1, k + 1), (i + 1, j - 1, k - 1),
        (i - 1, j + 1, k + 1), (i - 1, j + 1, k - 1), (i - 1, j - 1, k + 1), (i - 1, j - 1, k - 1) #corners of loop
        ]
    return list(set([(int(ni%L), int(nj%L), int(nk%L)) for ni, nj, nk in neighbors]))

def check_nn(cube1, cube2, L):
    neighbours1 = get_neighbors(cube1, L)
    return cube2 in neighbours1

def loop(lat_dic, cube1, cube2, L):
    loop = list(loop_nn_strong(cube1, L) + loop_nn_strong(cube2, L))
    loop.remove(cube1)
    loop.remove(cube2)
    loop_fluxes = np.zeros(len(loop))
    for i,cubes in enumerate(loop):
        loop_fluxes[i] = lat_dic[cubes][2]
    
    return np.all(np.abs(loop_fluxes) < 1e-10)

def check_cips(cip, L):
    check = 1
    for cips in cip:
        nns = get_neighbors(cips, L)
        check*= np.any(nns in cip)
    return check

def pair_check(cubes, pairs, L):
    nr_cip = 0
    cip =  []

    for cube in cubes:
        if cube in pairs:
            nr_cip += 1
            cip += [cube]

    if nr_cip%2==0 and check_cips(cip, L):
        return True
    else:
        return False

def update_pairs(lattice, pairs):
    for keys in pairs:
        if np.abs(lattice[keys][2]) < 1e-10:
            pairs.remove(keys)

def zeropaircheck(cubes, fluxes, pairs):
    # Step 1: Get indices of (near-)zero fluxes
    zeros = np.where(np.abs(fluxes) < 1e-10)[0]

    # Step 2: Get corresponding cubes
    zubes = set(cubes[z] for z in zeros)

    # Step 3: Build a lookup dictionary from pairs
    pair_dict = {p: q for p, q in zip(pairs[::2], pairs[1::2])}
    pair_dict.update({q: p for p, q in zip(pairs[::2], pairs[1::2])})  # make it bidirectional

    # Step 4: Check that each zub in zubes has its pair in zubes (if it's in pairs at all)
    check = all(
        pair_dict.get(zub, zub) in zubes
        for zub in zubes
    )

    return int(check)

def check_hedgehog(lattice, flipcoord, nref, L, pairs):
    lat_dic, lat_coords, spinvalues = lattice
    cubes = get_cubes(lat_dic, flipcoord)
    fluxes = np.zeros(len(cubes))

    #Fill in array with new fluxes of all cubes impacted by the flip of flipcoord
    for i, cube in enumerate(cubes):
        update_flux(lattice, cube, nref)
        fluxes[i] = lat_dic[cube][2]
    
    #Check wether these create a monopole or not
    #If all fluxes are zero and no pairs are broken up then there are no monopoles created so there is no problem 
    if np.all(np.abs(fluxes) < 1e-10) and pair_check(cubes, pairs, L):
        return True, 0, 0
    
    #If all fluxes sum to zero and there are only two nonzero values we have to check wether these are nn and isolated
    elif np.abs(np.sum(fluxes)) < 1e-10 and np.count_nonzero(np.abs(fluxes) > 1e-3) == 2 and zeropaircheck(cubes, fluxes, pairs):
        #Get the indices of the nonzero elements in fluxes to retreive the cubes in which the monopoles lie
        indices = np.nonzero(np.abs(fluxes) > 1e-3)[0]
        cube1 = cubes[indices[0]]
        cube2 = cubes[indices[1]]
        
        #check if these cubes are nearest neighbours and if all surrounding cubes do not contain monopoles
        if check_nn(cube1, cube2, L) and loop(lat_dic, cube1, cube2, L):
            return True, cube1, cube2
        else:
            return False, 0, 0
        
    #For all other cases there is no way we created none or isolated monopoles
    else:
        return False, 0, 0


#Now we set up the Metropolis step algorithm for our MCS
def metropolis_step(lattice, nref, J, acceptance, E, pairs):
    lat_dic, lat_coords, spinvalues = lattice
    L = len(lat_dic)**(1/3)
    
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
    check = check_hedgehog(lattice, flipcoord, nref, L, pairs)
    
    #first look if this new configuration respects the hedgehog constraint
    if check[0]:
        #if the first contraint is respected now calculate the probability of acception
        dE = E_added-E_removed

        if dE < 0 or np.random.rand() <= np.exp(-dE):
            #If the Metropolis step is accepted we only still need to flip the dictionary value and put the cubes into our pairlist
            flip_dic(lat_dic, flipcoord, cone)
            if check[1] !=0:
                pairs += [check[1], check[2]]
            acceptance[2] += 1
            return E-E_removed+E_added, pairs
        
        else:
            #If the Metropolis step is not accepted we need to flip the spinvalue and flux back to the old values
            spinvalues[lat_coords.index(flipcoord)] = OG
            for cube in cubes: 
                update_flux(lattice, cube, nref)
            update_pairs(lat_dic, pairs)
            acceptance[1] += 1
            return E, pairs
      
    else:
        #If the hedgehog constrained is not accepted we need to flip the spinvalue and flux back to the old values
        spinvalues[lat_coords.index(flipcoord)] = OG
        for cube in cubes: 
            update_flux(lattice, cube, nref)
        update_pairs(lat_dic, pairs)
        acceptance[0] += 1
        return E, pairs
    


def MCS(L, nref, J, n_steps, n_th, n, lattice_input=0):
    acceptance = [0,0,0] # hedgehog constraint denied, energy constraint denied, energy constraint accepted
    Nmem = n_th//n
    E, M = np.zeros(n), np.zeros(n)
    if lattice_input==0:
        lattice = initial_lattice(L)
    else:
        lattice = lattice_input
    e = energy(lattice, J)
    for i in range(n_steps):
        e = metropolis_step(lattice, nref, J, acceptance, e)
        if i > n_th:
            E[(i-n_th)//Nmem] += e
            M[(i-n_th)//Nmem] += magnetization(lattice)
    return E, M, acceptance, lattice


def hedgehog_constraint(lattice, flipcoord, nref, L, cone_in):
    lat_dic, lat_coords, spinvalues = lattice

    #Keep original value of flipcoord for later use
    OG = spinvalues[lat_coords.index(flipcoord)]

    #Define cone to give the flipped value a little nudge to avoid singularities if necessary 
    if cone_in == None:
        cone = get_cone(lat_coords, spinvalues, flipcoord, nref, len(lat_dic)**(1/3))
    else:
        cone = cone_in
    
    flip_values(lat_coords, spinvalues, flipcoord, cone)
    newvalue = spinvalues[lat_coords.index(flipcoord)]

    checks = check_hedgehog(lattice, flipcoord, nref, L)
    
    #now switch everyhting back to how it was
    return checks, OG, newvalue, cone


def cluster_check(nbr_OG, OG, eps):
    #these two vectors are either arrays or tuples so we wil write a general way to write if they are similar
    nbr_x, nbr_y, nbr_z = nbr_OG
    og_x, og_y, og_z = OG
    return (abs(nbr_x-og_x) < eps) * (abs(nbr_y -og_y) < eps) * (abs(nbr_z - og_z) < eps)


#Now we set up the Wolff cluster algorithm for our MCS
def Wolff_cluster(lattice, nref, J, pairs):
    lat_dic, lat_coords, spinvalues = lattice
    L = len(lat_dic)**(1/3)
    p_add = np.exp(-2*J)
    cluster_size = np.zeros(1)
    #Pick random coord from lat_coords to flip to start the cluster on
    flipcoord = random.choice(lat_coords)
    checks, OG, newvalue, cone = hedgehog_constraint(lattice, flipcoord, nref, L, None, pairs)

    if checks[0]:
    #if the hedgehog constraint is respected, start forming the cluster which must also respect the hedgehog constraint for every spin added
        cluster_size[0] += 1
        if checks[1] !=0:
            pairs += [checks[1], checks[2]]
        update_pairs(lat_dic, pairs)
        unvisited = deque([flipcoord]) #use a deque to efficiently track the unvisited cluster sites
        while unvisited: #while unvisited sites remain
            clustercoord = unvisited.pop()  #take one and remove from the unvisited list
            for nbr in spin_neighbours(clustercoord, L):
                nbr = tuple([ceil_half_int(x) for x in nbr])
                nbr_check, nbr_OG, nbr_new, _ = hedgehog_constraint(lattice, nbr, nref, L, cone, pairs)
                if nbr_check[0] and cluster_check(nbr_OG, OG, 1e-2) and np.random.rand() < p_add: 
                    cluster_size[0] += 1
                    if nbr_check[1] !=0:
                        pairs += [nbr_check[1], nbr_check[2]]
                    update_pairs(lat_dic, pairs)
                    unvisited.appendleft(nbr)
                else:
                    spinvalues[lat_coords.index(nbr)] = nbr_OG
                    cubes = get_cubes(lat_dic, nbr)
                    for cube in cubes:
                        update_flux(lattice, cube, nref)
                    
    
    else:
    #if the constrained is not respected we need to flip everything back
        spinvalues[lat_coords.index(flipcoord)] = OG
        cubes = get_cubes(lat_dic, flipcoord)
        for cube in cubes:
            update_flux(lattice, cube, nref)

    if len(pairs)%2!=0:
        print('STOP JE MIST EEN PAIR')
    elif np.abs(np.sum(pairs)) > 1e10:
        print('MAKKER KDENK NIE DAJE MONOPOLEN OKE ZIJN HOOR')
    return cluster_size, pairs


def WCS(L, nref, J, nsteps, nth, n, lattice_input=0):
    Nmem = nth//n
    M = np.zeros(n)
    if lattice_input==0:
        lattice = initial_lattice(L)
    else:
        lattice = lattice_input
    e = energy(lattice, J)
    for i in range(nsteps):
        clustersize = Wolff_cluster(lattice, nref, J)
        if i > nth:
            M[(i-nth)//Nmem] += magnetization(lattice)
    return M, lattice


# The following functions help us read out and write out our data
def write_to_file(lattice_out, nlast, filename):
    lattice, latcoords, spins = lattice_out
    data = {
        'lattice': lattice,
        'latcoords': latcoords,
        'spins': spins,
        'n': nlast
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    
def get_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Expecting a dict with keys: 'lattice', 'latcoords', 'spins', 'n'
    new_lattice = data['lattice']
    new_latcoords = data['latcoords']
    new_spins = data['spins']
    new_n = data['n']
    
    return new_lattice, new_latcoords, new_spins, new_n

def keys_to_matrix(lattice, L):
    fluxes = np.zeros((L, L, L))
    for keys in lattice.keys():
        fluxes[keys] = lattice[keys][2]
    return fluxes
