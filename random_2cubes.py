import numpy as np
import random

#Now we check for two random cubes if the initialization and calculations according to our original code works 

def vec():
    """
    Make a random normalized three-component vector
    """
    theta = np.pi * random.random()
    phi = 2* np.pi * random.random()
    return (np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)) 

def center(x):
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
    Make a desired amount of random oriented 3-component unit spin vectors
    """
    spins = []
    for i in range(N):
        spins += [vec()]
    return spins

def test_A(ni, nj, nref):

    dot_product = np.dot(nref, ni) + np.dot(nref, nj) + np.dot(ni, nj)
    cross_product = np.dot(nref, np.cross(ni, nj))
    
    A_ij = np.angle((1 + dot_product + 1j * cross_product)/ 
                    np.sqrt(2*(1+np.dot(nref, ni)) * (1 + np.dot(nref, nj))* (1+np.dot(ni,nj))))
    
    return A_ij

def gauge_pot(lat_coords, spinvalues, coordi, coordj, nref):
    """
    Calculate the gauge potential between two spins, this formula was taken from A. Vishwanath, O.I. Motrunich (2004)
    """
    ni, nj = np.array(spinvalues[lat_coords.index(coordi)]), np.array(spinvalues[lat_coords.index(coordj)])

    dot_product = np.dot(nref, ni) + np.dot(nref, nj) + np.dot(ni, nj)
    cross_product = np.dot(nref, np.cross(ni, nj))
    
    A_ij = np.angle((1 + dot_product + 1j *  cross_product) 
                    / np.sqrt(2 * (1+np.dot(nref, ni)) * (1+np.dot(nref, nj)) * (1+np.dot(ni,nj)) ) )
    
    return A_ij

def flux_side(lat_coords, spinvalues, side):
    """
    Calculate the flux through a given cube side
    """
    flux = gauge_pot(lat_coords, spinvalues, side[-1], side[0], nref)

    for spin in range(len(side)-1):
        flux += gauge_pot(lat_coords, spinvalues, side[spin], side[spin+1], nref)

    return center(flux)

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

indices = [(0,0,0), (0,1,0)]
lat_coords = []
coords000 = make_coords((0,0,0))
coords010 = make_coords((0, 1, 0))

lat_coords += coords000
lat_coords += coords010

cubes = {(0, 0, 0): [coords000, []], (0, 1, 0): [coords010, []]}
    
lat_coords = list(set(lat_coords))
spinvalues = make_spins(len(lat_coords))

for ind in indices:
    coordinates, spins = cubes[ind]
    for i in coordinates:
        index = lat_coords.index(i)
        spins += [spinvalues[index]]


#Now check wether the spins are not ambigously defined
coords000, values000 = cubes[(0, 0, 0)]
coords010, values010 = cubes[(0, 1, 0)]

for coord in coords000:
    if coord in coords010:
        print(values000[coords000.index(coord)])
        print(values000[coords000.index(coord)] == values010[coords010.index(coord)])


nref = vec()

#Now we will calculate the fluxes for each side of each cube

Flux000 = []
A000 = []

sides000 = get_sides((0,0,0))

for side in sides000:
    spins = []
    side_coords = sides000[side]
    for coord in side_coords:
        spins += [spinvalues[lat_coords.index(coord)]]
    flux = 0
    a = []
    for i in range(8):
        A =  test_A(spins[i], spins[(i+1)%8], nref)
        flux += A
        a.append(A)
    Flux000.append(center(flux))
    A000.append(a)
    
print(Flux000, np.sum(Flux000)/(2*np.pi))
for i in range(1,7):
    print(flux_side(lat_coords, spinvalues, sides000[i]))


Flux010 = []
A010 = []

sides010 = get_sides((0,1,0))

for side in sides010:
    spins = []
    side_coords = sides010[side]
    for coord in side_coords:
        spins += [spinvalues[lat_coords.index(coord)]]
    flux = 0
    a = []
    for i in range(8):
        A =  test_A(spins[i], spins[(i+1)%8], nref)
        flux += A
        a.append(A)
    Flux010.append(center(flux))
    A010.append(a)
    
print(Flux010, np.sum(Flux010)/(2*np.pi))
for i in range(1,7):
    print(flux_side(lat_coords, spinvalues, sides010[i]))


#Now review the overlapping spins, fluxes and gauge potentials: side_6 for 000 and side_5 for 010

print(Flux000[-1], '\n', A000[-1], '\n', 
      "================================================", '\n',
      Flux010[-2], '\n', A010[-2])


one = A000[-1]
two = A010[-2]
two.reverse()

for i in range(8):
    print(one[i]+two[i])
