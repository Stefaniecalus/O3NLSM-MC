import numpy as np
import random

def make_coords(indices):
    """
    Determine the spin coordinates based on cube indices
    """
    i,j,k = indices

    return [(0+i,0+j,0+k), (0.5+i,0+j,0+k), (1+i,0+j,0+k), (0+i,0.5+j,0+k), (0+i,1+j,0+k),
            (0+i,0+j,0.5+k), (0+i,0+j,1+k), (1+i,0.5+j,0+k), (1+i,1+j,0+k), (0.5+i,1+j,0+k),
            (1+i,0+j,0.5+k), (0+i,1+j,0.5+k), (1+i,1+j,0.5+k), (0.5+i,0+j,1+k), (1+i,0+j,1+k),
            (0+i,0.5+j,1+k), (0+i,1+j,1+k), (1+i,0.5+j,1+k), (0.5+i,1+j,1+k), (1+i,1+j,1+k)]

def test_A(ni, nj, nref):

    dot_product = np.dot(nref, ni) + np.dot(nref, nj) + np.dot(ni, nj)
    cross_product = np.dot(nref, np.cross(ni, nj))
    
    A_ij = np.angle((1 + dot_product + 1j * cross_product)/ 
                    np.sqrt(2*(1+np.dot(nref, ni)) * (1 + np.dot(nref, nj))* (1+np.dot(ni,nj))))
    
    return A_ij  

def vec():
    """
    Make a random normalized three-component vector
    """
    theta = np.pi * random.random()
    phi = 2*np.pi * random.random()
    return (np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)) 

def center(x):
    return (x % (2*np.pi)) - (2*np.pi)*((x % (2*np.pi)) // (((2*np.pi) + 1)//2))

#The cubes will lie next to each other along the y-axis

spins000 = [(-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), 
            (0, -1/np.sqrt(2), -1/np.sqrt(2)),
            (1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), 
            (-1/np.sqrt(2), 0, -1/np.sqrt(2)), 
            (-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)),
            (-1/np.sqrt(2), -1/np.sqrt(2), 0),
            (-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)),
            (1/np.sqrt(2), 0, -1/np.sqrt(2)), 
            (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), 
            (0, 1/np.sqrt(2), -1/np.sqrt(2)),
            (1/np.sqrt(2), -1/np.sqrt(2), 0), 
            (-1/np.sqrt(2), 1/np.sqrt(2), 0), 
            (1/np.sqrt(2), 1/np.sqrt(2), 0), 
            (0, -1/np.sqrt(2), 1/np.sqrt(2)), 
            (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), 
            (-1/np.sqrt(2), 0, 1/np.sqrt(2)), 
            (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), 
            (1/np.sqrt(2), 0, 1/np.sqrt(2)),
            (0, 1/np.sqrt(2), 1/np.sqrt(2)), 
            (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3))]

spins010 = [(-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)),
           (0, 1/np.sqrt(2), -1/np.sqrt(2)),
           (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)),
           (-1/np.sqrt(6), 2/np.sqrt(6), -1/np.sqrt(6)),
           (-1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)),
           (-1/np.sqrt(2), 1/np.sqrt(2), 0),
           (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),
           (1/np.sqrt(6), 2/np.sqrt(6), -1/np.sqrt(6)),
           (1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)),
           (0, 3/np.sqrt(10), -1/np.sqrt(10)),
           (1/np.sqrt(2), 1/np.sqrt(2), 0),
           (-1/np.sqrt(10), 3/np.sqrt(10), 0),
           (1/np.sqrt(10), 3/np.sqrt(10), 0),
           (0, 1/np.sqrt(2), 1/np.sqrt(2)),
           (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),
           (-1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6)),
           (-1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11)),
           (1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6)),
           (0, 3/np.sqrt(10), 1/np.sqrt(10)),
           (1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11))]

cubes = {(0, 0, 0): [make_coords((0, 0, 0)), spins000], (0, 1, 0): [make_coords((0, 1, 0)), spins010]}

#First we want to check wether the spinsvalues are the same for spincoordinates that overlap in cube 1 and cube 2
#hint: there should be 8 overlapping spins

coords000, values000 = cubes[0, 0, 0]
coords010, values010 = cubes[0, 1, 0]

for coord in coords000:
    if coord in coords010:
        print(values000[coords000.index(coord)] == values010[coords010.index(coord)])


#Now we will calculate the gauge_potentials for all sides from all cubes
#side_6 from 000 == side_5 from 010 

sides000 = {1: [[(0,0,0), (0,0.5,0), (0,1,0), (0.5,1,0), (1,1,0), (1,0.5,0), (1,0,0), (0.5,0,0)], 
               [(-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), ( -1/np.sqrt(2), 0, -1/np.sqrt(2)), 
                (-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (0, 1/np.sqrt(2), -1/np.sqrt(2)), 
                (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), 0, -1/np.sqrt(2)), 
                (1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (0, -1/np.sqrt(2), -1/np.sqrt(2))]], 
                
            2: [[(0,0,1), (0.5,0,1), (1,0,1), (1,0.5,1), (1,1,1), (0.5,1,1), (0,1,1), (0,0.5,1)], 
               [(-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (0, -1/np.sqrt(2), 1/np.sqrt(2)), 
                (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), 0, 1/np.sqrt(2)), 
                (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (0, 1/np.sqrt(2), 1/np.sqrt(2)), 
                (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), 0, 1/np.sqrt(2))]],
             
            3: [[(0,0,0), (0,0,0.5), (0,0,1), (0,0.5,1), (0,1,1), (0,1,0.5), (0,1,0), (0,0.5,0)], 
               [(-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(2), -1/np.sqrt(2), 0),
                (-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), 0, 1/np.sqrt(2)), 
                (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), 1/np.sqrt(2),0),
                (-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(2), 0, -1/np.sqrt(2))]],

            4: [[(1,0,0), (1,0.5,0), (1,1,0), (1,1,0.5), (1,1,1),(1,0.5,1), (1,0,1), (1,0,0.5)], 
               [(1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), 0, -1/np.sqrt(2)),
                (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), 1/np.sqrt(2),0), 
                (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), 0, 1/np.sqrt(2)),
                (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), -1/np.sqrt(2), 0)]],

            5: [[(0,0,0), (0.5,0,0), (1,0,0), (1,0,0.5), (1,0,1), (0.5,0,1), (0,0,1), (0,0,0.5)], 
               [(-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (0, -1/np.sqrt(2), -1/np.sqrt(2)),
                (1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), -1/np.sqrt(2), 0),
                (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (0, -1/np.sqrt(2), 1/np.sqrt(2)),
                (-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), -1/np.sqrt(2), 0)]],

            6: [[(0,1,0), (0,1,0.5), (0,1,1), (0.5,1,1), (1,1,1), (1,1,0.5), (1,1,0), (0.5,1,0)], 
               [(-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(2), 1/np.sqrt(2), 0), 
                (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (0, 1/np.sqrt(2), 1/np.sqrt(2)),  
                (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), 1/np.sqrt(2), 0),              
                (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (0, 1/np.sqrt(2), -1/np.sqrt(2))]]
}


sides010 = {1: [[(0,1,0), (0,1.5,0), (0,2,0), (0.5,2,0), (1,2,0), (1,1.5,0), (1,1,0), (0.5,1,0)], 
              [(-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(6), 2/np.sqrt(6), -1/np.sqrt(6)), 
               (-1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)), (0, 3/np.sqrt(10), -1/np.sqrt(10)), 
               (1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)), (1/np.sqrt(6), 2/np.sqrt(6), -1/np.sqrt(6)), 
               (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (0, 1/np.sqrt(2), -1/np.sqrt(2))]],     
    
           2: [[(0,1,1), (0.5,1,1), (1,1,1), (1,1.5,1), (1,2,1), (0.5,2,1), (0,2,1), (0,1.5,1)], 
              [(-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (0, 1/np.sqrt(2), 1/np.sqrt(2)), 
               (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6)), 
               (1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11)), (0, 3/np.sqrt(10), 1/np.sqrt(10)), 
               (-1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11)), (-1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6))]],
             
           3: [[(0,1,0), (0,1,0.5), (0,1,1), (0,1.5,1), (0,2,1), (0,2,0.5), (0,2,0), (0,1.5,0)], 
              [(-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(2), 1/np.sqrt(2), 0), 
               (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6)), 
               (-1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11)), (-1/np.sqrt(10), 3/np.sqrt(10), 0), 
               (-1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)), (-1/np.sqrt(6), 2/np.sqrt(6), -1/np.sqrt(6))]],

           4: [[(1,1,0), (1,1.5,0), (1,2,0), (1,2,0.5), (1,2,1),(1,1.5,1), (1,1,1), (1,1,0.5)], 
              [(1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(6), 2/np.sqrt(6), -1/np.sqrt(6)), 
               (1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)), (1/np.sqrt(10), 3/np.sqrt(10), 0), 
               (1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11)), (1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6)), 
               (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), 1/np.sqrt(2), 0)]],

           5: [[(0,1,0), (0.5,1,0), (1,1,0), (1,1,0.5), (1,1,1), (0.5,1,1), (0,1,1), (0,1,0.5)], 
              [(-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (0, 1/np.sqrt(2), -1/np.sqrt(2)), 
               (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), 1/np.sqrt(2), 0), 
               (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (0, 1/np.sqrt(2), 1/np.sqrt(2)), 
               (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), 1/np.sqrt(2), 0)]],

           6: [[(0,2,0), (0,2,0.5), (0,2,1), (0.5,2,1), (1,2,1), (1,2,0.5), (1,2,0), (0.5,2,0)], 
              [(-1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)), (-1/np.sqrt(10), 3/np.sqrt(10), 0), 
               (-1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11)), (0, 3/np.sqrt(10), 1/np.sqrt(10)), 
               (1/np.sqrt(11), 3/np.sqrt(11), 1/np.sqrt(11)), (1/np.sqrt(10), 3/np.sqrt(10), 0), 
               (1/np.sqrt(11), 3/np.sqrt(11), -1/np.sqrt(11)), (0, 3/np.sqrt(10), -1/np.sqrt(10))]]
}

Flux000 = []
A000 = []
nref = vec()

#We first look at the cubes seperately and then look into the neighbouring sides

for key in sides000:
    side, spins = sides000[key]
    flux = 0
    a = []
    for i in range(8):
        A =  test_A(spins[i], spins[(i+1)%8], nref)
        flux += A
        a.append(A)
    Flux000.append(center(flux))
    A000.append(a)

print(Flux000) #check if monopole is there

Flux010 = []
A010 = []

for key in sides010:
    side, spins = sides010[key]
    flux = 0
    a = []
    for i in range(8):
        A =  test_A(spins[i], spins[(i+1)%8], nref)
        flux += A
        a.append(A)
    Flux010.append(center(flux))
    A010.append(a)


print(Flux010) #check there is no monopole

#Now we check side6 in 000 and side5 000
#Their fluxes should be equal and opposite, and their gauge potentials should be summed to zero in reverse

print(Flux000[-1], '\n', A000[-1], '\n', 
      "================================================", '\n',
      Flux010[-2], '\n', A010[-2])


one = A000[-1]
two = A010[-2]
two.reverse()

for i in range(8):
    print(one[i]+two[i])
