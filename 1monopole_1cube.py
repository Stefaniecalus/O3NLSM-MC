import numpy as np
import random

def vec():
    """
    Make a random normalized three-component vector
    """
    theta = np.pi * random.random()
    phi = 2* np.pi * random.random()
    return (np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)) 

def test_A(ni, nj, nref):

    dot_product = np.dot(nref, ni) + np.dot(nref, nj) + np.dot(ni, nj)
    cross_product = np.dot(nref, np.cross(ni, nj))
    
    A_ij = np.angle((1 + dot_product + 1j * cross_product)/ 
                    np.sqrt(2*(1+np.dot(nref, ni)) * (1 + np.dot(nref, nj))* (1+np.dot(ni,nj))))
    
    return A_ij  

#test cube 
side_1 = {1: [[(0,0,0), (0,0.5,0), (0,1,0), (0.5,1,0), (1,1,0), (1,0.5,0), (1,0,0), (0.5,0,0)], 
              [(-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), ( -1/np.sqrt(2), 0, -1/np.sqrt(2)), 
               (-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (0, 1/np.sqrt(2), -1/np.sqrt(2)), 
               (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), 0, -1/np.sqrt(2)), 
               (1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (0, -1/np.sqrt(2), -1/np.sqrt(2))]]}     
    
side_2 = {2: [[(0,0,1), (0.5,0,1), (1,0,1), (1,0.5,1), (1,1,1), (0.5,1,1), (0,1,1), (0,0.5,1)], 
              [(-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (0, -1/np.sqrt(2), 1/np.sqrt(2)), 
               (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), 0, 1/np.sqrt(2)), 
               (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (0, 1/np.sqrt(2), 1/np.sqrt(2)), 
               (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), 0, 1/np.sqrt(2))]]}
             
side_3 = {3: [[(0,0,0), (0,0,0.5), (0,0,1), (0,0.5,1), (0,1,1), (0,1,0.5), (0,1,0), (0,0.5,0)], 
              [(-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(2), -1/np.sqrt(2), 0),
               (-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), 0, 1/np.sqrt(2)), 
               (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), 1/np.sqrt(2),0),
               (-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(2), 0, -1/np.sqrt(2))]]}

side_4 = {4: [[(1,0,0), (1,0.5,0), (1,1,0), (1,1,0.5), (1,1,1),(1,0.5,1), (1,0,1), (1,0,0.5)], 
              [(1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), 0, -1/np.sqrt(2)),
               (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), 1/np.sqrt(2),0), 
               (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), 0, 1/np.sqrt(2)),
               (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), -1/np.sqrt(2), 0)]]}

side_5 = {5: [[(0,0,0), (0.5,0,0), (1,0,0), (1,0,0.5), (1,0,1), (0.5,0,1), (0,0,1), (0,0,0.5)], 
              [(-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (0, -1/np.sqrt(2), -1/np.sqrt(2)),
               (1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(2), -1/np.sqrt(2), 0),
               (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (0, -1/np.sqrt(2), 1/np.sqrt(2)),
               (-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(2), -1/np.sqrt(2), 0)]]}

side_6 = {6: [[(0,1,0), (0,1,0.5), (0,1,1), (0.5,1,1), (1,1,1), (1,1,0.5), (1,1,0), (0.5,1,0)], 
              [(-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (-1/np.sqrt(2), 1/np.sqrt(2), 0), 
               (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (0, 1/np.sqrt(2), 1/np.sqrt(2)),  
               (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(2), 1/np.sqrt(2), 0),              
               (1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)), (0, 1/np.sqrt(2), -1/np.sqrt(2))]]}


nster_werkt = (0.6466697277920891, -0.4211833840278718, 0.6359424660896108)


for values in side_1.values():
    testcoord = values[0]
    testspins = values[1]

F_side1 = 0
for i in range(8):
    F_side1 += test_A(testspins[i], testspins[(i+1)%8], nster_werkt)
    print('A{}{} equals'.format(i+1, (i+2)%8), test_A(testspins[i], testspins[(i+1)%8], nster_werkt))
print('F_side1 equals', F_side1, 'with nvec=',nster_werkt)

for values in side_2.values():
    testcoord = values[0]
    testspins = values[1]
    
F_side2 = 0
for i in range(8):
    F_side2 += test_A(testspins[i], testspins[(i+1)%8], nster_werkt)
    print('A{}{} equals'.format(i+1, (i+2)%8), test_A(testspins[i], testspins[(i+1)%8], nster_werkt))
print('F_side2 equals', F_side2, 'with nvec=',nster_werkt)

for values in side_3.values():
    testcoord = values[0]
    testspins = values[1]
    
F_side3 = 0
for i in range(8):
    F_side3 += test_A(testspins[i], testspins[(i+1)%8], nster_werkt)
    print('A{}{} equals'.format(i+1, (i+2)%8), test_A(testspins[i], testspins[(i+1)%8], nster_werkt))
print('F_side3 equals', F_side3, 'with nvec=',nster_werkt)

for values in side_4.values():
    testcoord = values[0]
    testspins = values[1]
    
F_side4 = 0
for i in range(8):
    F_side4 += test_A(testspins[i], testspins[(i+1)%8], nster_werkt)
    print('A{}{} equals'.format(i+1, (i+2)%8), test_A(testspins[i], testspins[(i+1)%8], nster_werkt))
print('F_side4 equals', F_side4, 'with nvec=',nster_werkt)

for values in side_5.values():
    testcoord = values[0]
    testspins = values[1]
    
F_side5 = 0
for i in range(8):
    F_side5 += test_A(testspins[i], testspins[(i+1)%8], nster_werkt)
    print('A{}{} equals'.format(i+1, (i+2)%8), test_A(testspins[i], testspins[(i+1)%8], nster_werkt))
print('F_side5 equals', F_side5, 'with nvec=',nster_werkt)

for values in side_6.values():
    testcoord = values[0]
    testspins = values[1]
    
F_side6 = 0
for i in range(8):
    F_side6 += test_A(testspins[i], testspins[(i+1)%8], nster_werkt)
    print('A{}{} equals'.format(i+1, (i+2)%8), test_A(testspins[i], testspins[(i+1)%8], nster_werkt))
print('F_side6 equals', F_side6, 'with nvec=',nster_werkt)

center_mod = lambda x: (x % (2*np.pi)) - (2*np.pi)*((x % (2*np.pi)) // (((2*np.pi) + 1)//2))

F = [F_side1, F_side2, F_side3, F_side4, F_side5, F_side6]
for i in range(len(F)):
    F[i] = center_mod(F[i])
print(F, np.sum(F)/(2*np.pi))