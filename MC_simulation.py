from functions import *

#Do simulations
L = 3
nref = vec()
J_values = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
n_steps = 50000
magnetizations = []
energies = []


#this will only calculate J=0, chance 1 to J_values for full calculations
start_time = datetime.now()
for J in J_values:
    E, m = MCS(L, nref, J, n_steps)
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