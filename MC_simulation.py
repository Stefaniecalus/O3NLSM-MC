from func_flips import *
#from func_rot import * 

#Do simulations
L = 8
nref = vec()
J_values = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
n_steps = 50000
magnetizations = []
energies = []


#this will only calculate J=0, chance 1 to J_values for full calculations
start_time = datetime.now()
for index, J in enumerate(J_values):
    t1 = datetime.now()
    print("index = {}, J = {}".format(index, J))
    E, m = MCS(L, nref, J, n_steps)
    energies += [E]
    magnetizations += [m]
    t2 = datetime.now()
    print('Intermediate duration: {}'.format(t2-t1))

end_time = datetime.now()

DeltaT = end_time - start_time
print('Total duration: {}'.format(DeltaT))

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(8, 5))
axs[0].plot(J_values, magnetizations, marker='o', color='firebrick')
axs[0].set_ylabel("Magnetization per spin")

axs[1].plot(J_values, energies, marker='o', color='firebrick')
axs[1].set_ylabel("Energy of system")

for i in range(2):
    axs[i].set_xlabel("Exchange Interaction J")

plt.subplots_adjust(wspace=0.6, bottom=0.2)
fig.suptitle('MC simulations with nsteps = {}, L = {}'.format(n_steps, L))
fig.supxlabel('Duration time: {}'.format(end_time-start_time))
plt.savefig(f'Output/L={L}_nsteps={n_steps}.png')
plt.show()