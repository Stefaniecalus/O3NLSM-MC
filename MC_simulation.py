from func_flips import *
#from func_rot import * 

#Do simulations
L = 8
nref = vec()
J_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
n_steps = 5000
magnetizations = []
energies = []
accept = 0
decline_hedgehog = 0
decline_energy = 0


start_time = datetime.now()
for index, J in enumerate(J_values):
    t1 = datetime.now()
    print("index = {}, J = {}".format(index, J))
    E, m, acceptance = MCS(L, nref, J, n_steps)
    energies += [E]
    magnetizations += [m]
    
    #add the acceptance and decline rates: 
    # 0 means no constraint met, 1 means hedgehog constraint met, 2 means hedgehog and energy constraint met
    accept += acceptance[2]
    decline_hedgehog += acceptance[0]
    decline_energy += acceptance[1]
    t2 = datetime.now()
    print('Intermediate duration: {}'.format(t2-t1))


end_time = datetime.now()
DeltaT = end_time - start_time
print('Total duration: {}'.format(DeltaT))

#Calculate the mean acceptance rates over all J values:
accept = np.mean(accept)
decline_hedgehog = np.mean(decline_hedgehog)
decline_energy = np.mean(decline_energy)


print(magnetizations, energies, accept, decline_hedgehog, decline_energy )
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
fig.supxlabel('Duration time: {}\n Acceptance = {}, decline (hedgehog) = {}, decline (energy) = {}'.format(end_time-start_time, accept, decline_hedgehog, decline_energy))
# plt.savefig(f'Output/L={L}_nsteps={n_steps}.png')
plt.show()