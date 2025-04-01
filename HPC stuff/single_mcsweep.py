from func_flips import *
import argparse

###################################################################
# load the data from the csv file
###################################################################

parser = argparse.ArgumentParser(prog="single_mcsweep.py",
                                 description="Run a single MC simulation sweep with specified parameters.",
                                 epilog="Example usage: python single_mcsweep.py -L 16 -J 0.5 -n_steps 10000")

parser.add_argument("-L", '--L', type=int, required=True, const=6, nargs="?")
parser.add_argument("-J", '--J', type=float, required=True, const=0.3, nargs="?")
parser.add_argument("-nsteps", '--nsteps', type=int, required=True, const=10000, nargs="?")

args = parser.parse_args("--L --J --n_steps".split())

L = args.L
J = args.J
nsteps = args.nsteps

##################################################################
# run the simulation
##################################################################

E, m, acceptance = MCS(L, J, nsteps)

# write out data
E,m,acceptance = np.float64(-349.2040644852037),np.float64(0.7518431151459216),[50,1000,23]
data = np.array([E, m, acceptance[0], acceptance[1], acceptance[2]])
reshaped_data = data.reshape(1, data.shape[0])
formatter = "%1f %1f %d %d %d"
np.savetxt("HPC stuff/test_output.txt", reshaped_data, fmt=formatter)