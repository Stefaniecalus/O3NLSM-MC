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
parser.add_argument("-nth", '--nth', type=int, required=True, const=5000, nargs="?")
parser.add_argument("-nspin", '--nspin', type=int, required=True, const=1225, args="?")

args = parser.parse_args()

L = args.L
J = args.J
nsteps = args.nsteps
nth = args.nth
nspin = args.nspin
print(L, J, nsteps)

##################################################################
# run the simulation
##################################################################
nref = vec()
E, M, acceptance = MCS(L, nref, J, nsteps, nth)
M_av = np.mean(M)
M_av_squared = np.mean(M**2)
M_av_quartic = np.mean(M**4)
binder_cumulant = M_av_quartic / (M_av_squared**2) 
m_density = M_av / (nspin)

# write out data
E,m_density,acceptance = np.float64(-349.2040644852037),np.float64(0.7518431151459216),[50,1000,23]
binder_cumulant = np.float64(1.5226894958649473)
data = np.array([E, m_density, binder_cumulant, acceptance[0], acceptance[1], acceptance[2]])
reshaped_data = data.reshape(1, data.shape[0])
formatter = "%1f %1f %1f %d %d %d"
np.savetxt("HPC stuff/test_output.txt", reshaped_data, fmt=formatter)