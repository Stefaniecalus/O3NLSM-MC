from func_flips import *
import argparse
from pathlib import Path

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
parser.add_argument("-n", '--n', type=int, required=True, const=100, nargs="?")

args = parser.parse_args()
L = args.L
J = args.J
nsteps = args.nsteps
nth = args.nth
nspin = args.nspin
n = args.n
print(L, J, nsteps, nth, nspin, n)

##################################################################
# run the simulation
##################################################################
nref = vec()
E, M, acceptance = MCS(L, nref, J, nsteps, nth, n)
# E/M will be 112 + 8*n bytes each
M_av = np.sum(M)/nth
E_av = np.sum(E)/nth
M_av_squared = np.sum(M**2)/nth
E_av_squared = np.sum(E**2)/nth
M_av_quartic = np.sum(M**4)/nth
E_av_quartic = np.sum(E**4)/nth
binder_cumulant = M_av_quartic / (M_av_squared**2) 
binder_E = E_av_quartic / (E_av_squared**2)
m_density = M_av / (nspin)
m_var_dens = M_av_squared / (nspin)
E_last = E[-1] 

# write out data
data = np.array([E_last, m_density, m_var_dens, binder_cumulant, binder_E, acceptance[0], acceptance[1], acceptance[2]])
print(data)
reshaped_data = data.reshape(1, data.shape[0])
formatter = "%1f %1f %1f %1f %1f %d %d %d"
savepath = Path("Data", "L{L}_J{J}.txt".format(L=L,J=J))
np.savetxt(savepath, reshaped_data, fmt=formatter)

# test
# data = np.array([np.float64(123.321), np.float64(0.123), np.float64(0.456), np.float64(0.789), np.float64(0.012), 1, 2, 3])
# reshaped_data = data.reshape(1, data.shape[0])
# formatter = "%1f %1f %1f %1f %1f %d %d %d"
# L=16
# J=1.0
# savepath = Path("test", "L{L}_J{J}.txt".format(L=L,J=J))
# np.savetxt(savepath, reshaped_data, fmt=formatter)