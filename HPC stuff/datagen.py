import numpy as np
import csv

# Number of spins in lattice
Ls = [(6,1225),(8,2673),(10,4961),(12,8281),(14,12825),(16,18785)]
Js = np.linspace(0.0, 1.4, 15, endpoint=True, dtype=float)
steps_per_spin = 10
n = 100  # Split calculating averages and such into n parts

# Data collection
data = []
for (L, nspins) in Ls:
    nth = int(1 * nspins * steps_per_spin)  # For simulations before thermalization we don't need any data
    # nth = int(0)  # For simulations after thermalization we want the data
    for J in Js:
        data.append((L, f"{J:.1f}", nspins * steps_per_spin, nth, nspins, n, "L{L}_J{J:.1f}.txt".format(L=L, J=J)))

# Save to CSV directly in the "HPC stuff" folder
csv_file = "HPC stuff/mcvalues.csv"

header = ["L", "J", "nsteps", "nth", "nspin", "n", "file"]
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write the header
    writer.writerows(data)   # Write the data

print(f"Wrote {len(data)} rows to '{csv_file}'")
