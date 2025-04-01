import numpy as np

Ls = [6,8,10,12]
Js = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]

data = []
for L in Ls:
    for J in Js:
        data.append((L, J))

data
np.savetxt("HPC stuff/data.csv", data, delimiter=",", header="L,J", fmt="%d,%0.1f")