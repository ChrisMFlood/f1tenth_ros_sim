import numpy as np
import math
import matplotlib.pyplot as plt

scanParams = np.loadtxt('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanParameters.csv', delimiter=',')
scans = np.zeros((int(scanParams[3]),5,5))
for i in range(5):
    scans[:,i] = np.loadtxt(f'/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanData_{i}.csv', delimiter=',')
angles = np.arange(scanParams[0], scanParams[1], scanParams[2])

for i in range(5):
    for j in range(5):
        plt.polar(angles, scans[:,j,i])
    # plt.polar(angles, scans[:,0,i])
plt.show()
