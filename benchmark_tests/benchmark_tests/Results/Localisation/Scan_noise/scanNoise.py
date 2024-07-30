import numpy as np
import matplotlib.pyplot as plt

scanParams = np.loadtxt('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanParameters.csv', delimiter=',')
scans = np.zeros((int(scanParams[3]),5,5))
for i in range(5):
    scans[:,i] = np.loadtxt(f'/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanData_{i}.csv', delimiter=',')
angles = np.arange(scanParams[0], scanParams[1], scanParams[2])
hallWidth = 1.74
wallDistancesTests = np.array([[1.00, -0.83], [0.16, -1.36], [0.49, -0.64], [1.54, -0.30], [2.08, -1.01]]) #[front,side]

# Rotate the scans slightly to make them straight
rotatedScans = np.zeros((int(scanParams[3]), 5, 5))
rotationAngle = np.array([-0.015,-0.025,0.025,-0.05,-0.018])  # Adjust the rotation angle as needed
# rotationAngle = np.array([0.0,0.0,0.0,0.0,0.0])  # Adjust the rotation angle as needed

for i in range(5):
    for j in range(5):
        rotatedScans[:, i, j] = np.roll(scans[:, i, j], int(rotationAngle[i] / scanParams[2]))
scans = rotatedScans

testArray = np.zeros((int(scanParams[3]),5))
for i in range(5):
    cornerAngle1 = np.arctan2(wallDistancesTests[i,1], wallDistancesTests[i,0])
    cornerAngle2 = np.arctan2(wallDistancesTests[i,1]+hallWidth, wallDistancesTests[i,0])
    for angle in range(len(angles)):
        if angles[angle] <= cornerAngle1:
            testArray[angle,i] = max(0,min(2.5,abs(wallDistancesTests[i,1]/np.sin(angles[angle]))))
        elif angles[angle] > cornerAngle2:
            testArray[angle,i] = max(0,min(2.5,abs((hallWidth+wallDistancesTests[i,1])/np.sin(angles[angle]))))
        else:
            testArray[angle,i] = max(0,min(2.5,abs(wallDistancesTests[i,0]/np.cos(angles[angle]))))

error = np.zeros((int(scanParams[3]),5,5))
for i in range(5):
    for j in range(len(angles)):
        for k in range(5):
            error[j,i,k] = (-(testArray[j,i] - scans[j,i,k]))
# print(error)

rmse = np.zeros((5))
sigma = np.zeros((5))
print("RMSE")
for i in range(5):
    rmse[i] = np.sqrt(np.sum(error[50:1000,i,1]**2)/len(error[50:1000,i,1]))
    print(i,rmse[i])

def standardDeviation(array):
    return np.sqrt(np.sum(array**2)/len(array))

print("standard deviation")
for i in range(5):
    sigma[i] = standardDeviation(error[50:1000,i])
    print(i,sigma[i])
# print(error[50:1000,:].shape)

# aE = np.zeros((4750,5))
# for i in range(5):
#     aE[:,i] = error[50:1000,:,i].reshape(-1)
#     # print(aE[:,i])
# # aE = error[50:1000,:].reshape(-1,5)
# # print(aE.shape)
# print("standard deviation")
# for i in range(5):
#     print(i,standardDeviation(aE[:,i]))


# plt.figure()
for i in range(5):
    plt.figure(i)
    for j in range(5):
        # plt.plot(error[50:1000,i,j], label=f"Test {j}")
        plt.hist(error[50:1000,i,j], bins=100, label=f"Test {j}")
        
    # plt.hist(error[50:1000,i,0],bins=100, label=f"Test {i}")
    # plt.hist(error[:,i,0],bins=1000, label=f"Test {i}")
    # plt.hist(aE[:,i],bins=1000, label=f"Test {i}")


colors = ['r','g','b','y','m']
plt.figure()
for i in range(5):
    for j in range(5):
        plt.polar(angles[50:1000], scans[50:1000,i,j],color=colors[i])
        plt.polar(angles[50:1000], testArray[50:1000,i], color='black',linestyle='--',linewidth=1)
plt.show()




