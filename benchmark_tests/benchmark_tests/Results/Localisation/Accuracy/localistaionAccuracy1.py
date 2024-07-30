import numpy as np
import matplotlib.pyplot as plt
import math

def euler_from_quaternion(x, y, z, w):  

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return yaw_z # in radians

def getAngleDiff(arr1,arr2):
    v1 =np.array([np.cos(arr1), np.sin(arr1)]).reshape(2,-1)
    v2 =np.array([np.cos(arr2), np.sin(arr2)]).reshape(2,-1)
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2).reshape(1,-1)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2).reshape(1,-1)
    dot = np.array([v1[0]*v2[0] + v1[1]*v2[1]]).reshape(1,-1)
    angle = np.arccos(dot/(mag1*mag2))
    return angle

# res = np.loadtxt('benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/gbr_1.csv', delimiter=',')
res = np.loadtxt('benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/cornerHall_1.csv', delimiter=',')


time = res[:,0]+res[:,1]

realYawAngle = np.zeros(len(time))
pfYawAngle = np.zeros(len(time))

for i in range(len(time)):
    realYawAngle[i] = euler_from_quaternion(0,0,res[i,4],res[i,5])
    pfYawAngle[i] = euler_from_quaternion(0,0,res[i,8],res[i,9])

angleDiff = getAngleDiff(realYawAngle,pfYawAngle)

plt.plot(range(len(time)), res[:,2]-res[:,6], label='True Position X')
plt.show()
plt.plot(range(len(time)), res[:,3]-res[:,7], label='True Position Y')
plt.show()
plt.plot(range(len(time)), angleDiff[0], label='True Angle')
plt.show()




