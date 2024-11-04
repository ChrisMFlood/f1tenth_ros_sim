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
diffdrive = np.loadtxt('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/aut_diffdrive.csv', delimiter=',')[:,:2800]
mit = np.loadtxt('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/aut_mit.csv', delimiter=',')[:,:2800]

def meanSquareError(data):
    return np.sqrt(np.mean(data**2))

def standardDeviation(data):
    return np.std(data)

def numberOutliers(pError, hError):
    pError = np.array(pError).reshape(-1)
    hError = np.array(hError).reshape(-1)
    meanP = np.mean(pError)
    meanH = np.mean(hError)
    stdP = np.std(pError)
    stdH = np.std(hError)
    outliers = 0
    for i in range(len(hError)):

        if hError[i] > meanH + 2*stdH or hError[i] < meanH - 2*stdH or pError[i] > meanP + 2*stdP or pError[i] < meanP - 2*stdP:
            outliers +=1
    return outliers

def analysPFData(res,mit):
    time = res[:,0]+res[:,1]

    poseErrorRes = np.sqrt((res[:,2]-res[:,6])**2 + (res[:,3]-res[:,7])**2)
    poseErrorMit = np.sqrt((mit[:,2]-mit[:,6])**2 + (mit[:,3]-mit[:,7])**2)

    realYawAngle = np.zeros(len(time))
    pfYawAngle = np.zeros(len(time))

    for i in range(len(time)):
        realYawAngle[i] = euler_from_quaternion(0,0,res[i,4],res[i,5])
        pfYawAngle[i] = euler_from_quaternion(0,0,res[i,8],res[i,9])
    angleDiff = getAngleDiff(realYawAngle,pfYawAngle)

    realYawAngle = np.zeros(len(mit[:,0]))
    pfYawAngle = np.zeros(len(mit[:,0]))

    for i in range(len(mit[:,0])):
        realYawAngle[i] = euler_from_quaternion(0,0,mit[i,4],mit[i,5])
        pfYawAngle[i] = euler_from_quaternion(0,0,mit[i,8],mit[i,9])
    angleDiff1 = getAngleDiff(realYawAngle,pfYawAngle)
    a=2
    plt.figure()
    plt.plot(res[:int(len(res[:,0])/a),2],res[:int(len(res[:,0])/a),3], label='Path')
    plt.plot(res[:int(len(res[:,0])/a),6],res[:int(len(res[:,0])/a),7], label='DD')
    plt.plot(mit[:int(len(mit[:,0])/a),6],mit[:int(len(mit[:,0])/a),7], label='MIT')
    # plt.title('Path')
    plt.legend()
    plt.savefig('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/path.png')
    # plt.show()
    plt.figure()
    plt.plot(range(len(time)), res[:,2]-res[:,6], label='DD')
    plt.plot(range(len(mit[:,2])), mit[:,2]-mit[:,6], label='MIT')
    plt.title('X Error (m)')
    plt.legend()
    plt.savefig('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/x_error.png')
    # plt.show()
    plt.figure()
    plt.plot(range(len(time)), res[:,3]-res[:,7], label='True Position Y')
    plt.plot(range(len(mit[:,3])), mit[:,3]-mit[:,7], label='MIT')
    plt.title('Y Error (m)')
    plt.legend()
    plt.savefig('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/y_error.png')
    # plt.show()
    plt.figure()
    plt.plot(range(len(time)), poseErrorRes, label='DD')
    plt.plot(range(len(mit[:,0])), poseErrorMit, label='MIT')
    plt.title('Position Error (m)')
    plt.legend()
    plt.savefig('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/position_error.png')
    # plt.show()
    plt.figure()
    plt.plot(range(len(angleDiff[0])), angleDiff[0], label='DD')
    plt.plot(range(len(angleDiff1[0])), angleDiff1[0], label='MIT')
    plt.title('Heading Error (rad)')
    plt.legend()
    plt.savefig('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/heading_error.png')
    # plt.show()


    print('MIT')
    print('Mean Square Error P:', meanSquareError(poseErrorMit))
    print('Standard Deviation P:', standardDeviation(poseErrorMit))
    # print('Number of Outliers P:', numberOutliers(poseErrorMit))
    print('Mean Square Error H:', meanSquareError(angleDiff1))
    print('Standard Deviation H:', standardDeviation(angleDiff1))
    print('Number of Outliers:', numberOutliers(poseErrorMit,angleDiff1))

    print('DD')
    print('Mean Square Error P:', meanSquareError(poseErrorRes))
    print('Standard Deviation P:', standardDeviation(poseErrorRes))
    # print('Number of Outliers P:', numberOutliers(poseErrorRes))
    print('Mean Square Error H:', meanSquareError(angleDiff))
    print('Standard Deviation H:', standardDeviation(angleDiff))
    print('Number of Outliers:', numberOutliers(poseErrorRes,angleDiff))






def main():
    analysPFData(diffdrive,mit)
    

if __name__ == '__main__':
    main()




