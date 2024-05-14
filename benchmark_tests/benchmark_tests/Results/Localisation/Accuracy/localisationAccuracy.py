import csv
import numpy as np


def read_csv_file(file_path):
    times = []
    x = []
    y = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            if len(row) != 3:
                raise ValueError("Each row must have exactly three columns")
            times.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
    first_time = times[0]
    for i in range(len(times)):
        times[i] -= first_time
    return times, x, y


trueOdomFilePath = '/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/trueOdom.csv' 
pfOdomFilePath = '/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/pfOdom.csv'

trueTimeLong, trueXLong, trueYLong = read_csv_file(trueOdomFilePath)
pfTime, pfX, pfY = read_csv_file(pfOdomFilePath)

def find_closest_values(trueTime, trueX, trueY, pfTime):
    '''
    Find the indices of the closest times in the trueTime array to the times in the pfTime array.
    Use for comparing the true and pf odometry data with different publishing rates.
    '''
    closest_times = []
    closest_xs = []
    closest_ys = []
    for time in pfTime:
        differences = np.abs(np.array(trueTime) - time)  # Calculate absolute differences
        closest_index = np.argmin(differences)  # Find index of minimum difference
        closest_times.append(trueTime[closest_index])
        closest_xs.append(trueX[closest_index])
        closest_ys.append(trueY[closest_index])
    return closest_times, closest_xs, closest_ys

trueTime, trueX, trueY = find_closest_values(trueTimeLong, trueXLong, trueYLong, pfTime)
# for i in range(len(trueTime)):
#     print(trueTime[i], trueX[i], trueY[i])  # Or do something else with the closest values

# print(len(trueTime), len(pfTime))
# for i in range(len(trueTime)):
#     print(trueTime[i],  pfTime[i],i)  # Or do something else with the closest values

def compute_mean_euclidean_distance(pfX, pfY, trueX, trueY):
        '''
        Compute the mean Euclidean distance between the points (pfX[i], pfY[i]) and (trueX[i], trueY[i]).
        '''
        distances = []
        for i in range(len(pfX)):
            distance = np.sqrt((pfX[i] - trueX[i])**2 + (pfY[i] - trueY[i])**2)
            print(distance)
            distances.append(distance)
        mean_distance = np.mean(distances)
        return mean_distance

mean_distance = compute_mean_euclidean_distance(pfX, pfY, trueX, trueY)
print("Mean Euclidean Distance:", mean_distance)