import csv

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

trueTime, trueX, trueY = read_csv_file(trueOdomFilePath)
pfTime, pfX, pfY = read_csv_file(pfOdomFilePath)

# for i in range(len(trueTime)-1):
#     print(trueTime[i+1]-trueTime[i])

for i in range(len(pfTime)-1):
    print(pfTime[i+1]-pfTime[i])

