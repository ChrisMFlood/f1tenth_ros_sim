import numpy as np
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv, yaml
from PIL import Image
from matplotlib.collections import LineCollection
from transforms3d import euler
import trajectory_planning_helpers

class MapData:
    def __init__(self, map_name):
        self.map_name = map_name

        self.map_resolution = None
        self.map_origin = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        try:
            self.path = "/home/chris/Ros2PP/src/my_waypoint_follower/my_waypoint_follower/maps/"
            self.load_map_img()
        except:
            self.path = "../maps/"
            self.load_map_img()

    def load_map_img(self):
        with open(self.path + self.map_name + ".yaml", 'r') as file:
            map_yaml_data = yaml.safe_load(file)
            self.map_resolution = map_yaml_data["resolution"]
            self.map_origin = map_yaml_data["origin"]
            map_img_name = map_yaml_data["image"]

        self.map_img = np.array(Image.open(self.path + map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)
        if len(self.map_img.shape) > 2:
            self.map_img = self.map_img[:, :, 0]

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 1.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        
    def xy2rc(self, xs, ys):
        xs = (xs - self.map_origin[0]) / self.map_resolution
        ys = (ys - self.map_origin[1]) /self.map_resolution
        return xs, ys

    def pts2rc(self, pts):
        return self.xy2rc(pts[:,0], pts[:,1])
    
    def plot_map_img(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        plt.imshow(self.map_img, origin='lower', cmap='gray')

    def plot_map_img_left_right_flip(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        map_img = np.fliplr(self.map_img)
        plt.imshow(map_img, origin='lower', cmap='gray')

    def plot_map_img_T(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        map_img = np.transpose(self.map_img)
        plt.imshow(map_img, origin='lower', cmap='gray')

    def plot_map_img_rotate(self, k):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        map_img = np.rot90(self.map_img, k=k)
        plt.imshow(map_img, origin='lower', cmap='gray')

    def get_formatted_img(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0

        return self.map_img

    def plot_map_img_light_T(self):
        self.map_img[self.map_img == 1] = 220
        self.map_img[self.map_img == 0 ] = 255
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        map_img = np.transpose(self.map_img)
        plt.imshow(map_img, origin='lower', cmap='gray')

    def plot_map_img_light(self):
        self.map_img[self.map_img == 1] = 220
        self.map_img[self.map_img == 0 ] = 255
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        plt.imshow(self.map_img, origin='lower', cmap='gray')

def read_csv_file(file_path):
    times = np.array([])
    trueX = np.array([])
    trueY = np.array([])
    pfX = np.array([])
    pfY = np.array([])
    trueAngle = np.array([])
    pfAngle = np.array([])
    trueOrientation = np.empty((0,3))
    pfOrientation = np.empty((0,3))
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            times = np.append(times, float(row[0])+(float(row[1])/1e9))

            trueX = np.append(trueX, float(row[2]))
            trueY = np.append(trueY, float(row[3]))
            trueVec,trueTheta=euler.quat2axangle([float(row[5]),0.0,0.0,float(row[4])])
            trueAngle = np.append(trueAngle, trueTheta)
            trueOrientation = np.append(trueOrientation, np.array(trueVec).reshape(1, -1), axis=0)

            pfX = np.append(pfX, float(row[6]))
            pfY = np.append(pfY, float(row[7]))
            pfVec,pfTheta=euler.quat2axangle([float(row[9]),0.0,0.0,float(row[8])])
            pfAngle = np.append(pfAngle, pfTheta)
            pfOrientation = np.append(pfOrientation, np.array(pfVec).reshape(1, -1), axis = 0)

    first_time = times[1]
    for i in range(len(times)):
        times[i] -= first_time
    truePosition = np.array([trueX, trueY])
    pfPosition = np.array([pfX, pfY])
    return times, truePosition, trueOrientation.reshape(3,-1),trueAngle, pfPosition, pfOrientation.reshape(3,-1), pfAngle


def getError(arr1, arr2):
    error = arr2 - arr1
    return error


def getAngleDiff(arr1,arr2):
    v1 =np.array([np.cos(arr1), np.sin(arr1)]).reshape(2,-1)
    v2 =np.array([np.cos(arr2), np.sin(arr2)]).reshape(2,-1)
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2).reshape(1,-1)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2).reshape(1,-1)
    dot = np.array([v1[0]*v2[0] + v1[1]*v2[1]]).reshape(1,-1)
    angle = np.arccos(np.clip(dot/(mag1*mag2),-1,1))
    return angle

def errorAnalysis(error):
    #rmse
    squared_error = np.sum((error)**2,axis=0)
    total_error = np.sum(squared_error)
    rmse = sqrt(total_error/len(squared_error))
    #abs error
    abs_error = np.sqrt(squared_error)
    #quartiles
    upper_quartile = np.percentile(abs_error, 75)
    upper_quartile_index = np.where(abs_error == upper_quartile)
    print("upper q",upper_quartile)
    # print(upper_quartile_index)
    lower_quartile = np.percentile(abs_error, 25)
    lower_quartile_index = np.where(abs_error == lower_quartile)
    print("lower q", lower_quartile)
    # print(lower_quartile_index)
    median = np.percentile(abs_error, 50)
    median_index = np.where(abs_error == median)
    print("meadian" ,median)
    # print(median_index)
    max_error = np.max(abs_error)
    max_error_index = np.where(abs_error == max_error)
    print("max e", max_error)
    # print(max_error_index)
    min_error = np.min(abs_error)
    min_error_index = np.where(abs_error == min_error)
    print("min e", min_error)
    # print(min_error_index)
    #outliers
    iqr = upper_quartile - lower_quartile
    upper_bound = upper_quartile + 1.5*iqr
    lower_bound = lower_quartile - 1.5*iqr
    outliers = abs_error[(abs_error > upper_bound) | (abs_error < lower_bound)]
    # print("Outliers:", outliers)
    print("#Outliers:", len(outliers))
    outlier_indices = np.where((abs_error > upper_bound) | (abs_error < lower_bound)) 
    # print("Outlier Indices:", outlier_indices)

    return rmse

def main():

    file_path = '/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/gbr_1.csv'
    time, truePosition, trueOrientation, trueAngle, pfPosition, pfOrientation, pfAngle = read_csv_file(file_path)

    position_rmse = errorAnalysis(getError(truePosition, pfPosition))
    print("Position RMSE:", position_rmse)

    orientation_rmse = errorAnalysis(getError(trueOrientation, pfOrientation))
    print("Orientation RMSE:", orientation_rmse)
    ang = getAngleDiff(trueOrientation, pfOrientation)
    # print("Angle Diff:", ang)
    ang_rmse = errorAnalysis(ang)
    print("Angle RMSE:", ang_rmse)
    # for a in range(ang.size):
    #     if ang[0][a] != 0.0 and ang[0][a] != 2.0:
    #         print(str(a), ang[0][a])
    #     # print(str(a), ang[0][a])

    map_name = "gbr"

    # map_data = MapData(map_name)
    # map_data.plot_map_img()
    # ox, oy = map_data.xy2rc(truePosition[0], truePosition[1])
    # pf, py = map_data.xy2rc(pfPosition[0], pfPosition[1])
    # plt.plot(ox, oy, label='True Position', color='blue', linewidth=1.5)
    # plt.plot(pf, py, label='PF Position', color='red', linestyle='dashed', linewidth=0.9)
    # plt.legend()
    # plt.savefig('/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/0_gbr.svg')


    # plt.show()

    diffx = truePosition[1] - pfPosition[1]
    plt.plot(time, diffx, label='True Position', color='blue', linewidth=1.5)
    plt.show()

if __name__ == '__main__':

    main()