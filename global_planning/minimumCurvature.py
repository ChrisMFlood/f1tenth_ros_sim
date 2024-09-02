import trajectory_planning_helpers as tph

from argparse import Namespace
import numpy as np
import os
import cv2 as cv
from map_converter import getCentreLine
import yaml

class CentreLine:
    def __init__(self, map_name):
        self.track = np.loadtxt(f"maps/{map_name}_wl_centreline.csv", delimiter=',', skiprows=1)
        self.path = self.track[:, :2]
        self.widths = self.track[:, 2:4]
        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
        self.normvectors = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

def generateMinCurvaturePath(map_name):
    centreline = CentreLine(map_name)
    closed_path = np.row_stack([centreline.path, centreline.path[0]])
    closed_lengths = np.append(centreline.el_lengths, centreline.el_lengths[0])
    print(closed_lengths)
    vehicle = load_parameter_file("vehicle_params")
    coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(closed_path, closed_lengths)
    alpha, error = tph.opt_min_curv.opt_min_curv(centreline.track, centreline.normvectors, A, 1, 0, print_debug=True, closed=True)


def load_parameter_file(planner_name):
    file_name = f"params/{planner_name}.yaml"
    with open(file_name, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)    
    return params

def main():
    for file in os.listdir('maps/'):
        if file.endswith('.png'):
            map_name = file.split('.')[0]
            if not os.path.exists(f"maps/{map_name}_wl_centreline.csv"):
                print(f"Extracting centre line for: {map_name}")
                getCentreLine(map_name)
            print(f"Generating minimum curvature path for: {map_name}")
            generateMinCurvaturePath(map_name)

if __name__ == "__main__":
    main()