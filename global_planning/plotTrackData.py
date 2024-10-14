import trajectory_planning_helpers as tph
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from centerlineExtraction import getCentreLine
from iqp import generateMinCurvaturePath
from shortPath import generateShortestPath
import scipy

class Track:
	'''
		Generates:
		- centreline: maps/{map_name}_centreline.csv
		- shortest path: maps/{map_name}_short.csv
		- minimum curvature path: maps/{map_name}_minCurve.csv
		- shortest path based on minimum curvature path: maps/{map_name}_minCurve_short.csv
		- minimum curvature path based on shortest path: maps/{map_name}_short_minCurve.csv

		Plots:
		- all paths
		- curvature vs distance
		- heading vs distance
	'''
	def __init__(self, map_name):

		# Ensure the directory exists
		output_dir = f"data"
		os.makedirs(output_dir, exist_ok=True)

		if not os.path.exists(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_centreline.csv"):
			getCentreLine(map_name)
		if not os.path.exists(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_short.csv"):
			generateShortestPath(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_centreline.csv")
		if not os.path.exists(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_short_minCurve.csv"):
			generateMinCurvaturePath(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_short.csv")
		# if not os.path.exists(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_minCurve.csv"):
		# 	generateMinCurvaturePath(f"/home/chris/sim_ws/src/global_planning/maps/{map_name}_centreline.csv")

		# getCentreLine(map_name)
		# generateShortestPath(f"maps/{map_name}_centreline.csv")
		# generateMinCurvaturePath(f"maps/{map_name}_centreline.csv")
		# generateMinCurvaturePath(f"maps/{map_name}_short.csv")

		map_yaml_path = f"maps/{map_name}.yaml"
		map_img = plt.imread(f'maps/{map_name}.png')
		centreline = np.loadtxt(f"maps/{map_name}_centreline.csv", delimiter=',')
		# smoothed_centreline = np.loadtxt(f"/home/chris/sim_ws/src/global_planning/smooth_test/{map_name}_centerline.csv", delimiter=',')
		short = np.loadtxt(f"maps/{map_name}_short.csv", delimiter=',')
		# minCurve = np.loadtxt(f"maps/{map_name}_minCurve.csv", delimiter=',')
		shortMinCurve = np.loadtxt(f"maps/{map_name}_short_minCurve.csv", delimiter=',')
		# flip the image around x axis
		map_img = np.flipud(map_img)
		map_img = scipy.ndimage.distance_transform_edt(map_img)
		map_img = np.abs(map_img - 1)
		map_img[map_img!=0]=1
		
		with open(map_yaml_path, 'r') as yaml_stream:
			try:
				map_metadata = yaml.safe_load(yaml_stream)
				self.map_resolution = map_metadata['resolution']
				origin = map_metadata['origin']
			except yaml.YAMLError as ex:
				print(ex)

		self.orig_x = origin[0]
		self.orig_y = origin[1]

		startX = int((0-self.orig_x)/self.map_resolution)
		startY = int((0-self.orig_y)/self.map_resolution)
		centrelineX, centrelineY, centrelineS = self.processTrack(centreline)
		shortX, shortY, shortS = self.processTrack(short)
		# minCurveX, minCurveY, minCurveS = self.processTrack(minCurve)
		shortMinCurveX, shortMinCurveY, shortMinCurveS = self.processTrack(shortMinCurve)
		# smoothed_centrelineX, smoothed_centrelineY, smoothed_centrelineS = self.processTrack(smoothed_centreline)

		fig = plt.figure( num=f'{map_name}_racelines')
		plt.title(map_name)
		plt.imshow(map_img, cmap="gray", origin="lower")
		plt.plot(startX, startY, 'ro', label='Start')
		plt.plot(centrelineX,centrelineY, '--', label=f'Centreline ({centreline[-1, -1]:.2f}s)')
		plt.plot(shortX, shortY, label=f'Shortest Path ({short[-1, -1]:.2f}s)')
		# plt.plot(minCurveX, minCurveY, label=f'Minimum Curvature C ({minCurve[-1, -1]:.2f}s)')
		plt.plot(shortMinCurveX, shortMinCurveY, label=f'Minimum Curvature S({shortMinCurve[-1, -1]:.2f}s)')
		# plt.plot(smoothed_centrelineX, smoothed_centrelineY, label='Smoothed Centreline')
		plt.legend(loc='upper right')
		plt.savefig(f"{output_dir}/{map_name}_racelines.svg")
		plt.show()

		plt.figure( num=f'{map_name}_curvature')
		plt.title(f'Curvature vs Distance {map_name}')
		plt.plot(centrelineS, centreline[:, 5], label='Centreline')
		plt.plot(shortS, short[:, 5], label='Shortest Path')
		# plt.plot(minCurveS, minCurve[:, 5], label='Minimum Curvature (C)')
		plt.plot(shortMinCurveS, shortMinCurve[:, 5], label='Minimum Curvature')
		# plt.plot(smoothed_centrelineS, smoothed_centreline[:, 5], label='Smoothed Centreline')
		plt.legend(loc='upper right')
		plt.savefig(f"{output_dir}/{map_name}_curvature.svg")
		plt.show()

		plt.figure( num=f'{map_name}_heading')
		plt.title(f'Heading vs Distance {map_name}')
		plt.plot(centrelineS, centreline[:, 4], label='Centreline')
		plt.plot(shortS, short[:, 4], label='Shortest Path')
		# plt.plot(minCurveS, minCurve[:, 4], label='Minimum Curvature (C)')
		plt.plot(shortMinCurveS, shortMinCurve[:, 4], label='Minimum Curvature')
		# plt.plot(smoothed_centrelineS, smoothed_centreline[:, 4], label='Smoothed Centreline')
		plt.legend(loc='upper right')
		plt.savefig(f"{output_dir}/{map_name}_heading.svg")
		plt.show()



	def processTrack(self, track):
		x = track[:, 0]
		y = track[:, 1]
		s = track[:, 6]
		x -= self.orig_x
		y -= self.orig_y
		x /= self.map_resolution
		y /= self.map_resolution
		normS = s/s[-1]
		return x, y, normS

	

def main():
	for file in os.listdir('/home/chris/sim_ws/src/global_planning/maps/'):
		if file.endswith('.png'):
			map_name = file.split('.')[0]
			print(f"Extracting data for: {map_name}")
			track = Track(map_name)

if __name__ == '__main__':
	main()
			

