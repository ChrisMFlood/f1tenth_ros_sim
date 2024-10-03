import numpy as np
import scipy.ndimage
import trajectory_planning_helpers as tph
import matplotlib.pyplot as plt
import yaml
import scipy

class controlData:
	def __init__(self, map_name):


		map_img = plt.imread(f'/home/chris/sim_ws/src/global_planning/maps/{map_name}.png')
		map_img = np.flipud(map_img)
		map_img = scipy.ndimage.distance_transform_edt(map_img)
		map_img = np.abs(map_img - 1)
		map_img[map_img!=0]=1

		map_yaml_path = f"/home/chris/sim_ws/src/global_planning/maps/{map_name}.yaml"
		with open(map_yaml_path, 'r') as yaml_stream:
			try:
				map_metadata = yaml.safe_load(yaml_stream)
				self.map_resolution = map_metadata['resolution']
				origin = map_metadata['origin']
				self.orig_x = origin[0]
				self.orig_y = origin[1]
			except yaml.YAMLError as ex:
				print(ex)
		
		self.waypoints = np.loadtxt(f'/home/chris/sim_ws/src/global_planning/maps/{map_name}_short_minCurve.csv', delimiter=',', skiprows=1)
		wx, wy, ws = self.processTrack(self.waypoints[:, :2])
		self.stanley = np.loadtxt(f'/home/chris/sim_ws/src/control/Results/Data/{map_name}_stanley.csv', delimiter=',', skiprows=1)
		self.stanley_1 = self.stanley[self.stanley[:, 5] == 0]
		self.stanley_2 = self.stanley[self.stanley[:, 5] != 0]
		sx, sy, ss = self.processTrack(self.stanley_2[:, 1:3])
		self.pp = np.loadtxt(f'/home/chris/sim_ws/src/control/Results/Data/{map_name}_purepursuit.csv', delimiter=',', skiprows=1)
		self.pp_1 = self.pp[self.pp[:, 5] == 0]
		self.pp_2 = self.pp[self.pp[:, 5] != 0]
		px, py, ps = self.processTrack(self.pp_2[:, 1:3])


		plt.figure(num = f'{map_name}_paths')
		plt.title(f'{map_name} paths')
		plt.imshow(map_img, cmap="gray", origin="lower")
		plt.plot(wx, wy, label=f'Waypoints ({self.waypoints[-1, 9]}s)')
		plt.plot(sx, sy, label=f'Stanley ({self.stanley_2[-1, 0]-self.stanley_1[-1,0]}s)')
		plt.plot(px, py, label=f'Pure Pursuit ({self.pp_2[-1, 0]-self.pp_1[-1,0]}s)')
		plt.legend()
		plt.savefig(f"/home/chris/sim_ws/src/control/Results/Plots/{map_name}_paths.svg")
		plt.show()

		plt.figure(num = f'{map_name}_heading')
		plt.title(f'{map_name} heading')
		plt.plot(ws, self.waypoints[:, 4], label='Waypoints')
		plt.plot(ss, self.stanley_2[:,3], label='Stanley')
		plt.plot(ps, self.pp_2[:,3], label='Pure Pursuit')
		plt.legend()
		plt.savefig(f"/home/chris/sim_ws/src/control/Results/Plots/{map_name}_heading.svg")
		plt.show()

		plt.figure(num = f'{map_name}_speed')
		plt.title(f'{map_name} speed')
		plt.plot(ws, self.waypoints[:, 7], label='Waypoints')
		plt.plot(ss, self.stanley_2[:,4], label='Stanley')
		plt.plot(ps, self.pp_2[:,4], label='Pure Pursuit')
		plt.legend()
		plt.savefig(f"/home/chris/sim_ws/src/control/Results/Plots/{map_name}_speed.svg")
		plt.show()


	def processTrack(self, track):
		x = track[:, 0]
		y = track[:, 1]
		x -= self.orig_x
		y -= self.orig_y
		x /= self.map_resolution
		y /= self.map_resolution
		lengths = np.linalg.norm(np.diff(track, axis=0), axis=1)
		s = np.insert(np.cumsum(lengths), 0, 0)
		normS = s/s[-1]
		return x, y, normS
	
def main():
	controlData('mco')
	controlData('esp')
	controlData('gbr')
	controlData('aut')

if __name__ == "__main__":
	main()
