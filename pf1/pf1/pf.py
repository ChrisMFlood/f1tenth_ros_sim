#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import range_libc
import time
from threading import Lock

# TF
# import tf.transformations
# import tf
import tf_transformations

# messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

# TODO: Make a launch and params file


'''
These flags indicate several variants of the sensor model. Only one of them is used at a time.
'''
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4
  
class myNode(Node):
	def __init__(self):
		super().__init__("pf_node")  
		self.get_logger().info("pf_node has been started")
		
		# Params
		# self.WHICH_RM = 'bl'
		# self.WHICH_RM = 'cddt'
		# self.WHICH_RM = 'pcddt'
		# self.WHICH_RM = 'rm'
		# self.WHICH_RM = 'rmgpu'
		self.WHICH_RM = 'glt'
		self.SHOW_FINE_TIMING = True
		self.ranges = None
		self.ANGLE_STEP = 18
		self.THETA_DISCRETIZATION = 112

		## Lidar
		self.MAX_RANGE_METERS = 30.0
		##PF
		self.MAX_PARTICLES = 4000

		## Sensor Model
		self.Z_SHORT = 0.01
		self.Z_MAX = 0.07
		self.Z_RAND = 0.12
		self.Z_HIT = 0.75
		self.SIGMA_HIT = 8
		self.INV_SQUASH_FACTOR = 1/2.2
		## Sensor model variant??
		self.RANGELIB_VAR = 2
		## Motion model
		self.MOTION_DISPERSION_X = 0.025
		self.MOTION_DISPERSION_Y = 0.025
		self.MOTION_DISPERSION_THETA = 0.025

		self.sigmas = np.zeros(3)
		self.deltas_hat = np.zeros((self.MAX_PARTICLES,3))
		# self.alpha1 = 0.5
		# self.alpha2 = 0.5
		# self.alpha3 = 1
		# self.alpha4 = 0.1
		self.alpha1 = 0.5
		self.alpha2 = 0.015
		self.alpha3 = 1
		self.alpha4 = 0.1

		

		
		# Publishers
		self.particle_pub = self.create_publisher(PoseArray, 'particles', 10)
		self.expected_pose_pub = self.create_publisher(Odometry, 'expected_pose', 10)
		self.fake_scan_pub = self.create_publisher(LaserScan, 'fake_scan', 10)
		# Subscribers
		self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
		self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
		self.click_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.click_pose_callback, 10)
		# Services
		## Servers
		## Clients
		self.map_client = self.create_client(GetMap, '/map_server/map')


		# Variables
		self.weights = np.ones(self.MAX_PARTICLES)/float(self.MAX_PARTICLES)
		self.particle_indices = np.arange(self.MAX_PARTICLES)
		self.particles = np.zeros((self.MAX_PARTICLES, 3))
		# self.state_lock = Lock()
		## Map
		self.map_initialized = False
		## Lidar
		self.lidar_initialized = False
		self.angles = None
		## Odometry
		self.odom_initialized = False
		# self.deltas = np.zeros(3)
		# self.last_pose = None


		# Initialize
		self.get_omap()
		self.precompute_sensor_model()
		self.initialize_global()
		

	def get_omap(self):
		'''
		Fetch the occupancy grid map from the map_server instance, and initialize the correct
		RangeLibc method. Also stores a matrix which indicates the permissible region of the map
		'''
		self.get_logger().info('Get Map')
		while not self.map_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('Get map service not available, waiting...')
		req = GetMap.Request()
		future = self.map_client.call_async(req)
		rclpy.spin_until_future_complete(self, future)
		map_msg = future.result().map
		self.map_info = map_msg.info
		self.get_logger().info('Map received')

		oMap = range_libc.PyOMap(map_msg)
		self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

		# initialize range method
		self.get_logger().info('Initializing range method: ' + self.WHICH_RM)
		self.range_method = self.setRangeMethod(oMap)

		# 0: permissible, -1: unmapped, 100: blocked
		array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
		# 0: not permissible, 1: permissible
		self.permissible_region = np.zeros_like(array_255, dtype=bool)
		self.permissible_region[array_255==0] = 1
		self.map_initialized = True
		self.get_logger().info('Done loading map')

	def setRangeMethod(self,oMap):
		'''
		Set Rangelibc method based on the parameter.

		Input: Occupancy grid map
		'''
		if self.WHICH_RM == 'bl':
			range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
		elif 'cddt' in self.WHICH_RM:
			range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
			if self.WHICH_RM == 'pcddt':
				self.get_logger().info('Pruning...')
				range_method.prune()
		elif self.WHICH_RM == 'rm':
			range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
		elif self.WHICH_RM == 'rmgpu':
			range_method = range_libc.PyRayMarchingGPU(oMap, self.MAX_RANGE_PX)
		elif self.WHICH_RM == 'glt':
			range_method = range_libc.PyGiantLUTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
		return range_method
	
	def precompute_sensor_model(self):
		'''
		Generate and store a table which represents the sensor model. For each discrete computed
		range value, this provides the probability of measuring any (discrete) range.

		This table is indexed by the sensor model at runtime by discretizing the measurements
		and computed ranges from RangeLibc.

		TODO: Set model intrinsic parameters and look at short equation
		'''
		self.get_logger().info('Precomputing sensor model')
		# sensor model intrinsic parameters
		z_short = self.Z_SHORT
		z_max   = self.Z_MAX
		z_rand  = self.Z_RAND
		z_hit   = self.Z_HIT
		sigma_hit = self.SIGMA_HIT
		
		table_width = int(self.MAX_RANGE_PX) + 1
		self.sensor_model_table = np.zeros((table_width,table_width))

		# d is the computed range from RangeLibc
		for d in range(table_width):
			norm = 0.0
			# r is the observed range from the lidar unit
			for r in range(table_width):
				prob = 0.0
				z = float(r-d)
				# Normal distribution
				prob += z_hit * np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / (sigma_hit * np.sqrt(2.0*np.pi))
				# observed range is less than the predicted range - short reading
				if r < d:
					prob += 2.0 * z_short * (d - r) / float(d)
				# erroneous max range measurement
				if int(r) == int(self.MAX_RANGE_PX):
					prob += z_max
				# random measurement
				if r < int(self.MAX_RANGE_PX):
					prob += z_rand * 1.0/float(self.MAX_RANGE_PX)
				norm += prob
				self.sensor_model_table[int(r),int(d)] = prob
			# normalize
			self.sensor_model_table[:,int(d)] /= norm
		# upload the sensor model to RangeLib for ultra fast resolution
		if self.RANGELIB_VAR > 0:
			self.range_method.set_sensor_model(self.sensor_model_table)

	def initialize_global(self):
		'''
		Spread the particle distribution over the permissible region of the state space.
		'''
		self.get_logger().info('GLOBAL INITIALIZATION')
		# randomize over grid coordinate space
		# self.state_lock.acquire()
		permissible_x, permissible_y = np.where(self.permissible_region == 1)
		indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

		permissible_states = np.zeros((self.MAX_PARTICLES,3), dtype=np.float32)
		permissible_states[:,0] = permissible_y[indices]
		permissible_states[:,1] = permissible_x[indices]
		permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0
		self.get_logger().info('Converting to world coordinates')

		map_to_world(permissible_states, self.map_info)
		self.particles = permissible_states
		self.weights[:] = 1.0 / self.MAX_PARTICLES
		# self.state_lock.release()

	def scan_callback(self, msg: LaserScan):
		'''
		Initializes reused buffers, and stores the relevant laser scanner data for later use.

		TODO: Downsample the scan data to reduce computation time?
		'''
		self.scan = msg.ranges
		self.downsampled_ranges = np.array(msg.ranges[::self.ANGLE_STEP])
		# self.downsampled_ranges = np.copy(self.scan)
		
		self.angle_min = msg.angle_min
		self.angle_max = msg.angle_max

		if (not self.lidar_initialized):
			self.get_logger().info('Scan received')
			self.angles = np.linspace(self.angle_min, self.angle_max, len(self.scan),dtype=np.float32)
			# self.downsampled_angles = np.copy(self.angles)
			self.downsampled_angles = np.copy(self.angles[0::self.ANGLE_STEP]).astype(np.float32)
			self.lidar_initialized = True

	def odom_callback(self, msg: Odometry):
		'''
		Get change in odometery

		Odometry data is accumulated via dead reckoning, so it is very inaccurate on its own.

		TODO: Add noise to deltas
		'''
		orientation = quaternion_to_angle(msg.pose.pose.orientation)
		self.current_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, orientation])
		self.velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])

		if not self.odom_initialized:
			self.get_logger().info('First odom received')
			self.last_pose = self.current_pose
			self.odom_initialized = True
		else:
			delta_rot1 = np.arctan2(msg.pose.pose.position.y - self.last_pose[1], msg.pose.pose.position.x - self.last_pose[0]) - self.last_pose[2]
			delta_trans = np.sqrt((msg.pose.pose.position.x - self.last_pose[0])**2 + (msg.pose.pose.position.y - self.last_pose[1])**2)
			delta_rot2 = orientation - self.last_pose[2] - delta_rot1
			self.deltas = np.array([delta_rot1, delta_trans, delta_rot2])
			self.last_pose = self.current_pose
			self.first_sensor_update = True
			self.update()


	def update(self):
		'''
		Apply the MCL function to update particle filter state. 

		Ensures the state is correctly initialized, and acquires the state lock before proceeding.
		'''
		if self.lidar_initialized and self.odom_initialized and self.map_initialized:
			observation = np.copy(self.downsampled_ranges)
			deltas = np.copy(self.deltas)
			self.MCL(observation, deltas)
			self.publishExpectedPose()
			self.visualiseParticles()
			# self.publishFakeScan()

	def MCL(self, observation, deltas):
		'''
		Apply MCL to particles

		1. resample particle distribution to form the proposal distribution
		2. apply the motion model
		3. apply the sensor model
		4. normalize particle weights
		'''
		# print(self.weights)
		proposal_distribution = self.resample()
		self.motion_model(proposal_distribution, deltas)
		self.sensor_model(proposal_distribution, observation, self.weights)
		self.weights /= np.sum(self.weights)
		self.expected_pose = self.expectedPose()
		self.get_logger().info('Expected Pose: ' + str(self.expected_pose))
		self.particles = proposal_distribution

	def resample(self):
		'''
		Resample Particles
		TODO: KDL sampling
		'''
		proposal_indices = np.random.choice(self.particle_indices, self.MAX_PARTICLES, p=self.weights)
		proposal_distribution = self.particles[proposal_indices,:]
		return proposal_distribution
	
	def motion_model(self, particles, deltas):
		'''
		The motion model applies the odometry to the particle distribution. Since there the odometry
		data is inaccurate, the motion model mixes in gaussian noise to spread out the distribution.

		Vectorized motion model. Computing the motion model over all particles is thousands of times
		faster than doing it for each particle individually due to vectorization and reduction in
		function call overhead
		
		TODO this could be better, but it works for now
			- model provides bad estimates at high speed
		'''
		# particles[:,0] += deltas[1]*np.cos(particles[:,2]+deltas[0]) + np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_X,size=self.MAX_PARTICLES)
		# particles[:,1] += deltas[1]*np.sin(particles[:,2]+deltas[0]) + np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_Y,size=self.MAX_PARTICLES)
		# particles[:,2] += deltas[0] + deltas[2] + np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_THETA,size=self.MAX_PARTICLES)

		self.sigmas[0] = self.alpha1*deltas[0] + self.alpha2/np.max([deltas[1],0.1])
		self.sigmas[1] = self.alpha3*deltas[1] + self.alpha4*(deltas[0]+deltas[2])
		self.sigmas[2] = self.alpha1*deltas[2] + self.alpha2/np.max([deltas[1],0.1])

		self.deltas_hat[:,0] = deltas[0] + np.random.normal(loc=0.0,scale=self.sigmas[0]**2,size=self.MAX_PARTICLES)
		self.deltas_hat[:,1] = deltas[1] + np.random.normal(loc=0.0,scale=self.sigmas[1]**2,size=self.MAX_PARTICLES)
		self.deltas_hat[:,2] = deltas[2] + np.random.normal(loc=0.0,scale=self.sigmas[2]**2,size=self.MAX_PARTICLES)

		particles[:,0] += self.deltas_hat[:,1]*np.cos(particles[:,2]+self.deltas_hat[:,0]) #+ np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_X,size=self.MAX_PARTICLES)
		particles[:,1] += self.deltas_hat[:,1]*np.sin(particles[:,2]+self.deltas_hat[:,0]) #+ np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_Y,size=self.MAX_PARTICLES)
		particles[:,2] += self.deltas_hat[:,0] + self.deltas_hat[:,2] #+ np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_THETA,size=self.MAX_PARTICLES)

	def sensor_model(self, particles, obs, weights):
		'''
		This function computes a probablistic weight for each particle in the proposal distribution.
		These weights represent how probable each proposed (x,y,theta) pose is given the measured
		ranges from the lidar scanner.

		There are 4 different variants using various features of RangeLibc for demonstration purposes.
		- VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
		- VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
										optimizations to CDDT which simultaneously performs ray casting
										in two directions, reducing the amount of work by roughly a third
		'''
		num_rays = self.downsampled_angles.shape[0]
		self.ranges = np.zeros(num_rays*self.MAX_PARTICLES, dtype=np.float32)
		self.range_method.calc_range_repeat_angles(particles, self.downsampled_angles, self.ranges)
		self.range_method.eval_sensor_model(obs, self.ranges, weights, num_rays, self.MAX_PARTICLES)
		# weights = np.power(weights, self.INV_SQUASH_FACTOR)
		self.get_logger().info('Weights: '+str(np.max(weights)) +','+ str(np.min(weights)))

	def expectedPose(self):
		return np.dot(self.particles.transpose(), self.weights)
	
	def publishExpectedPose(self):
		Pose = Odometry()
		Pose.header.stamp = self.get_clock().now().to_msg()
		Pose.header.frame_id = 'map'
		Pose.pose.pose.position.x = self.expected_pose[0]
		Pose.pose.pose.position.y = self.expected_pose[1]
		Pose.pose.pose.orientation = angle_to_quaternion(self.expected_pose[2])
		Pose.twist.twist.linear.x = self.velocity[0]
		Pose.twist.twist.linear.y = self.velocity[1]
		Pose.twist.twist.angular.z = self.velocity[2]
		self.expected_pose_pub.publish(Pose)

	def publishFakeScan(self):
		'''
		Publish the fake scan data
		'''
		scan = LaserScan()
		scan.header.frame_id = 'ego_racecar/laser'
		scan.header.stamp = self.get_clock().now().to_msg()
		scan.angle_min = self.angle_min
		scan.angle_max = self.angle_max
		q=np.array([[self.expected_pose[0],self.expected_pose[1],self.expected_pose[2]]],dtype=np.float32)
		num_rays = self.downsampled_angles.shape[0]
		fake_ranges = np.zeros((num_rays), dtype=np.float32)
		self.range_method.calc_range_repeat_angles(q, self.downsampled_angles,fake_ranges)
		scan.ranges = fake_ranges.tolist()
		scan.range_min = 0.0
		scan.range_max = self.MAX_RANGE_METERS
		scan.angle_increment = float(self.downsampled_angles[1] - self.downsampled_angles[0])	
		self.fake_scan_pub.publish(scan)

	def click_pose_callback(self, msg: PoseWithCovarianceStamped):
		self.get_logger().info('SETTING POSE')
		self.get_logger().info(str(msg.pose.pose.position.x) + ' ' + str(msg.pose.pose.position.y) + ' ' + str(quaternion_to_angle(msg.pose.pose.orientation)))
		self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
		self.particles[:,0] = msg.pose.pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
		self.particles[:,1] = msg.pose.pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
		self.particles[:,2] = quaternion_to_angle(msg.pose.pose.orientation) + np.random.normal(loc=0.0,scale=0.25,size=self.MAX_PARTICLES)
		
		














	def visualiseParticles(self):
		'''
		Visualize the particles in rviz
		'''
		particles = PoseArray()
		particles.header.frame_id = 'map'
		particles.header.stamp = self.get_clock().now().to_msg()
		particles.poses = particles_to_poses(self.particles)
		self.particle_pub.publish(particles)
		# self.get_logger().info('Publishing particles')


class CircularArray(object):
	""" Simple implementation of a circular array.
		You can append to it any number of times but only "size" items will be kept
	"""
	def __init__(self, size):
		self.arr = np.zeros(size)
		self.ind = 0
		self.num_els = 0

	def append(self, value):
		if self.num_els < self.arr.shape[0]:
			self.num_els += 1
		self.arr[self.ind] = value
		self.ind = (self.ind + 1) % self.arr.shape[0]

	def mean(self):
		return np.mean(self.arr[:self.num_els])

	def median(self):
		return np.median(self.arr[:self.num_els])

class Timer:
	""" Simple helper class to compute the rate at which something is called.
		
		"smoothing" determines the size of the underlying circular array, which averages
		out variations in call rate over time.

		use timer.tick() to record an event
		use timer.fps() to report the average event rate.
	"""
	def __init__(self, smoothing):
		self.arr = CircularArray(smoothing)
		self.last_time = time.time()

	def tick(self):
		t = time.time()
		self.arr.append(1.0 / (t - self.last_time))
		self.last_time = t

	def fps(self):
		return self.arr.mean()

def angle_to_quaternion(angle):
	"""Convert an angle in radians into a quaternion _message_."""
	q = tf_transformations.quaternion_from_euler(0, 0, angle)
	q_out = Quaternion()
	q_out.x = q[0]
	q_out.y = q[1]
	q_out.z = q[2]
	q_out.w = q[3]
	return q_out

def quaternion_to_angle(q):
	"""Convert a quaternion _message_ into an angle in radians.
	The angle represents the yaw.
	This is not just the z component of the quaternion."""
	# x, y, z, w = q.x, q.y, q.z, q.w
	quat = [q.x, q.y, q.z, q.w]
	# roll, pitch, yaw = tf_transformations.euler_from_quaternion((x, y, z, w))
	roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
	return yaw

def rotation_matrix(theta):
	''' Creates a rotation matrix for the given angle in radians '''
	c, s = np.cos(theta), np.sin(theta)
	return np.matrix([[c, -s], [s, c]])

def particle_to_pose(particle):
	''' Converts a particle in the form [x, y, theta] into a Pose object '''
	pose = Pose()
	pose.position.x = float(particle[0])
	pose.position.y = float(particle[1])
	pose.orientation = angle_to_quaternion(particle[2])
	return pose

def particles_to_poses(particles):
	''' Converts a two dimensional array of particles into an array of Poses. 
		Particles can be a array like [[x0, y0, theta0], [x1, y1, theta1]...]
	'''
	return list(map(particle_to_pose, particles))

def map_to_world_slow(x,y,t,map_info):
	''' Converts given (x,y,t) coordinates from the coordinate space of the map (pixels) into world coordinates (meters).
		Provide the MapMetaData object from a map message to specify the change in coordinates.
		*** Logical, but slow implementation, when you need a lot of coordinate conversions, use the map_to_world function
	''' 
	scale = map_info.resolution
	angle = quaternion_to_angle(map_info.origin.orientation)
	rot = rotation_matrix(angle)
	trans = np.array([[map_info.origin.position.x],
					  [map_info.origin.position.y]])

	map_c = np.array([[x],
					  [y]])
	world = (rot*map_c) * scale + trans

	return world[0,0],world[1,0],t+angle

def map_to_world(poses, map_info):
	''' Takes a two dimensional numpy array of poses:
			[[x0,y0,theta0],
			 [x1,y1,theta1],
			 [x2,y2,theta2],
				   ...     ]
		And converts them from map coordinate space (pixels) to world coordinate space (meters).
		- Conversion is done in place, so this function does not return anything.
		- Provide the MapMetaData object from a map message to specify the change in coordinates.
		- This implements the same computation as map_to_world_slow but vectorized and inlined
	'''

	scale = map_info.resolution
	orientation = Quaternion()
	orientation.x = map_info.origin.orientation.x
	orientation.y = map_info.origin.orientation.y
	orientation.z = map_info.origin.orientation.z
	orientation.w = map_info.origin.orientation.w
	angle = quaternion_to_angle(orientation)

	# rotation
	c, s = np.cos(angle), np.sin(angle)
	# we need to store the x coordinates since they will be overwritten
	temp = np.copy(poses[:,0])
	poses[:,0] = c*poses[:,0] - s*poses[:,1]
	poses[:,1] = s*temp       + c*poses[:,1]

	# scale
	poses[:,:2] *= float(scale)

	# translate
	poses[:,0] += map_info.origin.position.x
	poses[:,1] += map_info.origin.position.y
	poses[:,2] += angle

def world_to_map(poses, map_info):
	''' Takes a two dimensional numpy array of poses:
			[[x0,y0,theta0],
			 [x1,y1,theta1],
			 [x2,y2,theta2],
				   ...     ]
		And converts them from world coordinate space (meters) to world coordinate space (pixels).
		- Conversion is done in place, so this function does not return anything.
		- Provide the MapMetaData object from a map message to specify the change in coordinates.
		- This implements the same computation as world_to_map_slow but vectorized and inlined
		- You may have to transpose the returned x and y coordinates to directly index a pixel array
	'''
	scale = map_info.resolution
	angle = -quaternion_to_angle(map_info.origin.orientation)

	# translation
	poses[:,0] -= map_info.origin.position.x
	poses[:,1] -= map_info.origin.position.y

	# scale
	poses[:,:2] *= (1.0/float(scale))

	# rotation
	c, s = np.cos(angle), np.sin(angle)
	# we need to store the x coordinates since they will be overwritten
	temp = np.copy(poses[:,0])
	poses[:,0] = c*poses[:,0] - s*poses[:,1]
	poses[:,1] = s*temp       + c*poses[:,1]
	poses[:,2] += angle


		
		

		

  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()