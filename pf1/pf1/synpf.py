#!/usr/bin/env python3

#git@github.com:ForzaETH/particle_filter.git
import rclpy
from rclpy.node import Node

import numpy as np
import tf_transformations
# import utils.utils as Utils
import range_libc
from threading import Lock

# Message types
from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped, PoseWithCovarianceStamped, TransformStamped, Quaternion, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, MapMetaData
from nav_msgs.srv import GetMap

# Define Datatypes
from enum import Enum
from typing import Optional

# Debug
from time import time
from collections import deque
import cProfile
import pstats

# Dynamic Reconfigure
# from dynamic_reconfigure.msg import Config


# TODO: Make a launch and params file
# TODO: Look at TUM box scan

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
		super().__init__("SynPF")
		self.get_logger().info("SynPF Node Started")

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

		##PF
		self.MAX_PARTICLES = 4000

		## Sensor Model
		self.MAX_RANGE_METERS = 30.0

		# self.Z_SHORT = 0.01
		# self.Z_MAX = 0.07
		# self.Z_RAND = 0.12
		# self.Z_HIT = 0.8
		# self.SIGMA_HIT = 8
		# self.LAM_SHORT = 0.01

		self.Z_SHORT = 0.1
		self.Z_MAX = 0.025
		self.Z_RAND = 0.025
		self.Z_HIT = 0.85
		self.SIGMA_HIT = 0.1
		self.LAM_SHORT = 0.25

		self.INV_SQUASH_FACTOR = 1/2.2
		### Sensor model variant??
		self.RANGELIB_VAR = 2

		## Motion model
		### MIT
		# self.MOTION_DISPERSION_X = 0.025
		# self.MOTION_DISPERSION_Y = 0.025
		# self.MOTION_DISPERSION_THETA = 0.025

		### Standard Odemetry
		# self.alpha1 = 0.5
		# self.alpha2 = 0.5
		# self.alpha3 = 1
		# self.alpha4 = 0.1

		### TUM/ETH
		self.alpha1 = 0.5
		self.alpha2 = 0.015
		self.alpha3 = 1
		self.alpha4 = 0.1


		


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
		self.first_sensor_update = True
		## Odometry
		self.odom_initialized = False
		self.last_pose = None
		self.sigmas = np.zeros(3)
		self.deltas_hat = np.zeros((self.MAX_PARTICLES,3))

		# Publishers
		self.particle_pub = self.create_publisher(PoseArray, 'particles', 10)
		self.expected_pose_pub = self.create_publisher(Odometry, 'expected_pose', 10)
		self.fake_scan_pub = self.create_publisher(LaserScan, 'fake_scan', 10)
		# Subscribers
		self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
		self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
		self.click_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.click_pose_callback, 10)
		# Clients
		self.map_client = self.create_client(GetMap, '/map_server/map')

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
		Generate and store a lookup table which represents the sensor model.

		For each discrete computed range value, this provides the probability of measuring that (discrete) range.

		This table is indexed by the sensor model at runtime by discretizing the measurements
		and computed ranges from RangeLibc.

		TODO: Set model intrinsic parameters
		'''
		self.get_logger().info('Precomputing sensor model')
		# sensor model intrinsic parameters
		z_short = self.Z_SHORT
		z_max   = self.Z_MAX
		z_rand  = self.Z_RAND
		z_hit   = self.Z_HIT
		# normalise sigma and lambda from meters to pixel space
		# [px] = [m] / [m/px]
		sigma_hit = self.SIGMA_HIT/self.map_info.resolution
		lam_short = self.LAM_SHORT/self.map_info.resolution

		table_width = int(self.MAX_RANGE_PX) + 1
		self.sensor_model_table = np.zeros((table_width,table_width))

		# compute normalizers for the gaussian and exponential distributions
		norm_gau = np.zeros((table_width,))
		norm_exp = np.zeros((table_width,))
		for d in range(table_width):
			sum_gau = 0
			sum_exp = 0
			for r in range(table_width):
				z = float(d-r)
				sum_gau += np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / (sigma_hit * np.sqrt(2.0*np.pi))
				if r <= d:
					sum_exp += ( lam_short * np.exp(-lam_short*r) )
			norm_gau[d] = 1/sum_gau
			norm_exp[d] = 1/sum_exp

		for d in range(table_width): #d is the computed range for particles
			norm = 0.0
			for r in range(table_width): #r is the measured range from the lidar
				prob = 0.0
				z = float(r-d)
				# Probability of hitting the intended object (Normal distribution)
				prob += z_hit * np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / (sigma_hit * np.sqrt(2.0*np.pi)) * norm_gau[d]
				# observed range is less than the predicted range - short reading
				if (r <= d):
					prob += z_short * ( lam_short * np.exp(-lam_short*r) ) * norm_exp[d]
				# erroneous max range measurement 
				if int(r) == int(self.MAX_RANGE_PX):
					prob += z_max
				# random measurement (uniform distribution at max range)
				if r < int(self.MAX_RANGE_PX):
					prob += z_rand * 1.0/float(self.MAX_RANGE_PX)
				norm += prob
				self.sensor_model_table[r,d] = prob
			# normalize
			self.sensor_model_table[:,d] /= norm
			# upload the sensor model to RangeLib for ultra fast resolution
		if self.RANGELIB_VAR > 0:
			self.range_method.set_sensor_model(self.sensor_model_table)

	def initialize_global(self):
		'''
		Spread the particle distribution over the permissible region of the state space.
		'''
		self.get_logger().info('GLOBAL INITIALIZATION / Lost Robot Initialization')
		# self.state_lock.acquire()
		# randomize over grid coordinate space
		permissible_x, permissible_y = np.where(self.permissible_region == 1)
		indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

		permissible_states = np.zeros((self.MAX_PARTICLES,3), dtype=np.float32)
		permissible_states[:,0] = permissible_y[indices]
		permissible_states[:,1] = permissible_x[indices]
		permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

		map_to_world(permissible_states, self.map_info)
		self.particles = permissible_states
		self.weights[:] = 1.0 / self.MAX_PARTICLES

	def click_pose_callback(self, msg: PoseWithCovarianceStamped):
		'''
		Initialize particles in the general region of the provided pose. Handle the initial pose message from RViz.
		'''
		self.get_logger().info('SETTING POSE')
		self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
		self.particles[:,0] = msg.pose.pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
		self.particles[:,1] = msg.pose.pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
		self.particles[:,2] = quaternion_to_angle(msg.pose.pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.MAX_PARTICLES)
		self.get_logger().info(str(self.particles[:,0]) + ' ' + str(self.particles[:,1]) + ' ' + str(self.particles[:,2]))

	def scan_callback(self, msg: LaserScan):
		'''
		Initializes reused buffers, and stores the relevant laser scanner data for later use.

		Downsample the scan data to reduce computation time
		'''
		self.scan = msg.ranges
		self.downsampled_ranges = np.array(msg.ranges[::self.ANGLE_STEP])
		
		if (not self.lidar_initialized):
			self.get_logger().info('First scan received')
			self.angle_min = msg.angle_min
			self.angle_max = msg.angle_max
			self.lidar_initialized = True

			self.angles = np.linspace(self.angle_min, self.angle_max, len(self.scan),dtype=np.float32)
			self.downsampled_angles = np.copy(self.angles[0::self.ANGLE_STEP]).astype(np.float32)

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
			self.update()
			self.last_pose = self.current_pose

	def update(self):
		'''
		Apply the MCL function to update particle filter state. 

		Ensures the state is correctly initialized, and acquires the state lock before proceeding.

		TODO: state lock stuff
		'''
		if self.lidar_initialized and self.odom_initialized and self.map_initialized:
			observation = np.copy(self.downsampled_ranges).astype(np.float32)
			deltas = np.copy(self.deltas).astype(np.float32)
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

		TODO: check if particle is in valid map region
		'''
		proposal_distribution = self.resample()
		self.motion_model(proposal_distribution, deltas)
		self.sensor_model(proposal_distribution, observation, self.weights)
		self.weights /= np.sum(self.weights)
		self.particles = proposal_distribution
		self.expected_pose = self.getExpectedPose()	
	
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
		Apply the motion model to the proposal distribution

		TODO: Look at simulator motion model
		'''

		self.sigmas[0] = self.alpha1*deltas[0] + self.alpha2/np.max([deltas[1],0.1])
		self.sigmas[1] = self.alpha3*deltas[1] + self.alpha4*(deltas[0]+deltas[2])
		self.sigmas[2] = self.alpha1*deltas[2] + self.alpha2/np.max([deltas[1],0.1])

		self.deltas_hat[:,0] = deltas[0] + np.random.normal(loc=0.0,scale=self.sigmas[0]**2,size=self.MAX_PARTICLES)
		self.deltas_hat[:,1] = deltas[1] + np.random.normal(loc=0.0,scale=self.sigmas[1]**2,size=self.MAX_PARTICLES)
		self.deltas_hat[:,2] = deltas[2] + np.random.normal(loc=0.0,scale=self.sigmas[2]**2,size=self.MAX_PARTICLES)

		# self.deltas_hat[:,0] = deltas[0] + np.random.normal(loc=0.0,scale=self.sigmas[0],size=self.MAX_PARTICLES)
		# self.deltas_hat[:,1] = deltas[1] + np.random.normal(loc=0.0,scale=self.sigmas[1],size=self.MAX_PARTICLES)
		# self.deltas_hat[:,2] = deltas[2] + np.random.normal(loc=0.0,scale=self.sigmas[2],size=self.MAX_PARTICLES)

		particles[:,0] += self.deltas_hat[:,1]*np.cos(particles[:,2]+self.deltas_hat[:,0]) #+ np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_X,size=self.MAX_PARTICLES)
		particles[:,1] += self.deltas_hat[:,1]*np.sin(particles[:,2]+self.deltas_hat[:,0]) #+ np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_Y,size=self.MAX_PARTICLES)
		particles[:,2] += self.deltas_hat[:,0] + self.deltas_hat[:,2] #+ np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_THETA,size=self.MAX_PARTICLES)

	def sensor_model(self, particles, observation, weights):
		'''
		This function computes a probablistic weight for each particle in the proposal distribution. These weights represent how probable each proposed (x,y,theta) pose is given the measured ranges from the lidar scanner.
		'''
		if self.first_sensor_update:
			self.first_sensor_update = False
			self.get_logger().info('First sensor update')
			self.num_rays = self.downsampled_angles.shape[0]
			self.ranges = np.zeros(self.num_rays*self.MAX_PARTICLES, dtype=np.float32)

		self.range_method.calc_range_repeat_angles(particles, self.downsampled_angles, self.ranges)
		self.range_method.eval_sensor_model(observation, self.ranges, weights, self.num_rays, self.MAX_PARTICLES)
		# weights = np.power(weights, self.INV_SQUASH_FACTOR)
		self.get_logger().info('Weights: '+str(np.max(weights)) +','+ str(np.min(weights)))

	def getExpectedPose(self):
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
	
def quaternion_to_angle(q):
	"""Convert a quaternion _message_ into an angle in radians.
	The angle represents the yaw.
	This is not just the z component of the quaternion."""
	# x, y, z, w = q.x, q.y, q.z, q.w
	quat = [q.x, q.y, q.z, q.w]
	# roll, pitch, yaw = tf_transformations.euler_from_quaternion((x, y, z, w))
	roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
	return yaw

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

def angle_to_quaternion(angle):
	"""Convert an angle in radians into a quaternion _message_."""
	q = tf_transformations.quaternion_from_euler(0, 0, angle)
	q_out = Quaternion()
	q_out.x = q[0]
	q_out.y = q[1]
	q_out.z = q[2]
	q_out.w = q[3]
	return q_out

		
		
				

				
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()