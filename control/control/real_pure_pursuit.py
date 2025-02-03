#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from control import utils as utils
  
class myNode(Node):
	def __init__(self):
		super().__init__("pure_pursuit")
		# Parameters
		self.declare_parameter("odom_topic","/pf/pose/odom")
		self.odom_topic = self.get_parameter("odom_topic").value

		self.declare_parameter("lookahead_distance", 1)
		self.declare_parameter("k", 0.5)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("min_speed", 0.1)
		self.declare_parameter('max_speed', 3.0)
		self.declare_parameter("max_steering_angle", 0.4)
		self.declare_parameter("map_name", "map3")
		self.declare_parameter('constant_lookahead_distance',0.7)
		self.declare_parameter('variable_lookahead_distance',0.3)

		self.lookahead_distance = self.get_parameter("lookahead_distance").value
		self.k = self.get_parameter("k").value
		self.wheel_base = self.get_parameter("wheel_base").value
		self.min_speed = self.get_parameter("min_speed").value
		self.max_speed = self.get_parameter("max_speed").value
		self.max_steering_angle = self.get_parameter("max_steering_angle").value
		self.map_name = self.get_parameter("map_name").value
		self.constant_lookahead = self.get_parameter("constant_lookahead_distance").value
		self.variable_lookahead = self.get_parameter('variable_lookahead_distance').value

		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
		# Publishers
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
		self.target_pub = self.create_publisher(Marker, "target", 10)
		self.waypoints_pub = self.create_publisher(PoseArray, '/waypoints', 10)
		self.ref_traj_pub = self.create_publisher(PoseArray, 'reference_trajectory', 10)
		self.calc_ref_traj_pub = self.create_publisher(PoseArray, 'calculated_trajectory', 10)

		# Waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_minCurve.csv', delimiter=',', skiprows=1)
		utils.publishTrajectory(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,4], self.waypoints_pub)

		# Variables 

		

	def odom_callback(self, msg: Odometry):
		self.odom = msg
		self.yaw = utils.euler_from_quaternion(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w) 
		self.pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw])
		self.speed = self.odom.twist.twist.linear.x


		self.closestPoint, self.nearestDistance, self.t1, self.nearestIndex = utils.nearest_point(self.pose[0:2], self.waypoints[:,0:2])
		self.lookAheadDistance = self.getLookAheadDistance(self.speed)
		self.targetPoint, self.targetIndex, self.t2 = self.getTargetWaypoint()
		utils.publishPoint(self.targetPoint[0],self.targetPoint[1],self.target_pub)
		self.steeringAngle, self.velocity = self.actuation()
		utils.pubishActuation(self.steeringAngle, self.velocity, self.cmd_pub, self.max_speed)
		
	
	def getLookAheadDistance(self, speed):
		# Speed
		# if speed < 0.1:
		# 	speed = 0.1
		lookAheadDistance = self.constant_lookahead + (speed/self.max_speed)*self.variable_lookahead
		# Curvature
		# lookAheadDistance = self.constant_lookahead + self.k/np.abs(self.waypoints[self.nearestIndex,5]) * 0.15
		return lookAheadDistance
	
	
	def getTargetWaypoint(self):
		if self.lookAheadDistance > self.nearestDistance:
			targetWaypoint, index, t = self.intersect_point(self.pose[0:2], self.lookAheadDistance, self.waypoints[:,0:2], self.nearestIndex, True)
		else:
			targetWaypoint, index, t = self.intersect_point(self.closestPoint[0:2], self.lookAheadDistance, self.waypoints[:,0:2], self.nearestIndex, True) 
		return targetWaypoint, index, t
	
	
	def actuation(self):
		speed = self.waypoints[self.nearestIndex,7]
		waypoint = np.dot(np.array([np.sin(-self.pose[2]), np.cos(-self.pose[2])]), self.targetPoint[:2] - self.pose[:2])
		LD = np.linalg.norm(self.targetPoint[:2] - self.pose[:2])
		radius = (LD**2) / (2 * waypoint)
		steering_angle = np.arctan(self.wheel_base / radius)
		return steering_angle, speed
	
	
	# @njit(cache=True)
	def intersect_point(self, point, radius, trajectory, t=0.0, wrap=False):
		"""
		starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
	
		Assumes that the first segment passes within a single radius of the point
	
		http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
		"""
		start_i = int(t)
		start_t = t % 1.0
		first_t = None
		first_i = None
		first_p = None
		trajectory = np.ascontiguousarray(trajectory)
		for i in range(start_i, trajectory.shape[0]-1):
			start = trajectory[i,:]
			end = trajectory[i+1,:]+1e-6
			V = np.ascontiguousarray(end - start)
	
			a = np.dot(V,V)
			b = 2.0*np.dot(V, start - point)
			c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
			discriminant = b*b-4*a*c
	
			if discriminant < 0:
				continue
			#   print "NO INTERSECTION"
			# else:
			# if discriminant >= 0.0:
			discriminant = np.sqrt(discriminant)
			t1 = (-b - discriminant) / (2.0*a)
			t2 = (-b + discriminant) / (2.0*a)
			if i == start_i:
				if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
					first_t = t1
					first_i = i
					first_p = start + t1 * V
					break
				if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
					first_t = t2
					first_i = i
					first_p = start + t2 * V
					break
			elif t1 >= 0.0 and t1 <= 1.0:
				first_t = t1
				first_i = i
				first_p = start + t1 * V
				break
			elif t2 >= 0.0 and t2 <= 1.0:
				first_t = t2
				first_i = i
				first_p = start + t2 * V
				break
		# wrap around to the beginning of the trajectory if no intersection is found1
		if wrap and first_p is None:
			for i in range(-1, start_i):
				start = trajectory[i % trajectory.shape[0],:]
				end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
				V = end - start
	
				a = np.dot(V,V)
				b = 2.0*np.dot(V, start - point)
				c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
				discriminant = b*b-4*a*c
	
				if discriminant < 0:
					continue
				discriminant = np.sqrt(discriminant)
				t1 = (-b - discriminant) / (2.0*a)
				t2 = (-b + discriminant) / (2.0*a)
				if t1 >= 0.0 and t1 <= 1.0:
					first_t = t1
					first_i = i
					first_p = start + t1 * V
					break
				elif t2 >= 0.0 and t2 <= 1.0:
					first_t = t2
					first_i = i
					first_p = start + t2 * V
					break
				
		return first_p, first_i, first_t
	

	







def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()