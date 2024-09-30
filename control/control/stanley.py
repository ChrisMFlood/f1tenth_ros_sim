#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
import math
import trajectory_planning_helpers as tph
  
class myNode(Node):
	def __init__(self):
		super().__init__("stanley") 
		# Parameters
		self.declare_parameter("k", 0.99)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("min_speed", 0.1)
		self.declare_parameter("max_steering_angle", 0.4)
		self.declare_parameter("map_name", "esp")

		self.k = self.get_parameter("k").value
		self.wheel_base = self.get_parameter("wheel_base").value
		self.min_speed = self.get_parameter("min_speed").value
		self.max_steering_angle = self.get_parameter("max_steering_angle").value
		self.map_name = self.get_parameter("map_name").value

		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
		# Publishers
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
		self.marker_pub = self.create_publisher(Marker, "/waypoint_marker", 10)

		# Waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)
		self.getHeading()

		# Variables  
		self.odom: Odometry = None

		

	def odom_callback(self, msg: Odometry):
		self.get_logger().info(f'Position: x={msg.pose.pose.position.x}, y={msg.pose.pose.position.y}')
		self.odom = msg
		self.yaw = self.euler_from_quaternion(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w) 
		self.pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw])
		
		self.get_logger().info(f'yaw={self.pose[2]}, heading={self.waypoints[self.getClosestPointOnPath(), 4]}, psi={self.psi[self.getClosestPointOnPath()]}')


	def getHeading(self):
		el_lengths = np.linalg.norm(np.diff(self.waypoints[:, :2], axis=0), axis=1)
		path = np.column_stack((self.waypoints[:, 1], self.waypoints[:, 0]))
		self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(path, el_lengths, False)
 
		
	def getClosestPointOnPath(self):
		distance = np.linalg.norm(self.waypoints[:, :2] - self.pose[:2], axis=1)
		closest_index = np.argmin(distance)
		return closest_index
	
	def euler_from_quaternion(self,x, y, z, w):  
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
		return yaw_z # in radians
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()