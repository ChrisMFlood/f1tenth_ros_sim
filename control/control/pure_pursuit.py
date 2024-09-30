#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
import math
  
class myNode(Node):
	def __init__(self):
		super().__init__("pure_pursuit")
		# Parameters
		self.declare_parameter("lookahead_distance", 1)
		self.declare_parameter("k", 0.5)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("min_speed", 0.1)
		self.declare_parameter("max_steering_angle", 0.4)
		self.declare_parameter("map_name", "esp")

		self.lookahead_distance = self.get_parameter("lookahead_distance").value
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
		# self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_centreline.csv', delimiter=',', skiprows=1)

		# Variables  
		self.odom: Odometry = None
		

	def odom_callback(self, msg: Odometry):
		self.get_logger().info(f'Position: x={msg.pose.pose.position.x}, y={msg.pose.pose.position.y}')
		self.odom = msg
		self.yaw = self.euler_from_quaternion(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w) 
		self.pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw])
		waypoint_index = self.get_closest_waypoint()
		self.waypoint = self.waypoints[waypoint_index]
		self.get_logger().info(f'Closest waypoint: x={self.waypoint[0]}, y={self.waypoint[1]}')
		self.visualiseMarker()
		self.actuation()

	def getClosestPointOnPath(self):
		distance = np.linalg.norm(self.waypoints[:, :2] - self.pose[:2], axis=1)
		closest_index = np.argmin(distance)
		return closest_index
	
	def getLookAheadDistance(self):
		lookahead_distance = 0.5 + self.k/np.abs(self.waypoints[self.closest_index,5]) * 0.15
		# cross_error = np.linalg.norm(self.waypoints[self.closest_index, :2]-self.pose[:2])
		# kappa = np.average(self.waypoints[self.closest_index:self.closest_index+5,5])
		# lookahead_distance = np.sqrt(2*cross_error/np.abs(kappa))
		lookahead_distance = np.clip(lookahead_distance, 0.5, 3)
		return lookahead_distance

	def get_closest_waypoint(self):
		self.closest_index = self.getClosestPointOnPath()
		allDistances = np.linalg.norm(self.waypoints[:, :2] - self.waypoints[self.closest_index, :2], axis=1)
		allDistances = np.roll(allDistances, -self.closest_index)
		self.lookahead_distance = self.getLookAheadDistance()
		distances = np.where(allDistances > self.lookahead_distance, allDistances, np.inf)[:int(len(allDistances)/4)]
		w=(np.argmin(distances)+self.closest_index)
		waypoint_index = w % len(self.waypoints)
		return waypoint_index
	
	def actuation(self):
		waypoint = np.dot(np.array([np.sin(-self.pose[2]), np.cos(-self.pose[2])]), self.waypoint[:2] - self.pose[:2])
		LD = np.linalg.norm(self.waypoint[:2] - self.pose[:2])
		radius = (LD**2) / (2 * waypoint)
		steering_angle = np.arctan(self.wheel_base / radius)
		steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
		speed = np.max([self.waypoints[self.closest_index,7],0.1])
		
		print(f"Speed: {speed}")
		cmd = AckermannDriveStamped()
		cmd.header.stamp = self.get_clock().now().to_msg()
		cmd.header.frame_id = "map"
		cmd.drive.steering_angle = steering_angle
		self.get_logger().info(f'Steering angle: {steering_angle}')
		cmd.drive.speed = speed
		# self.cmd_pub.publish(cmd)

	def visualiseMarker(self):
		marker = Marker()
		marker.header.frame_id = "map"
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.position.x = self.waypoint[0]
		marker.pose.position.y = self.waypoint[1]
		marker.pose.position.z = 0.0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.1
		marker.scale.y = 0.1
		marker.scale.z = 0.1
		marker.color.a = 1.0
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		np.arctan2
		self.marker_pub.publish(marker)
	
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