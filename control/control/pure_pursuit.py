#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
import time
  
class myNode(Node):
	def __init__(self):
		super().__init__("pure_pursuit")
		# Parameters
		self.declare_parameter("lookahead_distance", 1)
		self.declare_parameter("k", 0.5)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("min_speed", 0.1)
		self.declare_parameter("max_steering_angle", 0.4)
		self.declare_parameter("map_name", "mco")

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
		self.initial_position_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose",10)

		# Waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)

		# Variables 
		self.initial_position_ = False
		
		self.odom: Odometry = None
		self.lapDistance = self.waypoints[-1, 6]
		
		self.setInitialPosition()

		

	def odom_callback(self, msg: Odometry):
		self.odom = msg

		if self.start:
			# self.setInitialPosition()
			self.start_time = time.time()
			self.start = False

		self.yaw = self.euler_from_quaternion(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w) 
		self.pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw])
		self.speed = self.odom.twist.twist.linear.x
		self.current_time = time.time() - self.start_time
		self.getLapProgress()


		waypoint_index = self.get_closest_waypoint()
		self.waypoint = self.waypoints[waypoint_index]
		# self.get_logger().info(f'Closest waypoint: x={self.waypoint[0]}, y={self.waypoint[1]}')
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
		speed = np.max([self.waypoints[self.closest_index,7],0.1])
		waypoint = np.dot(np.array([np.sin(-self.pose[2]), np.cos(-self.pose[2])]), self.waypoint[:2] - self.pose[:2])
		LD = np.linalg.norm(self.waypoint[:2] - self.pose[:2])
		radius = (LD**2) / (2 * waypoint)
		steering_angle = np.arctan(self.wheel_base / radius)
		steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
		# print(f"Speed: {speed}")
		cmd = AckermannDriveStamped()
		cmd.header.stamp = self.get_clock().now().to_msg()
		cmd.header.frame_id = "map"
		cmd.drive.steering_angle = steering_angle
		# self.get_logger().info(f'Steering angle: {steering_angle}')
		cmd.drive.speed = speed
		self.cmd_pub.publish(cmd)

	def getLapProgress(self):
		distance = np.linalg.norm(self.waypoints[:, :2] - self.pose[:2], axis=1)
		closest_index = np.argmin(distance)
		self.closestPointOnPath = self.waypoints[closest_index]
		self.distance = self.closestPointOnPath[6]
		self.lapProgress = self.distance/self.lapDistance*100
		LP = self.lapProgress + 100*self.lapCount
		self.get_logger().info(f'Lap progress: {(self.lapProgress+self.lapCount*100):.2f}%')
		print(self.lapCount)

		# if (int(self.lapProgress)%1 == 100) and (int(self.prevDistance)%1 != 100):
		if ((self.lapProgress) == 100) and (int(self.prevDistance) != 100):
			self.lapCount += 1
			self.get_logger().info(f'Lap {self.lapCount} completed')

		self.prevDistance = self.lapProgress

		if LP <= 200:
			test=int(LP/100)
			temp = np.array([self.current_time, self.pose[0], self.pose[1], self.pose[2], self.speed, test])
			self.saveData[self.saveDataIndex] = temp
			self.saveDataIndex += 1

		if LP > 200:
			self.saveData = self.saveData[:self.saveDataIndex]
			dataPath = f'/home/chris/sim_ws/src/control/Results/Data/{self.map_name}_purepursuit.csv'
			with open(dataPath, 'wb') as fh:
				np.savetxt(fh, self.saveData, fmt='%0.16f', delimiter=',', header='time,x,y,yaw,speed,lap')
			# self.setInitialPosition()
			self.get_logger().info('Data saved')
			rclpy.shutdown()

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
  
	def setInitialPosition(self):
		self.start = True
		self.start_time = 0
		self.finish = 0
		self.distance = 0
		self.prevDistance = 0
		self.lapCount = 0

		self.saveData = np.zeros((50000, 6))
		self.saveDataIndex = 0
		if not self.initial_position_:
			initial_position = PoseWithCovarianceStamped()
			initial_position.header.frame_id = "map"
			initial_position.header.stamp = self.get_clock().now().to_msg()
			# initial_position.pose.pose.position.x = self.waypoints[0,0]
			# initial_position.pose.pose.position.y = self.waypoints[0,1]
			initial_position.pose.pose.position.x = 0.0
			initial_position.pose.pose.position.y = 0.0
			initial_position.pose.pose.position.z = 0.0
			initial_position.pose.pose.orientation.x = 0.0
			initial_position.pose.pose.orientation.y = 0.0
			initial_position.pose.pose.orientation.z = 0.0
			initial_position.pose.pose.orientation.w = 1.0

			cmd = AckermannDriveStamped()
			cmd.header.stamp = self.get_clock().now().to_msg()
			cmd.header.frame_id = "map"
			cmd.drive.speed = 0.0
			cmd.drive.acceleration = 0.0
			cmd.drive.jerk = 0.0
			cmd.drive.steering_angle = 0.0
			cmd.drive.steering_angle_velocity = 0.0

			self.initial_position_pub.publish(initial_position)
			self.cmd_pub.publish(cmd)
			self.get_logger().info('Initial position set')
			self.initial_position_ = True

def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()