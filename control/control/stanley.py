#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
import math
import trajectory_planning_helpers as tph
import time
from geometry_msgs.msg import PoseWithCovarianceStamped

# class Utils:
# 	def __init__(self):
		

	
  
class myNode(Node):
	def __init__(self):
		super().__init__("stanley") 
		self.get_logger().info('Stanley controller started')
		# Parameters
		self.declare_parameter("ke", 10)
		self.declare_parameter("kv", 0.0)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("min_speed", 0.1)
		self.declare_parameter("max_steering_angle", 0.4)
		self.declare_parameter("map_name", "aut")

		self.k = self.get_parameter("ke").value
		self.wheel_base = self.get_parameter("wheel_base").value
		self.min_speed = self.get_parameter("min_speed").value
		self.max_steering_angle = self.get_parameter("max_steering_angle").value
		self.map_name = self.get_parameter("map_name").value
		self.kv = self.get_parameter("kv").value

		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
		# Publishers
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
		# self.marker_pub = self.create_publisher(Marker, "/waypoint_marker", 10)
		self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
		self.initial_position_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose",10)

		# Waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)

		# Variables  
		self.odom: Odometry = None
		self.start = True
		self.start_time = 0
		self.finish = 0
		self.distance = 0
		self.prevDistance = 0
		self.lapDistance = self.waypoints[-1, 6]
		self.lapCount = 0

		self.saveData = np.zeros((50000, 6))
		self.saveDataIndex = 0

		self.initial_position_ = False

		self.setInitialPosition()



	def odom_callback(self, msg: Odometry):
		if self.start:
			self.start_time = time.time()
			self.start = False

		self.odom = msg
		self.yaw = self.euler_from_quaternion(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w) 
		self.pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw])
		self.speed = self.odom.twist.twist.linear.x
		self.current_time = time.time() - self.start_time
		self.getLapProgress()

		# self.distance = self.waypoints

		self.getCrossTrackError()
		self.getHeadingError()
		self.actuation()
		# self.visualiseMarker()

	def getLapProgress(self):
		distance = np.linalg.norm(self.waypoints[:, :2] - self.pose[:2], axis=1)
		closest_index = np.argmin(distance)
		self.closestPointOnPath = self.waypoints[closest_index]
		self.distance = self.closestPointOnPath[6]
		self.lapProgress = self.distance/self.lapDistance*100
		LP = self.lapProgress +100*self.lapCount
		self.get_logger().info(f'Lap progress: {(self.lapProgress+self.lapCount*100):.2f}%')

		if (int(self.lapProgress) == 100) and (int(self.prevDistance) != 100):
			self.lapCount += 1
			self.get_logger().info(f'Lap {self.lapCount} completed')

		self.prevDistance = self.lapProgress

		if LP <= 200:
			temp = np.array([self.current_time, self.pose[0], self.pose[1], self.pose[2], self.speed, int(LP/100)])
			self.saveData[self.saveDataIndex] = temp
			self.saveDataIndex += 1

		if LP > 200:
			self.saveData = self.saveData[:self.saveDataIndex]
			dataPath = f'/home/chris/sim_ws/src/control/Results/Data/{self.map_name}_stanley.csv'
			with open(dataPath, 'wb') as fh:
				np.savetxt(fh, self.saveData, fmt='%0.16f', delimiter=',', header='time,x,y,yaw,speed,lap')
			self.get_logger().info('Data saved')
			rclpy.shutdown()
			




	def getCrossTrackError(self):
		x = self.pose[0] + np.cos(self.pose[2])*self.wheel_base
		y = self.pose[1] + np.sin(self.pose[2])*self.wheel_base
		self.frontAxle = np.array([x, y])
		distance = np.linalg.norm(self.waypoints[:, :2]-self.frontAxle, axis=1)
		self.closest_index = np.argmin(distance)
		self.close_point = self.waypoints[self.closest_index]
		self.crossTrackError = np.dot([np.sin(-self.pose[2]), np.cos(-self.pose[2])],self.close_point[:2]-self.frontAxle)

	def getHeadingError(self):
		pathHeading = self.waypoints[self.closest_index, 4]
		carHeading = self.pose[2]
		if pathHeading < 0:
			pathHeading += 2*np.pi
		if carHeading < 0:
			carHeading += 2*np.pi
		self.headingError =  self.normalize_angle(pathHeading - carHeading)

	def actuation(self):
		speed = np.max([self.waypoints[self.closest_index,7],0.1])
		steering_angle = (self.headingError + math.atan2(self.k*self.crossTrackError,speed+(self.kv/np.abs(self.crossTrackError))))
		steering_angle = np.clip(self.normalize_angle(steering_angle), -self.max_steering_angle, self.max_steering_angle)
		cmd = AckermannDriveStamped()
		cmd.header.stamp = self.get_clock().now().to_msg()
		cmd.header.frame_id = "map"
		cmd.drive.steering_angle = steering_angle
		cmd.drive.speed = speed
		self.cmd_pub.publish(cmd)

	def visualiseMarker(self):
		marker = Marker()
		marker.header.frame_id = "map"
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		# marker.pose.position.x = self.waypoints[self.closest_index, 0]
		# marker.pose.position.y = self.waypoints[self.closest_index, 1]
		marker.pose.position.x = self.frontAxle[0]
		marker.pose.position.y = self.frontAxle[1]
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
		self.marker_pub.publish(marker)
	
	def euler_from_quaternion(self,x, y, z, w):  
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
		return yaw_z # in radians
	
	def normalize_angle(self, angle):
		if angle > np.pi:
			angle = angle - 2*np.pi
		if angle < -np.pi:
			angle = angle + 2*np.pi
		return angle
	
	def setInitialPosition(self):
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
	
class SaveData:
	def __init__(self):
		self.a =1
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()