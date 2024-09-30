#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
import math
import trajectory_planning_helpers as tph
  
class myNode(Node):
	def __init__(self):
		super().__init__("stanley") 
		# Parameters
		self.declare_parameter("ke", 6)
		self.declare_parameter("kv", 2)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("min_speed", 0.1)
		self.declare_parameter("max_steering_angle", 0.4)
		self.declare_parameter("map_name", "esp")

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

		# Waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)
		# self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_centreline.csv', delimiter=',', skiprows=1)

		# Variables  
		self.odom: Odometry = None

		

	def odom_callback(self, msg: Odometry):
		# self.get_logger().info(f'Position: x={msg.pose.pose.position.x}, y={msg.pose.pose.position.y}')
		self.odom = msg
		self.yaw = self.euler_from_quaternion(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w) 
		self.pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw])
		self.speed = self.odom.twist.twist.linear.x
		self.getClosestPointOnPath()
		self.getHeadingError()
		self.actuation()
		# self.visualiseMarker()
		




 
		
	def getClosestPointOnPath(self):
		x = self.pose[0] + np.cos(self.pose[2])*self.wheel_base
		y = self.pose[1] + np.sin(self.pose[2])*self.wheel_base
		self.frontAxle = np.array([x, y])
		distance = np.linalg.norm(self.waypoints[:, :2]-self.frontAxle, axis=1)
		self.closest_index = np.argmin(distance)
		self.close_point = self.waypoints[self.closest_index]
		side = self.getSide(self.waypoints[self.closest_index], self.frontAxle)
		self.crossTrackError = distance[self.closest_index]*side
		self.get_logger().info(f' crossTrackError: {self.crossTrackError}')


	# def getClosestPointOnPath(self):
	# 	x = self.pose[0] + np.cos(self.pose[2])*self.wheel_base
	# 	y = self.pose[1] + np.sin(self.pose[2])*self.wheel_base
	# 	self.frontAxle = np.array([x, y])
	# 	distance = np.linalg.norm(self.waypoints[:, :2]-self.frontAxle, axis=1)
	# 	self.closest_index = np.argmin(distance)
	# 	self.close_point = self.waypoints[self.closest_index]
	# 	self.crossTrackError = distance[self.closest_index]
	# 	self.get_logger().info(f' crossTrackError: {self.crossTrackError}')

	def getHeadingError(self):
		pathHeading = self.waypoints[self.closest_index, 4]
		carHeading = self.pose[2]
		if pathHeading < 0:
			pathHeading += 2*np.pi
		if carHeading < 0:
			carHeading += 2*np.pi
		self.headingError =  self.normalize_angle(pathHeading - carHeading)
		self.get_logger().info(f'pathHeading: {pathHeading}, carHeading: {carHeading}, headingError: {self.headingError}')

	def actuation(self):
		speed = np.max([self.waypoints[self.closest_index,7],0.1])
		print(f'CE{math.atan(self.k*self.crossTrackError/speed)}')
		print(f'HE{self.headingError}')
		steering_angle = (self.headingError + math.atan2(self.k*self.crossTrackError,speed+(self.kv/np.abs(self.k*self.crossTrackError))))
		steering_angle = np.clip(self.normalize_angle(steering_angle), -self.max_steering_angle, self.max_steering_angle)
		self.get_logger().info(f'Steering angle: {steering_angle}')
		print(f"Speed: {speed}")
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
	
	def getSide(self, pathPoint, carPoint):
		heading = pathPoint[4]
		# create a vector in the direction of the path
		pathVector = np.array([np.cos(heading), np.sin(heading)])
		# create a line along the path vector
		pp = pathPoint[:2]
		p1 = pp-pathVector
		p2 = pp+pathVector
		side = np.sign((p2[0]-p1[0])*(carPoint[1]-p1[1])-(p2[1]-p1[1])*(carPoint[0]-p1[0]))
		path=np.array([carPoint,p1,pp,p2])
		self.publish_path(path)
		print(f"Side: {side}")
		return side*-1
	
	def publish_path(self, path_points):
		marker_array = MarkerArray()
		for i, (x, y) in enumerate(path_points):
			point = Marker()
			point.header.frame_id = "map"
			point.header.stamp = self.get_clock().now().to_msg()
			point.ns = "path"
			point.id = i
			point.type = Marker.SPHERE
			point.action = Marker.ADD
			point.pose.position.x = x
			point.pose.position.y = y
			point.pose.position.z = 0.0
			point.pose.orientation.x = 0.0
			point.pose.orientation.y = 0.0
			point.pose.orientation.z = 0.0
			point.pose.orientation.w = 1.0
			point.scale.x = 0.1
			point.scale.y = 0.1
			point.scale.z = 0.1
			point.color.a = 1.0
			point.color.r = 1.0
			point.color.g = 0.0
			point.color.b = 0.0
			if i == 0:
				point.color.r = 0.0
				point.color.g = 1.0
			marker_array.markers.append(point)
		self.publisher.publish(marker_array)
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()