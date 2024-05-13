#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped  
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

class myNode(Node):
	def __init__(self):
		super().__init__("start_position")
		self.initial_position_ = False
		self.odom_subscriber_ = self.create_subscription(Odometry, "/ego_racecar/odom", self.odomCallback, 10)
		self.position_publisher_ = self.create_publisher(PoseWithCovarianceStamped, "/initialpose",10)
		self.speed_publisher_ = self.create_publisher(AckermannDriveStamped, "/drive",10)
		self.publishCallback()

	def serviceCallback(self, request, response):
		self.get_logger().info('service called')
		if(request.initial_position):
			self.publishCallback()
			if(self.initial_position_):
				response.success = True
				response.message = 'initial position set'
				return response

	def publishCallback(self):
		position_msg = PoseWithCovarianceStamped()
		position_msg.pose.pose.orientation.x = 0.0
		position_msg.pose.pose.orientation.y = 0.0
		position_msg.pose.pose.orientation.z = 0.0
		position_msg.pose.pose.orientation.w = 1.0
		position_msg.pose.pose.position.x = 0.0
		position_msg.pose.pose.position.y = 0.0
		position_msg.pose.pose.position.z = 0.0

		speed_msg = AckermannDriveStamped()
		speed_msg.drive.acceleration = 0.0
		speed_msg.drive.speed = 0.0
		speed_msg.drive.jerk = 0.0
		speed_msg.drive.steering_angle =0.0
		speed_msg.drive.steering_angle_velocity = 0.0

		self.get_logger().info('publishing initial position and speed')
		self.position_publisher_.publish(position_msg)
		self.speed_publisher_.publish(speed_msg)
	
	def odomCallback(self, msg: Odometry):
		self.publishCallback()
		self.get_logger().info('x: ' + str(msg.pose.pose.position.x) + ', y: ' + str(msg.pose.pose.position.y) + ', z: ' + str(msg.pose.pose.position.z))
		if (msg.pose.pose.position.x == 0.0 and msg.pose.pose.position.y == 0.0 and msg.pose.pose.position.z == 0.0 and msg.pose.pose.orientation.x == 0.0 and msg.pose.pose.orientation.y == 0.0 and msg.pose.pose.orientation.z == 0.0 and msg.pose.pose.orientation.w == 1.0 and msg.twist.twist.linear.x == 0.0 and msg.twist.twist.linear.y == 0.0 and msg.twist.twist.linear.z == 0.0 and msg.twist.twist.angular.x == 0.0 and msg.twist.twist.angular.y == 0.0 and msg.twist.twist.angular.z == 0.0):
			self.initial_position_ = True
			self.get_logger().info('at initial position')
			rclpy.shutdown()
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()