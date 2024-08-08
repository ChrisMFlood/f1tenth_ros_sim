#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf_transformations 
  
class myNode(Node):
	def __init__(self):
		super().__init__("orientation_listener")
		self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
		self.get_logger().info("Orientation Listener has been started.")
		
		
	def odom_callback(self, msg: Odometry):
		orientation = msg.pose.pose.orientation
		roll, pitch, yaw = self.quaternion_to_angle(orientation)
		self.get_logger().info(str(yaw))
		
	def quaternion_to_angle(self,q):
		"""Convert a quaternion _message_ into an angle in radians.
		The angle represents the yaw.
		This is not just the z component of the quaternion."""
		quat = [q.x, q.y, q.z, q.w]
		roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
		return roll, pitch, yaw
		
		
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()