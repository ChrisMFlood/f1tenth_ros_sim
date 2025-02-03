#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Joy
from ackermann_msgs.msg import AckermannDriveStamped
from control import utils as Utils
import numpy as np
  
class myNode(Node):
	def __init__(self):
		super().__init__("follow_the_gap")  

		# Parameters
		self.declare_parameter("max_lidar_dist", 10.0)
		self.declare_parameter("preprocess_conv_size", 3)
		self.declare_parameter("bubble_radius", 200)
		self.declare_parameter("best_point_conv_size", 80)
		self.declare_parameter("max_steer", 0.4)
		self.declare_parameter("fast_steering_angle", 0.0785)
		self.declare_parameter("straights_steering_angle", 0.174)
		self.declare_parameter("fast_speed", 6.0)
		self.declare_parameter("corners_speed", 3.0)
		self.declare_parameter("straights_speed", 8.0)
		self.declare_parameter('speed_gain',0.5)

		self.max_lidar_dist = self.get_parameter("max_lidar_dist").value
		self.preprocess_conv_size = self.get_parameter("preprocess_conv_size").value
		self.bubble_radius = self.get_parameter("bubble_radius").value
		self.best_point_conv_size = self.get_parameter("best_point_conv_size").value
		self.max_steer = self.get_parameter("max_steer").value
		self.fast_steering_angle = self.get_parameter("fast_steering_angle").value
		self.straights_steering_angle = self.get_parameter("straights_steering_angle").value
		self.fast_speed = self.get_parameter("fast_speed").value
		self.corners_speed = self.get_parameter("corners_speed").value
		self.straights_speed = self.get_parameter("straights_speed").value
		self.speed_gain = self.get_parameter("speed_gain").value
		
		# Subscribers
		self.lidar_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
		self.joy_sub = self.create_subscription(Joy, "/joy", self.joy_callback, 10)
		# Pulishers
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
		
	def joy_callback(self, msg: Joy):
		self.Joy7 = msg.buttons[7]
		
	def lidar_callback(self, laserScan: LaserScan):
		proc_ranges = self.preprocessScan(laserScan.ranges)

		closest = proc_ranges.argmin()
		min_index = closest - self.bubble_radius
		max_index = closest + self.bubble_radius
		if min_index < 0: min_index = 0
		if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
		proc_ranges[min_index:max_index] = 0

		gap_start, gap_end = self.find_max_gap(proc_ranges)

		best = self.find_best_point(gap_start, gap_end, proc_ranges)

		if self.Joy7:
			steering_angle = self.get_angle(best, len(proc_ranges))
			# TODO: Implement better speed control
			if abs(steering_angle) > self.straights_steering_angle:
				speed = self.corners_speed
			elif abs(steering_angle) > self.fast_steering_angle:
				speed = self.fast_speed
			else:
				speed = self.straights_speed
		else:
			steering_angle = 0
			speed = 0

		Utils.pubishActuation(steering_angle, speed*self.speed_gain, self.cmd_pub)

	def get_angle(self, range_index, range_len):
		lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
		steering_angle = lidar_angle / 2
		steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
		return steering_angle

	def find_best_point(self, start_i, end_i, ranges: LaserScan.ranges):
		averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.best_point_conv_size),'same') / self.best_point_conv_size
		return averaged_max_gap.argmax() + start_i

	def find_max_gap(self, free_space_ranges: LaserScan.ranges):
		masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
		slices = np.ma.notmasked_contiguous(masked)
		slice_lengths = np.zeros(len(slices))
		for i, slice in enumerate(slices):
			slice_lengths[i] = slice.stop - slice.start
		max_index = slice_lengths.argmax()
		return slices[max_index].start, slices[max_index].stop

	def preprocessScan(self, ranges: LaserScan.ranges):
		self.radians_per_elem = (2 * np.pi) / len(ranges)
		proc_ranges = np.array(ranges[135:-135])
		proc_ranges = np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), 'same') / self.preprocess_conv_size
		proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)
		return proc_ranges
	
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()