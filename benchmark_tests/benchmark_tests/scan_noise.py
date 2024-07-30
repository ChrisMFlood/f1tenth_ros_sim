#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import os
  
class myNode(Node):
	def __init__(self):
		super().__init__("scan_noise")  
		self.laserscan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
		self.noisy_laserscan_pub = self.create_publisher(LaserScan, "/noisy_scan", 10)

		self.saveScanData = saveScanData()
		self.count = 0

	def scan_callback(self, msg: LaserScan):
		self.saveScanData.scanParameters[0] = msg.angle_min
		self.saveScanData.scanParameters[1] = msg.angle_max
		self.saveScanData.scanParameters[2] = msg.angle_increment
		self.saveScanData.scanParameters[3] = len(msg.ranges)
		self.saveScanData.saveParameters()
		self.saveScanData.scanData[:,self.count] = msg.ranges
		self.get_logger().info(str(len(self.saveScanData.scanData)))
		self.count += 1
		if self.count == 5:
			self.saveScanData.saveScan()
			self.get_logger().info("Scan data saved")
			rclpy.shutdown()

class saveScanData:
	def __init__(self):
		self.scanData = np.zeros((1080,5))
		self.scanParameters = np.zeros((4,1))
		self.testNumber = 0
		# self.path = f'/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanData_{self.testNumber}.csv'
		# self.pPath = f'/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanParameters_{self.testNumber}.csv'
		
		self.desktop_path = os.path.expanduser("~/Desktop") #path on car to Desktop
		self.path = os.path.join(self.desktop_path, f"scanData_{self.testNumber}.csv")
		

	def saveScan(self):
		while os.path.exists(self.path):
			self.testNumber += 1
			self.path = os.path.join(self.desktop_path, f"scanData_{self.testNumber}.csv")
		np.savetxt(self.path, self.scanData, delimiter=',')

	def saveParameters(self):
		self.pPath = os.path.join(self.desktop_path, f"scanParameters.csv")
		np.savetxt(self.pPath, self.scanParameters, delimiter=',')
		
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()