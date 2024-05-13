#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry  
# from f110_interfaces.msg import CrashStatus
import numpy as np

class localisation_test_accuracy(Node):
	def __init__(self):
		super().__init__("localisation_test_accuracy")
		#Subscribers 
		self.trueOdom_subscriber_ = self.create_subscription(Odometry, "/ego_racecar/odom", self.trueOdomCallback, 10)
		self.pfOdom_subscriber_ = self.create_subscription(Odometry, "/pf/pose/odom", self.pfOdomCallback, 10)
		# self.collison_subscriber_ = self.create_subscription(CrashStatus, "ego_crash", self.collisionCallback, 1)
		
		# self.lap = 0

	def trueOdomCallback(self, msg: Odometry):
		time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		dataArray = np.array([time, x, y])
		# self.get_logger().info("True Odom: "+str(dataArray))
		self.saveToFile(dataArray, 'trueOdom')
		
	def pfOdomCallback(self, msg: Odometry):
		time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		dataArray = np.array([time, x, y])
		# self.get_logger().info("True Odom: "+str(dataArray))
		self.saveToFile(dataArray, 'pfOdom')

	def saveToFile(self,data, filename):
		path = '/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/'+filename+'.csv'
		# np.savetxt(fname=path, X=data, delimiter=',',newline='/n', fmt='%1.3f')
		f = open(path, 'a')
		f.write(str(data[0])+','+str(data[1])+','+str(data[2])+'\n')
		f.close()


		
  
def main(args=None):
	rclpy.init(args=args)
	node = localisation_test_accuracy()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()