#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry  
from f110_interfaces.msg import CrashStatus
from ackermann_msgs.msg import AckermannDriveStamped
from example_interfaces.msg import Bool


import numpy as np

class localisation_test_accuracy(Node):
	def __init__(self):
		super().__init__("localisation_test_accuracy")
		#Subscribers 
		self.trueOdom_subscriber_ = self.create_subscription(Odometry, "/ego_racecar/odom", self.trueOdomCallback, 10)
		self.pfOdom_subscriber_ = self.create_subscription(Odometry, "/pf/pose/odom", self.pfOdomCallback, 10)
		# self.collison_subscriber_ = self.create_subscription(CrashStatus, "ego_crash", self.collisionCallback, 10)
		# self.drive_subscriber_ = self.create_subscription(AckermannDriveStamped, "/drive", self.driveCallback, 10)
		self.done_subscriber_ = self.create_subscription(Bool, "/ego_done", self.doneCallback, 10)

		# timers
		# self.save_timer = self.create_timer(0.05, self.savedata)
		
		# obect to save data
		self.ds = saveToFile()

	def doneCallback(self, msg: Bool):
		self.get_logger().info(str(msg.data))
		if msg.data:
			self.ds.saveFlag = msg.data
			self.get_logger().info("Done")


	def trueOdomCallback(self, msg: Odometry):
		# time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		z = msg.pose.pose.orientation.z
		w = msg.pose.pose.orientation.w

		# if x == 0.1 and -1.0 <= y <= 1.5:
		# 	self.ds.lap = self.ds.lap + 1

		self.ds.tempData[2] = x
		self.ds.tempData[3] = y
		self.ds.tempData[4] = z
		self.ds.tempData[5] = w

		# dataArray = np.array([time, x, y])
		# self.get_logger().info("True Odom: "+str(dataArray))
		# self.saveToFile(dataArray, 'trueOdom')
		
	def pfOdomCallback(self, msg: Odometry):
		time_s = msg.header.stamp.sec 
		time_ns = msg.header.stamp.nanosec
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		z = msg.pose.pose.orientation.z
		w = msg.pose.pose.orientation.w

		self.ds.tempData[0] = time_s
		self.ds.tempData[1] = time_ns
		self.ds.tempData[6] = x
		self.ds.tempData[7] = y
		self.ds.tempData[8] = z
		self.ds.tempData[9] = w

		self.ds.saveData = np.vstack((self.ds.saveData, self.ds.tempData))
		if self.ds.saveFlag:
			self.ds.saveToFile(self.ds.saveData, 'gbr')
			self.get_logger().info('saving to file')
			# self.ds.saveFlag = False
			rclpy.shutdown()
			# self.ds.saveData = np.array([0,0,0,0,0,0,0,0,0])

		# dataArray = np.array([time, x, y])
		# self.get_logger().info("True Odom: "+str(dataArray))
		# self.saveToFile(dataArray, 'pfOdom')

	

class saveToFile:
	def __init__(self):
		self.lap = 0
		# 						  T,t,x,y,z,w,x,y,z,w
		self.saveData = np.array([0,0,0,0,0,0,0,0,0,0],dtype='f')
		self.tempData = np.array([0,0,0,0,0,0,0,0,0,0],dtype='f')
		self.saveFlag = False

	def saveToFile(self,data,map):
		np.savetxt(fname='/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/'+map+'_1.csv', 
			 X=data, 
			 delimiter=',',
			#  newline=' /n', 
			#  fmt='%-10f',
			 header='seconds, nanoseconds, trueX, trueY, trueZ, trueW, pfX, pfY, pfZ, pfW',
			 )
		# path = '/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/'+filename+'.csv'
		# # np.savetxt(fname=path, X=data, delimiter=',',newline='/n', fmt='%1.3f')
		# f = open(path, 'a')
		# f.write(str(data[0])+','+str(data[1])+','+str(data[2])+'\n')
		# f.close()



		
  
def main(args=None):
	rclpy.init(args=args)
	node = localisation_test_accuracy()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()