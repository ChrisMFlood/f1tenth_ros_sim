#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
import time
from scipy import sparse
from cvxpy import *

class State:
	x: float = 0.0
	y: float = 0.0
	yaw: float = 0.0
	v: float = 0.0
	yaw_rate: float = 0.0
	slip_angle: float = 0.0
  
class myNode(Node):
	def __init__(self):
		super().__init__("mpc") 
		self.get_logger().info('MPC controller started')
		# Parameters
		self.declare_parameter("map", 'mco')
		self.declare_parameter("dt", 0.1)
		self.declare_parameter("time_step_horizon", 10)
		
		self.map_name = self.get_parameter("map").value
		self.dt = self.get_parameter("dt").value
		self.time_step_horizon = self.get_parameter("time_step_horizon").value
		

		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
		# Publishers
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
		self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
		self.initial_position_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose",10)
		self.ref_traj_pub = self.create_publisher(PoseArray, "/reference_trajectory", 10)

		# Waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)
		self.waypointStepSize = np.mean(self.waypoints[1:, 6] - self.waypoints[:-1, 6])

		# Variables  
		self.odom: Odometry = None
		self.start = True
		self.start_time = 0
		self.finish = 0
		self.distance = 0
		self.prevDistance = 0
		self.lapDistance = self.waypoints[-1, 6]
		self.lapCount = 0

		self.referenceTrajectory = np.zeros((4, self.time_step_horizon+1))
		'''[x, y, yaw, speed]'''

		self.H = sparse.csc_matrix(np.zeros((6, 6)))
		self.g = sparse.csc_matrix(np.zeros((6, 1)))
		self.lb = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
		self.ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

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

		self.currentState = State()
		self.currentState.x = self.pose[0]
		self.currentState.y = self.pose[1]
		self.currentState.yaw = self.pose[2]
		self.currentState.v = self.speed
		self.currentState.yaw_rate = msg.twist.twist.angular.z
		
		self.getLapProgress()
		self.getReferenceTrajectory()
		# self.mpc()


	def mpc(self):	
		self.horizon = 10

	def getModelMatrix(self):
		A = sparse.csc

	def getReferenceTrajectory(self):
		self.closestIndex, self.closestPoint = self.getClosestPoint()
		travel_distance = abs(self.speed*self.dt)
		dind = np.max([travel_distance/self.waypointStepSize,1/self.time_step_horizon])
		ind_list = (self.closestIndex + np.insert(np.cumsum(np.repeat(dind, self.time_step_horizon)), 0, 0)).astype(int)
		ind_list[ind_list >= self.waypoints.shape[0]] -= self.waypoints.shape[0]
		self.referenceTrajectory = np.zeros((4, self.time_step_horizon+1))
		self.referenceTrajectory[0,:] = self.waypoints[ind_list,0]
		self.referenceTrajectory[1,:] = self.waypoints[ind_list,1]
		self.referenceTrajectory[2,:] = self.waypoints[ind_list,4]
		self.referenceTrajectory[3,:] = self.waypoints[ind_list,7]
		self.publishReferenceTrajectory()

	def publishReferenceTrajectory(self):
		pose_array = PoseArray()
		pose_array.header.frame_id = "map"
		pose_array.header.stamp = self.get_clock().now().to_msg()
		pose_array.poses = []
		for i in range(self.referenceTrajectory.shape[1]):
			pose = Pose()
			pose.position.x = self.referenceTrajectory[0,i]
			pose.position.y = self.referenceTrajectory[1,i]
			qw,qx,qy,qz = self.quaternion_from_euler(0,0,self.referenceTrajectory[2,i])
			pose.orientation.x = qx
			pose.orientation.y = qy
			pose.orientation.z = qz
			pose.orientation.w = qw
			pose_array.poses.append(pose)
		self.ref_traj_pub.publish(pose_array)
	
	# def getClosestPointOnPath(self):
	# 	'''
	# 	return closest point on the path and index and projection of the car on the path
	# 	'''
	# 	position = np.array([self.pose[0], self.pose[1]])
	# 	diffs = np.vstack((self.waypoints[1:, :2] - self.waypoints[:-1, :2], self.waypoints[0, :2] - self.waypoints[-1, :2])).reshape(-1, 2)
	# 	l2s   = diffs[:,0]**2 + diffs[:,1]**2
	# 	dots = np.diag(np.dot((position - self.waypoints[:, :2]), diffs.T))
	# 	t = dots / l2s
	# 	t[t<0.0] = 0.0
	# 	t[t>1.0] = 1.0
	# 	projections = self.waypoints[:,:2] + (t*diffs.T).T
	# 	distances = np.linalg.norm(projections - position, axis=1)
	# 	print(distances)

	def getClosestPoint(self):
		'''
		return closest point on the path and index

		:return:
		index, closest point
		'''
		distance = np.linalg.norm(self.waypoints[:, :2] - self.pose[:2], axis=1)
		closest_index = np.argmin(distance)
		return closest_index , self.waypoints[closest_index]
		


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
			



	def actuation(self):
		cmd = AckermannDriveStamped()
		cmd.header.stamp = self.get_clock().now().to_msg()
		cmd.header.frame_id = "map"
		cmd.drive.steering_angle = 0
		cmd.drive.speed = 0
		self.cmd_pub.publish(cmd)


	
	def euler_from_quaternion(self,x, y, z, w):  
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
		return yaw_z # in radians
	
	def quaternion_from_euler(self, roll, pitch, yaw):
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		cr = math.cos(roll * 0.5)
		sr = math.sin(roll * 0.5)
		cp = math.cos(pitch * 0.5)
		sp = math.sin(pitch * 0.5)
		w = cy * cr * cp + sy * sr * sp
		x = cy * sr * cp - sy * cr * sp
		y = cy * cr * sp + sy * sr * cp
		z = sy * cr * cp - cy * sr * sp
		return w,x,y,z
	
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