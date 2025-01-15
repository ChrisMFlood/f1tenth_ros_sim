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
from geometry_msgs.msg import PoseArray, Pose
from control import utils as utils

class myNode(Node):
	def __init__(self):
		super().__init__("stanley") 
		self.get_logger().info('Stanley controller started')
		# Parameters
		self.declare_parameter("odom_topic","/ego_racecar/odom")
		# self.declare_parameter("odom_topic","/pf/pose/odom")
		self.odom_topic = self.get_parameter("odom_topic").value
		self.declare_parameter("ke", 10)
		self.declare_parameter("kv", 10)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("min_speed", 0.1)
		self.declare_parameter("max_speed", 8)
		self.declare_parameter("max_steering_angle", 0.4)
		self.declare_parameter("map_name", "esp")

		self.k = self.get_parameter("ke").value
		self.wheel_base = self.get_parameter("wheel_base").value
		self.min_speed = self.get_parameter("min_speed").value
		self.max_speed = self.get_parameter("max_speed").value
		self.max_steering_angle = self.get_parameter("max_steering_angle").value
		self.map_name = self.get_parameter("map_name").value
		self.kv = self.get_parameter("kv").value

		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
		# Publishers
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
		self.waypoints_pub = self.create_publisher(PoseArray, '/waypoints', 10)

		# self.marker_pub = self.create_publisher(Marker, "/waypoint_marker", 10)
		self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
		self.initial_position_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose",10)
		self.ref_traj_pub = self.create_publisher(PoseArray, 'reference_trajectory', 10)
		self.calc_ref_traj_pub = self.create_publisher(PoseArray, 'calculated_trajectory', 10)

		# Waypoints
		self.waypoints = np.loadtxt(f'/home/chris/sim_ws/src/global_planning/maps/{self.map_name}_minCurve.csv', delimiter=',', skiprows=1)
		utils.publishTrajectory(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,4], self.waypoints_pub)

		# Variables  



	def odom_callback(self, msg: Odometry):
		self.odom = msg
		self.yaw = utils.euler_from_quaternion(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w) 
		self.pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw])
		self.speed = self.odom.twist.twist.linear.x

		self.crossTrackError = self.getCrossTrackError()
		self.headingError = self.getHeadingError()
		
		
		self.steeringAngle, self.velocity = self.actuation()
		# print(self.steeringAngle, self.velocity)
		utils.pubishActuation(self.steeringAngle, self.velocity, self.cmd_pub)

	def getCrossTrackError(self):
		x = self.pose[0] + np.cos(self.pose[2])*self.wheel_base
		y = self.pose[1] + np.sin(self.pose[2])*self.wheel_base
		self.frontAxle = np.array([x, y])
		self.nearest_point_front, self.nearest_dist, self.t1, self.targetIndex = utils.nearest_point(self.frontAxle, self.waypoints[:,0:2])
		vec_dist_nearest_point = self.nearest_point_front -  self.frontAxle
		front_axle_vec_rot_90 = np.array([math.sin(-self.yaw), math.cos(-self.yaw)])
		crossTrackError = np.dot(front_axle_vec_rot_90,vec_dist_nearest_point)
		print(f"corsstrack: {crossTrackError}")
		return crossTrackError
	
	def getHeadingError(self):
		pathHeading = self.waypoints[self.targetIndex, 4]
		carHeading = self.pose[2]
		if pathHeading < 0:
			pathHeading += 2*np.pi
		if carHeading < 0:
			carHeading += 2*np.pi
		headingError =  utils.normaliseAngle(pathHeading - carHeading)
		print(f"heading: {headingError}")
		return headingError
	
	def actuation(self):
		speed = self.waypoints[self.targetIndex,7] #+ (self.waypoints[self.targetIndex+1,7] - self.waypoints[self.targetIndex,7])*self.t1

		# theta_cte = math.atan2(self.k*self.crossTrackError, self.speed+self.kv)
		theta_cte = math.atan2(self.k*self.crossTrackError,speed+self.kv)
		steeringAngle = self.headingError + theta_cte
		# steeringAngle = theta_cte
		# steeringAngle = self.headingError
		steeringAngle = np.clip(steeringAngle, -self.max_steering_angle, self.max_steering_angle)
		print(f'steering angle: {steeringAngle}')
		return steeringAngle, speed






  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()