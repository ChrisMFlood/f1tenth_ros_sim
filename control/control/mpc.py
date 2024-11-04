#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from visualization_msgs.msg import MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
import cvxpy as CP
# from scipy.linalg import block_diag
from sympy import pprint
import math
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag
from scipy.sparse import csc_matrix

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray 

class State:
	def __init__(self):
		self.x: float = 0.0
		self.y: float = 0.0
		self.v: float = 0.0
		self.yaw: float = 0.0
		self.yaw_rate: float = 0.0
		self.slip_angle: float = 0.0
		self.delta: float = 0.0
  
class myNode(Node):
	def __init__(self):
		super().__init__("mpc") 
		self.get_logger().info('MPC controller started')
		# Parameters
		self.declare_parameter("map", 'aut')
		self.declare_parameter("dt", 0.1)
		self.declare_parameter("time_step_horizon", 8)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("MAX_SPEED", 8.0)
		self.declare_parameter("MIN_SPEED", 0.0)
		self.declare_parameter("MAX_ACCEL", 3.0)
		self.declare_parameter("MAX_STEER", 0.4189)
		self.declare_parameter("MIN_STEER", -0.4189)
		self.declare_parameter("MAX_DSTEER", np.pi)

		
		self.map_name = self.get_parameter("map").value
		self.dt = self.get_parameter("dt").value
		self.time_step_horizon = self.get_parameter("time_step_horizon").value
		self.WB = self.get_parameter("wheel_base").value
		self.MAX_DSTEER = self.get_parameter("MAX_DSTEER").value
		self.MAX_SPEED = self.get_parameter("MAX_SPEED").value
		self.MIN_SPEED = self.get_parameter("MIN_SPEED").value
		self.MAX_ACCEL = self.get_parameter("MAX_ACCEL").value
		self.MAX_STEER = self.get_parameter("MAX_STEER").value
		self.MIN_STEER = self.get_parameter("MIN_STEER").value

		self.declare_parameter("NX",4)
		self.NX = self.get_parameter("NX").value
		'''X = [x, y, v, yaw]'''
		self.declare_parameter("NU",2)
		self.NU = self.get_parameter("NU").value
		'''U = [acceleration, steering speed,]'''

		# self.R: list = field(default_factory=lambda: np.diag([0.01, 100.0]))
		# self.Rd: list = field(default_factory=lambda: np.diag([0.01, 100.0]))
		# self.Q: list = field(default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))
		# self.Qf: list = field(default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))

		self.R =  np.diag([100.0, 0.01])
		self.Rd =  np.diag([100.0, 0.01])
		self.Q =  np.diag([13.5, 13.5, 5.5, 13.0])
		self.Qf =  np.diag([13.5, 13.5, 5.5, 13.0])

		self.odelta_v = None
		self.odelta = None
		self.oa = None
		self.init_flag = 0
		

		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
		# Publishers
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
		self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
		self.initial_position_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose",10)
		self.ref_traj_pub = self.create_publisher(PoseArray, "/reference_trajectory", 10)
		self.cal_traj_pub = self.create_publisher(PoseArray, "/calculated_trajectory", 10)

		# Waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)
		self.waypointStepSize = np.mean(self.waypoints[1:,6] - self.waypoints[:-1,6])

		self.current_state = State()
		# Initial mpc
		self.mpc_init()

	def odom_callback(self, msg: Odometry):
		self.current_state.x = msg.pose.pose.position.x
		self.current_state.y = msg.pose.pose.position.y
		self.current_state.v = msg.twist.twist.linear.x
		self.current_state.yaw = self.euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
		self.current_state.yaw_rate = msg.twist.twist.angular.z
		self.pose = np.array([self.current_state.x, self.current_state.y, self.current_state.yaw])
		x0 = np.array([self.current_state.x, self.current_state.y, self.current_state.v, self.current_state.yaw])

		ref_path = self.calc_ref_trajectory(self.current_state, self.waypoints)
		self.publishReferenceTrajectory(ref_path,self.ref_traj_pub)
		(
			self.oa,
			self.odelta_v,
			ox,
			oy,
			oyaw,
			ov,
			state_predict,
		) = self.mpc_control(ref_path, x0, self.oa, self.odelta_v)
		# print("oa", self.oa)
		# print("odelta_v", self.odelta_v)
		# print("ox", ox)
		# print("oy", oy)
		# print("oyaw", oyaw)
		# print("ov", ov)
		# print("state_predict", state_predict)

		p = np.zeros_like(ref_path)
		p[0, :] = ox
		p[1, :] = oy
		p[2, :] = ov
		p[3, :] = oyaw
		# self.publishReferenceTrajectory(p, self.cal_traj_pub)

		steer_output = self.odelta_v[0]
		speed_output = self.current_state.v + self.oa[0] * self.dt

		cmd = AckermannDriveStamped()
		cmd.header.stamp = self.get_clock().now().to_msg()
		cmd.header.frame_id = "map"
		cmd.drive.speed = speed_output
		cmd.drive.steering_angle = steer_output
		print(f'steering = {steer_output}, speed = {speed_output}')

		# self.cmd_pub.publish(cmd)

	def mpc_init(self):
		'''
		Create MPC quadratic optimization problem using cp, solver: OSQP
		'''

		# Initialize and create vectors for the optimization problem
		self.xk = CP.Variable((self.NX, self.time_step_horizon + 1),name='x')
		'''car state vectors'''
		self.uk = CP.Variable((self.NU, self.time_step_horizon),name='u')
		'''control input vectors'''
		# Objective function
		self.objective = 0.0
		'''objective function'''
		# Constraints
		self.constraints = []
		'''constraints list'''
		# Initialise reference
		self.xk0 = CP.Parameter((self.NX,),name='x0')
		self.xk0.value = np.zeros((self.NX,))

		self.reference_trajectory = CP.Parameter((self.NX, self.time_step_horizon + 1),name='xref')
		self.reference_trajectory.value = np.zeros((self.NX, self.time_step_horizon + 1))

		# cost function weights
		self.R_block = block_diag(tuple([self.R] * self.time_step_horizon))
		self.Rd_block = block_diag(tuple([self.Rd] * (self.time_step_horizon-1)))
		self.Q_block = block_diag(tuple([self.Q] * self.time_step_horizon + [self.Qf]))
		# print('-----------------------------------------')
		# print("R")
		# pprint((self.R_block.toarray()), use_unicode=True, wrap_line=False, full_prec=False)
		# print('-----------------------------------------')
		# print("Rd")
		# pprint((self.Rd_block.toarray()), use_unicode=True, wrap_line=False, full_prec=False)
		# print('-----------------------------------------')
		# print("Q")
		# pprint((self.Q_block.toarray()), use_unicode=True, wrap_line=False, full_prec=False)
		# print('-----------------------------------------')
		
		# Objective function
		self.objective += CP.quad_form(CP.vec(self.reference_trajectory - self.xk), (self.Q_block))
		self.objective += CP.quad_form(CP.vec(self.uk), (self.R_block))
		self.objective += CP.quad_form(CP.vec(CP.diff(self.uk, axis=1)), (self.Rd_block))

		# Constraints
		A_block  = []
		B_block = []
		C_block = []

		predicted_path = np.zeros((self.NX, self.time_step_horizon + 1))
		for k in range(self.time_step_horizon):
			A, B, C = self.get_model_matrix(predicted_path[2, k], predicted_path[3, k], 0.0)
			A_block.append(A)
			B_block.append(B)
			C_block.append(C)
		A_block = block_diag(tuple(A_block))
		B_block = block_diag(tuple(B_block))
		C_block = np.array(tuple(C_block)).reshape(-1,)

		m, n = A_block.shape
		self.Annz_k = CP.Parameter(A_block.nnz, name='A')
		data = np.ones(self.Annz_k.size)
		rows = A_block.row * n + A_block.col
		cols = np.arange(self.Annz_k.size)
		Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))
		self.Annz_k.value = A_block.data
		self.Ak_ = CP.reshape(Indexer @ self.Annz_k, (m, n), order="C")
		print('-----------------------------------------')
		print("A")
		pprint((self.Ak_.value).astype(int), wrap_line=False)

		m, n = B_block.shape
		self.Bnnz_k = CP.Parameter(B_block.nnz, name='B')
		data = np.ones(self.Bnnz_k.size)
		rows = B_block.row * n + B_block.col
		cols = np.arange(self.Bnnz_k.size)
		Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
		self.Bk_ = CP.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
		self.Bnnz_k.value = B_block.data
		print('-----------------------------------------')
		print("B")
		pprint((self.Bk_.value).astype(int), wrap_line=False)

		self.Ck_ = CP.Parameter(C_block.shape, name='C')
		self.Ck_.value = C_block
		print('-----------------------------------------')
		print("C")
		pprint((self.Ck_.value).astype(int), wrap_line=False)
		print('-----------------------------------------')

		self.constraints += [CP.vec(self.xk[:, 1:]) == self.Ak_ @ CP.vec(self.xk[:, :-1]) + self.Bk_ @ CP.vec(self.uk) + (self.Ck_)]
		self.constraints += [self.xk[:, 0] == self.xk0]
		self.constraints += [CP.abs(CP.diff(self.uk[1, :]))<= self.MAX_DSTEER * self.dt]
		self.constraints += [self.xk[2, :] <= self.MAX_SPEED]
		self.constraints += [self.xk[2, :] >= self.MIN_SPEED]
		self.constraints += [CP.abs(self.uk[0, :]) <= self.MAX_ACCEL]
		self.constraints += [CP.abs(self.uk[1, :]) <= self.MAX_STEER]
		# self.constraints += [self.uk[0, :] <= 0.4]
		# self.constraints += [self.uk[0, :] >= 0.4]

		self.MPC_prob = CP.Problem(CP.Minimize(self.objective), self.constraints)
		print(self.MPC_prob)

	def calc_ref_trajectory(self, state: State, waypoints):
		"""
		calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
		using the current velocity, calc the T points along the reference path
		:param cx: Course X-Position
		:param cy: Course y-Position
		:param cyaw: Course Heading
		:param sp: speed profile
		:dl: distance step
		:pind: Setpoint Index
		:return: reference trajectory ref_traj, reference steering angle
		"""
		cx = waypoints[:, 0]
		cy = waypoints[:, 1]
		cyaw = waypoints[:, 4]
		sp = waypoints[:, 7]
		# Create placeholder Arrays for the reference trajectory for T steps
		ref_traj = np.zeros((self.NX, self.time_step_horizon + 1))
		ncourse = len(cx)

		# Find nearest index/setpoint from where the trajectories are calculated
		_, _, _, ind = self.nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)
		# ind,_ = self.getClosestPoint()

		# Load the initial parameters from the setpoint into the trajectory
		ref_traj[0, 0] = cx[ind]
		ref_traj[1, 0] = cy[ind]
		ref_traj[2, 0] = sp[ind]
		ref_traj[3, 0] = cyaw[ind]

		# based on current velocity, distance traveled on the ref line between time steps
		travel = abs(state.v) * self.dt
		# dind = np.max([travel / self.waypointStepSize, 1/self.time_step_horizon])
		# dind = np.max([travel / self.waypointStepSize, 2])
		dind =2

		ind_list = int(ind) + np.insert(
			np.cumsum(np.repeat(dind, self.time_step_horizon)), 0, 0
		).astype(int)
		ind_list[ind_list >= ncourse] -= ncourse
		ref_traj[0, :] = cx[ind_list]
		ref_traj[1, :] = cy[ind_list]
		ref_traj[2, :] = sp[ind_list]
		cyaw[cyaw - state.yaw > 4.5] = np.abs(
			cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
		)
		cyaw[cyaw - state.yaw < -4.5] = np.abs(
			cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
		)
		ref_traj[3, :] = cyaw[ind_list]
		ref_traj[3, :] = self.normalize_angle(cyaw[ind_list])
		# print(ref_traj[3, :])
		return ref_traj
	
	def mpc_control(self, ref_path, x0, oa, od):
		"""
		MPC control with updating operational point iteraitvely
		:param ref_path: reference trajectory in T steps
		:param x0: initial state vector
		:param oa: acceleration of T steps of last time
		:param od: delta of T steps of last time
		"""
		if oa is None or od is None:
			oa = [0.0] * self.time_step_horizon
			od = [0.0] * self.time_step_horizon

		# Call the Motion Prediction function: Predict the vehicle motion for x-steps
		path_predict = self.predict_motion(x0, oa, od, ref_path)
		self.publishReferenceTrajectory(path_predict, self.cal_traj_pub)
		poa, pod = oa[:], od[:]

		# Run the MPC optimization: Create and solve the optimization problem
		mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
			ref_path, path_predict, x0
		)

		return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

	def predict_motion(self, x0, oa, od, xref):
		path_predict = xref * 0.0
		for i, _ in enumerate(x0):
			path_predict[i, 0] = x0[i]

		state = State()
		state.x = x0[0]
		state.y = x0[1]
		state.v = x0[2]
		state.yaw = x0[3]

		for (ai, di, i) in zip(oa, od, range(1, self.time_step_horizon + 1)):
			state = self.update_state(state, ai, di)
			path_predict[0, i] = state.x
			path_predict[1, i] = state.y
			path_predict[2, i] = state.v
			path_predict[3, i] = state.yaw

		return path_predict

	def update_state(self, state: State, a, delta):
		# input check
		if delta >= self.MAX_STEER:
			delta = self.MAX_STEER
		elif delta <= -self.MAX_STEER:
			delta = -self.MAX_STEER

		state.x = state.x + state.v * math.cos(state.yaw) * self.dt
		state.y = state.y + state.v * math.sin(state.yaw) * self.dt
		state.yaw = (
			state.yaw + (state.v / self.WB) * math.tan(delta) * self.dt
		)
		state.v = state.v + a * self.dt

		if state.v > self.MAX_SPEED:
			state.v = self.MAX_SPEED
		elif state.v < self.MIN_SPEED:
			state.v = self.MIN_SPEED

		return state

	def mpc_prob_solve(self, ref_traj, path_predict, x0):
		self.xk0.value = x0

		A_block = []
		B_block = []
		C_block = []
		for t in range(self.time_step_horizon):
			A, B, C = self.get_model_matrix(
				path_predict[2, t], path_predict[3, t], 0.0
			)
			A_block.append(A)
			B_block.append(B)
			C_block.extend(C)

		A_block = block_diag(tuple(A_block))
		B_block = block_diag(tuple(B_block))
		C_block = np.array(C_block).reshape(-1,)

		self.Annz_k.value = A_block.data
		self.Bnnz_k.value = B_block.data
		self.Ck_.value = C_block

		self.reference_trajectory.value = ref_traj

		# Solve the optimization problem in CVXPY
		# Solver selections: cvxpy.OSQP; cvxpy.GUROBI
		self.MPC_prob.solve(solver=CP.OSQP, verbose=False, warm_start=True)

		if (
			self.MPC_prob.status == CP.OPTIMAL
			or self.MPC_prob.status == CP.OPTIMAL_INACCURATE
		):
			ox = np.array(self.xk.value[0, :]).flatten()
			oy = np.array(self.xk.value[1, :]).flatten()
			ov = np.array(self.xk.value[2, :]).flatten()
			oyaw = np.array(self.xk.value[3, :]).flatten()
			oa = np.array(self.uk.value[0, :]).flatten()
			odelta = np.array(self.uk.value[1, :]).flatten()

		else:
			print("Error: Cannot solve mpc..")
			oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

		return oa, odelta, ox, oy, oyaw, ov





		
		


	def get_model_matrix(self, v, phi, delta):
		"""
		Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
		Linear System: Xdot = Ax +Bu + C
		State vector: x=[x, y, v, yaw]
		:param v: speed
		:param phi: heading angle of the vehicle
		:param delta: steering angle: delta_bar
		:return: A, B, C
		"""

		# State (or system) matrix A, 4x4
		A = np.zeros((self.NX, self.NX))
		A[0, 0] = 1.0
		A[1, 1] = 1.0
		A[2, 2] = 1.0
		A[3, 3] = 1.0
		A[0, 2] = self.dt * math.cos(phi)
		A[0, 3] = -self.dt * v * math.sin(phi)
		A[1, 2] = self.dt * math.sin(phi)
		A[1, 3] = self.dt * v * math.cos(phi)
		A[3, 2] = self.dt * math.tan(delta) / self.WB

		# Input Matrix B; 4x2
		B = np.zeros((self.NX, self.NU))
		B[2, 0] = self.dt
		B[3, 1] = self.dt * v / (self.WB * math.cos(delta) ** 2)

		C = np.zeros(self.NX)
		C[0] = self.dt * v * math.sin(phi) * phi
		C[1] = -self.dt * v * math.cos(phi) * phi
		C[3] = -self.dt * v * delta / (self.WB * math.cos(delta) ** 2)

		return A, B, C

	# @njit(cache=True)
	def nearest_point(self, point, trajectory):
		"""
		Return the nearest point along the given piecewise linear trajectory.
		Args:
			point (numpy.ndarray, (2, )): (x, y) of current pose
			trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
				NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
		Returns:
			nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
			nearest_dist (float): distance to the nearest point
			t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
			i (int): index of nearest point in the array of trajectory waypoints
		"""
		diffs = trajectory[1:,:] - trajectory[:-1,:]
		l2s   = diffs[:,0]**2 + diffs[:,1]**2
		dots = np.empty((trajectory.shape[0]-1, ))
		for i in range(dots.shape[0]):
			dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
		t = dots / l2s
		t[t<0.0] = 0.0
		t[t>1.0] = 1.0
		projections = trajectory[:-1,:] + (t*diffs.T).T
		dists = np.empty((projections.shape[0],))
		for i in range(dists.shape[0]):
			temp = point - projections[i]
			dists[i] = np.sqrt(np.sum(temp*temp))
		min_dist_segment = np.argmin(dists)
		return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

		




	



		



			






	
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
		angle[angle >= 2*np.pi] -= 2*np.pi
		angle[angle < 0] += 2*np.pi
		# if angle >= 2*np.pi:
		# 	angle = angle - 2*np.pi
		# if angle < 0:
		# 	angle = angle + 2*np.pi		
		# if angle < 0 or angle >= 2*np.pi:
		# 	self.normalize_angle(angle)
		return angle
	
	def getClosestPoint(self):
		'''
		return closest point on the path and index

		:return:
		index, closest point
		'''
		distance = np.linalg.norm(self.waypoints[:, :2] - self.pose[:2], axis=1)
		closest_index = np.argmin(distance)
		return closest_index , self.waypoints[closest_index]
	
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

	def publishReferenceTrajectory(self, referenceTrajectory, publisher):
		pose_array = PoseArray()
		pose_array.header.frame_id = "map"
		pose_array.header.stamp = self.get_clock().now().to_msg()
		pose_array.poses = []
		for i in range(referenceTrajectory.shape[1]):
			pose = Pose()
			pose.position.x = referenceTrajectory[0,i]
			pose.position.y = referenceTrajectory[1,i]
			qw,qx,qy,qz = self.quaternion_from_euler(0,0,referenceTrajectory[3,i])
			pose.orientation.x = qx
			pose.orientation.y = qy
			pose.orientation.z = qz
			pose.orientation.w = qw
			pose_array.poses.append(pose)
		publisher.publish(pose_array)
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()