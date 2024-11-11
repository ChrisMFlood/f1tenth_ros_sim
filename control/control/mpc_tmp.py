#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.publisher import Publisher
import numpy as np
from visualization_msgs.msg import MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
import cvxpy
# from scipy.linalg import block_diag
from sympy import pprint
import math
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag
from scipy.sparse import csc_matrix
from dataclasses import dataclass, field

@dataclass
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
		super().__init__("MPC_Node")  
		self.get_logger().info('MPC Node Started')
		# Parameters
		self.declare_parameter("odom_topic","/ego_racecar/odom")
		self.odom_topic = self.get_parameter("odom_topic").value

		self.declare_parameter("map", 'esp')
		self.declare_parameter("dt", 0.1)
		self.declare_parameter("time_step_horizon", 5)
		self.declare_parameter("wheel_base", 0.33)
		self.declare_parameter("MAX_SPEED", 9.0)
		self.declare_parameter("MIN_SPEED", 0.0)
		self.declare_parameter("MAX_ACCEL", 10)
		self.declare_parameter("MAX_STEER", 0.41)
		self.declare_parameter("MIN_STEER", -0.41)
		self.declare_parameter("MAX_DSTEER", np.pi)

		
		self.map_name = self.get_parameter("map").value
		self.dt = self.get_parameter("dt").value
		self.step_horizon = self.get_parameter("time_step_horizon").value
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

		# Publishers
		self.waypoints_pub = self.create_publisher(PoseArray, 'waypoints', 10)
		self.ref_traj_pub = self.create_publisher(PoseArray, 'reference_trajectory', 10)
		self.calc_ref_traj_pub = self.create_publisher(PoseArray, 'calculated_trajectory', 10)
		self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
		# Subscribers
		self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)

		# Load waypoints
		self.waypoints = np.loadtxt(f'src/global_planning/maps/{self.map_name}_short_minCurve.csv', delimiter=',', skiprows=1)
		'''[x_m,y_m,w_tr_right_m,w_tr_left_m,psi,kappa,s,velocity,acceleration,time]'''
		self.waypointStepSize = np.mean(self.waypoints[1:,6] - self.waypoints[:-1,6])
		self.publishTrajectory(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,4], self.waypoints_pub)

		# Variables
		self.state = State()
		self.oa = None
		self.odelta_v = None

		self.Rk = np.diag([0.01, 100])  # input cost matrix
		self.Rdk = np.diag([0.01, 0.01])  # input difference cost matrix
		self.Qk = np.diag([13.5, 13.5, 13.5, 13.0])  # state cost matrix
		self.Qfk = np.diag([13.5, 13.5, 7.5, 13.0])  # state final cost matrix

		# Init MPC
		self.init_mpc()

	def odom_callback(self, msg: Odometry):
		self.odom = msg
		self.state.x = msg.pose.pose.position.x
		self.state.y = msg.pose.pose.position.y
		self.state.v = msg.twist.twist.linear.x
		self.state.yaw = self.euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
		self.state.yaw_rate = msg.twist.twist.angular.z

		self.ref_traj = self.calc_ref_trajectory(self.state, self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,4], self.waypoints[:,7])
		# self.publishTrajectory(self.ref_traj[0,:], self.ref_traj[1,:], self.ref_traj[3,:], self.ref_traj_pub)

		x0 = np.array([self.state.x, self.state.y, self.state.v, self.state.yaw])
		# Solve the Linear MPC Control problem
		self.oa,self.odelta_v,ox,oy,oyaw,ov,state_predict = self.linear_mpc_control_kinematic(self.ref_traj, x0, self.oa, self.odelta_v)
		# self.publishTrajectory(ox,oy,oyaw,self.calc_ref_traj_pub)


		cmd = AckermannDriveStamped()
		cmd.header.stamp = self.get_clock().now().to_msg()
		cmd.header.frame_id = 'map'
		cmd.drive.steering_angle = self.odelta_v[0]
		cmd.drive.speed = self.oa[0]*self.dt + self.state.v
		self.cmd_pub.publish(cmd)

	def linear_mpc_control_kinematic(self, ref_path, x0, oa, od):
		"""
		MPC contorl with updating operational point iteraitvely
		:param ref_path: reference trajectory in T steps
		:param x0: initial state vector
		:param a_old: acceleration of T steps of last time
		:param delta_old: delta of T steps of last time
		:return: acceleration and delta strategy based on current information
		"""

		if oa is None or od is None:
			oa = [0.0] * self.step_horizon
			od = [0.0] * self.step_horizon

		# Call the Motion Prediction function: Predict the vehicle motion for x-steps
		path_predict = self.predict_motion_kinematic(x0, oa, od, ref_path)
		poa, pod = oa[:], od[:]

		# Run the MPC optimization: Create and solve the optimization problem
		mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve_kinematic(
			ref_path, path_predict, x0
		)

		return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
	
	def mpc_prob_solve_kinematic(self, ref_traj, path_predict, x0):
		self.x0k.value = x0

		A_block = []
		B_block = []
		C_block = []
		for t in range(self.step_horizon):
			A, B, C = self.get_kinematic_model_matrix(
				path_predict[2, t], path_predict[3, t], 0.0
			)
			A_block.append(A)
			B_block.append(B)
			C_block.extend(C)

		A_block = block_diag(tuple(A_block))
		B_block = block_diag(tuple(B_block))
		C_block = np.array(C_block)

		self.Annz_k.value = A_block.data
		self.Bnnz_k.value = B_block.data
		self.Ck_.value = C_block

		self.ref_traj_k.value = ref_traj

		# Solve the optimization problem in CVXPY
		# Solver selections: cvxpy.OSQP; cvxpy.GUROBI
		self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

		if (
			self.MPC_prob.status == cvxpy.OPTIMAL
			or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
		):
			ox = self.get_nparray_from_matrix(self.xk.value[0, :])
			oy = self.get_nparray_from_matrix(self.xk.value[1, :])
			ov = self.get_nparray_from_matrix(self.xk.value[2, :])
			oyaw = self.get_nparray_from_matrix(self.xk.value[3, :])
			oa = self.get_nparray_from_matrix(self.uk.value[0, :])
			odelta = self.get_nparray_from_matrix(self.uk.value[1, :])

		else:
			print("Error: Cannot solve mpc..")
			oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

		return oa, odelta, ox, oy, oyaw, ov
	
	def get_nparray_from_matrix(self, x):
		return np.array(x).flatten()
	
	def predict_motion_kinematic(self, x0, oa, od, xref):
		path_predict = xref * 0.0
		for i, _ in enumerate(x0):
			path_predict[i, 0] = x0[i]

		state = State()
		state.x = x0[0]
		state.y = x0[1]
		state.v = x0[2]
		state.yaw = x0[3]
		
		for (ai, di, i) in zip(oa, od, range(1, self.step_horizon + 1)):
			state: State = self.update_state_kinematic(state, ai, di)
			path_predict[0, i] = state.x
			path_predict[1, i] = state.y
			path_predict[2, i] = state.v
			path_predict[3, i] = state.yaw

		return path_predict
	
	def update_state_kinematic(self, state: State, a, delta):

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
		

	def init_mpc(self):
		"""
		Create MPC quadratic optimization problem using cvxpy, solver: OSQP
		Will be solved every iteration for control.
		More MPC problem information here: https://osqp.org/docs/examples/mpc.html

		xref: reference trajectory (desired trajectory: [x, y, v, yaw])
		path_predict: predicted states in T steps
		x0: initial state
		dref: reference steer angle
		:return: optimal acceleration and steering strateg
		"""
		# Initialize and create vectors for the optimization problem
		self.xk = cvxpy.Variable(
			(self.NX, self.step_horizon + 1)
		)  # Vehicle State Vector
		self.uk = cvxpy.Variable(
			(self.NU, self.step_horizon)
		)  # Control Input vector
		objective = 0.0  # Objective value of the optimization problem, set to zero
		constraints = []  # Create constraints array

		# Initialize reference vectors
		self.x0k = cvxpy.Parameter((self.NX,))
		self.x0k.value = np.zeros((self.NX,))

		# Initialize reference trajectory parameter
		self.ref_traj_k = cvxpy.Parameter((self.NX, self.step_horizon + 1))
		self.ref_traj_k.value = np.zeros((self.NX, self.step_horizon + 1))

		# Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
		R_block = block_diag(tuple([self.Rk] * self.step_horizon))

		# Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
		Rd_block = block_diag(tuple([self.Rdk] * (self.step_horizon - 1)))

		# Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
		Q_block = [self.Qk] * (self.step_horizon)
		Q_block.append(self.Qfk)
		Q_block = block_diag(tuple(Q_block))

		# Formulate and create the finite-horizon optimal control problem (objective function)
		# The FTOCP has the horizon of T timesteps

		# Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
		objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

		# Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
		objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

		# Objective 3: Difference from one control input to the next control input weighted by Rd
		objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

		# Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
		# Evaluate vehicle Dynamics for next T timesteps
		A_block = []
		B_block = []
		C_block = []
		# init path to zeros
		path_predict = np.zeros((self.NX, self.step_horizon + 1))
		for t in range(self.step_horizon):
			A, B, C = self.get_kinematic_model_matrix(
				path_predict[2, t], path_predict[3, t], 0.0
			)
			A_block.append(A)
			B_block.append(B)
			C_block.extend(C)

		A_block = block_diag(tuple(A_block))
		B_block = block_diag(tuple(B_block))
		C_block = np.array(C_block)
		print(A_block.shape, B_block.shape, C_block.shape)

		# [AA] Sparse matrix to CVX parameter for proper stuffing
		# Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
		m, n = A_block.shape
		self.Annz_k = cvxpy.Parameter(A_block.nnz)
		data = np.ones(self.Annz_k.size)
		rows = A_block.row * n + A_block.col
		cols = np.arange(self.Annz_k.size)
		Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

		# Setting sparse matrix data
		self.Annz_k.value = A_block.data

		# Now we use this sparse version instead of the old A_ block matrix
		self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

		# Same as A
		m, n = B_block.shape
		self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
		data = np.ones(self.Bnnz_k.size)
		rows = B_block.row * n + B_block.col
		cols = np.arange(self.Bnnz_k.size)
		Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
		self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
		self.Bnnz_k.value = B_block.data

		# No need for sparse matrices for C as most values are parameters
		self.Ck_ = cvxpy.Parameter(C_block.shape)
		self.Ck_.value = C_block

		print(self.Ak_.shape, self.Bk_.shape, self.Ck_.shape)

		# Add dynamics constraints to the optimization problem
		constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + (self.Ck_)]
		constraints += [cvxpy.abs(cvxpy.diff(self.uk[1, :])) <= self.MAX_DSTEER * self.dt]

		# Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
		constraints += [self.xk[:, 0] == self.x0k]
		constraints += [self.xk[2, :] <= self.MAX_SPEED]
		constraints += [self.xk[2, :] >= self.MIN_SPEED]
		constraints += [cvxpy.abs(self.uk[0, :]) <= self.MAX_ACCEL]
		constraints += [cvxpy.abs(self.uk[1, :]) <= self.MAX_STEER]

		# Create the optimization problem in CVXPY and setup the workspace
		# Optimization goal: minimize the objective function
		self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
		print(self.MPC_prob.objective)

	def get_kinematic_model_matrix(self, v, phi, delta):
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

	def calc_ref_trajectory(self, state: State, cx, cy, cyaw, sp):
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

		# Create placeholder Arrays for the reference trajectory for T steps
		ref_traj = np.zeros((self.NX, self.step_horizon + 1))
		ncourse = len(cx)

		# Find nearest index/setpoint from where the trajectories are calculated
		_, _, _, ind = self.nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

		# Load the initial parameters from the setpoint into the trajectory
		ref_traj[0, 0] = cx[ind]
		ref_traj[1, 0] = cy[ind]
		ref_traj[2, 0] = sp[ind]
		ref_traj[3, 0] = cyaw[ind]

		# based on current velocity, distance traveled on the ref line between time steps
		travel = abs(state.v) * self.dt
		# dind = np.max([travel / self.waypointStepSize,1])
		dind = int(travel / self.waypointStepSize)
		# dind = 2
		ind_list = int(ind) + np.insert(
			np.cumsum(np.repeat(dind, self.step_horizon)), 0, 0
		).astype(int)
		ind_list[ind_list >= ncourse] -= ncourse
		ref_traj[0, :] = cx[ind_list]
		ref_traj[1, :] = cy[ind_list]
		ref_traj[2, :] = sp[ind_list]
		
		# cyaw[cyaw - state.yaw > 4.5] = np.abs(
		# 	cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
		# )
		# cyaw[cyaw - state.yaw < -4.5] = np.abs(
		# 	cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
		# )
		angle_thres = 4.5
		for i in ind_list:
			# self.get_logger().info(f'cyaw: {cyaw[i]}- syaw: {state.yaw} = {self.angle_diff(cyaw[i], state.yaw)}, i = {i}, {cyaw[i] - state.yaw}')
			if (cyaw[i] - state.yaw) > angle_thres:
				cyaw[i] -= 2*np.pi
				# print(cyaw[i] - state.yaw)
			if (state.yaw - cyaw[i]) > angle_thres:
				cyaw[i] += 2*np.pi
				# print(cyaw[i] - state.yaw)
		ref_traj[3, :] = cyaw[ind_list]

		return ref_traj
		
	# def angle_diff(self,a, b):
	# 	'''Computes the shortest distance between two angles'''
	# 	a = self.normalize(a)
	# 	b = self.normalize(b)
	# 	d1 = a-b
	# 	d2 = 2*np.pi - np.abs(d1)
	# 	if(d1 > 0):
	# 		d2 *= -1.0
	# 	if(np.abs(d1) < np.abs(d2)):
	# 		return d1
	# 	else:
	# 		return d2
		
	def angle_diff(self, a, b):
		'''Computes the shortest distance between two angles'''
		a = self.normalize(a)
		b = self.normalize(b)
		d1 = a - b
		d2 = 2 * np.pi - np.abs(d1)
		d2[d1 > 0] *= -1.0
		return np.where(np.abs(d1) < np.abs(d2), d1, d2)
		
	def normalize(self,z):
		'''Normalizes an angle to between [-pi, pi]'''
		# if -np.pi <= z <= np.pi:
		# 	return z
		return np.arctan2(np.sin(z), np.cos(z))
	
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

	def publishTrajectory(self, x,y,yaw, publisher):
		poseArray = PoseArray()
		poseArray.header.frame_id = 'map'
		poseArray.poses = []
		for i in range(x.shape[0]):
			pose = Pose()
			pose.position.x = x[i]
			pose.position.y = y[i]
			qw,qx,qy,qz = self.quaternion_from_euler(0,0,yaw[i])
			pose.orientation.w = qw
			pose.orientation.x = qx
			pose.orientation.y = qy
			pose.orientation.z = qz
			poseArray.poses.append(pose)
		publisher.publish(poseArray)

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
	
	def euler_from_quaternion(self,x, y, z, w):  
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
		return yaw_z # in radians
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()