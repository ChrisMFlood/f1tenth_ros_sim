import numpy as np
import math
import rclpy
from rclpy.node import Node, Publisher


from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseArray, Pose, Quaternion, PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped



def publishTrajectory(x,y,yaw, publisher: Publisher):
	if publisher.get_subscription_count() >= 1:
		poseArray = PoseArray()
		poseArray.header.frame_id = 'map'
		poseArray.poses = []
		for i in range(x.shape[0]):
			pose = Pose()
			pose.position.x = x[i]
			pose.position.y = y[i]
			qw,qx,qy,qz = quaternion_from_euler(0,0,yaw[i])
			pose.orientation.w = qw
			pose.orientation.x = qx
			pose.orientation.y = qy
			pose.orientation.z = qz
			poseArray.poses.append(pose)
		publisher.publish(poseArray)

def publishPoint(x,y,publisher: Publisher):
		marker = Marker()
		marker.header.frame_id = "map"
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.position.x = x
		marker.pose.position.y = y
		marker.pose.position.z = 0.0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.1
		marker.scale.y = 0.1
		marker.scale.z = 0.1
		marker.color.a = 1.0
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		publisher.publish(marker)

def publishPose(x,y,yaw, publisher: Publisher):
	pose = Pose()
	pose.position.x = x
	pose.position.y = y
	qw,qx,qy,qz = quaternion_from_euler(0,0,yaw)
	pose.orientation.w = qw
	pose.orientation.x = qx
	pose.orientation.y = qy
	pose.orientation.z = qz
	publisher.publish(pose)
	
def quaternion_from_euler(roll, pitch, yaw):
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

def euler_from_quaternion(x, y, z, w):  
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)
	return yaw_z # in radians

def setInitialPosition(position_publisher: Publisher, drive_publisher: Publisher):
	position = PoseWithCovarianceStamped()
	position.pose.pose.position.x = 0.0
	position.pose.pose.position.y = 0.0
	position.pose.pose.position.z = 0.0
	position.pose.pose.orientation.x = 0.0
	position.pose.pose.orientation.y = 0.0
	position.pose.pose.orientation.z = 0.0
	position.pose.pose.orientation.w = 1.0
	cmd = AckermannDriveStamped()
	cmd.drive.speed = 0.0
	cmd.drive.acceleration = 0.0
	cmd.drive.jerk = 0.0
	cmd.drive.steering_angle = 0.0
	cmd.drive.steering_angle_velocity = 0.0
	drive_publisher.publish(cmd)
	position_publisher.publish(position)
	print('Initial position set')

# def nearest_point(point, trajectory):
# 	"""
# 	Return the nearest point along the given piecewise linear trajectory.

# 	Args:
# 		point (numpy.ndarray, (2, )): (x, y) of current pose
# 		trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
# 			NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

# 	Returns:
# 		nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
# 		nearest_dist (float): distance to the nearest point
# 		t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
# 		i (int): index of nearest point in the array of trajectory waypoints
# 	"""
# 	diffs = trajectory[1:,:] - trajectory[:-1,:]
# 	l2s   = diffs[:,0]**2 + diffs[:,1]**2
# 	dots = np.empty((trajectory.shape[0]-1, ))
# 	for i in range(dots.shape[0]):
# 		dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
# 	t = dots / l2s
# 	t[t<0.0] = 0.0
# 	t[t>1.0] = 1.0
# 	projections = trajectory[:-1,:] + (t*diffs.T).T
# 	dists = np.empty((projections.shape[0],))
# 	for i in range(dists.shape[0]):
# 		temp = point - projections[i]
# 		dists[i] = np.sqrt(np.sum(temp*temp))
# 	min_dist_segment = np.argmin(dists)
# 	return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def nearest_point(point, trajectory):
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
    diffs = trajectory[1:] - trajectory[:-1]
    l2s = np.sum(diffs**2, axis=1)
    dots = np.einsum('ij,ij->i', point - trajectory[:-1], diffs)
    t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1] + t[:, np.newaxis] * diffs
    dists = np.linalg.norm(point - projections, axis=1)
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

MIN_STEERING_ANGLE = -0.4
MAX_STEERING_ANGLE = 0.4
# MAX_SPEED = 8
MIN_SPEED = 0

def pubishActuation(steering_angle, speed, cmd_publisher: Publisher, max_speed=8):
	steering_angle = np.clip(steering_angle, MIN_STEERING_ANGLE, MAX_STEERING_ANGLE)
	speed = np.clip(speed, MIN_SPEED, max_speed)

	cmd = AckermannDriveStamped()
	cmd.drive.steering_angle = steering_angle
	cmd.drive.speed = speed
	cmd_publisher.publish(cmd)

def normaliseAngle(angle):
	if angle > np.pi:
			angle = angle - 2*np.pi
	if angle < -np.pi:
			angle = angle + 2*np.pi
	return angle


# def getLapProgress(waypoints, pose, lapDistance, lapCount, current_time, speed, saveData, saveDataIndex, map_name):
# 	distance = np.linalg.norm(waypoints[:, :2] - pose[:2], axis=1)
# 	closest_index = np.argmin(distance)
# 	self.closestPointOnPath = self.waypoints[closest_index]
# 	self.distance = self.closestPointOnPath[6]
# 	self.lapProgress = self.distance/self.lapDistance*100
# 	LP = self.lapProgress +100*self.lapCount
# 	self.get_logger().info(f'Lap progress: {(self.lapProgress+self.lapCount*100):.2f}%')
# 	if (int(self.lapProgress) == 100) and (int(self.prevDistance) != 100):
# 		self.lapCount += 1
# 		self.get_logger().info(f'Lap {self.lapCount} completed')
# 	self.prevDistance = self.lapProgress
# 	if LP <= 200:
# 		temp = np.array([self.current_time, self.pose[0], self.pose[1], self.pose[2], self.speed, int(LP/100)])
# 		self.saveData[self.saveDataIndex] = temp
# 		self.saveDataIndex += 1
# 	if LP > 200:
# 		self.saveData = self.saveData[:self.saveDataIndex]
# 		dataPath = f'/home/chris/sim_ws/src/control/Results/Data/{self.map_name}_stanley.csv'
# 		with open(dataPath, 'wb') as fh:
# 			np.savetxt(fh, self.saveData, fmt='%0.16f', delimiter=',', header='time,x,y,yaw,speed,lap')
# 		self.get_logger().info('Data saved')
# 		rclpy.shutdown()
