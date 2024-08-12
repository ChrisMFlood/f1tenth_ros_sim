import numpy as np
import time
from threading import Lock

# TF
# import tf.transformations
# import tf
import tf_transformations

# messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

def map_to_world(poses, map_info):
	''' Takes a two dimensional numpy array of poses:
			[[x0,y0,theta0],
			 [x1,y1,theta1],
			 [x2,y2,theta2],
				   ...     ]
		And converts them from map coordinate space (pixels) to world coordinate space (meters).
		- Conversion is done in place, so this function does not return anything.
		- Provide the MapMetaData object from a map message to specify the change in coordinates.
		- This implements the same computation as map_to_world_slow but vectorized and inlined
	'''

	scale = map_info.resolution
	orientation = Quaternion()
	orientation.x = map_info.origin.orientation.x
	orientation.y = map_info.origin.orientation.y
	orientation.z = map_info.origin.orientation.z
	orientation.w = map_info.origin.orientation.w
	angle = quaternion_to_angle(orientation)

	# rotation
	c, s = np.cos(angle), np.sin(angle)
	# we need to store the x coordinates since they will be overwritten
	temp = np.copy(poses[:,0])
	poses[:,0] = c*poses[:,0] - s*poses[:,1]
	poses[:,1] = s*temp       + c*poses[:,1]

	# scale
	poses[:,:2] *= float(scale)

	# translate
	poses[:,0] += map_info.origin.position.x
	poses[:,1] += map_info.origin.position.y
	poses[:,2] += angle
	
def quaternion_to_angle(q):
	"""Convert a quaternion _message_ into an angle in radians.
	The angle represents the yaw.
	This is not just the z component of the quaternion."""
	# x, y, z, w = q.x, q.y, q.z, q.w
	quat = [q.x, q.y, q.z, q.w]
	# roll, pitch, yaw = tf_transformations.euler_from_quaternion((x, y, z, w))
	roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
	return yaw

def particle_to_pose(particle):
	''' Converts a particle in the form [x, y, theta] into a Pose object '''
	pose = Pose()
	pose.position.x = float(particle[0])
	pose.position.y = float(particle[1])
	pose.orientation = angle_to_quaternion(particle[2])
	return pose

def particles_to_poses(particles):
	''' Converts a two dimensional array of particles into an array of Poses. 
		Particles can be a array like [[x0, y0, theta0], [x1, y1, theta1]...]
	'''
	return list(map(particle_to_pose, particles))

def angle_to_quaternion(angle):
	"""Convert an angle in radians into a quaternion _message_."""
	q = tf_transformations.quaternion_from_euler(0, 0, angle)
	q_out = Quaternion()
	q_out.x = q[0]
	q_out.y = q[1]
	q_out.z = q[2]
	q_out.w = q[3]
	return q_out