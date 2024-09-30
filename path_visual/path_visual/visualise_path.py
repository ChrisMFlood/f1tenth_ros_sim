import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import csv

class PathVisualizer(Node):
	def __init__(self):
		super().__init__('path_visualizer')
		self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
		self.centreLine = self.read_csv('/home/chris/sim_ws/src/global_planning/maps/esp_centreline.csv')
		self.raceline = self.read_csv('/home/chris/sim_ws/src/global_planning/maps/esp_short_minCurve.csv')

		# self.publish_path(self.centreLine)
		self.publish_path(self.raceline)
		# self.publisher.destroy()


	def read_csv(self, filename):
		track = []
		with open(filename, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			next(csvreader)  # Skip the header row
			for row in csvreader:
				if row[0].startswith('#'):
					continue  # Skip comment lines
				x, y = float(row[0]), float(row[1])
				track.append((x, y))
		return track

	def publish_path(self,track):
		path = MarkerArray()
		for i, (x, y) in enumerate(track):
			point = Marker()
			point.header.frame_id = "map"
			point.header.stamp = self.get_clock().now().to_msg()
			point.ns = "path"
			point.id = i
			point.type = Marker.SPHERE
			point.action = Marker.ADD
			point.pose.position.x = x
			point.pose.position.y = y
			point.pose.position.z = 0.0
			point.pose.orientation.x = 0.0
			point.pose.orientation.y = 0.0
			point.pose.orientation.z = 0.0
			point.pose.orientation.w = 1.0
			point.scale.x = 0.1
			point.scale.y = 0.1
			point.scale.z = 0.1
			point.color.a = 1.0
			point.color.r = 1.0
			point.color.g = 0.0
			point.color.b = 0.0
			path.markers.append(point)
		self.publisher.publish(path)

def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()