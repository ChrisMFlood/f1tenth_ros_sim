#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cvxpy

from cvxpy.atoms.affine.wraps import psd_wrap
  
class myNode(Node):
	def __init__(self):
		super().__init__("MPC")  
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()