#!/usr/bin/env python
import time
import rospy
from barefoot_msgs.msg import Wheelduino, Cartduino



class DataLogger(object):
	def __init__(self):
		self.wheel_data = Wheelduino()
		self.cart_data  = Cartduino()


	def cart_callback(self,data):
		self.wheel_data = data
		return 0

	def wheel_callback(self,data):
		print data
		return 0




if __name__ == '__main__':
	rospy.init_node("data_logger")
	rospy.loginfo("Starting the data_logger nodes")


	datalogger = DataLogger()

	cart_sub  = rospy.Subscriber("/cartduino_topic", Cartduino, datalogger.cart_callback)
	wheel_sub = rospy.Subscriber("/wheelduino_topic", Wheelduino, datalogger.wheel_callback)

	rate = rospy.Rate(50)

	rospy.spin()
