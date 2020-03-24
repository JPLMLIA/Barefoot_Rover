#!/usr/bin/env python
import rospy
import time
from barefoot_msgs.msg import Cartduino
from cartduino import RigArduino


if __name__ == "__main__":

	device 	= rospy.get_param("/cartduino/device")
	baud 	= rospy.get_param("/cartduino/baud")
	timeout = rospy.get_param("/cartduino/timeout")
	delay 	= rospy.get_param("/cartduino/delay")
	data_rate = rospy.get_param("/cartduino/rate")

	rigarduino = RigArduino(device, baud, timeout, delay)
	#/dev/ttyS5

	rospy.init_node("cartduino")
	rospy.loginfo("Starting cartduino node")

	pub = rospy.Publisher("/cartduino_topic", Cartduino, queue_size=1)

	data = Cartduino()

	data.cartduino_ms 	= 0.0
	data.z_height 		= 0.0
	data.speed 			= 0
	data.current 		= 0
	data.cartduino_imu 	= [0.0,0.0,0.0]
	data.pxflow 		= [0,0]

	rate = rospy.Rate(data_rate)


	while not rospy.is_shutdown():

		rigarduino.read()

		data.cartduino_ms 	= rigarduino.get_timestamp()
		data.z_height 		= rigarduino.get_height()
		data.speed 			= rigarduino.get_speed()
		data.current 		= rigarduino.get_current()
		#data.cartduino_imu = rigarduino.get_imu()
		#data.pxflow 		= rigarduino.get_pxflow()
		#print (data)
		pub.publish(data)
		rate.sleep()
