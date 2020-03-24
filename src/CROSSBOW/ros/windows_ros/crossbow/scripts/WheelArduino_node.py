#!/usr/bin/env python
import rospy
import time
from barefoot_msgs.msg import Wheelduino
from WheelArduino import WheelArduino_obj


if __name__ == "__main__":
	device 	= rospy.get_param("/wheelduino/device")
	baud 	= rospy.get_param("/wheelduino/baud")
	timeout = rospy.get_param("/wheelduino/timeout")
	delay 	= rospy.get_param("/wheelduino/delay")
	data_rate = rospy.get_param("/wheelduino/rate")

	rospy.init_node("Wheelduino")
	rospy.loginfo("Starting wheelduino node")

	pub = rospy.Publisher("/wheelduino_topic", Wheelduino, queue_size=1)

	data = Wheelduino()
	wheelduino = WheelArduino_obj(device, baud, timeout, delay)


	data.wheelduino_ms 		= 0.0
	data.wheelduino_imu 	= [0.0,0.0,0.0]
	data.r 					= 0.0
	data.theta 				= 0.0
	data.phi 				= 0.0
	data.rot 				= 0.0



	rate = rospy.Rate(data_rate)


	while not rospy.is_shutdown():
		wheelduino.read()

		data.wheelduino_ms	=wheelduino.get_time()
		data.wheelduino_imu =wheelduino.imu_calculations()
		data.r 				=wheelduino.get_r()
		data.theta 			=wheelduino.get_theta()
		data.phi 			=wheelduino.get_phi()
		data.rot 			=wheelduino.get_rot()
		#print (data)
		pub.publish(data)
		rate.sleep()