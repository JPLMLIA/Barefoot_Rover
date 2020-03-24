#!/usr/bin/env python
import rospy
import time
from barefoot_msgs.msg import Wheelduino
from Wheel import WheelArduino
from websocket import create_connection
import json

if __name__ == "__main__":
	device 	= rospy.get_param("/wheelduino/device", "COM4")
	baud 	= rospy.get_param("/wheelduino/baud", 115200)
	timeout = rospy.get_param("/wheelduino/timeout", 1.5)
	delay 	= rospy.get_param("/wheelduino/delay", 0.05)
	data_rate = rospy.get_param("/wheelduino/rate", 10)

	rospy.init_node("wheelduino")
	rospy.loginfo("Starting wheelduino node")

	pub = rospy.Publisher("/wheelduino", Wheelduino, queue_size=1)

	data = Wheelduino()
	wheelduino = WheelArduino(device, baud, timeout, delay)


	data.wheelduino_ms 		= 0.0
	data.wheel_x			= 0.0
	data.wheel_y			= 0.0
	data.wheel_z			= 0.0
	data.r 					= 0.0
	data.theta 				= 0.0
	data.phi 				= 0.0
	data.rho 				= 0.0

	try:
		ws = create_connection("ws://localhost:9090")
		print (ws)
	except Exception as e:
		print (e)

	rate = rospy.Rate(data_rate)

	try:
		print "sending init json"
		init_json = wheelduino.build_json_init("wheelduino", "crossbow_msgs/Wheelduino")
		print init_json
		ws.send(json.dumps(init_json))
	except Exception as e:
		print e

	while not rospy.is_shutdown():
		wheelduino.read()

		data.wheelduino_ms	=wheelduino.get_time()
		data.wheel_x			=wheelduino.imu[0]
		data.wheel_y			=wheelduino.imu[1]
		data.wheel_x			=wheelduino.imu[2]
		data.r 				=wheelduino.get_r()
		data.theta 			=wheelduino.get_theta()
		data.phi 			=wheelduino.get_phi()
		data.rho 			=wheelduino.get_rot()
		#print (data)
		pub.publish(data)
		try:
			ws.send(json.dumps(wheelduino.get_json_data()))
		except:
			pass
		rate.sleep()