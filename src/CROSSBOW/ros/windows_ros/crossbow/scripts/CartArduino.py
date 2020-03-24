#!/usr/bin/env python
import rospy
import time
from barefoot_msgs.msg import Cartduino
from Cart import RigArduino
from websocket import create_connection
import json

if __name__ == "__main__":

	device 	= rospy.get_param("/cartduino/device", "COM5")
	baud 	= rospy.get_param("/cartduino/baud", 115200)
	timeout = rospy.get_param("/cartduino/timeout", 1.5)
	delay 	= rospy.get_param("/cartduino/delay", 0.05)
	data_rate = rospy.get_param("/cartduino/rate", 10)

	rigarduino = RigArduino(device, baud, timeout, delay)
	#/dev/ttyS5

	rospy.init_node("cartduino")
	rospy.loginfo("Starting cartduino node")

	pub = rospy.Publisher("/cartduino", Cartduino, queue_size=1)

	try:
		ws = create_connection("ws://localhost:9090")
		print (ws)
	except Exception as e:
		print (e)


	data = Cartduino()

	data.cartduino_ms 	= 0.0
	data.z_height 		= 0.0
	data.speed 			= 0
	data.current 		= 0
	data.cart_x		 	= 0.0
	data.cart_x		 	= 0.0
	data.cart_x		 	= 0.0
	data.pxflow_1 		= 0
	data.pxflow_2		= 0

	rate = rospy.Rate(data_rate)
	
	try:
		print "sending init json"
		init_json = rigarduino.build_json_init("cartduino", "crossbow_msgs/Cartarduino")
		print init_json
		ws.send(json.dumps(init_json))
	except Exception as e:
		print e

	while not rospy.is_shutdown():

		rigarduino.read()

		data.cartduino_ms 	= rigarduino.get_timestamp()
		data.z_height 		= rigarduino.get_height()
		data.speed 			= rigarduino.get_speed()
		data.current 		= rigarduino.get_current()

		#TODO - FIX PXFLOW DATA TYPE ERROR

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#data.cartduino_imu = rigarduino.get_imu()
		#data.pxflow 		= rigarduino.get_pxflow()
		#print (data)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		try:
			ws.send(json.dumps(rigarduino.get_json_data()))
		except:
			pass
		pub.publish(data)
		rate.sleep()
