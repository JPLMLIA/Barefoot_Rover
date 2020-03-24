#!/usr/bin/env python
import sys, os
import rospy
import time
from barefoot_msgs.msg import Labjack
from Ground_truth import StringPot, testConnection
from websocket import create_connection
import json



if __name__ == "__main__":
	rospy.init_node("LabJack")
	rospy.loginfo("Starting LabJack node")

	pub = rospy.Publisher("/Labjack", Labjack, queue_size=1)

	rate = rospy.Rate(10)

	try:
		ws = create_connection("ws://localhost:9090")
		print (ws)
	except Exception as e:
		print (e)

	data = Labjack()
	stringPot = StringPot()
	
	data.labjack_time = 0.0
	data.raw_voltage = 0.0
	data.distance_travelled = 0.0

	try:
		print "sending init json"
		print stringPot.build_json_init("labjack", "crossbow_msgs/Labjack")
		ws.send(json.dumps(stringPot.build_json_init("labjack", "crossbow_msgs/Labjack")))
	except Exception as e:
		print e

	while not rospy.is_shutdown():
		#Read and then Display
		stringPot.read()

		data.labjack_time = stringPot.get_time()
		data.raw_voltage = stringPot.get_raw_voltage()
		data.distance_travelled = stringPot.get_distance_travelled()

		pub.publish(data)

		#print stringPot.json_data
		try:
			ws.send(json.dumps(stringPot.json_data))
		except:
			pass
		rate.sleep()





