#!/usr/bin/env python

import rospy
import time
from barefoot_msgs.msg import ForceTorqueSensor
from ForceTorqueSensor import ATI_Task


if __name__ == "__main__":
	rospy.init_node("ForceTorqueSensor")
	rospy.loginfo("Starting ForceTorqueSensor node")

	pub = rospy.Publisher("/ForceTorqueSensor_topic", ForceTorqueSensor, queue_size=1)

	data = ForceTorqueSensor()
	ATI_Task = ATI_Task()

	data.ati_time 	= 0.0
	data.mdata 	= [0.0, 0.0, 0.0]
	data.xdata 	= [0.0, 0.0, 0.0]

	rate = rospy.Rate(10)

	while not rospy.is_shutdown():
		ForceTorqueSensor.read()

		data.ati_time 	= ATI_Task.get_time()
		data.mdata		= ATI_Task.get_mdata()
		data.xdata 		= ATI_Task.get_xdata()

		pub.publish(data)
		rate.sleep()





