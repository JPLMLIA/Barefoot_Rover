import rospy
import time
from barefoot_msgs.msg import Ati
from ForceTorqueSensor import ATI_Task
from websocket import create_connection
import json


if __name__ == "__main__":
	rospy.init_node("ati")
	rospy.loginfo("Starting the ati node")

	pub = rospy.Publisher("/ati", Ati, queue_size=1)
	try:
		ws = create_connection("ws://localhost:9090")
		print (ws)
	except Exception as e:
		print (e)
	ati = Ati()
	ATI_Task = ATI_Task()

	ati.ati_time = 0.0
	ati.m_Fx = 0.0
	ati.m_Fy = 0.0
	ati.m_Fz = 0.0
	ati.m_Tx = 0.0
	ati.m_Ty = 0.0
	ati.m_Tz = 0.0

	ati.c_Fx = 0.0
	ati.c_Fy = 0.0
	ati.c_Fz = 0.0
	ati.c_Tx = 0.0
	ati.c_Ty = 0.0
	ati.c_Tz = 0.0

	rate = rospy.Rate(10)
	try:
		ws.send(json.dumps(ATI_Task.build_json_init("fts", "crossbow_msgs/Ati")))
	except Exception as e:
		print e

	while not rospy.is_shutdown():
		ATI_Task.read()

		ati.ati_time 	= ATI_Task.get_time()
		ati.m_Fx = ATI_Task.mdata[0]
		ati.m_Fy = ATI_Task.mdata[1]
		ati.m_Fz = ATI_Task.mdata[2]
		ati.m_Tx = ATI_Task.mdata[3]
		ati.m_Ty = ATI_Task.mdata[4]
		ati.m_Tz = ATI_Task.mdata[5]
		
		#rospy.loginfo(ati)

		pub.publish(ati)
		#print ATI_Task.json_data
		try:
			ws.send(json.dumps(ATI_Task.json_data))
		except:
			pass
		rate.sleep()





