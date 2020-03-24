#!/usr/bin/env python
import rospy
from barefoot_msgs.msg import Cartduino
def callback(data):
  rospy.loginfo(rospy.get_caller_id() + data.z_height)
    
def listener():
  rospy.init_node('listener', anonymous=True)

  rospy.Subscriber("chatter", Cartduino, callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()

if __name__ == '__main__':
  listener()