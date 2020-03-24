#!/usr/bin/env python

import argparse
import rospy
from barefoot_rover_ros.msg import XtsScan

verbose=False;
ros_topic='XtsBus'

def callback(scanData):

    global verbose
    global ros_topic

    print('%d (%d,%d)' % 
          (scanData.frameNumber, scanData.yCoils, scanData.xCoils))

    # Un-comment the below for debugging, but it likely will result in lost messages since
    # ROS queues are limited in size...
    if verbose:
        for i,coil in enumerate(scanData.frameData):
            print ('%d : %s' % (i, ', '.join(str(v) for v in coil.xCoil)))


def listener():
    rospy.init_node('listener', anonymous=True)     
    rospy.Subscriber(ros_topic, XtsScan, callback)

    print('Listening for <XtsScan> messages on topic <%s>' % ros_topic)

    rospy.spin()


def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description='Example app which listens on the ROS message bus for XTS data messages.')

    parser.add_argument('-v', '--verbose', required=False, action='store_true', help='Display received messages, this may cause lost data and is intended for debugging purposes only.')
    parser.add_argument('ros_topic', nargs='?', help='The name of the ros topic which to subscribe.', default='XtsBus')

    return vars(parser.parse_args())

if __name__ == '__main__':

    args=parse_args()
    verbose=args['verbose']
    ros_topic=args['ros_topic']

    listener()
