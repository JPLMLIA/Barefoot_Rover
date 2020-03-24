#!/usr/bin/env python

import argparse
import sys
from timeit import default_timer as timer

import rospy

from barefoot_rover_ros.msg import XtsScan, XtsXCoil

def testTalker(ros_topic, input_file_name):
    start = timer()
    pub = rospy.Publisher(ros_topic, XtsScan, queue_size=1000)
    rospy.init_node('testTalker', anonymous=True)

    rate = rospy.Rate(50) # 50hz to match the Xts API
    
    frameCount = 0
    msg = XtsScan()
    testInput = open(input_file_name, 'r')
    line = testInput.readline()
    while(line):
        pref = line[0:4]
        if pref == 'Capt' or pref == 'gain' or pref == 'numb' or pref == 'freq' or pref == 'time':
            pass
        else:
            pref = line[0:7]
            val = int(line.split('=')[1])

            if pref == 'frameno':
                msg.frameNumber = val
            elif pref == 'xcoilmi':
                xmin = val
            elif pref == 'xcoilma':
                msg.xCoils = val - xmin + 1
            elif pref == 'ycoilmi':
                ymin = val
            elif pref == 'ycoilma':
                msg.yCoils = val - ymin + 1
    
                # the subsequent lines are the data
                msg.frameData = processData(testInput, msg.yCoils)

                # now publish
                pub.publish(msg)
                frameCount += 1
                print('published frame %d' % msg.frameNumber)

        line = testInput.readline()
    end = timer()
    print('Published %d frames in %f seconds.' % (frameCount, end - start))
             
def processData(inputSource, yCoils):
    # skip the next three lines
    frameData =[]
    for i in range(3):
        inputSource.readline()
    for i in range(yCoils):
        line = inputSource.readline()
        coil = XtsXCoil()
        coil.xCoil = [float(v) for v in line.split('\t')[1:]]
        frameData.append(coil)

    return frameData

def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description='Example app which publishes XTS data messages from an input file on the ROS message bus.')

    parser.add_argument('input_file_name', help='The name of the input file containing xts records.')
    parser.add_argument('ros_topic', nargs='?', help='The name of the ros topic which to subscribe.', default='XtsBus')

    return vars(parser.parse_args())

if __name__ == '__main__':

    args=parse_args()
    input_file_name=args['input_file_name']
    ros_topic=args['ros_topic']

    try:
        testTalker(ros_topic, input_file_name)
    except rospy.ROSInterruptException:
        pass

