#!/usr/bin/env python
from datetime import datetime
import rospy
import time
import json
from std_msgs.msg import String


class RunObject(object):
	def __init__(self):
		self.cmd_list = ['new', 'gullie', 'd', 'rock', 'bedrock', 'dune', 'pebble', 'start', 'bs', 'be', 'rs', 're', 'gs', 'ge', 'end', 'n', 'v', 'f']

	def parse_input(self, arg):
		cmd = arg.split(",")[0]
		vals = arg.split(",")[1:]
		if (cmd in self.cmd_list):
			pass
		elif (cmd == "h"):
			self.print_help()
			return 0
		else:
			print "Not a valid command... enter 'h' to see all option"
			return 0

		return 1

	def print_help(self):
		print('---------------------------------')
		print('    Crossbow test run commands   ')
		print('---------------------------------')
		print(' new : start a new test run      ')
		print(' gullie : Add gullies to run')
		print(' rock, <x,y,z..> : Add rock with locations x,y,z...')
		print(' bedrock, <x,y,z..> : Add bedrock with locations x,y,z...')
		print(' dune, <type> : Adds dunes of types sharp or smooth..')
		print(' pebble, <type> : Adds pebbles of types sparse or dense')
		print(' start : Starts recording data for the run')
		print(' bs : Marks the start time of a bedrock object')
		print(' be : Marks the end time of a bedrock object')
		print(' rs : Marks the start time of a rock object')
		print(' re : Marks the end time of a rock object')
		print(' gs : Marks the start time of a gullie object')
		print(' ge : Marks the end time of a gullie object')
		print(' n, <string> : Adds notes to run')
		print(' d: Displays current data object')
		print(' f, <foldername> : Set the foldername for the run data')
		print(' m, <material> : Set the material of the run')
		print(' v : Move all files to specified folder')
		print(' end : Ends the current test run, stops recording data')
		print(' ctrl + x : Stop the program')
		print(' h : Display help menu')
		print('---------------------------------')


if __name__ == "__main__":
	runobject = RunObject()
	rospy.init_node('command_input')
	pub = rospy.Publisher('crossbow_commands', String, queue_size=1)
	while not rospy.is_shutdown():
		arg = raw_input("Enter in commands for test runs....\n")
		if (runobject.parse_input(arg)):
			pub.publish(arg 	)
		time.sleep(0.5)
