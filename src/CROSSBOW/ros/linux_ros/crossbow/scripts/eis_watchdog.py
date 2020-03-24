#!/usr/bin/env python3
import time
import rospy
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from parse_test import parse_idf
from crossbow_msgs.msg import EIS
import os

class Eis_handler(object):
	def __init__(self):
		self.files_checked = []

		self.eis = EIS()
		self.eis.peaks 				= False
		self.eis.timesteps 			= 4
		self.eis.exp_time			= 0.0
		self.eis.times 				= [0.0]*self.eis.timesteps
		self.eis.real_primary 		= [0.0]*self.eis.timesteps
		self.eis.imaginary_primary 	= [0.0]*self.eis.timesteps
		self.eis.frequency_primary	= [0.0]*self.eis.timesteps
		
		patterns = "*"
		ignore_patterns = ""
		ignore_directories = False
		case_sensitive = True
		my_event_handler = PatternMatchingEventHandler(
								patterns,
								ignore_patterns,
								ignore_directories,
								case_sensitive
								)

		my_event_handler.on_created		= self.on_created
		my_event_handler.on_deleted 	= self.on_deleted
		my_event_handler.on_modified 	= self.on_modified
		my_event_handler.on_moved 		= self.on_moved

		abs_path = str((os.path.dirname(os.path.realpath(__file__))))
		path = abs_path + "/eis_data/"
		
		go_recursively = True
		self.my_observer = Observer()
		self.my_observer.schedule(
						my_event_handler,
						path,
						recursive = go_recursively)

		self.my_observer.start()

	def on_created(self, event):
		#print(str(event.src_path) + " has been created!")
		return

	def on_deleted(self, event):
		#print(str(event.src_path) + " has been deleted!")
		return

	def on_modified(self, event):
		if (event.src_path == "eis_data/"):
			#the folder being modified should be ignored
			return
		if (str(event.src_path) in self.files_checked):
			#print ("file has already been read from")
			return

		#print(str(event.src_path) + " has been modified!")
		try:
			eis_data = parse_idf(event.src_path)
			if eis_data != None:
				self.eis.peaks 				= False if (eis_data['Peaks'] == 'false') else True
				self.eis.timesteps 			= int(eis_data['timesteps'])
				self.eis.times 				= eis_data['times']
				self.eis.real_primary 		= eis_data['real_primary']
				self.eis.imaginary_primary 	= eis_data['imaginary_primary']
				self.eis.frequency_primary	= eis_data['frequency_primary']
				
				pub.publish(self.eis)
		except Exception as e:
			rospy.loginfo(e)

		self.files_checked.append(str(event.src_path))

	def on_moved(self, event):
		#print(str(event.src_path) + " has been moved!")
		return

	def cleanup(self):
		rospy.loginfo("Stopping the eis node")
		rospy.loginfo("Merging observer handler")
		self.my_observer.stop()
		self.my_observer.join()

if __name__ == "__main__":
	rospy.init_node('eis')
	rospy.loginfo('starting the eis node')
	eis_handler = Eis_handler()
	pub = rospy.Publisher('eis', EIS, queue_size = 1)
	rospy.on_shutdown(eis_handler.cleanup)
	rospy.spin()