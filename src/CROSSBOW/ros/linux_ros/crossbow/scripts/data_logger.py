#!/usr/bin/env python
from datetime import datetime
import rospy
import time
import numpy as np
from crossbow_msgs.msg import Labjack, Cartarduino, Ati, Wheelduino
from std_msgs.msg import String
import os
import json
import shutil


class DataLogger(object):
	def __init__(self):
		
		self.timestamp_string = self.now_name()

		abs_path = str((os.path.dirname(os.path.realpath(__file__))))
		self.eis_data_location = abs_path + '/eis_data'
		self.save_data_location = abs_path + '/test_save_data/'
		self.filepath = self.save_data_location + self.timestamp_string
		#self.move_folder_location = "/mnt/c/Users/jplba/workspace/Barefoot_data/"
		self.move_folder_location = "/mnt/c/Users/jplba/Desktop/Barefoot_data/"
		self.webcam_folder_location = '/mnt/c/Users/jplba/Documents/crossbow_ros/src/crossbow/scripts/webcam_images/'

		self.cartduino_data 		= [0.0]*10
		self.fts_data 				= [0.0]*7
		self.fts_data_cal			= [0.0]*7
		self.labjack_data 			= [0.0]*3
		self.wheelduino_data 		= [0.0]*8

		self.cartduino_save_data	= []
		self.fts_save_data 			= []
		self.fts_save_data_cal		= []
		self.labjack_save_data 		= []
		self.wheelduino_save_data 	= []

		self.running = 0

	def utctime(self):
		return (datetime.now()-datetime.utcfromtimestamp(0)).total_seconds()

	def now_name(self):
		nstr = (datetime.isoformat(datetime.now())).replace(':','-')
		return nstr[:nstr.find('.')]
	
	def cartduino_callback(self, data):
		self.cartduino_data[0] 						= self.utctime()		
		self.cartduino_data[1]						= data.cartduino_ms
		self.cartduino_data[2]						= data.z_height
		self.cartduino_data[3]						= data.current
		self.cartduino_data[4]						= data.speed
		self.cartduino_data[5]						= data.cart_x
		self.cartduino_data[6]						= data.cart_y
		self.cartduino_data[7]						= data.cart_z
		self.cartduino_data[8]						= data.pxflow_1
		self.cartduino_data[9]						= data.pxflow_2	
		
		if self.running: self.cartduino_save_data.append(np.array(self.cartduino_data))

	def fts_callback(self, data):
		self.fts_data[0]							= self.utctime()
		self.fts_data[1]							= data.m_Fx
		self.fts_data[2]							= data.m_Fy
		self.fts_data[3]							= data.m_Fz
		self.fts_data[4]							= data.m_Tx
		self.fts_data[5]							= data.m_Ty
		self.fts_data[6]							= data.m_Tz

		self.fts_data_cal[0]						= self.utctime()
		self.fts_data_cal[1]						= data.c_Fx
		self.fts_data_cal[2]						= data.c_Fy
		self.fts_data_cal[3]						= data.c_Fz
		self.fts_data_cal[4]						= data.c_Tx
		self.fts_data_cal[5]						= data.c_Ty
		self.fts_data_cal[6]						= data.c_Tz

		if self.running: self.fts_save_data.append(np.array(self.fts_data))
		if self.running: self.fts_save_data_cal.append(np.array(self.fts_data_cal))

	def labjack_callback(self, data):
		self.labjack_data[0]				 		= self.utctime()
		self.labjack_data[1]					 	= data.raw_voltage
		self.labjack_data[2]			 			= data.distance_travelled
		if self.running: self.labjack_save_data.append(np.array(self.labjack_data))
	
	def wheelduino_callback(self, data):
		self.wheelduino_data[0]						= self.utctime()
		self.wheelduino_data[1]						= data.wheel_x
		self.wheelduino_data[2]						= data.wheel_y
		self.wheelduino_data[3]						= data.wheel_z
		self.wheelduino_data[4]						= data.r
		self.wheelduino_data[5]						= data.theta
		self.wheelduino_data[6]						= data.phi
		self.wheelduino_data[7]						= data.rho
		if self.running:  self.wheelduino_save_data.append(np.array(self.wheelduino_data))
	
	def save_files(self):
		np.save((self.filepath + '_ati.npy'), self.fts_save_data)
		np.save((self.filepath + '_atx_x.npy'), self.fts_save_data_cal)
		np.save((self.filepath + '_cart.npy'), self.cartduino_save_data)
		np.save((self.filepath + '_wheel.npy'), self.wheelduino_save_data)
		np.save((self.filepath + '_stringpot.npy'), self.labjack_save_data)
		with open(self.save_data_location + 'Notes.json', 'w') as outfile:
			json.dump(self.run_data, outfile)

	def cleanup(self):
		self.save_files()
		print("Saving data to: " + str(self.filepath )) 
		self.move_files()
		return

	def command_callback(self, message):
		cmd = message.data.split(",")[0]
		vals = message.data.split(",")[1:]
		if (cmd == "new"):
			self.new_run()

		elif (cmd =="rock" or cmd =="bedrock" or cmd == "gullie" or cmd == "dune" or cmd == "pebble"):
			self.create_feature(cmd, vals)
		elif (cmd == "m"):
			self.run_data['material'] = vals[0]
		elif (cmd =="f"):
			self.create_folder(vals[0])
		elif (cmd =="n"):
			self.add_notes(vals)

		elif (cmd == "rs"):
			self.mark_feature('rock', 0)
		elif (cmd == "re"):
			self.mark_feature('rock', 1)
		elif (cmd == "bs"):
			self.mark_feature('bedrock', 0)
		elif (cmd == "be"):
			self.mark_feature('bedrock', 1)
		elif (cmd == "gs"):
			self.mark_feature('gullie', 0)
		elif (cmd == "ge"):
			self.mark_feature('gullie', 1)

		elif (cmd == "d"):
			print(self.run_data)

		elif (cmd == "start"):
			self.start_run()
		elif (cmd == 'end'):
			self.end_run()

		elif (cmd == "v"):
			self.move_files()

	def new_run(self):
		if (len(os.listdir(self.save_data_location)) > 0):
			print("Cannot start new run without moving/removing data in the temp data folders!")
			return

		print "Creating new run...."
		self.run_data = {
			'utc_time' : 'None',   
			'type': 'None',                 #Slip, hydration, etc. 
			'notes' : 'None',
			'material' : 'None',
			'foldername' : 'None',
			'run_features_location' : {
				'rock': 'None',
				'bedrock': 'None',
				'gullie': 'None',
				'dune' : 'None',
				'pebble' : 'None'
			},
			'run_features_time' : {
				'rock_start' : 'None',
				'bedrock_start' : 'None',
				'rock_end' : 'None',
				'bedrock_end' : 'None',
				'gullie_start' : 'None',
				'gullie_end' : 'None'
			}
		}

	def create_feature(self, feature, location):
		if (feature == "bedrock"):
			self.run_data['run_features_location']['bedrock'] = location
			print('creating feature: %s at: %s ....' %(feature, location))
		elif (feature == "rock"):
			self.run_data['run_features_location']['rock'] = location
			print('creating feature: %s at: %s ....' %(feature, location))
		elif (feature =="gullie"):
			self.run_data['run_features_location']['gullie'] = location
			print('creating feature: %s at: %s ....' %(feature, location))
		elif (feature == "dune"):
			self.run_data['run_features_location']['dune'] = location[0]
			print('creating feature: %s of type: %s ....' %(feature, location[0]))

	def create_folder(self, name):
		self.run_data['foldername'] = name
		temp = self.move_folder_location + name
		self.cur_move_folder_loc = temp
		if not os.path.exists(temp):
			print ("Creating folder at %s ..." % (temp))
			os.makedirs(temp)
			os.makedirs(temp + '/EIS')
			os.makedirs(temp + '/' + self.timestamp_string)

	def add_notes(self, note):
		if self.run_data['notes'] == 'None':
			self.run_data['notes'] = [note]
		else:
			self.run_data['notes'].append(note)

	def start_run(self):
		if (self.run_data['foldername'] == 'None'):
			print("Must enter in a foldername before starting run!")
			return 

		# wipe the EIS and webcam folders before starting a new run, these devices will be
		# running constantly and we want data only from the runtime of the test
		for f in os.listdir(self.eis_data_location):
			os.remove(self.eis_data_location + '/' + f)
		for f in os.listdir(self.webcam_folder_location):
			os.remove(self.webcam_folder_location + f)

		print("Starting collecting data...")
		self.run_data['utc_time'] = self.utctime()
		self.running = 1

	def mark_feature(self, object, selector):
		if not self.running:
			print("Data collection has not been started! Need to start collection to annotate data")
			return
		param = object + '_end' if (selector) else object + "_start"
		print('Adding annotation to %s' %(param))
		if self.run_data['run_features_time'][param] == "None":
			self.run_data['run_features_time'][param] = [self.utctime()]
		else:
			self.run_data['run_features_time'][param].append(self.utctime())

	def end_run(self):
		if not self.running:
			print("Cannot end data collection, start command never received")
			return

		print('Ending run....')
		self.running = 0
		self.save_files()        #at the end of the run, output all the numpy vectors to files to be postprocessed

	def move_files(self):
		## NEED TO MOVE THE FOLLOWING
		## .dat (pressuregrid)

		print("moving files to %s" %(self.cur_move_folder_loc))
		for f in os.listdir(self.save_data_location):
			shutil.move(self.save_data_location + f, self.cur_move_folder_loc + "/" + f)

		for f in os.listdir(self.eis_data_location):
			shutil.move(self.eis_data_location + "/" +  f, self.cur_move_folder_loc + "/EIS/" + f )

		for f in os.listdir(self.webcam_folder_location):
			shutil.move(self.webcam_folder_location + f, self.cur_move_folder_loc + '/' + self.timestamp_string + '/')

		self.run_data = {}

if __name__ == "__main__":	
	rospy.init_node("data_logger")
	rospy.loginfo("Starting data_logger node")

	dataLogger = DataLogger()

	wheelduino_sub = rospy.Subscriber('/wheelduino', Wheelduino, dataLogger.wheelduino_callback)
	cartduiino_sub = rospy.Subscriber('/cartduino', Cartarduino, dataLogger.cartduino_callback)
	fts_sub = rospy.Subscriber('/fts', Ati, dataLogger.fts_callback)
	labjack_sub = rospy.Subscriber('/labjack', Labjack, dataLogger.labjack_callback)
	input_sub = rospy.Subscriber('crossbow_commands', String, dataLogger.command_callback)

	rospy.on_shutdown(dataLogger.cleanup)

	rate = rospy.Rate(10) 
	while not rospy.is_shutdown():
		rate.sleep()

