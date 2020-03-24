#Labjack functionality in support of string potentiometer for odometry

import u3
import barefoot_time as bt
import numpy as np
import threading
import time
import json

#save both voltage and approximated position relative to rest (0V)
class StringPot(object):
	def __init__(self,delay=0.05,filename=None):
			
		#self.K_MM = 627.907	#scaling factor measured for SG1-120-3 stringpot
		self.K_MM = 641.42 #scaling re-done on 2018-12-20
		self.json_data = {
			"op" : "publish",
			"topic" : "labjack",
			"msg" : {
				"labjack_time" : 0,
				"raw_voltage" : 0,
				"distance_travelled" : 0
			}
		}
		self.json_init = {
			"op": "advertise",
			"topic": "",
			"type": ""
		}
		
		try:
			self.d = u3.U3()	#should only be single one
			self.d.configIO(FIOAnalog = 3)	#from example scripts. may set both FIO0 and FIO1 to analog
		except u3.LabJackException:
			print("[ERR] Could not connec to LabJack. Check connections and try again")
			self.d = None
		
		self.raw_voltage = 0.0
		self.curr_time = bt.utctime()
	
	def build_json_init(self, topic, msg_type):
		self.json_init['topic'] = topic
		self.json_init['type'] = msg_type
		return self.json_init

	def read(self):
	
		self.raw_voltage = self.d.getAIN(0,32)		#32 setting for 0-3.6V range (default is single-ended AI limited to 0-2.5V)
		self.distance_travelled = self.raw_voltage*self.K_MM
		self.curr_time = bt.utctime()

		self.json_data['msg']['raw_voltage'] = self.raw_voltage
		self.json_data['msg']['labjack_time'] = self.curr_time
		self.json_data['msg']['distance_travelled'] = self.distance_travelled



	def get_time(self):
		return self.curr_time

	def get_raw_voltage(self):
		return self.raw_voltage

	def get_distance_travelled(self):
		return self.distance_travelled
		
#test script to gracefully catch connection errors and exit 
def testConnection():
	try:
		d = u3.U3()
		print("Connected Successfully")
	except u3.LabJackException:
		print("Could not open LabJack")
		