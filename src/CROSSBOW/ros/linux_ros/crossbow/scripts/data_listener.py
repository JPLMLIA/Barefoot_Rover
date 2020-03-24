#!/usr/bin/env python

import rospy
import time	
from crossbow_msgs.msg import Labjack, Cartarduino, Ati, Wheelduino, EIS
from barefoot_rover_ros.msg import XtsScan

class DataListener(object):
	def __init__(self):
		self.data_vec = {
			'cartduino_timestamp': 0,
			'z_height': 0,
			'wheel_current' : 0,
			'wheel_speed' : 0,
			'cart_x' : 0.0,
			'cart_y' : 0.0,
			'cart_z' : 0.0,
			'pxflow_1' : 0.0,
			'pxflow_2' : 0.0,
			
			'ati_timestamp' : 0.0,
			'Fx': 0.0,
			'Fy': 0.0,
			'Fz': 0.0,
			'Tx': 0.0,
			'Ty': 0.0,
			'Tz': 0.0,

			'labjack_temp_raw_voltage': 0,
			'labjack_timestamp': 0,
			'distance_travelled': 0,
			
			'wheelduino_timestamp' : 0.0,
			'wheel_x': 0.0,
			'wheel_y': 0.0,
			'wheel_z': 0.0,
			'r' : 0.0,
			'phi' : 0.0,
			'rho' : 0.0,
			'theta' : 0.0,

			'eis_peaks' : False,
			'eis_timesteps' : 0,
			'eis_exp_time' : 0.0,
			'eis_times_0' : 0.0,
			'eis_times_1' : 0.0,
			'eis_times_2' : 0.0,
			'eis_times_3' : 0.0,
			'eis_real_primary_0' : 0.0,
			'eis_real_primary_1' : 0.0,
			'eis_real_primary_2' : 0.0,
			'eis_real_primary_3' : 0.0,
			'eis_imaginary_primary_0' : 0.0,
			'eis_imaginary_primary_1' : 0.0,
			'eis_imaginary_primary_2' : 0.0,
			'eis_imaginary_primary_3' : 0.0,
			'eis_frequency_primary_0' : 0.0,
			'eis_frequency_primary_1' : 0.0,
			'eis_frequency_primary_2' : 0.0,
			'eis_frequency_primary_3' : 0.0,
		}
		
		self.xiroku_data = None

	def cartduino_callback(self, data):
		self.data_vec['cartduino_timestamp']		= data.cartduino_ms
		self.data_vec['z_height']					= data.z_height
		self.data_vec['wheel_current']				= data.current
		self.data_vec['wheel_speed']				= data.speed
		self.data_vec['cart_x']						= data.cart_x
		self.data_vec['cart_y']						= data.cart_y
		self.data_vec['cart_z']						= data.cart_z
		self.data_vec['pxflow_1']					= data.pxflow_1
		self.data_vec['pxflow_2']					= data.pxflow_2

	def fts_callback(self, data):
		self.data_vec['ati_timestamp'] 				= data.ati_time
		self.data_vec['Fx']							= data.m_Fx
		self.data_vec['Fy']							= data.m_Fy
		self.data_vec['Fz']							= data.m_Fz
		self.data_vec['Tx']							= data.m_Tx
		self.data_vec['Ty']							= data.m_Ty
		self.data_vec['Tz']							= data.m_Tz

	def labjack_callback(self, data):
		self.data_vec['labjack_timestamp'] 			= data.labjack_time
		self.data_vec['labjack_temp_raw_voltage'] 	= data.raw_voltage
		self.data_vec['distance_travelled'] 		= data.distance_travelled

	def wheelduino_callback(self, data):
		self.data_vec['wheelduino_timestamp']		= data.wheelduino_ms
		self.data_vec['wheel_x']					= data.wheel_x
		self.data_vec['wheel_y']					= data.wheel_y
		self.data_vec['wheel_z']					= data.wheel_z
		self.data_vec['r']							= data.r
		self.data_vec['phi']						= data.phi
		self.data_vec['rot']						= data.rho
		self.data_vec['theta']						= data.theta

	def eis_callback(self, data):
		self.data_vec['eis_peaks']					= data.peaks
		self.data_vec['eis_timesteps']				= data.timesteps
		self.data_vec['eis_exp_time'] 				= data.exp_time
		self.data_vec['eis_times_0']				= data.times[0]
		self.data_vec['eis_times_1']				= data.times[1]
		self.data_vec['eis_times_2']				= data.times[2]
		self.data_vec['eis_times_3']				= data.times[3]
		self.data_vec['eis_real_primary_0']			= data.real_primary[0]
		self.data_vec['eis_real_primary_0']			= data.real_primary[1]
		self.data_vec['eis_real_primary_0']			= data.real_primary[2]
		self.data_vec['eis_real_primary_0']			= data.real_primary[3]
		self.data_vec['eis_imaginary_primary_0']	= data.imaginary_primary[0]
		self.data_vec['eis_imaginary_primary_0']	= data.imaginary_primary[1]
		self.data_vec['eis_imaginary_primary_0']	= data.imaginary_primary[2]
		self.data_vec['eis_imaginary_primary_0']	= data.imaginary_primary[3]
		self.data_vec['eis_frequency_primary_0']	= data.frequency_primary[0]
		self.data_vec['eis_frequency_primary_0']	= data.frequency_primary[1]
		self.data_vec['eis_frequency_primary_0']	= data.frequency_primary[2]
		self.data_vec['eis_frequency_primary_0']	= data.frequency_primary[3]

	def xiroku_callback(self, data):
		self.xiroku_data = data

if __name__ == "__main__":	
	rospy.init_node("data_listener")
	rospy.loginfo("Starting data_listener node")

	dataListener = DataListener()

	wheelduino_sub = rospy.Subscriber('/wheelduino', Wheelduino, dataListener.wheelduino_callback)
	cartduiino_sub = rospy.Subscriber('/cartduino', Cartarduino, dataListener.cartduino_callback)
	fts_sub = rospy.Subscriber('/fts', Ati, dataListener.fts_callback)
	labjack_sub = rospy.Subscriber('/labjack', Labjack, dataListener.labjack_callback)
	eis_sub = rospy.Subscriber('/eis', EIS, dataListener.eis_callback)
	xiroku_sub = rospy.Subscriber('XtsBus', XtsScan, dataListener.xiroku_callback )
	rate = rospy.Rate(10) 
	while not rospy.is_shutdown():
		
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Insert code for real time data stream here. Everything else runs async and on callbacks
		# so no need to deal with threading etc. the dataListener.data_vec dictionary contains all
		# the data from all devices except for the xiroku, which is in dataListener.xiroku_data
		# The callbacks will happen every time the data gets an updated value, which is generally 10Hz,
		# except for the xiroku which is much faster
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		rate.sleep()  #depending on how fast you want your code to run you can remove this sleep
