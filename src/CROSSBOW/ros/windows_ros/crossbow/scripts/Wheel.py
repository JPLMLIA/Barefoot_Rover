import rospy
import time
import serial
from datetime import datetime
import numpy as np

IND_WHEELDUINO_IMU 		= 0
IND_WHEELDUINO_R		= 3
IND_WHEELDUINO_THETA	= 4
IND_WHEELDUINO_PHI		= 5
IND_WHEELDUINO_ROT		= 6


class WheelArduino(object):
	def __init__(self,portname,baudrate=115200,timeout=1.5,delay=0.05):
		self.ser = serial.Serial()
		self.ser.baudrate = baudrate
		self.ser.port=portname
		self.ser.timeout=timeout
		self.ser.delay=delay
		self.curr_time = (datetime.now()-datetime.utcfromtimestamp(0)).total_seconds()
		self.last_time = self.curr_time
		self.serial_data = [0] * 3
		self.imu = [0.0] *3
		self.ax = 0
		self.ay = 0
		self.az = 0
		self.json_data = {
			"op" : "publish",
			"topic" : "wheelduino",
			"msg" : {
				"wheelduino_ms" : 0.0,
				"wheel_x" : 0.0,
				"wheel_y" : 0.0,
				"wheel_z" : 0.0,
				"r" : 0.0,
				"theta" : 0.0,
				"phi" : 0.0,
				"rho" : 0.0
			}
		}
		self.json_init = {
			"op": "advertise",
			"topic": "",
			"type": ""
		}


		self.connect()

	def build_json_init(self, topic, msg_type):
		self.json_init['topic'] = topic
		self.json_init['type'] = msg_type
		return self.json_init

	def connect(self):
		self.ser.open()
		self.ser.rtscts = True
		time.sleep(1.0)

	def disconnect(self):
		self.ser.close()
		time.sleep(1.0)

	def reset(self):
		self.disconnect()
		self.connect()

	def read(self):
		self.ser.write('R'.encode())
		time.sleep(self.ser.delay)	#by default allows for 20Hz
		data_line = self.ser.readline()
		#rospy.loginfo(data_line)
		if len(data_line)>0:
			line_arr = data_line.split('\t'.encode())
			if len(line_arr)!=3:	#incomplete line
				#print('[WARNING-Cart] Cart incomplete line at {}, got {}/9 values'.format(bt.utctime(),len(line_arr)))
				#print('Cart msg: '+str(data_line))	#should trigger timeout on next run if ADXL345 failed to initialize
				raise Exception("[Wheel Arduino] Incomplete serial packet structure")
			else:
				self.serial_data = line_arr
				self.imu_calculations()


	def get_imu(self):
		return [self.serial_data[IND_WHEELDUINO_IMU], self.serial_data[IND_WHEELDUINO_IMU + 1], self.serial_data[IND_WHEELDUINO_IMU + 2]]
	
	def imu_calculations(self):
		line_arr=self.serial_data

		self.ax = float(line_arr[0])-0.4993
		self.ay = float(line_arr[1])-0.0772
		self.az = float(line_arr[2])+1.4496
		self.imu = [self.ax, self.ay, self.ax]
		return self.imu

	def get_r(self):
		self.r = np.sqrt(self.ax**2+self.ay**2+self.az**2)    #spherical coordinates conversion
		return self.r

	def get_theta(self):
		self.theta = np.arccos(self.az/self.r)*180.0/np.pi
		return self.theta

	def get_phi(self):
		self.phi =np.sign(self.ay)*np.arccos(self.ax/np.sqrt(self.ax**2+self.ay**2))*180/np.pi
		return self.phi

	def get_rot(self):
		if self.ax>=0:
			self.rot = (np.arctan(self.az/(self.ax+0.00000000001))*180/np.pi) % 360
		else:
			self.rot = (np.arctan(self.az/(self.ax+0.00000000001))*180/np.pi+180) % 360
		return self.rot

	def get_time(self):
		self.curr_time = (datetime.now()-datetime.utcfromtimestamp(0)).total_seconds()
		self.freq = 1./(self.curr_time-self.last_time)
		self.last_time = self.curr_time
		return self.curr_time

	def get_json_data(self):
		self.json_data["msg"]["wheelduino_ms"] = self.curr_time
		self.json_data["msg"]["wheel_x"] = self.ax
		self.json_data["msg"]["wheel_y"] = self.ay
		self.json_data["msg"]["wheel_z"] = self.ay
		self.json_data["msg"]["r"] = self.r
		self.json_data["msg"]["theta"] = self.theta
		self.json_data["msg"]["phi"] = self.phi
		self.json_data["msg"]["rho"] = self.rot
		return self.json_data
		