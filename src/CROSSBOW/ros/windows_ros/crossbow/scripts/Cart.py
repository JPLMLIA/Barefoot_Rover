import rospy
import time
import serial

IND_CARTDUINO_MS 	= 0
IND_Z_HEIGHT 		= 1
IND_SPEED 			= 2
IND_CURRENT 		= 3
IND_CARTDUINO_IMU 	= 4
IND_PXFLOW 			= 7

class RigArduino(object):
	def __init__(self,portname,baudrate=115200,timeout=1.5,delay=0.05):
		self.ser = serial.Serial()
		self.ser.baudrate = baudrate
		self.ser.port=portname
		self.ser.timeout=timeout
		self.ser.delay=delay
		self.serial_data = [0] * 9

		self.json_data = {
			"op" : "publish",
			"topic" : "cartduino",
			"msg" : {
				"cartduino_ms" : 0.0,
				"z_height" : 0.0,
				"speed" : 0.0,
				"current" : 0.0, 
				"cart_x" : 0.0, 
				"cart_y" : 0.0, 
				"cart_z" : 0.0, 
				"pxflow_1" : 0.0, 
				"pxflow_2" : 0.0
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
			if len(line_arr)!=9:	#incomplete line
				#print('[WARNING-Cart] Cart incomplete line at {}, got {}/9 values'.format(bt.utctime(),len(line_arr)))
				#print('Cart msg: '+str(data_line))	#should trigger timeout on next run if ADXL345 failed to initialize
				raise Exception("[Cartduino] Incomplete serial packet structure")
			else:
				self.serial_data = line_arr
				#print self.serial_data

	def get_timestamp(self):
		return float(self.serial_data[IND_CARTDUINO_MS])

	def get_height(self):
		return float(self.serial_data[IND_Z_HEIGHT])

	def get_speed(self):
		return float(self.serial_data[IND_SPEED])

	def get_current(self):
		return float(self.serial_data[IND_CURRENT])

	def get_imu(self):
		return [float(self.serial_data[IND_CARTDUINO_IMU]), 
				float(self.serial_data[IND_CARTDUINO_IMU + 1]), 
				float(self.serial_data[IND_CARTDUINO_IMU + 2])
				]
	
	def get_pxflow(self):
		return [self.serial_data[IND_PXFLOW], self.serial_data[IND_PXFLOW + 1]]

	def get_json_data(self):
		self.json_data['msg']['cartduino_ms'] = self.get_timestamp()
		self.json_data['msg']['z_height'] = self.get_height()
		self.json_data['msg']['speed'] = self.get_speed()
		self.json_data['msg']['current'] = self.get_current()
		self.json_data['msg']['cart_x'] = self.get_imu()[0]
		self.json_data['msg']['cart_y'] = self.get_imu()[1]
		self.json_data['msg']['cart_z'] = self.get_imu()[2]
		#self.json_data['msg']['pxflow_1'] = self.get_pxflow()[0]
		#self.json_data['msg']['pxflow_2'] = self.get_pxflow()[1]
		#print self.json_data
		return self.json_data
		

