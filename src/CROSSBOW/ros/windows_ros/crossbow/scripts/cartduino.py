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
		self.connect()

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
		rospy.loginfo(data_line)
		if len(data_line)>0:
			line_arr = data_line.split('\t'.encode())
			if len(line_arr)!=9:	#incomplete line
				#print('[WARNING-Cart] Cart incomplete line at {}, got {}/9 values'.format(bt.utctime(),len(line_arr)))
				#print('Cart msg: '+str(data_line))	#should trigger timeout on next run if ADXL345 failed to initialize
				raise Exception("[Cartduino] Incomplete serial packet structure")
			else:
				self.serial_data = line_arr

	def get_timestamp(self):
		return float(self.serial_data[IND_CARTDUINO_MS])

	def get_height(self):
		return float(self.serial_data[IND_Z_HEIGHT])

	def get_speed(self):
		return int(self.serial_data[IND_SPEED])

	def get_current(self):
		return int(self.serial_data[IND_CURRENT])

	def get_imu(self):
		return [self.serial_data[IND_CARTDUINO_IMU], self.serial_data[IND_CARTDUINO_IMU + 1], self.serial_data[IND_CARTDUINO_IMU + 2]]
	
	def get_pxflow(self):
		return [self.serial_data[IND_PXFLOW], self.serial_data[IND_PXFLOW + 1]]

