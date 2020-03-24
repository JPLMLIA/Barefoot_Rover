import numpy as np
from PyDAQmx import *
import ctypes
import barefoot_time as bt
import rospy

#default calibration array, as provided directly by ATI:
calib_SI1500240 = np.matrix([
[5.850068468,0.581182962,9.417016553,-163.9635653,-9.235987679,155.6377572],
[-11.40534156,184.1008543,7.850696406,-94.4901843,1.873133449,-90.2865238],
[262.1905169,5.900631418,263.3752605,5.323605345,267.0500841,-1.327472316],
[-0.421274998,6.076968507,-16.37120829,-3.457143502,17.07521405,-3.041908634],
[19.17771388,0.418029824,-9.555486577,5.177512026,-8.902097586,-5.065882526],
[0.621066816,-9.642869337,0.707577674,-9.724916988,0.281312445,-9.342177686]])

#running with additional rotation to account for mounting offset on CROSSBOW:
calib_SI1500240_rot45 = np.matrix([
[-3.923312804,130.5382417,12.21055695,-182.7739443,-5.209451449,46.27953513],
[-12.20298053,129.8199899,-1.102693888,49.05232882,7.853261293,-173.8763137],
[262.1905169,5.900631418,263.3752605,5.323605345,267.0500841,-1.327472316],
[13.2572861,4.59425024,-18.33485923,1.214053118,5.786579375,-5.732503817],
[13.86385761,-3.999644972,4.812143153,6.106107353,-18.36643067,-1.433448251],
[0.621066816,-9.642869337,0.707577674,-9.724916988,0.281312445,-9.342177686]])


class ATI_Task(Task):
	def __init__(self,rate=20,samplesN=10,calib=calib_SI1500240_rot45):
		Task.__init__(self)
		self.N = 6										#number of channels
		self.samplesN = samplesN						#samples per channel
		self.rate = rate								#samples per second (detected freq should match this)
		#unlike other sensors, ATI is hardware limited and not limited by serial port, so the rate should be determined ahead of time
		self.buffer = np.zeros([self.samplesN,self.N])	#for buffer output
		
		self.mdata = np.zeros([1,self.N])				#mean data w/o calibratino file applied (might just be direct gauge values?)
		self.xdata = np.zeros([1,self.N])				#mean data with calibration applied
		self.calib = calib
		
		self.last_time = bt.utctime() #barefoot UTC time
		self.curr_time = self.last_time
		self.freq = 0
		
		self.json_data = {
			"op" : "publish",
			"topic" : "fts",
			"msg" : {
				"ati_time" : 0.0,
				"m_Fx" : 0.0, 
				"m_Fy" : 0.0,  
				"m_Fz" : 0.0, 
				"m_Tx" : 0.0, 
				"m_Ty" : 0.0, 
				"m_Tz" : 0.0,
				"c_Fx" : 0.0, 
				"c_Fy" : 0.0,  
				"c_Fz" : 0.0, 
				"c_Tx" : 0.0, 
				"c_Ty" : 0.0, 
				"c_Tz" : 0.0
				}
		}

		self.json_init = {
			"op": "advertise",
			"topic": "",
			"type": ""
		}
		#setting up 6 channels
		for chan in range(self.N):
			self.CreateAIVoltageChan(
				physicalChannel = "Dev1/ai{}".format(chan),
				nameToAssignToChannel = "",
				terminalConfig = DAQmx_Val_Cfg_Default,
				minVal = -10.0,
				maxVal = 10.0,
				units = DAQmx_Val_Volts,
				customScaleName = None)
		
		#sampling and timing
		self.CfgSampClkTiming(
			source = "",
			rate = self.rate,
			activeEdge = DAQmx_Val_Rising,
			#sampleMode = DAQmx_Val_FiniteSamps,
			sampleMode = DAQmx_Val_ContSamps,
			sampsPerChan = self.samplesN)
		self.readvar = int32()	#need a blank variable for type matching
		

	def build_json_init(self, topic, msg_type):
		self.json_init['topic'] = topic
		self.json_init['type'] = msg_type
		return self.json_init

	#calling read prior to StartTask() will incur performance hit (start/stop initialized per read call if task not already started)
	def read(self):
		#GroupByScanNumber: interleaved, one set of samples at a time
		#GroupByChannel: non-interleaved, one set of channels at a time
		self.ReadAnalogF64(
			numSampsPerChan = self.samplesN,
			timeout = 10.0,
			fillMode = DAQmx_Val_GroupByScanNumber,
			readArray = self.buffer,
			arraySizeInSamps = self.N*self.samplesN,
			sampsPerChanRead = ctypes.byref(self.readvar),
			reserved = None)
		
		self.mdata = self.buffer.mean(axis=0)
		self.xdata = self.calib.dot(self.mdata)
		
		self.curr_time = bt.utctime()
		self.freq = 1./(self.curr_time-self.last_time+0.0000001)
		self.last_time = self.curr_time
		
		#print (self.mdata)
		#self.xdata = self.xdata.tolist()[0]
		
		self.json_data['msg']['ati_time'] = self.curr_time
	
		self.json_data['msg']['m_Fx'] = self.mdata[0]
		self.json_data['msg']['m_Fy'] = self.mdata[1]
		self.json_data['msg']['m_Fz'] = self.mdata[2]
		self.json_data['msg']['m_Tx'] = self.mdata[3]
		self.json_data['msg']['m_Ty'] = self.mdata[4]
		self.json_data['msg']['m_Tz'] = self.mdata[5]


		'''
		self.json_data['msg']['cFx'] = self.xdata[0]
		self.json_data['msg']['cFy'] = self.xdata[1]
		self.json_data['msg']['cFz'] = self.xdata[2]
		self.json_data['msg']['cTx'] = self.xdata[3]
		self.json_data['msg']['cTy'] = self.xdata[4]
		self.json_data['msg']['cTz'] = self.xdata[5]
		'''
		return 0

	def get_time(self):
		return self.curr_time

	def get_mdata(self):
		return self.mdata

	def get_xdata(self):
		return self.xdata
		
