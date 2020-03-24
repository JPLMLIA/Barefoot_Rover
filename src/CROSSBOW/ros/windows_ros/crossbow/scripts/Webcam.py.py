import cv2
import time
import rospy
from datetime import datetime
import os


class Webcam(object):
	def __init__(self):
		self.cameras = []
		self.camera_loc = ['front', 'back', 'side']
		try:
			self.camera0 = cv2.VideoCapture(0)
			self.cameras.append(self.camera0)
		except:
			pass

		### TODO uncomment these and test with multiple webcameras 
		'''
		try:
			self.camera1 = cv2.VideoCapture(1)
			self.cameras.append(self.camera1)
		except:
			pass

		try:
			self.camera2 = cv2.VideoCapture(2)
			self.cameras.append(self.camera2)
		except:
			pass
		'''
		self.abs_path = str((os.path.dirname(os.path.realpath(__file__))))

	def utctime(self):
		return (datetime.now()-datetime.utcfromtimestamp(0)).total_seconds()

	def now_name(self):
		nstr = (datetime.isoformat(datetime.now())).replace(':','-')
		return nstr[:nstr.find('.')]

	def take_pic(self):
		for i in range(len(self.cameras)):
			return_value, image = self.cameras[i].read()
			loc = self.abs_path +  "\\webcam_images\\" + self.now_name() + '.jpg'
			print loc
			cv2.imwrite(loc, image)

	def cleanup(self):
		for camera in self.cameras:
			del(camera)



if __name__ == "__main__":
	rospy.init_node("webcam")
	webcam = Webcam()

	rospy.on_shutdown(webcam.cleanup)

	rate = rospy.Rate(1)

	while not rospy.is_shutdown():
		webcam.take_pic()
		rate.sleep()

