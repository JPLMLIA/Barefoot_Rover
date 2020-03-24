#!/usr/bin/env python
 
#looking through 10 video ids and seeing if the streams are active
 
import cv2
import numpy as np
import time

if __name__=='__main__':
	N = 10
	IMG_H = 240
	
	cam_list = [None] * N
	cam_id_list = []
	for idx in range(10):
		print("Trying id {}".format(idx))
		cam_list[idx] = cv2.VideoCapture(idx)
		if cam_list[idx].isOpened():
			print("...Found!")
			cam_list[idx].set(cv2.CAP_PROP_FRAME_WIDTH,IMG_H/cam_list[idx].get(cv2.CAP_PROP_FRAME_HEIGHT)*cam_list[idx].get(cv2.CAP_PROP_FRAME_WIDTH))
			cam_list[idx].set(cv2.CAP_PROP_FRAME_HEIGHT,IMG_H)
			cam_id_list.extend([idx])
		else:
			print("...Not found...")
			cam_list[idx] = None
				
	Nc = sum(x is not None for x in cam_list)
	print("{} webcams found with ids: {}".format(Nc,cam_id_list))
	
	#preview the available webcams in order:
	img_list = [None] * Nc
	
	last_time = time.time()
	while True:		#always cast to 640x480 to align
		for idx in range(Nc):
			ret_val,img_r = cam_list[idx].read()
			cv2.putText(img_r,str(cam_id_list[idx]),(20,IMG_H-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
			
			img_list[idx] = img_r.copy()
		out_img = img_list[0]
		if Nc>1:
			for idx in range(Nc-1):
				out_img = np.concatenate((out_img,img_list[idx+1]),axis=1)
				
		curr_time = time.time()
		dur = curr_time-last_time	#frame duration in seconds 
		fps = 1.0/dur
		cv2.putText(out_img,('%.2f fps' % fps),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
		
		cv2.imshow('cams',out_img)
		keypress = cv2.waitKey(1)
		if keypress==27:
			break
		last_time = curr_time
	
	#cleanup:
	for idx in range(Nc):
		cam_list[idx].release()
	cv2.destroyAllWindows()