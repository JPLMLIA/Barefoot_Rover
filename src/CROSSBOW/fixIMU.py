#script to update IMU (correctly this time)
from barefoot_cartduino import *
import sys
import os

if __name__=="__main__":
	pathname = sys.argv[1]
	fname_arr = os.listdir(pathname)
	imu_arr = [fname for fname in fname_arr if fname.endswith('_wheel_v1.npy')]
	if len(imu_arr)>0:
		arr = np.load(pathname+'/'+imu_arr[0])
		arr2 = updateAccel(arr)
		fname2 = imu_arr[0][:-5]+'3.npy'
		np.save(pathname+'/'+fname2,arr2)
		print('Saved as '+fname2)
	else:
		print('[ERR] Not found')