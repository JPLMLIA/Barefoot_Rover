#command line script to generate summary plot 
#author: rayma
#creation date: 2018-06-06

#TODO: also generate other comprehensive plots here...

from barefoot_process import *
import sys
import os

if __name__=='__main__':
	if len(sys.argv)>1:
		pathname = sys.argv[1]
		
		#extract timestamp
		fname_arr = os.listdir(pathname)
		#expects consistent naming convention based around the xiroku pressure pad .dat file
		dat_arr = [fname[0:-4] for fname in fname_arr if fname.endswith('.dat')]
		if len(dat_arr)>0:
			if len(dat_arr)>1:
				print('[ERR] More than one set of data files found')
			else:
				timestamp_str = dat_arr[0]
				bp = BarefootPlot(pathname+'/'+timestamp_str,'xiroku_zero.npz')
				bp.plotAllSummary()
		else:
			print('[ERR] Expected .dat file not found')
	else:
		print('[ERR] Expected directory of test run to process')
