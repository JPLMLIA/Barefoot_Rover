#command line script to generate slip summary plot
#author: rayma
#creation date: 2018-07-13

from barefoot_process import *
import sys
import os

if __name__=='__main__':
	if len(sys.argv)>1:
		pathname = sys.argv[1]
		plotOdometry(pathname)	#EZ
	else:
		print('[ERR] Expected directory of test run to process')