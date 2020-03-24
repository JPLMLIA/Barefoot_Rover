from barefoot_process import *
import sys

if __name__=="__main__":
	if len(sys.argv)>1:
		pathname = sys.argv[1]
	if len(sys.argv)>2:
		hydra = float(sys.argv[2])
	else:
		hydra = 0.0
		
	plotRollingEIS(pathname,hydra)