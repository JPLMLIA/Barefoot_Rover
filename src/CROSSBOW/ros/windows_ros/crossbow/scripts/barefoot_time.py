#author: @rayma
#creation date: 2018-02-21

#consistent utc timestamping procedure to try and keep synchronization consistent
from datetime import datetime

#generates a timestamp string used for automatic naming
def now_name():
	nstr = (datetime.isoformat(datetime.now())).replace(':','-')
	return nstr[:nstr.find('.')]

#time since epoch
def utctime():
	return (datetime.now()-datetime.utcfromtimestamp(0)).total_seconds()
	
#returns utc time from Xiroku time string (appended to each measurement)
def utctimefromxiroku(time_str):
	dt = datetime.strptime(time_str,'%Y/%m/%d %H:%M:%S.%f')
	return (dt-datetime.utcfromtimestamp(0)).total_seconds()
	
#returns utc time from EIS time string (in .idf files)
def utctimefromeis(time_str):
	dt = datetime.strptime(time_str,'%m/%d/%Y %I:%M:%S %p')
	return (dt-datetime.utcfromtimestamp(0)).total_seconds()