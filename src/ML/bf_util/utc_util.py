"""This is duplicated in CROSSBOW/barefoot_time.py
"""
from datetime import datetime
def utc_time_from_xiroku(time_str):
    """[summary]
    
    [extended_summary]
    
    Parameters
    ----------
    time_str : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    dt = datetime.strptime(time_str,'%Y/%m/%d %H:%M:%S.%f')
    return (dt-datetime.utcfromtimestamp(0)).total_seconds()
