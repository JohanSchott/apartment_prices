

import time


def get_time_stamp(year, month, day):
    """
    Return time stamp

    Parameters
    ----------
    year : int

    month : int

    day : int

    """
    time_tuple = (year, month, day, 0, 0, 0, 0, 1, -1)
    time_stamp = time.mktime(time_tuple)
    return time_stamp



