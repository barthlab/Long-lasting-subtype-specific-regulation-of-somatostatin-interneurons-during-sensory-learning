import glob
import numpy as np
import os
import os.path as path
from collections import defaultdict
from typing import List, Callable, Optional, Dict, Any, Iterable
import re
from datetime import datetime, timedelta
import math

from src.basic.utils import *
from src.config import *


def parser_start_time_from_filename(filename: str) -> datetime:
    match = re.search(r"(\d{2})_(\d{2})_(\d{2})_T_(\d{2})_(\d{2})_(\d{2})", filename)
    if match:
        month, day, year, hour, minute, second = map(int, match.groups())
        return datetime(2000 + year, month, day, hour, minute, second)
    else:
        raise ValueError(f"Could not parse start time from filename: {filename}")


def time2tomorrow(now_time: datetime):
    tomorrow = now_time + timedelta(days=1)
    midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
    time_to_midnight = midnight - now_time
    return time_to_midnight


def time2yesterday(now_time: datetime):
    midnight = datetime(now_time.year, now_time.month, now_time.day, 0, 0, 0)
    time_to_midnight = now_time - midnight
    return time_to_midnight


