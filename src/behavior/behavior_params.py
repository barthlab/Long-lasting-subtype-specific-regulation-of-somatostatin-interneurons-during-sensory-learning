import numpy as np
import os
import os.path as path
from enum import Enum
from datetime import datetime, timedelta

from src.basic.terminology import *


TXT_FILE_COL = ('TimeS', 'PokeStatus', 'LickStatus', 'Phase', 'RandDelayMS')

# Go NoGo
GO_PHASE_CODE = 3
NOGO_PHASE_CODE = 9
LICK_STATUS_CODE = 2

# plotting config
D_TIME = 0.02  # s
D_DAY = 0.05  # day
BEHAVIOR_RANGE = (-1, 4)  # s
DAY_TEXT_SIZE = 5

# lick cmap
GRAY3 = ["#000000", "#969696", "#ffffff"]
BINARY = ['white', 'black']


class BehaviorTrialType(Enum):
    Go = 1
    NoGo = 0


MISALIGNED_MICE_RECORDING_START = {
    "M031": datetime(2021, 11, 9, 12, 0, 0)
}
