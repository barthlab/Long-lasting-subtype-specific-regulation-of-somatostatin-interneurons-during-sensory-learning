import numpy as np
import os
import os.path as path
from terminology import *


# average policy
REPLACE_BAD_VALUE_FLAG = False
DEBUG_FLAG = True


# path to data
ROOT_PATH = (r"C:\Users\maxyc\PycharmProjects\Long-lasting-subtype-specific-regulation-"
             r"of-somatostatin-interneurons-during-sensory-learning")
CALCIUM_DATA_PATH = "data/Calcium imaging"


# data lost
LOST_DATA = {
    "M087": {"corrupted": [SatDay.SAT9, SatDay.SAT10]},
    "M088": {"lost": [SatDay.SAT7, ]},
    "M017": {"lost": [SatDay.ACC1, ]}
}


# experiment related
EXP_LIST = ("Ai148_PSE", "Ai148_SAT", "Calb2_SAT")
EXP2DAY = {
    "Ai148_PSE": PseDay,
    "Ai148_SAT": SatDay,
    "Calb2_SAT": SatDay,
}

# 2p recording parameters
FS_2P = 5.1  # Hz
DT = 1/FS_2P  # s
HDT = DT/2  # s
GLOBAL_BASELINE_PERCENTILE = 20  # %
GLOBAL_BASELINE_WINDOW = int(FS_2P*60)  # frames
SESSION_FRAMES_2P = 3061
SESSION_DURATION = 60 * 10  # s
FRAME_LOST_THRESHOLD = 20  # pixel
FRAME_INTERPOLATE_THRESHOLD = 2  # frame
TIMEPOINTS_MS2S_FLAG = True
AP_DURATION = 0.5  # s

# Split parameter
TRIAL_RANGE = (-2, 4)  # s
TRIAL_BASELINE_RANGE = (-1, 0)  # s
BLOCK_PRE_TRIAL = -3  # s
BLOCK_POST_TRIAL = 5  # s
LAST_BLOCK_LEN = 10 + BLOCK_POST_TRIAL  # s


# color setting
EVENT2COLOR = {
    EventType.Puff: "green",
    EventType.Blank: "red",
    BlockType.PreBlock: "lightskyblue",
    BlockType.InterBlock: "deepskyblue",
    BlockType.PostBlock: "blue"
}

# fluorescence plotting
DY_DF_F0 = 0.5
