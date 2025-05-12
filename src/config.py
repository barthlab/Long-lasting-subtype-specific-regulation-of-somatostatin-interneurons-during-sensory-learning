import numpy as np
import os
import os.path as path

from src.basic.terminology import *


# average policy
REPLACE_BAD_VALUE_FLAG = False
DEBUG_FLAG = False
FIGURE_SHOW_FLAG = 0


# path to data
ROOT_PATH = (r"C:\Users\maxyc\PycharmProjects\Long-lasting-subtype-specific-regulation-"
             r"of-somatostatin-interneurons-during-sensory-learning")
CALCIUM_DATA_PATH = "data/Calcium imaging"
BEHAVIOR_DATA_PATH = "data/Behavior"
FEATURE_DATA_PATH = "data/Feature"
FEATURE_EXTRACTED_PATH = "data/Extracted Feature"
FIGURE_PATH = "figures"

# data lost
LOST_DATA = {
    "M087": {"corrupted": [SatDay.SAT9, SatDay.SAT10]},
    "M088": {"lost": [SatDay.SAT7, ]},
    "M017": {"lost": [SatDay.ACC1, ]},
}
# session corruption
LOST_SESSION = {
    SessionUID(exp_id="Calb2_SAT", mice_id="M085", fov_id=4, day_id=SatDay.SAT1, session_in_day=0): 15,
    SessionUID(exp_id="Calb2_SAT", mice_id="M099", fov_id=4, day_id=SatDay.SAT3, session_in_day=0): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M037", fov_id=2, day_id=PseDay.ACC3, session_in_day=0): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M037", fov_id=2, day_id=PseDay.ACC3, session_in_day=1): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M046", fov_id=2, day_id=PseDay.ACC2, session_in_day=0): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M046", fov_id=2, day_id=PseDay.ACC2, session_in_day=1): 19,
    SessionUID(exp_id="Ai148_SAT", mice_id="M023", fov_id=1, day_id=SatDay.SAT9, session_in_day=0): 19,
}


# experiment related
EXP_LIST = ("Ai148_SAT", "Ai148_PSE", "Calb2_SAT")
EXP2DAY = {
    "Ai148_PSE": PseDay,
    "Ai148_SAT": SatDay,
    "Calb2_SAT": SatDay,
}
Ai148_FLAG = {
    "Ai148_PSE": True,
    "Ai148_SAT": True,
    "Calb2_SAT": False,
}
BASELINE_DAYS = "ACC456"

# Calb2 related
CALB2_EXPRESSION_FILE_NAME = "Expression_Calb2_SAT.xlsx"
CALB2_THRESHOLD = 200  # A.U.
CALB2_MINIMAL = 100  # A.U.
CALB2_TICKS = (120, 200, 1000)  # A.U.


def CALB2_RESIZE_FUNC(x):
    return np.log10(x - CALB2_MINIMAL)


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

# Feature extraction related
TEST_EVOKED_PERIOD = 2  # s
TEST_STD_RATIO = 5  # 5 std pretty good
RESPONSE_THRESHOLD = 0.01  # p




