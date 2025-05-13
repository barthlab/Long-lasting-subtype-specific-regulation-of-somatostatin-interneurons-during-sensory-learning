import numpy as np
import os
import os.path as path

from src.basic.terminology import *
from src.data_status import *


# average policy
REPLACE_BAD_VALUE_FLAG = False
DEBUG_FLAG = True
FIGURE_SHOW_FLAG = 0


# path to data
ROOT_PATH = (r"C:\Users\maxyc\PycharmProjects\Long-lasting-subtype-specific-regulation-"
             r"of-somatostatin-interneurons-during-sensory-learning")
CALCIUM_DATA_PATH = "data/Calcium imaging"
BEHAVIOR_DATA_PATH = "data/Behavior"
FEATURE_DATA_PATH = "data/Feature"
FEATURE_EXTRACTED_PATH = "data/Extracted Feature"
FIGURE_PATH = "figures"
SELECTED_CLUSTERING_FEATURE_JSON_PATH = path.join(
    FEATURE_EXTRACTED_PATH, "Calb2_SAT", "mitten_feature_Top25_ACC456_features.json")


# experiment related
EXP_LIST = ("Ai148_SAT", "Ai148_PSE", "Calb2_SAT", "Calb2_PSE")
EXP2DAY = {
    "Ai148_PSE": PseDay,
    "Ai148_SAT": SatDay,
    "Calb2_SAT": SatDay,
    "Calb2_PSE": PseDay,
}
Ai148_FLAG = {
    "Ai148_PSE": True,
    "Ai148_SAT": True,
    "Calb2_SAT": False,
    "Calb2_PSE": False,
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
SESSION_FRAME_NUMER_2P_MO = 3061
SESSION_FRAME_NUMER_2P_MATT = 3065
SESSION_DURATION = 60 * 10  # s
FRAME_LOST_THRESHOLD = 20  # pixel
FRAME_INTERPOLATE_THRESHOLD = 2  # frame
TIMEPOINTS_MS2S_FLAG = True
AP_DURATION = 0.5  # s
SESSION_FRAME_NUMBER = {
    "Ai148_PSE": SESSION_FRAME_NUMER_2P_MO,
    "Ai148_SAT": SESSION_FRAME_NUMER_2P_MO,
    "Calb2_SAT": SESSION_FRAME_NUMER_2P_MO,
    "Calb2_PSE": SESSION_FRAME_NUMER_2P_MATT,
}

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

# common feature
EVOKED_RESPONSE_FEATURE = "Evoked-Response"
FOLD_CHANGE_ACC456_FEATURE = "Fold-Change from ACC456"


