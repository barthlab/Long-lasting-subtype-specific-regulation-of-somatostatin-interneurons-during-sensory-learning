import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.interpolate import splprep, splev

from src.data_manager import *
from src.basic.utils import *
from src.config import *

plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 8

# color setting
EVENT2COLOR = {
    EventType.Puff: "Green",
    EventType.Blank: "red",
    BlockType.PreBlock: "lightskyblue",
    BlockType.InterBlock: "deepskyblue",
    BlockType.PostBlock: "blue"
}
OTHER_COLORS = {
    "puff": "gray",
    "annotate": 'gray',
    "SAT": "lightskyblue",
    "PSE": 'lightpink'
}
CELLTYPE2COLOR = {
    CellType.Unknown: "gray",
    CellType.Calb2_Pos: "magenta",
    CellType.Calb2_Neg: "black",
    CellType.Put_Calb2_Neg: "brown",
    CellType.Put_Calb2_Pos: "pink",
}


# fluorescence plotting
DY_DF_F0 = 0.5
DISPLAY_MAX_DF_F0_Calb2 = 2.5
DISPLAY_MAX_DF_F0_Ai148 = 3.5
DISPLAY_MIN_DF_F0 = -0.1
AVG_MAX_DF_F0_Calb2 = 0.75
AVG_MAX_DF_F0_Ai148 = 1.45
AVG_MAX_FOLD_Calb2 = 3
AVG_MAX_FOLD_Ai148 = 3

LW_SMALL_DF_F0 = 0.5
LW_MEDIUM_DF_F0 = 1
LW_LARGE_DF_F0 = 2

ALPHA_DEEP_DF_F0 = 0.8
ALPHA_LIGHT_DF_F0 = 0.3


# smooth flow
BSPLINE_SMOOTH_FACTOR = 0.1
NUM_SMOOTH_POINTS = 100