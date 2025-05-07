import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from scipy.interpolate import splprep, splev
import matplotlib.patches as mpatches

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
FEATURE_LABEL2COLOR ={
    # Response probability (favorite) - vibrant red
    "response probability feature": "#E63946",

    # In-trial activity - blue spectrum
    "in-trial activity peak features": "#1D3557",
    "in-trial activity timing features": "#457B9D",
    "in-trial activity center of mass features": "#A8DADC",
    "in-trial activity area under curve features": "#48CAE4",

    # Spontaneous activity - green/yellow spectrum
    "spontaneous activity peak features": "#2A9D8F",
    "spontaneous activity scale features": "#8AB17D",
    "spontaneous activity center of mass features": "#E9C46A",
    "spontaneous activity area under curve features": "#F4A261"
}

# fluorescence plotting
DY_DF_F0 = 0.5
DISPLAY_MAX_DF_F0_Calb2 = 2.5
DISPLAY_MAX_DF_F0_Ai148 = 3.5
DISPLAY_MIN_DF_F0_Calb2 = -0.1
DISPLAY_MIN_DF_F0_Ai148 = 1.5*DISPLAY_MAX_DF_F0_Ai148*DISPLAY_MIN_DF_F0_Calb2/DISPLAY_MAX_DF_F0_Calb2
AVG_MAX_DF_F0_Calb2 = 1.2
AVG_MAX_DF_F0_Ai148 = 2.3
AVG_MAX_FOLD_Calb2 = 3
AVG_MAX_FOLD_Ai148 = 3.7

LW_SMALL_DF_F0 = 0.5
LW_MEDIUM_DF_F0 = 1
LW_LARGE_DF_F0 = 2

ALPHA_DEEP_DF_F0 = 0.8
ALPHA_LIGHT_DF_F0 = 0.3


# smooth flow
BSPLINE_SMOOTH_FACTOR = 0.1
NUM_SMOOTH_POINTS = 100


# Statistical test
STATISTICAL_FONTSIZE = 5
SIGNIFICANT_P = 0.05
SIGNIFICANT_ALPHA = 0.05
TEXT_OFFSET_SCALE = 1.1
SIGNIFICANT_TRACE_Y_EXTENT = (-0.09, -0.06)
SIGNIFICANT_TRACE_POOLING_WINDOW = 0.2  # s
SIGNIFICANT_TRACE_COLORMAP = ListedColormap(['lightgray', 'black'])
SIGNIFICANT_ANOVA_BAR_HEIGHT_SAT = 2
SIGNIFICANT_ANOVA_BAR_HEIGHT_PSE = 3.3


