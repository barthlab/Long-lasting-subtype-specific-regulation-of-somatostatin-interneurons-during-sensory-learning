import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, Normalize
from scipy.interpolate import splprep, splev
import matplotlib.patches as mpatches

from src.data_manager import *
from src.basic.utils import *
from src.config import *

plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 8

plt.rcParams.update({
    'xtick.labelsize': 7,      # X-axis tick labels
    'ytick.labelsize': 7,      # Y-axis tick labels
    'axes.labelsize': 7,       # X and Y axis labels
    'axes.titlesize': 7,       # Plot title
    'legend.fontsize': 7,      # Legend font size
    'figure.titlesize': 8      # Figure title (suptitle)
})


# color setting
EVENT2COLOR = {
    EventType.Puff: "Green",
    EventType.Blank: "red",
    BlockType.PreBlock: "lightskyblue",
    BlockType.InterBlock: "deepskyblue",
    BlockType.PostBlock: "blue"
}
OTHER_COLORS = {
    "water": "lightblue",
    "nowater": "gray",
    "puff": "gray",
    "annotate": 'gray',
    "SAT": "lightskyblue",
    "PSE": 'lightpink'
}
CELLTYPE2COLOR = {
    CellType.Unknown: "black",
    CellType.Calb2_Pos: "magenta",
    CellType.Calb2_Neg: "black",
    CellType.Put_Calb2_Neg: "gray",
    CellType.Put_Calb2_Pos: "#ff80ff",
}
FEATURE_LABEL2COLOR = {
    # Response probability (favorite) - vibrant red
    "response probability features": "#E63946",

    # In-trial activity - blue spectrum
    "in-trial activity peak features": "#1D3557",
    "in-trial activity timing features": "#457B9D",
    "in-trial activity center of mass features": "#A8DADC",
    "in-trial activity area under curve features": "#48CAE4",

    # Spontaneous activity - green/yellow spectrum
    "spontaneous activity peak features": "#2A9D8F",
    "spontaneous activity scale features": "#8AB17D",
    # "spontaneous activity center of mass features": "#E9C46A",
    "spontaneous activity area under curve features": "#F4A261"
}
PERIOD_NAME2COLOR = {
    # Block-related (cool colors)
    "inter-trial blocks": "#66c2a5",  # soft teal
    "init&final blocks": "#3288bd",  # medium blue
    "final block": "#1f78b4",  # deeper blue
    "init block": "#a6cee3",  # light blue
    # "after-blank blocks": "#5eaaa8",  # blue-green
    # "after-stim blocks": "#007c91",  # deep teal

    # Trial-period related (warm colors)
    "pre-trial period": "#fdae61",  # warm orange
    "evoked-trial period": "#d7191c",  # bright red
    "decay-trial period": "#f46d43",  # reddish-orange
    "post-trial period": "#fee08b",  # pale yellow-orange
}

# fluorescence plotting
DY_DF_F0 = 0.5
DISPLAY_SINGLE_DF_F0_RANGE = {
    "zscore": (-1., 3.),
    "Calb2_SAT": (-0.2, 6),
    "Calb2_PSE": (-0.1, 1.7),
    "Ai148_SAT": (-0.2, 6),
    "Ai148_PSE": (-0.2, 2.5),
}
DISPLAY_AVG_DF_F0_RANGE = {
    "Calb2_SAT": (0., 1.2),
    "Calb2_PSE": (0., 1.9),
    "Ai148_SAT": (0., 2.3),
    "Ai148_PSE": (0., 2.3),
}
DISPLAY_FOLD_CHANGE_RANGE = {
    "Calb2_SAT": (0., 3),
    "Calb2_PSE": (0., 3.7),
    "Ai148_SAT": (0., 3.7),
    "Ai148_PSE": (0., 3.7),
}

LW_SMALL_DF_F0 = 0.5
LW_MEDIUM_DF_F0 = 1
LW_LARGE_DF_F0 = 2

ALPHA_DEEP_DF_F0 = 0.8
ALPHA_LIGHT_DF_F0 = 0.3


# smooth flow
BSPLINE_SMOOTH_FACTOR = 0.1
NUM_SMOOTH_POINTS = 100


# Statistical test
STATISTICAL_FONTSIZE = 3
SIGNIFICANT_P = 0.05
SIGNIFICANT_P_EXTRA = 0.01
SIGNIFICANT_ALPHA = 0.05
TEXT_OFFSET_SCALE = 1.1
# SIGNIFICANT_TRACE_Y_EXTENT = (-0.09, -0.06)
# SIGNIFICANT_TRACE_POOLING_WINDOW = 0.2  # s
# SIGNIFICANT_TRACE_COLORMAP = ListedColormap(['lightgray', 'black'])
SIGNIFICANT_ANOVA_BAR_HEIGHT_SAT = 2
SIGNIFICANT_ANOVA_BAR_HEIGHT_PSE = 3.3


# clustering
CLUSTER_COLORLIST = ["#ff80ff", "gray", "#AEAE00", ] + list(mcolors.TABLEAU_COLORS.values())*2


# experiment related plotting params
PLOTTING_TRIALS_CONFIG = {
    "stimulus_trial": {"trial_type": EventType.Puff},
}

PLOTTING_DAYS_CONFIG = {
    "Calb2_SAT": {
        # "sortbymice": (["ACC4", "ACC5", "ACC6", f"SAT1", f"SAT5", f"SAT9"], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", "SAT1", "SAT5", "SAT9"], "ACC6"),
        "two_days": (["ACC456", "SAT12", "SAT56", "SAT910"], "ACC456"),
    },
    "Calb2_PSE": {
        # "sortbymice": (["ACC4", "ACC5", "ACC6", "PSE1", "PSE3", "PSE5", ], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", "PSE1", "PSE2", "PSE3", "PSE4", "PSE5", ], "ACC6"),
        "two_days": (["ACC456", 'PSE12', 'PSE45'], "ACC456"),
    },
    "Ai148_SAT": {
        # "sortbymice": (["ACC4", "ACC5", "ACC6", f"SAT1", f"SAT5", f"SAT9"], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", "SAT1", "SAT5", "SAT9"], "ACC6"),
        "two_days": (["ACC456", "SAT12", "SAT56", "SAT910"], "ACC456"),
    },
    "Ai148_PSE": {
        # "sortbymice": (["ACC4", "ACC5", "ACC6", f"PSE1", f"PSE5", f"PSE9"], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", "PSE1", "PSE5", "PSE9"], "ACC6"),
        "two_days": (["ACC456", "PSE12", "PSE56", "PSE910"], "ACC456"),
    }
}
PLOTTING_GROUPS_CONFIG = {
    "Calb2_SAT": {
        "single_days": (["ACC456", "SAT1", "SAT5", "SAT9"], "ACC6"),
        "two_days": (["ACC456", "SAT12", "SAT56", "SAT910"], "ACC456"),
    },
    "Calb2_PSE": {
        "single_days": (["ACC456", "PSE1", "PSE3", "PSE5", ], "ACC6"),
        "two_days": (["ACC456", 'PSE12', 'PSE45'], "ACC456"),
    },
    "Ai148_SAT": {
        "single_days": (["ACC456", "SAT1", "SAT5", "SAT9"], "ACC6"),
        "two_days": (["ACC456", "SAT12", "SAT56", "SAT910"], "ACC456"),
    },
    "Ai148_PSE": {
        "single_days": (["ACC456", "PSE1", "PSE5", "PSE9"], "ACC6"),
        "two_days": (["ACC456", "PSE12", "PSE56", "PSE910"], "ACC456"),
    }
}


# trial_marker
TRIAL_MARKER = {
    EventType.Puff: {"color": "black", "marker": '|', "s": 20, "lw": 1.5, "clip_on": False},
    EventType.Blank: {"edgecolors": "black", "facecolors": "none", "marker": "o", "s": 40, "lw": 1.5, "clip_on": False}
}



