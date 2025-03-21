import colorist
import os
import os.path as path
import numpy as np
from scipy.io import loadmat, savemat
import scipy
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'

DEBUG_FLAG = False

session_length = 3061
session_duration = 60 * 10 * 1000  # ms


TotalFeatureNames = ['Spt_Peak_Mean', 'Spt_Prom_Mean', 'Spt_ISI_Mean', 'Spt_Len_Mean', 'Spt_Height_Mean', 'Spt_Peak_Median', 'Spt_Prom_Median', 'Spt_ISI_Median', 'Spt_Len_Median', 'Spt_Height_Median', 'Spt_Freq', 't2_0s1s_RPoP', 't2_0s1s_RBoB', 't2_0s1s_RPmRB', 't2_0s1s_RoA', 't2_0s1s_RP_Pre_Peak', 't2_0s1s_RP_Evoked_Peak', 't2_0s1s_RP_Late_Peak', 't2_0s1s_RP_Post_Peak', 't2_0s1s_RB_Pre_Peak', 't2_0s1s_RB_Evoked_Peak', 't2_0s1s_RB_Late_Peak', 't2_0s1s_RB_Post_Peak', 'P_Pre_Peak', 'P_Evoked_Peak', 'P_Late_Peak', 'P_Post_Peak', 'B_Pre_Peak', 'B_Evoked_Peak', 'B_Late_Peak', 'B_Post_Peak', 't2_0s1s_R_Pre_Peak', 't2_0s1s_R_Evoked_Peak', 't2_0s1s_R_Late_Peak', 't2_0s1s_R_Post_Peak', 'A_Pre_Peak', 'A_Evoked_Peak', 'A_Late_Peak', 'A_Post_Peak', 't2_0s1s_RPmRB_Evoked_Peak', 'PmB_Evoked_Peak', 't2_1s4s_RPoP', 't2_1s4s_RBoB', 't2_1s4s_RPmRB', 't2_1s4s_RoA', 't2_1s4s_RP_Pre_Peak', 't2_1s4s_RP_Evoked_Peak', 't2_1s4s_RP_Late_Peak', 't2_1s4s_RP_Post_Peak', 't2_1s4s_RB_Pre_Peak', 't2_1s4s_RB_Evoked_Peak', 't2_1s4s_RB_Late_Peak', 't2_1s4s_RB_Post_Peak', 't2_1s4s_R_Pre_Peak', 't2_1s4s_R_Evoked_Peak', 't2_1s4s_R_Late_Peak', 't2_1s4s_R_Post_Peak', 't2_1s4s_RPmRB_Evoked_Peak', 't2_0s3s_RPoP', 't2_0s3s_RBoB', 't2_0s3s_RPmRB', 't2_0s3s_RoA', 't2_0s3s_RP_Pre_Peak', 't2_0s3s_RP_Evoked_Peak', 't2_0s3s_RP_Late_Peak', 't2_0s3s_RP_Post_Peak', 't2_0s3s_RB_Pre_Peak', 't2_0s3s_RB_Evoked_Peak', 't2_0s3s_RB_Late_Peak', 't2_0s3s_RB_Post_Peak', 't2_0s3s_R_Pre_Peak', 't2_0s3s_R_Evoked_Peak', 't2_0s3s_R_Late_Peak', 't2_0s3s_R_Post_Peak', 't2_0s3s_RPmRB_Evoked_Peak', 't3_0s1s_RPoP', 't3_0s1s_RBoB', 't3_0s1s_RPmRB', 't3_0s1s_RoA', 't3_0s1s_RP_Pre_Peak', 't3_0s1s_RP_Evoked_Peak', 't3_0s1s_RP_Late_Peak', 't3_0s1s_RP_Post_Peak', 't3_0s1s_RB_Pre_Peak', 't3_0s1s_RB_Evoked_Peak', 't3_0s1s_RB_Late_Peak', 't3_0s1s_RB_Post_Peak', 't3_0s1s_R_Pre_Peak', 't3_0s1s_R_Evoked_Peak', 't3_0s1s_R_Late_Peak', 't3_0s1s_R_Post_Peak', 't3_0s1s_RPmRB_Evoked_Peak', 't3_1s4s_RPoP', 't3_1s4s_RBoB', 't3_1s4s_RPmRB', 't3_1s4s_RoA', 't3_1s4s_RP_Pre_Peak', 't3_1s4s_RP_Evoked_Peak', 't3_1s4s_RP_Late_Peak', 't3_1s4s_RP_Post_Peak', 't3_1s4s_RB_Pre_Peak', 't3_1s4s_RB_Evoked_Peak', 't3_1s4s_RB_Late_Peak', 't3_1s4s_RB_Post_Peak', 't3_1s4s_R_Pre_Peak', 't3_1s4s_R_Evoked_Peak', 't3_1s4s_R_Late_Peak', 't3_1s4s_R_Post_Peak', 't3_1s4s_RPmRB_Evoked_Peak', 't3_0s3s_RPoP', 't3_0s3s_RBoB', 't3_0s3s_RPmRB', 't3_0s3s_RoA', 't3_0s3s_RP_Pre_Peak', 't3_0s3s_RP_Evoked_Peak', 't3_0s3s_RP_Late_Peak', 't3_0s3s_RP_Post_Peak', 't3_0s3s_RB_Pre_Peak', 't3_0s3s_RB_Evoked_Peak', 't3_0s3s_RB_Late_Peak', 't3_0s3s_RB_Post_Peak', 't3_0s3s_R_Pre_Peak', 't3_0s3s_R_Evoked_Peak', 't3_0s3s_R_Late_Peak', 't3_0s3s_R_Post_Peak', 't3_0s3s_RPmRB_Evoked_Peak', 't5_0s1s_RPoP', 't5_0s1s_RBoB', 't5_0s1s_RPmRB', 't5_0s1s_RoA', 't5_0s1s_RP_Pre_Peak', 't5_0s1s_RP_Evoked_Peak', 't5_0s1s_RP_Late_Peak', 't5_0s1s_RP_Post_Peak', 't5_0s1s_RB_Pre_Peak', 't5_0s1s_RB_Evoked_Peak', 't5_0s1s_RB_Late_Peak', 't5_0s1s_RB_Post_Peak', 't5_0s1s_R_Pre_Peak', 't5_0s1s_R_Evoked_Peak', 't5_0s1s_R_Late_Peak', 't5_0s1s_R_Post_Peak', 't5_0s1s_RPmRB_Evoked_Peak', 't5_1s4s_RPoP', 't5_1s4s_RBoB', 't5_1s4s_RPmRB', 't5_1s4s_RoA', 't5_1s4s_RP_Pre_Peak', 't5_1s4s_RP_Evoked_Peak', 't5_1s4s_RP_Late_Peak', 't5_1s4s_RP_Post_Peak', 't5_1s4s_RB_Pre_Peak', 't5_1s4s_RB_Evoked_Peak', 't5_1s4s_RB_Late_Peak', 't5_1s4s_RB_Post_Peak', 't5_1s4s_R_Pre_Peak', 't5_1s4s_R_Evoked_Peak', 't5_1s4s_R_Late_Peak', 't5_1s4s_R_Post_Peak', 't5_1s4s_RPmRB_Evoked_Peak', 't5_0s3s_RPoP', 't5_0s3s_RBoB', 't5_0s3s_RPmRB', 't5_0s3s_RoA', 't5_0s3s_RP_Pre_Peak', 't5_0s3s_RP_Evoked_Peak', 't5_0s3s_RP_Late_Peak', 't5_0s3s_RP_Post_Peak', 't5_0s3s_RB_Pre_Peak', 't5_0s3s_RB_Evoked_Peak', 't5_0s3s_RB_Late_Peak', 't5_0s3s_RB_Post_Peak', 't5_0s3s_R_Pre_Peak', 't5_0s3s_R_Evoked_Peak', 't5_0s3s_R_Late_Peak', 't5_0s3s_R_Post_Peak', 't5_0s3s_RPmRB_Evoked_Peak', 't10_0s1s_RPoP', 't10_0s1s_RBoB', 't10_0s1s_RPmRB', 't10_0s1s_RoA', 't10_0s1s_RP_Pre_Peak', 't10_0s1s_RP_Evoked_Peak', 't10_0s1s_RP_Late_Peak', 't10_0s1s_RP_Post_Peak', 't10_0s1s_RB_Pre_Peak', 't10_0s1s_RB_Evoked_Peak', 't10_0s1s_RB_Late_Peak', 't10_0s1s_RB_Post_Peak', 't10_0s1s_R_Pre_Peak', 't10_0s1s_R_Evoked_Peak', 't10_0s1s_R_Late_Peak', 't10_0s1s_R_Post_Peak', 't10_0s1s_RPmRB_Evoked_Peak', 't10_1s4s_RPoP', 't10_1s4s_RBoB', 't10_1s4s_RPmRB', 't10_1s4s_RoA', 't10_1s4s_RP_Pre_Peak', 't10_1s4s_RP_Evoked_Peak', 't10_1s4s_RP_Late_Peak', 't10_1s4s_RP_Post_Peak', 't10_1s4s_RB_Pre_Peak', 't10_1s4s_RB_Evoked_Peak', 't10_1s4s_RB_Late_Peak', 't10_1s4s_RB_Post_Peak', 't10_1s4s_R_Pre_Peak', 't10_1s4s_R_Evoked_Peak', 't10_1s4s_R_Late_Peak', 't10_1s4s_R_Post_Peak', 't10_1s4s_RPmRB_Evoked_Peak', 't10_0s3s_RPoP', 't10_0s3s_RBoB', 't10_0s3s_RPmRB', 't10_0s3s_RoA', 't10_0s3s_RP_Pre_Peak', 't10_0s3s_RP_Evoked_Peak', 't10_0s3s_RP_Late_Peak', 't10_0s3s_RP_Post_Peak', 't10_0s3s_RB_Pre_Peak', 't10_0s3s_RB_Evoked_Peak', 't10_0s3s_RB_Late_Peak', 't10_0s3s_RB_Post_Peak', 't10_0s3s_R_Pre_Peak', 't10_0s3s_R_Evoked_Peak', 't10_0s3s_R_Late_Peak', 't10_0s3s_R_Post_Peak', 't10_0s3s_RPmRB_Evoked_Peak', 't20_0s1s_RPoP', 't20_0s1s_RBoB', 't20_0s1s_RPmRB', 't20_0s1s_RoA', 't20_0s1s_RP_Pre_Peak', 't20_0s1s_RP_Evoked_Peak', 't20_0s1s_RP_Late_Peak', 't20_0s1s_RP_Post_Peak', 't20_0s1s_RB_Pre_Peak', 't20_0s1s_RB_Evoked_Peak', 't20_0s1s_RB_Late_Peak', 't20_0s1s_RB_Post_Peak', 't20_0s1s_R_Pre_Peak', 't20_0s1s_R_Evoked_Peak', 't20_0s1s_R_Late_Peak', 't20_0s1s_R_Post_Peak', 't20_0s1s_RPmRB_Evoked_Peak', 't20_1s4s_RPoP', 't20_1s4s_RBoB', 't20_1s4s_RPmRB', 't20_1s4s_RoA', 't20_1s4s_RP_Pre_Peak', 't20_1s4s_RP_Evoked_Peak', 't20_1s4s_RP_Late_Peak', 't20_1s4s_RP_Post_Peak', 't20_1s4s_RB_Pre_Peak', 't20_1s4s_RB_Evoked_Peak', 't20_1s4s_RB_Late_Peak', 't20_1s4s_RB_Post_Peak', 't20_1s4s_R_Pre_Peak', 't20_1s4s_R_Evoked_Peak', 't20_1s4s_R_Late_Peak', 't20_1s4s_R_Post_Peak', 't20_1s4s_RPmRB_Evoked_Peak', 't20_0s3s_RPoP', 't20_0s3s_RBoB', 't20_0s3s_RPmRB', 't20_0s3s_RoA', 't20_0s3s_RP_Pre_Peak', 't20_0s3s_RP_Evoked_Peak', 't20_0s3s_RP_Late_Peak', 't20_0s3s_RP_Post_Peak', 't20_0s3s_RB_Pre_Peak', 't20_0s3s_RB_Evoked_Peak', 't20_0s3s_RB_Late_Peak', 't20_0s3s_RB_Post_Peak', 't20_0s3s_R_Pre_Peak', 't20_0s3s_R_Evoked_Peak', 't20_0s3s_R_Late_Peak', 't20_0s3s_R_Post_Peak', 't20_0s3s_RPmRB_Evoked_Peak']
CoolFeatureNames = [
    'Spt_Peak_Mean', 'Spt_Prom_Mean', 'Spt_ISI_Mean', 'Spt_Len_Mean', 'Spt_Height_Mean', 'Spt_Peak_Median',
    'Spt_Prom_Median', 'Spt_ISI_Median', 'Spt_Len_Median', 'Spt_Height_Median', 'Spt_Freq',
    'P_Pre_Peak', 'P_Evoked_Peak', 'P_Late_Peak', 'P_Post_Peak',
    'B_Pre_Peak', 'B_Evoked_Peak', 'B_Late_Peak', 'B_Post_Peak',
    'A_Pre_Peak', 'A_Evoked_Peak', 'A_Late_Peak', 'A_Post_Peak',
    't2_0s1s_RPoP', 't3_0s1s_RPoP', 't5_0s1s_RPoP', 't10_0s1s_RPoP', 't20_0s1s_RPoP',
    't2_0s1s_RoA', 't3_0s1s_RoA', 't5_0s1s_RoA', 't10_0s1s_RoA', 't20_0s1s_RoA', ]


PlasticityFeature = "P_Evoked_Peak"


def shout_out(message: str, end=False):
    if not end:
        print()
    message_len = len(message)
    print("#"*(message_len+18))
    print("#"*8, message, "#"*8)
    print("#"*(message_len+18))
    if end:
        print()


def get_visualize_feature_label_names(data_dict, select=0):
    code = data_dict['dataset name'].split("_")[-1]
    if select == 0:
        list_names = [(feature_name + f" ({code}4~{code}6)", "min max") for feature_name in TotalFeatureNames]
    elif select == 1:
        list_names = [(feature_name + f" (ACC4~ACC6)", "min max") for feature_name in TotalFeatureNames]
    elif select == 2:
        list_names = [(feature_name + f" (ACC4~ACC6)", "min max") for feature_name in CoolFeatureNames]
    elif select == -1:
        list_names = []
    else:
        raise NotImplementedError

    list_names += [
        (f"{PlasticityFeature} ({code}1 / ACC4~ACC6)", "log"),
        (f"{PlasticityFeature} ({code}5 / ACC4~ACC6)", "log"),
        (f"{PlasticityFeature} ({code}10 / ACC4~ACC6)", "log"),
    ]
    if "Calb2" in data_dict['dataset name']:
        list_names.append(("Calb2", "scalar"), )
    return list_names


DotScheme = {
    "SST-Calb2": {"color": "magenta", "type": 1, 'hatch': False},
    "SST-Calb2 (TP)": {"color": "magenta", "type": 1, 'hatch': True},
    "SST-Calb2 (FP)": {"color": "magenta", "type": 0, 'hatch': True},
    "SST-O": {"color": "black", "type": 1, 'hatch': False},
    "SST-O2": {"color": "pink", "type": 0, 'hatch': True},

    "SST-Calb2 (Put.)": {"color": "magenta", "type": 0, 'hatch': True},
    "SST-Calb2 (FP) (Put.)": {"color": "#AEAE00", "type": 0, 'hatch': True},
    "SST-O (Put.)": {"color": "gray", "type": 0, 'hatch': True},
    "SST-O2 (Put.)": {"color": "#AEAE00", "type": 0, 'hatch': True},

    "Unknown": {"color": "blue", "type": 0}
}


RemovedCellIdx = {
    "Calb2_SAT": [46, 28, 29, 33, 34],  # cell 43?
    "Ai148_SAT": [],
    "Ai148_PSE": list(np.arange(38, 93))}


def FeatureNameConvertor(feature_names):
    """
    Converts a list of feature names into more descriptive names.

    Parameters:
    feature_names (list of str): List of feature names to be converted.

    Returns:
    list of str: List of converted feature names.
    """
    new_names = []
    for feature_name_acc in feature_names:
        feature_name = feature_name_acc.split(" (ACC4")[0]
        if "Spt" in feature_name:
            if feature_name == "Spt_Freq":
                new_names.append("Spontaneous Frequency")
            else:
                p1, p2, p3 = feature_name.split("_")
                assert p1 == "Spt"
                if p2 == "Peak":
                    new_names.append(f"Spontaneous\nAbsolute Peak {p3}")
                elif p2 == "Prom":
                    new_names.append(f"Spontaneous\nProminence {p3}")
                elif p2 == "Prom":
                    new_names.append(f"Spontaneous\nProminence {p3}")
                elif p2 == "ISI":
                    new_names.append(f"Spontaneous\nInter-Event Interval {p3}")
                elif p2 == "Len":
                    new_names.append(f"Spontaneous\nDuration {p3}")
                elif p2 == "Height":
                    new_names.append(f"Spontaneous\nRelative Peak {p3}")
                else:
                    raise NotImplementedError
        elif "Peak" in feature_name:
            p1, p2, p3 = feature_name.split("_")
            assert p3 == "Peak"
            if p1 == "P":
                prefix = "Stimulus"
            elif p1 == "B":
                prefix = "Blank"
            elif p1 == "A":
                prefix = "Any"
            else:
                raise NotImplementedError
            if p2 == "Pre":
                new_names.append(f"Pre-{prefix} Peak")
            elif p2 == "Post":
                new_names.append(f"Post-{prefix} Peak")
            elif p2 == "Evoked":
                new_names.append(f"{prefix}-Evoked Peak")
            elif p2 == "Late":
                new_names.append(f"{prefix}-Related Peak")
            else:
                raise NotImplementedError
        else:
            p1, p2, p3 = feature_name.split("_")
            assert p2 == "0s1s"
            threshold = p1[1:]
            if p3 == "RPoP":
                new_names.append(f"Stimulus Trial\nResponse Probability ({threshold} MAD)")
            elif p3 == "RoA":
                new_names.append(f"All Trial\nResponse Probability ({threshold} MAD)")
            else:
                raise NotImplementedError
    return new_names