import numpy as np
import os
import os.path as path
from scipy.signal import find_peaks

from src.config import *
from src.basic.utils import *
from src.basic.terminology import *
from src.data_manager import *


OPTIONS_TIME_RANGE = {
    "pre-trial period": {"start_t": -2, "end_t": 0},  # s
    "trial evoked period": {"start_t": 0, "end_t": 2},  # s
    "post-trial period": {"start_t": 2, "end_t": 4},  # s
}

OPTIONS_TRIAL_GROUP = {
    "stimulus trial only": {"trial_type": EventType.Puff},
    "blank trial only": {"trial_type": EventType.Blank},
    "all trial included": {},
}

OPTIONS_SPONT_BLOCK_GROUP = {
    "init block": {"block_type": BlockType.PreBlock},
    "final block": {"block_type": BlockType.PostBlock},
    "inter-trial blocks": {"block_type": BlockType.InterBlock},
    "init&final blocks": {"block_type": (BlockType.PreBlock, BlockType.PostBlock)},
}


# element func: trial period
def element_feature_compute_trial_period_activity_peak(
        single_trial: Trial, start_t: float, end_t: float,
):
    trial_clip = single_trial.df_f0.segment(start_t=start_t, end_t=end_t, new_origin_t=0, relative_flag=True).v
    return np.max(trial_clip)


def element_feature_compute_trial_period_activity_auc(
        single_trial: Trial, start_t: float, end_t: float,
):
    trial_clip = single_trial.df_f0.segment(start_t=start_t, end_t=end_t, new_origin_t=0, relative_flag=True)
    return np.trapz(trial_clip.v, x=trial_clip.t_aligned)


def element_feature_compute_trial_period_peak_latency(
        single_trial: Trial, start_t: float, end_t: float,
):
    trial_clip = single_trial.df_f0.segment(start_t=start_t, end_t=end_t, new_origin_t=0, relative_flag=True)
    peak_frame = np.argmax(trial_clip.v)
    return trial_clip.t_aligned[peak_frame]


def element_feature_compute_trial_period_activity_com(
        single_trial: Trial, start_t: float, end_t: float,
):
    trial_clip = single_trial.df_f0.segment(start_t=start_t, end_t=end_t, new_origin_t=0, relative_flag=True)
    x, y = trial_clip.t_aligned, trial_clip.v
    return np.trapz(y * x, x=x)/np.trapz(y, x=x)


def element_feature_compute_trial_period_responsive_probability(
        single_trial: Trial, start_t: float, end_t: float,
        ratio_std: float, trail_baseline_std: float
):
    trial_clip = single_trial.df_f0.segment(start_t=start_t, end_t=end_t, relative_flag=True).v
    return np.max(trial_clip) >= ratio_std * trail_baseline_std


OPTIONS_ELEMENT_FEATURE_TRIAL_PERIOD = {
    "peak": element_feature_compute_trial_period_activity_peak,
    "auc": element_feature_compute_trial_period_activity_auc,
    "latency": element_feature_compute_trial_period_peak_latency,
    "com": element_feature_compute_trial_period_activity_com,
}


OPTIONS_STD_RATIO = {
    f"{i}std": {"ratio_std": i} for i in [2, 3, 5, 10]
}


# element func: blocks
def element_feature_compute_block_event_amplitude(
        single_block: SpontBlock, overall_baseline_std: float, ratio_std: float,
):
    peaks, properties = find_peaks(single_block.df_f0.v, prominence=ratio_std*overall_baseline_std)
    peak_prominences = properties["prominences"]
    if len(peaks) > 0:
        return nan_mean(peak_prominences)
    else:
        return 0


def element_feature_compute_block_event_count(
        single_block: SpontBlock, overall_baseline_std: float, ratio_std: float,
):
    peaks, properties = find_peaks(single_block.df_f0.v, prominence=ratio_std*overall_baseline_std)
    return len(peaks)


def element_feature_compute_block_activity_auc(
        single_block: SpontBlock,
):
    return np.trapz(single_block.df_f0.v, x=single_block.df_f0.t_zeroed)


def element_feature_compute_block_activity_com(
        single_block: SpontBlock,
):
    x, y = single_block.df_f0.t_zeroed, single_block.df_f0.v
    return np.trapz(y * x, x=x)/np.trapz(y, x=x)


OPTIONS_ELEMENT_FEATURE_SPONT_BLOCK_OVERALL_STD_RELATED = {
    "amplitude": element_feature_compute_block_event_amplitude,
    "count": element_feature_compute_block_event_count,
}

OPTIONS_ELEMENT_FEATURE_SPONT_BLOCK_BASIC = {
    "auc": element_feature_compute_block_activity_auc,
    "com": element_feature_compute_block_activity_com,
}


def feature_name_to_y_axis_label(feature_name: str) -> str:
    split_names = feature_name.split(" || ")
    if split_names[0] == "response prob":
        return "response prob (%)"
    elif split_names[0] in ("peak", "amplitude"):
        return r"activity peak ($\Delta F/F_0$)"
    elif split_names[0] in ("latency", ):
        return "peak latency (s)"
    elif split_names[0] in ("count", ):
        return "# event"
    elif split_names[0] in ("com", ):
        return "activity center (s)"
    elif split_names[0] in ("auc", ):
        return "AUC (A.U.)"
    else:
        raise NotImplementedError


def feature_name_to_title_short(feature_name: str) -> str:
    split_names = feature_name.split(" || ")[1:]
    if len(split_names) > 1:
        return f"{split_names[0]}\n{' | '.join(split_names[1:])}"
    else:
        return split_names[0]


def feature_name_to_file_name(feature_name: str) -> str:
    tmp_strs = feature_name.split(" || ")
    for i in range(1, len(tmp_strs)):
        tmp_strs[i] = tmp_strs[i].split(" ")[0].replace("-", "").replace("&", "")
    if tmp_strs[0] == "response prob":
        tmp_strs[0] = "prob"
    return " ".join(tmp_strs)


def feature_name_to_label(feature_name: str) -> str:
    split_names = feature_name.split(" || ")
    prefix, postfix = split_names[0], " ".join(split_names[1:])
    if prefix == "response prob":
        return "response probability feature"
    elif prefix == "peak":
        return r"in-trial activity peak features"
    elif prefix == "amplitude":
        return r"spontaneous activity peak features"
    elif prefix == "latency":
        return "in-trial activity timing features"
    elif prefix == "count":
        return r"spontaneous activity scale features"
    elif prefix == "com":
        if "block" in postfix:
            return "spontaneous activity center of mass features"
        else:
            return "in-trial activity center of mass features"
    elif prefix == "auc":
        if "block" in postfix:
            return "spontaneous activity area under curve features"
        else:
            return "in-trial activity area under curve features"
    else:
        raise NotImplementedError


def feature_name_to_period_name(feature_name: str) -> str:
    for possible_period_name in ("pre-trial period", "trial evoked period", "post-trial period",
                                 "inter-trial blocks", "init&final blocks", "final block", "init block"):
        if possible_period_name in feature_name:
            return possible_period_name

