"""
This script processes calcium imaging data to extract features from trials and spontaneous periods.
It includes functions for segmenting sessions, extracting features, and archiving the results.

Functions
---------
session_segmentation:
    Segments the session data into trials and spontaneous periods.
    Detrends the calcium signals and calculates the median absolute deviation (MAD).

session_feature_extraction:
    Extracts features from the segmented session data.
    Calculates peak values, spontaneous event properties, and response fractions.

mice_segmentation:
    Segments the data for each mouse into sessions and extracts features.
    Handles specific cases for different mice.

read_mice:
    Reads and processes data for each mouse from the dataset.
    Calls mice_segmentation to segment and extract features.

archive_features:
    Archives the extracted features into an Excel file.
    Organizes the data by mice, FOV, and cell.

main:
    Main entry point of the script.
    Processes each dataset and archives the features.
"""
from __init__ import *
from argparse import Namespace
from calcium_data_preprocess import *
import xlwt
from scipy.signal import medfilt, find_peaks
from figures import preprocess_vis, neurosis_analysis, selective_plotting

plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 24


def session_segmentation(
        session_data: Namespace,
        drop_range=20, drop_threshold=3,
        trial_range=-15, trial_length=100, baseline_range=-5,
        spont_pre_duration=100 * 1000, spont_post_duration=500 * 1000, kernel_size=25,
):
    """
    Segments the session data into trials and spontaneous periods.
    Detrends the calcium signals and calculates the median absolute deviation (MAD).

    Parameters
    ----------
    session_data : Namespace
        The session data containing raw calcium traces and other metadata.
    drop_range : int, optional
        The frame range for dropping trials (default is 20).
    drop_threshold : int, optional
        The dropped frame threshold for dropping trials (default is 3).
    trial_range : int, optional
        The frame range for trial selection (default is -15).
    trial_length : int, optional
        The frame number of each trial (default is 100).
    baseline_range : int, optional
        The relative frame for calculating the baseline of df/f0 (default is -5).
    spont_pre_duration : int, optional
        The duration of the pre-stimulus spontaneous period in milliseconds (default is 100000).
    spont_post_duration : int, optional
        The start time of the post-stimulus spontaneous period in milliseconds (default is 500000).
    kernel_size : int, optional
        The median filter kernel size for trend calculation (default is 25).

    Returns
    -------
    None
    """
    # calculate global_baseline, trend, and mad(median absolute deviation)
    global_baseline = []
    for trial_id, trial_onset_time in enumerate(session_data.puff_times):
        trial_onset_frame = int(np.floor(session_length * trial_onset_time / session_duration))
        global_baseline.append(session_data.raw_trace[:, trial_onset_frame + baseline_range:trial_onset_frame])
    session_data.global_baseline = np.mean(np.concatenate(global_baseline, axis=-1), axis=-1, keepdims=True)
    session_data.df_f0 = (session_data.raw_trace - session_data.global_baseline) / session_data.global_baseline
    session_data.trend = np.stack([medfilt(session_data.df_f0[cell_id], kernel_size=kernel_size, )
                                   for cell_id in range(session_data.cell_num)], axis=0)
    session_data.calcium = session_data.df_f0 - session_data.trend  # detrend traces
    session_data.mad = np.median(np.abs(session_data.calcium -
                                        np.median(session_data.calcium, axis=-1, keepdims=True)), axis=-1)

    # extract trials
    session_data.trials = []
    pre_frames = int(np.floor(session_length * spont_pre_duration / session_duration))
    all_sponts = [session_data.calcium[:, :pre_frames], ]
    for trial_id, (trial_onset_time, trial_type) in enumerate(zip(session_data.puff_times, session_data.puff_types)):
        trial_onset_frame = int(np.floor(session_length * trial_onset_time / session_duration))

        # drop trials
        drop_flag = np.sum(session_data.dropped_frames[trial_onset_frame - drop_range:
                                                       trial_onset_frame + drop_range]) >= drop_threshold

        trial_flag = (trial_type == "real") or (trial_type == "Puff")  # trial type

        # make clips
        tmp_signal = session_data.calcium[:, trial_onset_frame + trial_range: trial_onset_frame + trial_length]

        # extract trial
        tmp_trial = Namespace(
            trial_id=trial_id,
            calcium={
                "Pre": tmp_signal[:, :-trial_range],  # -3s ~ 0s relative to trial onset
                "Evoked": tmp_signal[:, -trial_range: -trial_range + 5],  # 0s ~ 1s relative to trial onset
                "Late": tmp_signal[:, -trial_range + 5: -trial_range + 20],  # 1s ~ 4s relative to trial onset
                "Post": tmp_signal[:, -trial_range: -trial_range + 15],  # 0s ~ 3s
                "Spont": tmp_signal[:, -trial_range + 20: trial_length],  # 4s ~ 17s relative to trial onset
            },
            drop_flag=drop_flag,
            puff_trial=trial_flag,
            trial_onset_frame=-trial_range,
        )
        session_data.trials.append(tmp_trial)
        all_sponts.append(tmp_trial.calcium["Spont"])

    # concatenate the spontaneous periods
    post_frames = int(np.floor(session_length * spont_post_duration / session_duration))
    all_sponts.append(session_data.calcium[:, post_frames:])
    session_data.sponts = np.concatenate(all_sponts, axis=-1)


def session_feature_extraction(
        session_data: Namespace,
):
    """
      Extracts features from the segmented session data.
      Calculates peak values, spontaneous event properties, and response fractions.

      Parameters
      ----------
      session_data : Namespace
          The session data containing segmented trials and spontaneous periods.

      Returns
      -------
      None
    """
    #  Initialization
    n_cell = session_data.cell_num
    session_data.cells = {i: {} for i in range(n_cell)}

    #  Per trial Features
    for trial in session_data.trials:
        trial.cells = {i: {} for i in range(n_cell)}
        for cell_id in range(n_cell):
            for window_name in ("Pre", "Evoked", "Late", "Post"):
                trial.cells[cell_id][f"{window_name}_Peak"] = nanmax(trial.calcium[window_name][cell_id])

    n_frame_spont = session_data.sponts.shape[1]
    trial_pos = [int(np.floor(session_length * trial_onset_time / session_duration))
                 for trial_onset_time in session_data.puff_times]
    for cell_id in range(n_cell):
        #  Spontaneous Features
        spont_features = {}
        tmp_spont = session_data.sponts[cell_id]
        peaks_pos, properties = find_peaks(tmp_spont, prominence=10 * session_data.mad[cell_id], width=1, )

        # filter near-trial spontaneous events
        filtered_index = []
        for peak_id in range(len(peaks_pos)):
            filtered_index.append(np.all([peak_neighbor not in trial_pos
                                          for peak_neighbor in np.arange(-5, 5) + peaks_pos[peak_id]]))
        peaks_pos = peaks_pos[filtered_index]

        peaks_num = len(peaks_pos)
        peaks_value = tmp_spont[peaks_pos]
        peaks_prom = properties["prominences"][filtered_index]
        peaks_height = peaks_value - properties['width_heights'][filtered_index]
        peaks_isi = peaks_pos[1:] - peaks_pos[:-1]
        peaks_length = (properties['right_ips'] - properties['left_ips'])[filtered_index]

        for func_name, func in zip(("Mean", "Median"), (nanmean, nanmedian)):
            spont_features[f"Spt_Peak_{func_name}"] = func(peaks_value)
            spont_features[f"Spt_Prom_{func_name}"] = func(peaks_prom)
            spont_features[f"Spt_ISI_{func_name}"] = func(peaks_isi)
            spont_features[f"Spt_Len_{func_name}"] = func(peaks_length)
            spont_features[f"Spt_Height_{func_name}"] = func(peaks_height)

        spont_features[f"Spt_Freq"] = peaks_num / n_frame_spont
        session_data.cells[cell_id].update(spont_features)

        # per trial features
        for response_threshold in (2, 3, 5, 10, 20):
            for critical_period, period_name in zip(("Evoked_Peak", "Late_Peak", "Post_Peak"),
                                                    ("0s1s", "1s4s", "0s3s")):
                criterion_name = f"t{response_threshold}_{period_name}"
                response_flags = [
                    (trial.cells[cell_id][critical_period] / session_data.mad[cell_id]) >= response_threshold
                    for trial in session_data.trials]

                RP = [trial for trial_id, trial in enumerate(session_data.trials)
                      if trial.puff_trial and response_flags[trial_id]]
                RB = [trial for trial_id, trial in enumerate(session_data.trials)
                      if (not trial.puff_trial) and response_flags[trial_id]]
                NP = [trial for trial_id, trial in enumerate(session_data.trials)
                      if trial.puff_trial and (not response_flags[trial_id])]
                NB = [trial for trial_id, trial in enumerate(session_data.trials)
                      if (not trial.puff_trial) and (not response_flags[trial_id])]
                R, N, B, P = RP + RB, NP + NB, RB + NB, RP + NP
                A = RP + RB + NP + NB

                # fraction features
                session_data.cells[cell_id][f"{criterion_name}_RPoP"] = len(RP) / len(P)
                session_data.cells[cell_id][f"{criterion_name}_RBoB"] = len(RB) / len(B)
                session_data.cells[cell_id][f"{criterion_name}_RPmRB"] = len(RP) / len(P) - len(RB) / len(B)
                session_data.cells[cell_id][f"{criterion_name}_RoA"] = len(R) / len(A)

                for type_name, type_trials in zip(("RP", "RB", "P", "B", "R", "A"), (RP, RB, P, B, R, A)):
                    for window_name in ("Pre", "Evoked", "Late", "Post"):
                        all_data = [trial.cells[cell_id][f"{window_name}_Peak"] for trial in type_trials]
                        if type_name in ("RP", "RB", "R"):
                            session_data.cells[cell_id][f"{criterion_name}_{type_name}_{window_name}_Peak"] = nanmean(
                                all_data)
                        else:
                            session_data.cells[cell_id][f"{type_name}_{window_name}_Peak"] = nanmean(all_data)

                session_data.cells[cell_id][f"{criterion_name}_RPmRB_Evoked_Peak"] = session_data.cells[cell_id][
                                                                                         f"{criterion_name}_RP_Evoked_Peak"] - \
                                                                                     session_data.cells[cell_id][
                                                                                         f"{criterion_name}_RB_Evoked_Peak"]
                session_data.cells[cell_id][f"PmB_Evoked_Peak"] = session_data.cells[cell_id][
                                                                      f"P_Evoked_Peak"] - \
                                                                  session_data.cells[cell_id][
                                                                      f"B_Evoked_Peak"]
    if DEBUG_FLAG:
        print(list(session_data.cells[0].keys()))
        exit()


def mice_segmentation(mice_name, fov_id, mice_data: Namespace):
    """
    Segments the data for each mouse into sessions and extracts features.
    Handles specific cases for different mice.

    Parameters
    ----------
    mice_name : str
        The name of the mouse.
    fov_id : int
        The field of view (FOV) identifier.
    mice_data : Namespace
        The data for the mouse, including calcium traces, dropped frames, puff types, and puff times.

    Returns
    -------
    dict
        A dictionary containing the segmented trials and spontaneous periods for each session.
    """
    n_cell, n_frames = mice_data.calcium.shape
    assert len(mice_data.puff_types.keys()) == len(mice_data.puff_times.keys()), \
        f"Arduino.xlsx file doesn't match Arduino time point.xlsx in {mice_name} FOV{fov_id}"
    n_sessions = int(n_frames / session_length)
    session_per_day = 1 if n_sessions <= 16 else 2
    days_in_data = list(range(16))
    days_to_extract = list(range(16))
    segmentation_result = {}

    # remove invalid data
    if mice_name == "M087 - Matt":
        days_to_extract.remove(14)  # remove SAT9
        days_to_extract.remove(15)  # remove SAT10
    elif mice_name == "M088 - Matt":
        days_in_data.remove(12)  # SAT7 lost
    elif mice_name == "M017":
        days_in_data.remove(0)  # ACC1 lost
    # segment trials and spontaneous data
    for extract_day in days_to_extract:
        if extract_day in days_in_data:
            day_id = days_in_data.index(extract_day)
            segmentation_result[extract_day] = []
            for session_id in range(session_per_day):
                prev_session_num = session_per_day * day_id + session_id
                frame_on, frame_off = prev_session_num * session_length, (prev_session_num + 1) * session_length
                session_data = Namespace(
                    mice_name=mice_name, fov_id=fov_id,
                    cell_num=mice_data.cell_num,
                    raw_trace=mice_data.calcium[:, frame_on:frame_off],
                    dropped_frames=mice_data.dropped_frames[frame_on:frame_off],
                    puff_types=mice_data.puff_types[2 * day_id + session_id][:, 0],
                    puff_times=mice_data.puff_times[2 * day_id + session_id][:, 0],
                )

                session_segmentation(session_data)
                session_feature_extraction(session_data)
                segmentation_result[extract_day].append(session_data)
    return segmentation_result


def read_mice(mice_name, dataset_path):
    """
    Reads and processes data for each mouse from the dataset.

    Parameters
    ----------
    mice_name : str
        The name of the mouse.
    dataset_path : str
        The path to the dataset.

    Returns
    -------
    dict
        A dictionary containing the segmented trials and spontaneous periods for each field of view (FOV).
    """
    mice_features = {}
    for fov_name in os.listdir(path.join(dataset_path, mice_name)):
        fov_id = int(fov_name[3:])
        fov_path = path.join(dataset_path, mice_name, fov_name)

        # files existence check
        for file_name in ("Arduino.xlsx", f"Arduino P6_{fov_id}.xlsx", "Arduino time point.xlsx",
                          f"Arduino time point P6_{fov_id}.xlsx", "Fall.mat"):
            if not path.exists(path.join(fov_path, file_name)):
                print(f"File {path.join(fov_path, file_name)} not found!")
        print(f"Processing {mice_name} FOV{fov_id}")

        # extract data from files
        mice_data = Namespace()
        raw_data = loadmat(path.join(fov_path, "Fall.mat"))
        raw_data, mice_data.dropped_frames, _ = drop_frames(raw_data)
        mice_data.calcium = df_f_calculation(raw_data)
        mice_data.puff_times = read_xlsx(path.join(fov_path, "Arduino time point.xlsx"), header=0)
        mice_data.puff_types = read_xlsx(path.join(fov_path, "Arduino.xlsx"), header=None)
        mice_data.cell_num = mice_data.calcium.shape[0]

        # extract trials and spontaneous period and features
        mice_features[fov_name] = mice_segmentation(mice_name, fov_id, mice_data)
    return mice_features


def archive_features(all_features, dataset_name, save_path):
    """
    Archives the extracted features into an Excel file.

    Parameters
    ----------
    all_features : dict
        A dictionary containing the features for all mice.
    dataset_name : str
        The name of the dataset.
    save_path : str
        The path where the Excel file will be saved.

    Returns
    -------
    None
    """
    wb = xlwt.Workbook()
    code = "PSE" if "PSE" in dataset_name else "SAT"
    titles = ["Mice", "FOV", "Cell"] + [f"ACC{j + 1}" for j in range(6)] + [f"{code}{j + 1}" for j in range(10)]
    for feature_name in TotalFeatureNames:
        ws = wb.add_sheet(feature_name)
        for i, title in enumerate(titles):
            ws.write(0, i, title)
        cell_cnt = 1
        for mice_name in all_features.keys():
            for fov_name in all_features[mice_name].keys():
                fov_data = all_features[mice_name][fov_name]
                valid_days = list(fov_data.keys())
                cell_num = fov_data[valid_days[0]][0].cell_num
                for cell_id in range(cell_num):
                    for i, title_value in enumerate([mice_name, fov_name, f"cell {cell_cnt}"]):
                        ws.write(cell_cnt, i, title_value)
                    for extract_day in valid_days:
                        tmp_list = []
                        for session_data in fov_data[extract_day]:
                            tmp_list.append(session_data.cells[cell_id].get(feature_name, np.nan))
                        tmp_value = nanmean(tmp_list)
                        ws.write(cell_cnt, extract_day + 3, "NaN" if np.isnan(tmp_value) else float(tmp_value))
                    cell_cnt += 1
    wb.save(path.join(save_path, f"{dataset_name}.xls"))


def main():
    """
    Main entry point of the script.
    Processes each dataset and archives the features.

    Returns
    -------
    None
    """
    raw_data_path = path.join("..", "data", "Calcium imaging")
    for dataset_name in os.listdir(raw_data_path):
        dataset_path = path.join(raw_data_path, dataset_name)
        if path.isdir(dataset_path):
            if dataset_name in ("Calb2_SAT", "Ai148_SAT", "Ai148_PSE"):
                all_features = {}
                for mice_name in os.listdir(dataset_path):
                    all_features[mice_name] = read_mice(mice_name, dataset_path)

                if DEBUG_FLAG:
                    preprocess_vis(all_features, dataset_name)
                    neurosis_analysis(all_features, dataset_name)
                    selective_plotting(all_features, dataset_name)
                else:
                    archive_features(all_features, dataset_name, raw_data_path)


if __name__ == "__main__":
    main()
