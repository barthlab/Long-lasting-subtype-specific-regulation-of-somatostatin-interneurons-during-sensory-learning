"""
This module provides functions to preprocess data for the Calb2 paper project.
It includes functions to read and process data from Excel files, annotate the data,
and perform renormalization.

Functions:
- read_mice_fov(dataset_name): Reads mice field of view data from an Excel file.
- read_data(dataset_name): Reads raw feature data from an Excel file.
- annotate_data(data_dict): Annotates the data with day labels and fold changes.
- read_expression(dataset_name): Reads Calb2 expression data from an Excel file.
- get_renormalize_data(dataset_name): Renormalizes data based on reference peaks.
- get_data_main(dataset_name): Main function to read and annotate data.
"""

from __init__ import *
import pandas as pd


def read_mice_fov(dataset_name):
    """
    Reads mice field of view (FOV) data from an Excel file.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        tuple: A tuple containing two lists, mice_list and fov_list.
    """
    xl_file = pd.ExcelFile(path.join("..", "data", f"Feature_{dataset_name}.xls"))
    if DEBUG_FLAG:
        print(f"reading dataset {dataset_name}")
    mice_list, fov_list = [], []
    feature_name = xl_file.sheet_names[0]

    tmp_matrix = xl_file.parse(feature_name, header=0).to_numpy()
    n_cell, n_day_p3 = tmp_matrix.shape
    for cell_id in range(n_cell):
        assert tmp_matrix[cell_id, 2] == f"cell {cell_id + 1}", \
            f"wrong format at dataset {dataset_name} feature {feature_name} cell {cell_id}"
        mice_list.append(tmp_matrix[cell_id, 0])
        fov_list.append(tmp_matrix[cell_id, 0]+tmp_matrix[cell_id, 1])
    mice_list = np.delete(mice_list, RemovedCellIdx[dataset_name], axis=0)
    fov_list = np.delete(fov_list, RemovedCellIdx[dataset_name], axis=0)
    return mice_list, fov_list


def read_data(dataset_name):
    """
    Reads raw feature data from an Excel file.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        dict: A dictionary containing the feature data.
    """
    xl_file = pd.ExcelFile(path.join("..", "data", f"Feature_{dataset_name}.xls"))
    if DEBUG_FLAG:
        print(f"reading dataset {dataset_name}")
    data_dict = {"dataset name": dataset_name}
    for feature_name in xl_file.sheet_names:
        if feature_name in TotalFeatureNames:
            tmp_matrix = xl_file.parse(feature_name, header=0).to_numpy()
            n_cell, n_day_p3 = tmp_matrix.shape
            for cell_id in range(n_cell):
                assert tmp_matrix[cell_id, 2] == f"cell {cell_id+1}", \
                    f"wrong format at dataset {dataset_name} feature {feature_name} cell {cell_id}"
            if DEBUG_FLAG:
                print(f"feature {feature_name} {n_cell} cells {n_day_p3-3} days")
            value_matrix = np.array(tmp_matrix[:, 3:], dtype=np.float64)

            # remove cells
            value_matrix = np.delete(value_matrix, RemovedCellIdx[dataset_name], axis=0)

            # replace nan value with 0
            value_matrix[np.isnan(value_matrix)] = 0.

            data_dict[feature_name] = value_matrix
        else:
            print(f"Cant read {feature_name} in {dataset_name}, not included in total feature names")
    return data_dict


def annotate_data(data_dict):
    """
    Annotates the data with day labels and fold changes.

    Args:
        data_dict (dict): The dictionary containing the feature data.

    Returns:
        dict: A dictionary containing the annotated data.
    """
    annotated_dict = {"dataset name": data_dict["dataset name"]}
    code = data_dict['dataset name'].split("_")[-1]
    day_labels = [f"ACC{i+1}" for i in range(6)] + [f"{code}{i+1}" for i in range(10)]
    # annotate feature for each day & fold change
    for feature_name in TotalFeatureNames:
        assert data_dict[feature_name].shape[1] == 16, f"16 days data required, {data_dict[feature_name].shape[1]} days found in {feature_name}"
        # each day features
        for day_id, day_name in enumerate(day_labels):
            annotated_dict[feature_name+f" ({day_name})"] = data_dict[feature_name][:, day_id]
        for duration in range(1, 3):
            for day_id, day_name in enumerate(day_labels[:-duration]):
                annotated_dict[feature_name+f" ({day_name}~{day_labels[day_id+duration]})"] = \
                    np.nanmean(data_dict[feature_name][:, day_id:day_id+duration+1], axis=-1)
        # fold change features
        for baseline_day in ("ACC1~ACC3", "ACC4~ACC6"):
            for day_id, day_name in enumerate(day_labels):
                annotated_dict[feature_name + f" ({day_name} / {baseline_day})"] = data_dict[feature_name][:, day_id] / \
                        annotated_dict[feature_name+f" ({baseline_day})"]
                annotated_dict[feature_name + f" ({day_name} - {baseline_day})"] = data_dict[feature_name][:, day_id] - \
                        annotated_dict[feature_name+f" ({baseline_day})"]
            for duration in range(1, 3):
                for day_id, day_name in enumerate(day_labels[:-duration]):
                    annotated_dict[feature_name + f" ({day_name}~{day_labels[day_id + duration]} / {baseline_day})"] = \
                        annotated_dict[feature_name+f" ({day_name}~{day_labels[day_id+duration]})"] / \
                        annotated_dict[feature_name+f" ({baseline_day})"]
                    annotated_dict[feature_name + f" ({day_name}~{day_labels[day_id + duration]} - {baseline_day})"] = \
                        annotated_dict[feature_name+f" ({day_name}~{day_labels[day_id+duration]})"] - \
                        annotated_dict[feature_name+f" ({baseline_day})"]
    if DEBUG_FLAG:
        print("Annotated: ", list(annotated_dict.keys()))
    return annotated_dict


def read_expression(dataset_name):
    """
    Reads Calb2 expression data from an Excel file.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        dict: A dictionary containing the expression data.
    """
    xl_file = pd.ExcelFile(path.join("..", "data", f"Expression_{dataset_name}.xlsx"))
    if DEBUG_FLAG:
        print(f"reading expression {dataset_name}")
    tmp_matrix = xl_file.parse("CR", header=None).to_numpy()
    col_names = tmp_matrix[0, 2:]
    expression_dict = {}
    for col_id, col_name in enumerate(col_names):
        value_matrix = np.array(tmp_matrix[1:, col_id+2], dtype=np.float64)
        value_matrix = np.delete(value_matrix, RemovedCellIdx[dataset_name], axis=0)  # remove cells
        expression_dict[f"Calb2 {col_name}"] = value_matrix

    if DEBUG_FLAG:
        print(f"expression loaded: {list(expression_dict.keys())}")

    # criterion for Calb2 positive
    expression_dict["Calb2"] = expression_dict["Calb2 Mean"] > 200.
    n_total = len(expression_dict["Calb2"])
    n_pos = np.sum(expression_dict["Calb2"])
    n_neg = n_total - n_pos
    print(f"Total: {n_total}, SST-Calb2: {n_pos}, SST-O: {n_neg}")
    return expression_dict


def get_renormalize_data(dataset_name):
    """
    Access to the renormalized data after 10 days acclimation

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        tuple: A tuple containing the renormalized reference peaks and labels.
    """
    if dataset_name == "Ai148_SAT":
        labels = [0 for _ in range(66)] + [1 for _ in range(67, 83)] + [0, 1, 0, 1, 0, ] + [1 for _ in range(88, 99)]
        ref_peaks = {}
        ref_peaks["ACC6"] = [0 for _ in range(66)] + [0.139544485, 0.046322121, 0.855240838, 0.492000584, 3.37527756, 0.227258264, 0.838615126, 0.474377163, 0.143400788, 0.444430457, 0.610413967, 0.509878463, 0.243368414, 0.213450992, 0.047587304, 0.024460085, 0, 0.013303627, 0, 0.105958523, 0, 0.066365219, 0.249555668, 0.060398022, 0.062752507, 0.608731632, 0.089421566, 0.08033853, 0.157313473, 0.060171939, 0.214190962, 0.098305458]
        ref_peaks["SAT10"] = [0 for _ in range(66)] + [0.136150849, 0.039884041, 0.388858042, 0.315982099, 1.646886984, 0.604597942, 0.233800434, 0.166338692, 0.053254511, 0.218576498, 0.056366097, 0.239029099, 0.110834751, 0.126350974, 0.099566144, 0.051689689, 0, 0.010360005, 0, 0.110025363, 0, 0.048224142, 0.003259297, 0.11217232, 0.120339366, 0.075371834, 0.086020935, 0.078216152, 0.053112407, 0.069398922, 0.041065653, 0.085796739]
        ref_peaks["HC1"] = [0 for _ in range(66)] + [0.305959926, 0.1773871, 0.769957026, 0.523126141, 2.31752659, 0.482925754, 0.133046447, 1.110244137, 0.191291034, 0.756847967, 0.107460745, 0.807680619, 0.184148509, 0.977911439, 0.054239368, 0.01479087, 0, 0.060381124, 0, 0.128173399, 0, 0.032363116, 0.514009837, 0.443269303, 3.702558274, 0.298434118, 0.104403684, 0.094200488, 0.178432365, 0.277817539, 0.132868949, 0.129021311]

    elif dataset_name == "Calb2_SAT":
        labels = [0 for _ in range(59)] + [1. for _ in range(60, 85)]
        ref_peaks = {}
        ref_peaks["ACC6"] = [0 for _ in range(59)] + [0.03999719, 0.205036762, 0.1518237, 0.041138888, 0.370249571, 0.278156364, 0.053702751, 0.928533767, 0.048354374, 0.26433057, - 0.0038356, 0.279549049, 0.046388117, 0.526424588, 0.068323334, 0.078007035, 0.018702155, 0.044311362, 0.128779142, 0.074120118, 0.004892661, 0.04534995, 0.995268538, 0.082406125, 0.178145019]
        ref_peaks["SAT10"] = [0 for _ in range(59)] + [0.148065655, 0.121800624, 1.79983136, 0.075047506, 0.865650297, 0.077721868, 0.090561511, 0.849133894, 0.793044998, 0.539838584, 0.037872713, 0.146994175, 0.096543622, 0.06217866, 0.010552428, 0.001663915, 0.002960194, 0.01472772, 0.022156276, 0.023405503, - 0.00262426, 0.135872804, 0.073056157, 0.0228016, - 0.055850294]
        ref_peaks["HC1"] = [0 for _ in range(59)] + [1.174414604, 2.974574069, 0.00935474, 2.37071929, 0.479673841, 1.310539283, 4.1585953, 0.355731694, 0.329267482, 0.664120509, 0.14700847, 5.61421906, 2.502696598, 1.984786453, 0.234027598, 0.967430909, 1.31179283, 0.048670348, 0.097793143, 0.130747116, 0.080209889, 0.094130465, 0.030836878, 0.201803106, 2.682589105]
    else:
        raise NotImplementedError
    # check validity of data read
    for cell_id in range(len(labels)):
        if labels[cell_id] == 1:
            for day_name in ref_peaks.keys():
                assert ref_peaks[day_name][cell_id] != 0, f"{day_name} {cell_id} {ref_peaks[day_name][cell_id]}"
        else:
            for day_name in ref_peaks.keys():
                assert ref_peaks[day_name][cell_id] == 0, f"{day_name} {cell_id} {ref_peaks[day_name][cell_id]}"

    labels = np.delete(labels, RemovedCellIdx[dataset_name], axis=0)  # remove cells

    for day_name in ref_peaks.keys():
        ref_peaks[day_name] = np.delete(ref_peaks[day_name], RemovedCellIdx[dataset_name], axis=0)
    ref_peaks["SAT10 div ACC6"] = ref_peaks["SAT10"]/ref_peaks["ACC6"]
    ref_peaks["HC1 div ACC6"] = ref_peaks["HC1"]/ref_peaks["ACC6"]
    ref_peaks["HC1 div SAT10"] = ref_peaks["HC1"]/ref_peaks["SAT10"]
    return ref_peaks, labels


def get_data_main(dataset_name):
    """
    Main function to read and annotate data.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        dict: A dictionary containing the annotated data.
    """
    feature_dict = read_data(dataset_name)
    annotated_dict = annotate_data(feature_dict)
    if dataset_name == "Calb2_SAT":
        annotated_dict.update(read_expression(dataset_name))
    return annotated_dict




