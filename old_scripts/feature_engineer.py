"""
Feature Engineering and Clustering Script

This script is responsible for feature selection and searching.

Key Functions:
- get_cluster_features: Generates a list of feature names based on a provided fetch_id.
- main: The main function that performs clustering and generates plots for the given features.
"""


from __init__ import *
from data_preprocess import get_data_main
from figures import *
from clusters import *


def get_cluster_features(fetch_id=None):
    """
    Generates a list of feature names based on the provided fetch_id.

    Parameters:
    fetch_id (int, optional): An identifier to select a specific set of features.
                              If None, returns a default list of feature names.

    Returns:
    tuple: A tuple containing:
        - feature_name (str): A short identifier for the selected feature set.
        - add_list_names (list): A list of feature names with the suffix " (ACC4~ACC6)".
    """
    # # all features
    # list_names = [feature_name + f" (ACC4~ACC6)" for feature_name in TotalFeatureNames]

    # chosen list_names
    list_names = [feature_name + f" (ACC4~ACC6)" for feature_name in [
        'P_Pre_Peak', 'P_Evoked_Peak', 'P_Late_Peak', 'P_Post_Peak',
        'B_Pre_Peak', 'B_Evoked_Peak', 'B_Late_Peak', 'B_Post_Peak',
        'A_Pre_Peak', 'A_Evoked_Peak', 'A_Late_Peak', 'A_Post_Peak',
        't2_0s1s_RPoP', 't3_0s1s_RPoP', 't5_0s1s_RPoP', 't10_0s1s_RPoP', 't20_0s1s_RPoP',
        't2_0s1s_RoA', 't3_0s1s_RoA', 't5_0s1s_RoA', 't10_0s1s_RoA', 't20_0s1s_RoA',
    ]]

    # Feature categories
    spont_features = [
        'Spt_Peak_Mean', 'Spt_Prom_Mean', 'Spt_ISI_Mean', 'Spt_Len_Mean', 'Spt_Height_Mean', 'Spt_Peak_Median',
        'Spt_Prom_Median', 'Spt_ISI_Median', 'Spt_Len_Median', 'Spt_Height_Median', 'Spt_Freq', ]
    evoked_features = [
        'P_Pre_Peak', 'P_Evoked_Peak', 'P_Late_Peak', 'P_Post_Peak',
        'B_Pre_Peak', 'B_Evoked_Peak', 'B_Late_Peak', 'B_Post_Peak',
        'A_Pre_Peak', 'A_Evoked_Peak', 'A_Late_Peak', 'A_Post_Peak',
    ]
    prob_features = [
        't2_0s1s_RPoP', 't3_0s1s_RPoP', 't5_0s1s_RPoP', 't10_0s1s_RPoP', 't20_0s1s_RPoP',
        't2_0s1s_RoA', 't3_0s1s_RoA', 't5_0s1s_RoA', 't10_0s1s_RoA', 't20_0s1s_RoA',
    ]

    # Return specific feature set based on fetch_id
    if fetch_id is not None:
        for feature_id, (feature_name, list_names) in enumerate(zip(
                ("spt", "evo", "pro", "se", "sp", "ep", "sep"),
                (spont_features, evoked_features, prob_features, spont_features+evoked_features, spont_features+prob_features, evoked_features+prob_features,
                    spont_features+evoked_features+prob_features
                 )
        )):
            if feature_id == fetch_id:
                add_list_names = [feature_name + f" (ACC4~ACC6)" for feature_name in list_names]
                return feature_name, add_list_names
    else:
        return "default", list_names


def main(dataset_name, fetch_id):
    """
    Main function to perform clustering and generate plots for the given dataset.

    Parameters:
    dataset_name (str): The name of the dataset to process.
    fetch_id (int): The identifier to select a specific set of features.
    """
    # Create directories for saving figures
    save_pth = os.path.join("..", "figures", dataset_name)
    os.makedirs(save_pth, exist_ok=True)

    # Extract code from dataset name
    code = dataset_name.split("_")[-1]
    data_dict = get_data_main(dataset_name)

    # Clustering parameters
    clustering_params = {
        "solver_name": 'UMAP',
        "n_neighbors": 8,
        "random_seed_num": 10,
        "normalization": 'zscore',
    }
    os.makedirs(path.join(save_pth, "clusters"), exist_ok=True)
    new = True

    if new:
        # Get features for clustering
        f_name, clustered_feature = get_cluster_features(fetch_id)
        # Save cluster results
        task_id = save_cluster_results(data_dict=data_dict, clustered_features=clustered_feature,
                                       save_path=os.path.join(save_pth, "clusters"), **clustering_params)
        # Load cluster results
        result_dict = load_cluster_result(save_path=os.path.join(save_pth, "clusters"), task_id=task_id)

        # Plot multiple decomposition data
        plot_multiple_decomposition_data(save_path=os.path.join(save_pth, f"all_random_seed_{f_name}.jpg"),
                                         label_name="labels", data_dict=data_dict,
                                         decomposition_dict=result_dict)
    else:
        task_id = 'new'
        result_dict = load_cluster_result(save_path=path.join(save_pth, "clusters"), task_id=task_id)

    # Get labeling
    data_dict["labels"], decomposed_data = labeling(dataset_name, result_dict)

    # Plot decomposition data
    plot_decomposition_data(save_path=os.path.join(save_pth, "clusters_labels.jpg"), label_name="labels",
                            data_dict=data_dict, decomposed_data=decomposed_data,)

    if dataset_name == "Calb2_SAT":
        plot_decomposition_data(save_path=path.join(save_pth, "clusters_Calb2.jpg"), label_name="Calb2",
                                data_dict=data_dict, decomposed_data=decomposed_data)
        plot_histogram(path.join(save_pth, "hist_calb2.jpg"), data_dict['labels'], data_dict['Calb2'])
        plot_pie(path.join(save_pth, "pie_calb2.jpg"), data_dict['labels'], data_dict['Calb2'])

    # # Code for renormalization dataset
    # re_norm_dict, _ = get_renormalize_data(dataset_name)
    # for feature_name in ("ACC6", "SAT10", "HC1", "SAT10 div ACC6", "HC1 div ACC6", "HC1 div SAT10"):
    #     color_decomposition(path.join(save_pth, f"re_norm_{feature_name}.jpg"), feature_name,
    #                         re_norm_dict[feature_name], decomposed_data,
    #                         calb2=data_dict['Calb2'] if dataset_name=="Calb2_SAT" else None)
    #
    # mice_list, fov_list = read_mice_fov(dataset_name)
    # label_decomposition(path.join(save_pth, f"label_mice.jpg"), mice_list, decomposed_data)
    # label_decomposition(path.join(save_pth, f"label_fov.jpg"), fov_list, decomposed_data)


if __name__ == "__main__":
    # Run main function for different datasets and feature sets
    for f_id in range(9):
        main("Calb2_SAT", f_id)
        main("Ai148_PSE", f_id)
        main("Ai148_SAT", f_id)
