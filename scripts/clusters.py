"""
This module provides functions and classes for clustering and dimensionality reduction
using various techniques such as PCA, t-SNE, UMAP, and custom methods. It also includes
functions for saving and loading clustering results, and labeling clusters.

Classes:
    Plain: A placeholder class for plain data transformation.

Functions:
    param2str(*args): Converts parameters to a string representation.
    predict_test(input_data, input_feature_names, output_data): Performs logistic regression and prints feature weights.
    save_cluster_results(data_dict, clustered_features, save_path, solver_name="UMAP", random_seed_num=5, normalization="min_max", **kwargs): Saves clustering results to .mat files.
    load_cluster_result(save_path, task_id=None): Loads clustering results from .mat files.
    labeling(dataset_name, result_dict): Labels clusters based on the dataset name.
"""

from __init__ import *
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import re
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression


class Plain:
    """
    A placeholder class for plain data transformation.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def fit_transform(x):
        """
        Returns the input data as is, after asserting it has 2 features.

        Args:
            x (numpy.ndarray): Input data with shape (n_samples, 2).

        Returns:
            numpy.ndarray: The input data.
        """
        assert x.shape[-1] == 2
        return x


# Dictionary mapping solver names to their corresponding classes
SOLVERS = {
    "PCA": PCA,
    "tSNE": TSNE,
    "UMAP": UMAP,
    "Plain": Plain
}


def param2str(*args):
    """
    Converts parameters to a string representation.

    Args:
        *args: Variable length argument list of parameters.

    Returns:
        str: String representation of the parameters.
    """
    final_str = ""
    for param_id, param in enumerate(args):
        if param_id > 0:
            final_str += "__"
        if isinstance(param, str):
            final_str += param
        elif isinstance(param, dict):
            tmp_str = ""
            for key, value in param.items():
                tmp_str += str(key) + str(value) + "_"
            final_str += tmp_str[:-1]
        elif isinstance(param, list):
            tmp_str = ""
            for value in param:
                tmp_str += str(value) + "_"
            final_str += tmp_str[:-1]
        else:
            final_str += str(param)
    return re.sub(r'[^A-Z0-9_]', '', final_str)


def predict_test(input_data, input_feature_names, output_data):
    """
    Performs logistic regression and prints feature weights and sorted feature names.

    Args:
        input_data (numpy.ndarray): Input data for the model.
        input_feature_names (list): List of feature names.
        output_data (numpy.ndarray): Output data for the model.
    """
    shout_out("Prediction Test")
    clf = LogisticRegression(random_state=2, penalty='l2').fit(input_data, output_data)
    feature_weights = clf.coef_[0]
    feature_sort_index = np.argsort(np.abs(feature_weights))[::-1]
    sorted_feature_names = [input_feature_names[feature_id] for feature_id in feature_sort_index]
    print(input_feature_names)
    print(sorted_feature_names)
    print(feature_weights[feature_sort_index])
    print(clf.score(input_data, output_data))


def save_cluster_results(data_dict, clustered_features, save_path,
                         solver_name="UMAP",
                         random_seed_num=5,
                         normalization="min_max",
                         **kwargs):
    """
    Saves clustering results to .mat files.

    Args:
        data_dict (dict): Dictionary containing the dataset.
        clustered_features (list): List of features to be clustered.
        save_path (str): Path to save the results.
        solver_name (str): Name of the solver to use for dimensionality reduction.
        random_seed_num (int): Number of random seeds to use.
        normalization (str): Normalization method to use.
        **kwargs: Additional keyword arguments for the solver.

    Returns:
        str: Task ID of the saved clustering results.
    """
    # preparation
    assert solver_name in SOLVERS.keys(), f"Solver {solver_name} not found!"
    n_components = 2
    code = data_dict['dataset name'].split("_")[-1]
    cluster_array = np.stack([data_dict[key] for key in clustered_features], axis=-1)
    if normalization == "min_max":
        normed_data = ((cluster_array - np.nanmin(cluster_array, axis=0, keepdims=True)) /
                       (np.nanmax(cluster_array, axis=0, keepdims=True) -
                        np.nanmin(cluster_array, axis=0, keepdims=True) + 1e-5))
    elif normalization == "zscore":
        normed_data = ((cluster_array - np.nanmean(cluster_array, axis=0, keepdims=True)) /
                       np.nanstd(cluster_array, axis=0, keepdims=True))
    else:
        raise NotImplementedError

    predict_test(input_data=normed_data,
                 input_feature_names=clustered_features,
                 output_data=data_dict[f"{PlasticityFeature} ({code}5~{code}6 / ACC4~ACC6)"] >= 1)

    task_id = str(time.monotonic_ns())
    os.makedirs(path.join(save_path, task_id), exist_ok=True)
    for random_seed_id in range(random_seed_num):
        mat_file_path = path.join(save_path, task_id, f"{random_seed_id}.mat")
        decomposed_data = SOLVERS[solver_name](n_components=n_components,
                                               random_state=random_seed_id, **kwargs).fit_transform(normed_data)
        savemat(mat_file_path,
                {'decomposed_data': decomposed_data,
                 'solver_name': solver_name,
                 'random_seed_id': random_seed_id,
                 'normalization': normalization,
                 'features': clustered_features,
                 'other_kwargs': kwargs})
    shout_out(f"Clustering Saved {task_id}", end=True)
    return task_id


def load_cluster_result(save_path, task_id=None):
    """
    Loads clustering results from .mat files.

    Args:
        save_path (str): Path to load the results from.
        task_id (str, optional): Task ID of the clustering results. If None, loads the latest task.

    Returns:
        dict: Dictionary containing the loaded clustering results.
    """
    if task_id is None:
        task_idx = [int(d) for d in os.listdir(save_path) if os.path.isdir(path.join(save_path, d))]
        task_id = str(sorted(task_idx)[-1])

    shout_out(f"Loading Clustering {task_id}")
    result_dict = {}
    for file_name in os.listdir(path.join(save_path, task_id)):
        if ".mat" in file_name:
            mat_dict = loadmat(path.join(save_path, task_id, file_name))
            result_dict[int(mat_dict['random_seed_id'])] = mat_dict['decomposed_data']
            if len(result_dict.keys()) == 1:
                print(mat_dict['features'])
    return result_dict


def labeling(dataset_name, result_dict):
    """
    Labels clusters based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        result_dict (dict): Dictionary containing the clustering results.

    Returns:
        tuple: Tuple containing the labels and decomposed data.
    """
    if dataset_name == "Calb2_SAT":
        decomposed_data = result_dict[9]
        decomposed_data[:, 0] *= -1
        decomposed_data = decomposed_data[:, ::-1]
        labels = SpectralClustering(n_clusters=3, random_state=0).fit(decomposed_data).labels_
        labels[np.argwhere(labels == 1)] = 3
        labels[np.argwhere(labels == 2)] = 1
        labels[np.argwhere(labels == 3)] = 2
    elif dataset_name == "Ai148_SAT":
        decomposed_data = result_dict[0]
        labels = SpectralClustering(n_clusters=3, random_state=0).fit(decomposed_data).labels_
        labels[np.argwhere(labels == 0)] = 3
        labels[np.argwhere(labels == 1)] = 0
        labels[np.argwhere(labels == 2)] = 1
        labels[np.argwhere(labels == 3)] = 2
    elif dataset_name == "Ai148_PSE":
        decomposed_data = result_dict[2]
        labels = SpectralClustering(n_clusters=3, random_state=0).fit(decomposed_data).labels_
        labels[np.argwhere(labels == 1)] = 3
        labels[np.argwhere(labels == 2)] = 1
        labels[np.argwhere(labels == 3)] = 2
    else:
        raise NotImplementedError
    for i in range(5):
        print(f"cluster {i}", f"n={np.sum(labels == i)}", ": ", np.argwhere(labels == i)[:, 0])
    return labels, decomposed_data
