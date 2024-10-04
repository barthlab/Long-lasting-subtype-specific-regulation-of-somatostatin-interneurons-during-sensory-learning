from __init__ import *
from data_preprocess import get_data_main, get_renormalize_data
from figures import *
from clusters import *


def multiple_ttest(ssto, outlier, sstcalb2):
    """
    Perform multiple t-tests and print the results.

    Args:
        ssto (array-like): Data for SST-O.
        outlier (array-like): Data for outliers.
        sstcalb2 (array-like): Data for SST-Calb2.
    """
    from statsmodels.stats.multitest import multipletests
    # _, poutlier = ttest_ind(sstcalb2, outlier,)
    # _, po = ttest_ind(sstcalb2, ssto)
    # _, pvals_corrected, _, _ = multipletests([poutlier, po], alpha=0.05, method='bonferroni')
    # print("paired t-test with Bonferroni correction:")
    # print(f"Corrected p-value for fold change SST-Calb2 (Put.) < SST-O (Put.): {pvals_corrected[1]:.3f}")
    # print(f"Corrected p-value for fold change SST-Calb2 (Put.) < Outlier: {pvals_corrected[0]:.3f}")
    _, pinc = ttest_ind(sstcalb2, np.concatenate([ssto, outlier]))
    print(f"With outlier included in SST-O, p-value for fold change SST-Calb2 (Put.) < SST-O: {pinc:.3f}")


def main(dataset_name):
    """
    Main function to process the dataset and generate various plots.

    Args:
        dataset_name (str): The name of the dataset to process.
    """
    save_pth = path.join("..", "figures", dataset_name)
    os.makedirs(save_pth, exist_ok=True)

    code = dataset_name.split("_")[-1]
    data_dict = get_data_main(dataset_name)

    task_id = 'new'
    result_dict = load_cluster_result(save_path=path.join(save_pth, "clusters"), task_id=task_id)
    data_dict["labels"], decomposed_data = labeling(dataset_name, result_dict)

    plot_decomposition_data(save_path=path.join(save_pth, "clusters_labels.jpg"), label_name="labels",
                            data_dict=data_dict, decomposed_data=decomposed_data, )
    if dataset_name == "Calb2_SAT":
        plot_decomposition_data(save_path=path.join(save_pth, "clusters_Calb2.jpg"), label_name="Calb2",
                                data_dict=data_dict, decomposed_data=decomposed_data)
        plot_histogram(path.join(save_pth, "hist_calb2.jpg"), data_dict['labels'], data_dict['Calb2'])
        plot_pie(path.join(save_pth, "pie_calb2.jpg"), data_dict['labels'], data_dict['Calb2'])

    plain_day_names = (("ACC1", "ACC2", "ACC3"), ("ACC4", "ACC5", "ACC6"),
                       (f"{code}1~{code}2",), (f"{code}5~{code}6",), (f"{code}9~{code}10",),)
    stacked_day_names = ((("ACC1", "ACC2", "ACC3"), ("ACC4", "ACC5", "ACC6"),),
                        ((f"{code}1~{code}2",), (f"{code}5~{code}6",), (f"{code}9~{code}10",)),)
    if dataset_name == "Calb2_SAT":
        for feature_name in CoolFeatureNames:
            feature_plot(path.join(save_pth, f"Feature_{feature_name}_gt_ACC4~6.jpg"), {
                "SST-O": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["Calb2"] == 0].reshape(-1),
                "SST-Calb2": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["Calb2"] == 1].reshape(-1),
            })
            feature_plot(path.join(save_pth, f"Feature_{feature_name}_put_ACC4~6.jpg"), {
                "SST-O2 (Put.)": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["labels"] == 2].reshape(-1),
                "SST-O (Put.)": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["labels"] == 0].reshape(-1),
                "SST-Calb2 (Put.)": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["labels"] == 1].reshape(-1),
            })

        for day_set in stacked_day_names:
            data_list, name_list = [], []
            for days in day_set:
                all_data = np.stack([data_dict[f"{PlasticityFeature} ({day} / ACC4~ACC6)"] for day in days], axis=-1)
                data_list.extend([all_data[data_dict["Calb2"] == 0].reshape(-1),
                                  all_data[data_dict["Calb2"] == 1].reshape(-1),
                                  None])
                name_list.extend(["SST-O", "SST-Calb2", None])
            bar_plot_minus(path.join(save_pth, f"FoldChange {day_set} gt div ACC4~ACC6.jpg"), data_list, name_list)

        for day_set in stacked_day_names:
            data_list, name_list = [], []
            for days in day_set:
                all_data = np.stack([data_dict[f"{PlasticityFeature} ({day} / ACC4~ACC6)"] for day in days], axis=-1)
                data_list.extend([
                    all_data[(data_dict["labels"] == 0) | (data_dict["labels"] == 2)].reshape(-1),
                    all_data[data_dict["labels"] == 1].reshape(-1),
                    None, ])
                name_list.extend(["SST-O (Put.)", "SST-Calb2 (Put.)", None, ])
            bar_plot_minus(path.join(save_pth, f"FoldChange {day_set} put chodl_include div ACC4~ACC6.jpg"), data_list,
                           name_list)

        for days in plain_day_names:
            all_data = np.stack([data_dict[f"{PlasticityFeature} ({day} / ACC4~ACC6)"] for day in days], axis=-1)
            print(days)
            _, p_value = ttest_ind(all_data[data_dict["Calb2"] == 1].reshape(-1),
                                   all_data[data_dict["Calb2"] == 0].reshape(-1), alternative="less")
            print(f"t-test for fold change SST-Calb2 < SST-O: {p_value:.3f}")
            multiple_ttest(all_data[data_dict["labels"] == 0].reshape(-1),
                           all_data[data_dict["labels"] == 2].reshape(-1),
                           all_data[data_dict["labels"] == 1].reshape(-1), )
    else:
        for feature_name in CoolFeatureNames:
            feature_plot(path.join(save_pth, f"{feature_name}_ACC4~6.jpg"), {
                "SST-O2 (Put.)": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["labels"] == 2].reshape(-1),
                "SST-O (Put.)": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["labels"] == 0].reshape(-1),
                "SST-Calb2 (Put.)": data_dict[feature_name + " (ACC4~ACC6)"][data_dict["labels"] == 1].reshape(-1),
            })
        for day_set in stacked_day_names:
            data_list, name_list = [], []
            for days in day_set:
                all_data = np.stack([data_dict[f"{PlasticityFeature} ({day} / ACC4~ACC6)"] for day in days], axis=-1)
                data_list.extend([
                    all_data[(data_dict["labels"] == 0) | (data_dict["labels"] == 2)].reshape(-1),
                    all_data[data_dict["labels"] == 1].reshape(-1),
                    None, ])
                name_list.extend(["SST-O (Put.)", "SST-Calb2 (Put.)", None, ])
            bar_plot_minus(path.join(save_pth, f"FoldChange {day_set} put chodl_include div ACC4~ACC6.jpg"),
                           data_list, name_list)

        for days in plain_day_names:
            all_data = np.stack([data_dict[f"{PlasticityFeature} ({day} / ACC4~ACC6)"] for day in days], axis=-1)
            print(days)
            multiple_ttest(all_data[data_dict["labels"] == 0].reshape(-1),
                           all_data[data_dict["labels"] == 2].reshape(-1),
                           all_data[data_dict["labels"] == 1].reshape(-1), )


if __name__ == "__main__":
    main("Calb2_SAT")
    main("Ai148_PSE")
    main("Ai148_SAT")
