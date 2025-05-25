from src import *


if __name__ == "__main__":
    # _exp = "Calb2_SAT"
    # mitten_behavior = BehaviorExperiment(exp_id=_exp)
    # mitten_imaging = Experiment(exp_id=_exp)
    # mitten_feature = FeatureDataBase("mitten_feature", mitten_imaging.image)
    #
    # plot_behavior.plot_daily_bar_graph(
    #     mitten_behavior, mitten_imaging,
    #     img_groups=[{"Calb2": 0}, {"Calb2": 1}], color_groups=["black"],
    #     col_days={
    #         "M081": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2],
    #         "M085": [SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3],
    #         "M087": [SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3],
    #         "M106": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5],
    #     },
    #     save_name=path.join("9_behavior", f"{_exp}_all_mice_summary_bars.png"))
    # plot_behavior.plot_daily_summary(
    #     mitten_behavior, mitten_imaging,
    #     img_groups=[{"Calb2": 0}, {"Calb2": 1}], color_groups=["black"],
    #     col_days={
    #         "M047": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2],
    #         "M017": [SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3],
    #         "M028": [SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3],
    #         "M023": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5],
    #         "M104": [SatDay.ACC6, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6],
    #         "M086": [SatDay.ACC6, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6],
    #         "M032": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
    #         "M027": [SatDay.ACC6, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8],
    #         "M031": [SatDay.ACC6, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8],
    #     },
    #     save_name=path.join("9_behavior", f"{_exp}_all_mice_summary.png"))

    xls = pd.ExcelFile(path.join(FINAL_CLUSTERING_RESULT_PATH, "Ai148_SAT_cluster_label.xlsx"), engine='openpyxl')
    df = pd.read_excel(xls, sheet_name="label")
    tmp_cells_uid = synthesize_cell_uid_list({_k: df[_k].tolist()
                                              for _k in ("exp_id", "mice_id", "fov_id", "cell_id")})
    label_array = df['value'].to_numpy(dtype=int)
    assert len(label_array) == len(tmp_cells_uid)
    label_dict = {cell_uid: label_id for cell_uid, label_id in zip(tmp_cells_uid, label_array)}

    _exp = "Ai148_SAT"
    mitten_behavior = BehaviorExperiment(exp_id=_exp)
    mitten_imaging = Experiment(exp_id=_exp)


    for single_cs in mitten_imaging.cell_sessions:
        single_cs.cluster_id = label_dict[single_cs.cell_uid]

    plot_behavior.plot_daily_bar_graph(
        mitten_behavior, mitten_imaging,
        img_groups=[{"cluster_id": 0}, {"cluster_id": 1}, {"cluster_id": 2}, ],
        color_groups=[CLUSTER_COLORLIST[0], CLUSTER_COLORLIST[1], CLUSTER_COLORLIST[2], ],
        col_days={
            # "M047": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2],
            "M017": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4],
            "M028": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4],
            "M023": [SatDay.ACC6, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6],
            "M104": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
            "M086": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
            "M032": [SatDay.ACC6, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8],
            "M027": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8, SatDay.SAT9],
            "M031": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8, SatDay.SAT9],
        },
        save_name=path.join("9_behavior", f"{_exp}_all_mice_summary_bars.png"))
    plot_behavior.plot_daily_summary(
        mitten_behavior, mitten_imaging,
        img_groups=[{"cluster_id": 0}, {"cluster_id": 1}, {"cluster_id": 2}, ],
        color_groups=[CLUSTER_COLORLIST[0], CLUSTER_COLORLIST[1], CLUSTER_COLORLIST[2], ],
        col_days={
            # "M047": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2],
            "M017": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4],
            "M028": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4],
            "M023": [SatDay.ACC6, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6],
            "M104": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
            "M086": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
            "M032": [SatDay.ACC6, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8],
            "M027": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8, SatDay.SAT9],
            "M031": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8, SatDay.SAT9],
        },
        save_name=path.join("9_behavior", f"{_exp}_all_mice_summary.png"))
    exit()
