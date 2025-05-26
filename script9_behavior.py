from src import *


if __name__ == "__main__":
    col_days_refs = {
        # Calb2_SAT
        "M081": [SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5],
        "M085": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
        "M087": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8, SatDay.SAT9],

        # Ai148_SAT
        # "M047": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2],
        # "M017": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4],
        # "M028": [SatDay.ACC6, SatDay.ACC6, SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4],
        "M023": [SatDay.ACC6, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6],
        "M104": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
        "M086": [SatDay.ACC6, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7],
        "M032": [SatDay.ACC6, SatDay.SAT4, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8],
        "M027": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8, SatDay.SAT9],
        "M031": [SatDay.ACC6, SatDay.SAT5, SatDay.SAT6, SatDay.SAT7, SatDay.SAT8, SatDay.SAT9],
    }

    for _exp in ("Ai148_SAT", ):
    # for _exp in ("Calb2_SAT", "Ai148_SAT"):
        mitten_behavior = BehaviorExperiment(exp_id=_exp)
        mitten_imaging = Experiment(exp_id=_exp)

        if _exp == "Ai148_SAT":
            xls = pd.ExcelFile(path.join(FINAL_CLUSTERING_RESULT_PATH, "Ai148_SAT_cluster_label.xlsx"),
                               engine='openpyxl')
            df = pd.read_excel(xls, sheet_name="label")
            tmp_cells_uid = synthesize_cell_uid_list({_k: df[_k].tolist()
                                                      for _k in ("exp_id", "mice_id", "fov_id", "cell_id")})
            label_array = df['value'].to_numpy(dtype=int)
            assert len(label_array) == len(tmp_cells_uid)
            label_dict = {cell_uid: label_id for cell_uid, label_id in zip(tmp_cells_uid, label_array)}
            for single_cs in mitten_imaging.cell_sessions:
                single_cs.cluster_id = label_dict[single_cs.cell_uid]

            img_groups = [{"cluster_id": 0}, {"cluster_id": 2}, {"cluster_id": 1}, ]
            color_groups = [CLUSTER_COLORLIST[0], CLUSTER_COLORLIST[2], CLUSTER_COLORLIST[1], ]

        elif _exp == "Calb2_SAT":
            xls = pd.ExcelFile(path.join(FINAL_CLUSTERING_RESULT_PATH, "Calb2_SAT_features.xlsx"),
                               engine='openpyxl')
            df = pd.read_excel(xls, sheet_name="Calb2")
            tmp_cells_uid = synthesize_cell_uid_list({_k: df[_k].tolist()
                                                      for _k in ("exp_id", "mice_id", "fov_id", "cell_id")})
            label_array = df['value'].to_numpy(dtype=int)
            assert len(label_array) == len(tmp_cells_uid)
            label_dict = {cell_uid: label_id for cell_uid, label_id in zip(tmp_cells_uid, label_array)}
            for single_cs in mitten_imaging.cell_sessions:
                single_cs.cell_type = label_dict[single_cs.cell_uid]

            img_groups = [{"cell_type": 1}, {"cell_type": 0}, ]
            color_groups = [CELLTYPE2COLOR[CellType.Calb2_Pos], CELLTYPE2COLOR[CellType.Calb2_Neg], ]
        else:
            raise NotImplementedError

        plot_behavior.plot_daily_bar_graph(
            mitten_behavior.mice, mitten_imaging.mice,
            img_groups=img_groups,
            color_groups=color_groups,
            col_days=col_days_refs,
            save_name=path.join("9_behavior", f"{_exp}", "all_mice_summary_bars.png"))
        plot_behavior.plot_daily_summary(
            mitten_behavior.mice, mitten_imaging.mice,
            img_groups=img_groups,
            color_groups=color_groups,
            col_days=col_days_refs,
            save_name=path.join("9_behavior", f"{_exp}", "all_mice_summary.png"))

        plot_behavior.plot_daily_bar_graph(
            mitten_behavior.mice, mitten_imaging.mice,
            img_groups=[{}, ],
            color_groups=['black', ],
            col_days=col_days_refs,
            save_name=path.join("9_behavior", f"{_exp}", "all_mice_together_summary_bars.png"))
        plot_behavior.plot_daily_summary(
            mitten_behavior.mice, mitten_imaging.mice,
            img_groups=[{}, ],
            color_groups=['black', ],
            col_days=col_days_refs,
            save_name=path.join("9_behavior", f"{_exp}", "all_mice_together_summary.png"))

        for mitten_behavior_mice in mitten_behavior.mice:
            plot_behavior.plot_daily_bar_graph(
                [mitten_behavior_mice],
                [mitten_imaging.get_mice(mitten_behavior_mice.mice_uid)],
                img_groups=[{}, ],
                color_groups=['black', ],
                col_days=col_days_refs,
                save_name=path.join("9_behavior", f"{_exp}",
                                    f"{mitten_behavior_mice.mice_uid.in_short()}_summary_bars.png"))
            plot_behavior.plot_daily_summary(
                [mitten_behavior_mice],
                [mitten_imaging.get_mice(mitten_behavior_mice.mice_uid)],
                img_groups=[{}, ],
                color_groups=['black', ],
                col_days=col_days_refs,
                save_name=path.join("9_behavior", f"{_exp}",
                                    f"{mitten_behavior_mice.mice_uid.in_short()}_summary.png"))
    exit()
