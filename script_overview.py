import matplotlib.pyplot as plt

from src import *


def _every_fov(single_exp: Experiment, additional_info: FeatureDataBase):
    for single_mice in single_exp.mice:
        for single_fov in single_mice.fovs:
            visualize.plot_image(
                Image(single_exp.exp_id, single_fov.cell_sessions),
                path.join(single_exp.exp_id, f"{single_fov.str_uid}.png"),
                additional_info=additional_info
            )


def _heatmap_calb2(single_image: Image, feature_db: FeatureDataBase):
    trials_config = {
        "stimulus_trial": {"trial_type": EventType.Puff},
        # "responsive_stimulus_trial_2std": {"trial_type": EventType.Puff, "responsiveness": 1},
    }
    col_config = {
        "sortbymice": (["ACC4", "ACC5", "ACC6", f"SAT1", f"SAT5", f"SAT8"], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", "SAT1", "SAT5", "SAT8"], "ACC6"),
        "multiple_days": (['ACC123', "ACC456", 'SAT123', 'SAT456', 'SAT789'], "ACC456")
    }
    for trials_name, trials_criteria in trials_config.items():
        for col_name, (col_criteria, sort_day) in col_config.items():
            save_name = f"{single_image.exp_id}_{trials_name}_{col_name}"
            days_dict = {col_name: ADV_SAT[col_name] for col_name in col_criteria}
            print(save_name)
            plot_overview.plot_heatmap_overview_cellwise(
                single_image.select(Calb2=1), feature_db, save_name=path.join("replot", f"{single_image.exp_id}_heatmap", f"{save_name}_Calb2_pos.png"),
                days_dict=days_dict, trials_criteria=trials_criteria,
                sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Pos]
            )
            plot_overview.plot_heatmap_overview_cellwise(
                single_image.select(Calb2=0), feature_db, save_name=path.join("replot", f"{single_image.exp_id}_heatmap", f"{save_name}_Calb2_neg.png"),
                days_dict=days_dict, trials_criteria=trials_criteria,
                sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Neg]
            )
            plot_overview.plot_heatmap_overview_cellwise(
                single_image, feature_db, save_name=path.join("replot", f"{single_image.exp_id}_heatmap", f"{save_name}_all_cell.png"),
                days_dict=days_dict, trials_criteria=trials_criteria,
                sorting=(sort_day, brightness), theme_color='black'
            )


def _heatmap_ai148(single_image: Image, feature_db: FeatureDataBase):
    _exp = "SAT" if feature_db.SAT_flag else "PSE"
    trials_config = {
        "stimulus_trial": {"trial_type": EventType.Puff},
        # "responsive_stimulus_trial_2std": {"trial_type": EventType.Puff, "responsiveness": 1},
    }
    col_config = {
        "sortbymice": (["ACC4", "ACC5", "ACC6", f"{_exp}1", f"{_exp}5", f"{_exp}8"], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", f"{_exp}1", f"{_exp}5", f"{_exp}8"], "ACC6"),
        "multiple_days": (['ACC123', "ACC456", f"{_exp}123", f"{_exp}456", f"{_exp}789"], "ACC456")
    }
    for trials_name, trials_criteria in trials_config.items():
        for col_name, (col_criteria, sort_day) in col_config.items():
            save_name = f"{single_image.exp_id}_{trials_name}_{col_name}"
            if feature_db.SAT_flag:
                days_dict = {col_name: ADV_SAT[col_name] for col_name in col_criteria}
            else:
                days_dict = {col_name: ADV_PSE[col_name] for col_name in col_criteria}
            plot_overview.plot_heatmap_overview_cellwise(
                single_image, feature_db, save_name=path.join("replot", f"{single_image.exp_id}_heatmap", f"{save_name}_all_cell.png"),
                days_dict=days_dict, trials_criteria=trials_criteria,
                sorting=(sort_day, brightness), theme_color='black'
            )


def _peak_complex_calb2(single_image: Image, feature_db: FeatureDataBase):
    def compute_normalized_peak(single_cs: CellSession):
        baseline = feature_db.get("evoked_peak", BASELINE_DAYS).get_cell(single_cs.cell_uid)
        return getattr(single_cs, "evoked_peak")/baseline

    feature_db.compute_DayWiseFeature("evoked_peak_normalized", compute_normalized_peak)

    bar_configs = {
        "single_day": (["ACC456", "SAT1", "SAT5", "SAT8"], "SAT5"),
        "multiple_day": (["ACC456", "SAT123", "SAT456", "SAT789"], "SAT456"),
    }
    for bar_name, (bar_config, trace_end) in bar_configs.items():
        for by_mouse_flag in (True, False):
            save_name = path.join("replot", f"{single_image.exp_id}_peak_complex", f"{single_image.exp_id}_{bar_name}_{trace_end}_bymouse{by_mouse_flag}.png")
            plot_overview.plot_peak_complex(
                select_criteria_list=[{"Calb2": 0}, {"Calb2": 1}], feature_db=feature_db,
                bar_feature_name="evoked_peak", curve_feature_name="evoked_peak_normalized",
                save_name=save_name, days_ref=ADV_SAT,
                bar_days=bar_config,
                trace_days=["ACC456", trace_end,],
                image_color=[CELLTYPE2COLOR[CellType.Calb2_Neg], CELLTYPE2COLOR[CellType.Calb2_Pos]],
                kwargs_dict=[{"ACC456": {"ls": ":"}, trace_end: {}}, {"ACC456": {"ls": ":"}, trace_end: {}}],
                trials_criteria={"trial_type": EventType.Puff}, by_mouse_flag=by_mouse_flag
            )


def _peak_complex_ai148(single_image: Image, feature_db: FeatureDataBase):
    def compute_normalized_peak(single_cs: CellSession):
        baseline = feature_db.get("evoked_peak", BASELINE_DAYS).get_cell(single_cs.cell_uid)
        return getattr(single_cs, "evoked_peak")/baseline

    _exp = "SAT" if feature_db.SAT_flag else "PSE"
    feature_db.compute_DayWiseFeature("evoked_peak_normalized", compute_normalized_peak)

    bar_configs = {
        "single_day": (["ACC456", f"{_exp}1", f"{_exp}5", f"{_exp}8"], f"{_exp}5"),
        "multiple_day": (["ACC456", f"{_exp}123", f"{_exp}456", f"{_exp}789"], f"{_exp}456"),
    }

    for bar_name, (bar_config, trace_end) in bar_configs.items():
        for by_mouse_flag in (True, False):
            save_name = path.join("replot", f"{single_image.exp_id}_peak_complex", f"{single_image.exp_id}_{bar_name}_{trace_end}_bymouse{by_mouse_flag}.png")
            plot_overview.plot_peak_complex(
                select_criteria_list=[{}], feature_db=feature_db,
                bar_feature_name="evoked_peak", curve_feature_name="evoked_peak_normalized",
                save_name=save_name, days_ref=ADV_SAT if feature_db.SAT_flag else ADV_PSE,
                bar_days=bar_config,
                trace_days=["ACC456", trace_end,],
                image_color=["black", ],
                kwargs_dict=[{"ACC456": {"ls": ":"}, trace_end: {}}, {"ACC456": {"ls": ":"}, trace_end: {}}],
                trials_criteria={"trial_type": EventType.Puff}, by_mouse_flag=by_mouse_flag
            )


if __name__ == "__main__":
    # # # plot every fov
    # for exp_id in EXP_LIST:
    #     mitten = Experiment(exp_id=exp_id)
    #     mitten_data = mitten.image
    #     mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
    #     mitten_feature.compute_DayWiseFeature("responsiveness", compute_trial_responsive)
    #     _every_fov(mitten, None if mitten_feature.Ai148_flag else mitten_feature)
    # exit()
    #
    # plot Calb2 figure
    # for exp_id in ("Calb2_SAT",):
    #     mitten = Experiment(exp_id=exp_id)
    #     mitten_data = mitten.image
    #     mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
    #     mitten_feature.compute_DayWiseFeature("responsiveness", compute_trial_responsive)
    #     mitten_feature.compute_DayWiseFeature("evoked_peak", compute_trial_evoked_peak)
    #     _heatmap_calb2(mitten_data, mitten_feature)
    #     _peak_complex_calb2(mitten_data, mitten_feature)

    # # plot Ai148 figure
    for exp_id in ("Ai148_SAT", "Ai148_PSE"):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature.compute_DayWiseFeature("responsiveness", compute_trial_responsive)
        mitten_feature.compute_DayWiseFeature("evoked_peak", compute_trial_evoked_peak)
        _heatmap_ai148(mitten_data, mitten_feature)
        # _peak_complex_ai148(mitten_data, mitten_feature)

