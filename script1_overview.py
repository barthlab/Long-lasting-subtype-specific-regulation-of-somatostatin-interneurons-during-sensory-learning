from src import *


def _every_fov(single_exp: Experiment, feature_db: FeatureDataBase):
    for single_mice in single_exp.mice:
        for single_fov in single_mice.fovs:
            visualize.plot_image(Image(single_exp.exp_id, single_fov.cell_sessions), feature_db,
                                 path.join("1_raw_data", single_exp.exp_id, f"{single_fov.str_uid}.png"),)


def _heatmap_vis(single_image: Image, feature_db: FeatureDataBase):
    for trials_name, trials_criteria in PLOTTING_TRIALS_CONFIG.items():
        for col_name, (col_criteria, sort_day) in PLOTTING_DAYS_CONFIG[single_image.exp_id].items():

            save_path = path.join("2_overview", f"{single_image.exp_id}",
                                  f"{single_image.exp_id}_{trials_name}_{col_name}")
            days_dict = {col_name: feature_db.days_ref[col_name] for col_name in col_criteria}
            if not single_image.Ai148_flag:
                plot_overview.plot_heatmap_overview_cellwise(
                    single_image.select(Calb2=1), feature_db, save_name=save_path + f"_Calb2_pos.png",
                    days_dict=days_dict, trials_criteria=trials_criteria,
                    sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Pos]
                )
                plot_overview.plot_heatmap_overview_cellwise(
                    single_image.select(Calb2=0), feature_db, save_name=save_path + f"_Calb2_neg.png",
                    days_dict=days_dict, trials_criteria=trials_criteria,
                    sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Neg]
                )
            plot_overview.plot_heatmap_overview_cellwise(
                single_image, feature_db, save_name=save_path + "_all_cell.png",
                days_dict=days_dict, trials_criteria=trials_criteria,
                sorting=(sort_day, brightness), theme_color='black'
            )


def _peak_complex_vis(single_image: Image, feature_db: FeatureDataBase):
    basic_evoked_response_and_fold_change(feature_db)
    for bar_name, (bar_config, trace_end) in PLOTTING_GROUPS_CONFIG[single_image.exp_id].items():
        for by_mouse_flag in (True, False):
            by_which_post_fix = "by_mouse" if by_mouse_flag else "by_cell"
            save_name = path.join("2_overview", f"{single_image.exp_id}",
                                  f"{single_image.exp_id}_{bar_name}_{trace_end}_{by_which_post_fix}.png")
            if single_image.Ai148_flag:
                plot_overview.plot_peak_complex(
                    select_criteria_list=[{}], feature_db=feature_db,
                    bar_feature_name=EVOKED_RESPONSE_FEATURE, curve_feature_name=FOLD_CHANGE_ACC456_FEATURE,
                    save_name=save_name, days_ref=feature_db.days_ref,
                    bar_days=bar_config,
                    trace_days=["ACC456", trace_end, ],
                    image_color=["black", ],
                    kwargs_dict=[{"ACC456": {"ls": ":"}, trace_end: {}}, {"ACC456": {"ls": ":"}, trace_end: {}}],
                    trials_criteria={"trial_type": EventType.Puff}, by_mouse_flag=by_mouse_flag
                )
            else:
                plot_overview.plot_peak_complex(
                    select_criteria_list=[{"Calb2": 0}, {"Calb2": 1}], feature_db=feature_db,
                    bar_feature_name=EVOKED_RESPONSE_FEATURE, curve_feature_name=FOLD_CHANGE_ACC456_FEATURE,
                    save_name=save_name, days_ref=feature_db.days_ref,
                    bar_days=bar_config,
                    trace_days=["ACC456", trace_end,],
                    image_color=[CELLTYPE2COLOR[CellType.Calb2_Neg], CELLTYPE2COLOR[CellType.Calb2_Pos]],
                    kwargs_dict=[{"ACC456": {"ls": ":"}, trace_end: {}}, {"ACC456": {"ls": ":"}, trace_end: {}}],
                    trials_criteria={"trial_type": EventType.Puff}, by_mouse_flag=by_mouse_flag
                )


if __name__ == "__main__":
    for exp_id in ("Calb2_PSE", "Calb2_SAT", "Ai148_SAT", "Ai148_PSE"):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        _every_fov(mitten, mitten_feature)
        _heatmap_vis(mitten_data, mitten_feature)
        _peak_complex_vis(mitten_data, mitten_feature)


