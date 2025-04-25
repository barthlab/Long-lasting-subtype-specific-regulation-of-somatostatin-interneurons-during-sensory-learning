from src import *


def _every_fov(single_exp: Experiment, additional_info: FeatureDataBase):
    for single_mice in single_exp.mice:
        for single_fov in single_mice.fovs:
            visualize.plot_image(
                Image(single_exp.exp_id, single_fov.cell_sessions),
                path.join(single_exp.exp_id, f"{single_fov.str_uid}.png"),
                additional_info=additional_info
            )


def _heatmap(single_image: Image, feature_db: FeatureDataBase):
    def brightness(ts: TimeSeries) -> float:
        return np.mean(ts.segment(start_t=0, end_t=1.5, relative_flag=True).v)

    plot_overview.plot_heatmap_overview_cellwise(
        single_image, feature_db, save_name=path.join("replot", f"{single_image.exp_id}.jpg"),
        cols_ref=ADV_SAT,
        # cols_names=['ACC4', "ACC5", "ACC6", 'SAT1', 'SAT5', 'SAT9'],
        cols_names=['ACC123', "ACC456", 'SAT123', 'SAT456', 'SAT789'],
        trials_criteria={"trial_type": EventType.Puff},
        sorting=("ACC456", brightness), theme_color='black'
    )


if __name__ == "__main__":
    # for exp_id in EXP_LIST:
    # for exp_id in ("Ai148_SAT", "Calb2_SAT"):
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature.compute_DayWiseFeature("responsiveness", compute_trial_responsive)
        _heatmap(mitten_data, mitten_feature)

    #     # plot_overview.plot_daywise_evoked_shape(mitten_data, "test")
    #     exit()
        # visualize.plot_plasticity_manifold(mitten_feature, "SAT123456.jpg")
        # print(mitten_feature.get("EvokedPeak").value.shape)
        # print(mitten_feature.cell_types)

        # for start_cell_cnt in range(0, len(mitten_data.cells_uid), 4):
        #     selected_cell_lists = mitten_data.cells_uid[start_cell_cnt: start_cell_cnt+4]
        #     visualize.plot_image(
        #         mitten_data.select(
        #             day_id=lambda x: x in SAT_PLOT_DAYS,
        #             cell_uid=lambda x: x in selected_cell_lists,
        #         ),
        #         path.join(mitten_data.exp_id+"_toprint", f"slice_{start_cell_cnt}.png"),
        #         additional_info=mitten_feature
        #     )
        #     if start_cell_cnt > 2:
        #         exit()
