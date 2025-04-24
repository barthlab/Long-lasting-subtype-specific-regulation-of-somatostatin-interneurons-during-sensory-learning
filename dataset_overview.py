from src import *


if __name__ == "__main__":
    # for exp_id in EXP_LIST:
    # for exp_id in ("Ai148_SAT", "Calb2_SAT"):
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature.compute_DayWiseFeature("responsiveness", compute_trial_responsive)
        print(mitten_feature.get("responsiveness").value)
        exit()
        # visualize.plot_plasticity_manifold(mitten_feature, "SAT123456.jpg")
        # print(mitten_feature.get("EvokedPeak").value.shape)
        # print(mitten_feature.cell_types)

        for start_cell_cnt in range(0, len(mitten_data.cells_uid), 4):
            selected_cell_lists = mitten_data.cells_uid[start_cell_cnt: start_cell_cnt+4]
            visualize.plot_image(
                mitten_data.select(
                    day_id=lambda x: x in SAT_PLOT_DAYS,
                    cell_uid=lambda x: x in selected_cell_lists,
                ),
                path.join(mitten_data.exp_id+"_toprint", f"slice_{start_cell_cnt}.png"),
                additional_info=mitten_feature
            )
            if start_cell_cnt > 2:
                exit()
