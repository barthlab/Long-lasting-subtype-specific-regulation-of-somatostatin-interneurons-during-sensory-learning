from src import *


if __name__ == "__main__":
    # for exp_id in EXP_LIST:
    # for exp_id in ("Ai148_SAT", "Calb2_SAT"):
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = feature_manager.FeatureDataBase("mitten_feature", mitten_data)
        # visualize.plot_plasticity_manifold(mitten_feature, "SAT123456.jpg")
        # print(mitten_feature.get("EvokedPeak").value.shape)
        # print(mitten_feature.cell_types)

        for single_mice in mitten.mice:
            for single_fov in single_mice.fovs:  # type: int, FOV
                visualize.plot_image(
                    Image(mitten.exp_id, single_fov.cell_sessions),
                    path.join(mitten_data.exp_id, f"{single_fov.exp_id}_FOV{single_fov.fov_id}.png"),
                    additional_info=mitten_feature
                )
                exit()
