from src import *


if __name__ == "__main__":
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature_names = feature_prepare(mitten_feature)
        basic_evoked_response_and_fold_change(mitten_feature)
        visualize.plot_plasticity_manifold(mitten_feature,
                                           "plasticity_calb2_intensity.png")

