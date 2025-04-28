from src import *


if __name__ == "__main__":
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature.compute_DayWiseFeature("responsiveness", compute_trial_responsive)
        mitten_feature.compute_DayWiseFeature("evoked_peak", compute_trial_evoked_peak)