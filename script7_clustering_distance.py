from src import *


# def _distance_distribution_visualization(feature_db: FeatureDataBase):
#     # full
#     full_vector = get_feature_vector(feature_db, all_feature_names,
#                                      "ACC456", "Full_ACC456")
#
#     # # top 30
#     # top30_mitten_features = all_feature_names[:30]
#     # top30_vector = get_feature_vector(feature_db, top30_mitten_features,
#     #                                   "ACC456", "Top30_ACC456")
#     #
#     # # waveform
#     # waveform_vector = get_waveform_vector(feature_db, "ACC456", "waveform_ACC456")
#     #
#     # # response probability feature
#     # rp_feature_names = [feature_name for feature_name in all_feature_names
#     #                     if feature_name_to_label(feature_name) == "response probability features"]
#     # rp_vector = get_feature_vector(feature_db, rp_feature_names,
#     #                                "ACC456", "RP_ACC456")
#     #
#     # # in-trial metric feature
#     # in_trial_feature_names = [feature_name for feature_name in all_feature_names
#     #                           if "in-trial" in feature_name_to_label(feature_name)]
#     # in_trial_vector = get_feature_vector(feature_db, in_trial_feature_names,
#     #                                      "ACC456", "in_trial_ACC456")
#     #
#     # # spontaneous metric feature
#     # spont_feature_names = [feature_name for feature_name in all_feature_names
#     #                        if "spontaneous" in feature_name_to_label(feature_name)]
#     # spont_vector = get_feature_vector(feature_db, spont_feature_names,
#     #                                   "ACC456", "spont_ACC456")
#
#     for _tmp_vector in [full_vector, ]:
#         _tmp_save_name = path.join("8_justification", feature_db.ref_img.exp_id + f"_distance_{_tmp_vector.name}")
#         plot_feature.plot_vector_space_distance_calb2(
#             _tmp_vector, save_name1=_tmp_save_name+"_scatter.png", save_name2=_tmp_save_name+"_",
#         )


if __name__ == "__main__":
    all_feature_names = json_load(SORTED_FEATURE_NAMES_JSON_PATH["Calb2_SAT"])
    top25_features = all_feature_names[:25]

    calb2_exp = Experiment(exp_id="Calb2_SAT")
    calb2_feature_db = FeatureDataBase("justify_feature", calb2_exp.image)
    feature_prepare(calb2_feature_db)

    calb2_vector = get_feature_vector(calb2_feature_db, top25_features,
                                      "ACC456", "Top25_ACC456")
    calb2_vector.prepare_embedding()

    _tmp_save_name = path.join("8_justification", "n_cluster")
    plot_clustering.plot_embedding_n_neighbor_distribution(calb2_vector, _tmp_save_name)




