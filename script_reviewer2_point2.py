from src import *


def _distance_distribution_visualization(vector_list: List[VectorSpace], feature_db: FeatureDataBase):
    for _tmp_vector in vector_list:
        _tmp_save_name = path.join("distance", feature_db.name,
                                   feature_db.ref_img.exp_id + f"_{_tmp_vector.name}")
        # plot_feature.plot_vector_space_distance_calb2(
        #     _tmp_vector, save_name1=_tmp_save_name+"_scatter.png", save_name2=_tmp_save_name+"_distance.png",
        # )
        plot_clustering.plot_umap_space_distance_calb2(
            _tmp_vector, save_name1=_tmp_save_name+"_best_clustering.png",
            save_name2=_tmp_save_name+"_umap_distance_matrix.png",
        )


def _clustering_examples_visualization(vector_list: List[VectorSpace], feature_db: FeatureDataBase):
    for _tmp_vector in vector_list:
        print([(single_embed.params, single_embed.score)
               for single_embed in _tmp_vector.get_embeddings(top_k=20)])
        _tmp_save_name = path.join("clustering", feature_db.name,
                                   feature_db.ref_img.exp_id + f"_{_tmp_vector.name}_summary.png")
        plot_clustering.plot_embedding_summary(_tmp_vector, save_name=_tmp_save_name)


if __name__ == "__main__":
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature_names = feature_prepare(mitten_feature)
        mitten_pvalues = mitten_feature.pvalue_ttest_ind_calb2(mitten_feature_names)
        top25_mitten_features = list(mitten_pvalues.keys())[:25]
        print(top25_mitten_features)
        top25_vector = get_feature_vector(mitten_feature, top25_mitten_features,
                                          "ACC456", "Top25_ACC456")
        top25_vector.prepare_embedding()
        full_vector = get_feature_vector(mitten_feature, list(mitten_pvalues.keys()),
                                         "ACC456", "Full_ACC456")
        full_vector.prepare_embedding()
        waveform_vector = get_waveform_vector(mitten_feature, "ACC456", "waveform_ACC456")
        waveform_vector.prepare_embedding()
        _distance_distribution_visualization([top25_vector, full_vector, waveform_vector], mitten_feature)
        # _clustering_examples_visualization([top25_vector, full_vector, waveform_vector], mitten_feature)

