from src import *


def _clustering_examples_visualization(vec_space: VectorSpace, feature_db: FeatureDataBase):
    print([(single_embed.params, single_embed.score)
           for single_embed in vec_space.get_embeddings(top_k=20)])
    _tmp_save_name = path.join("clustering_summary", feature_db.ref_img.exp_id + f"_{vec_space.name}_summary.png")
    plot_clustering.plot_embedding_summary(vec_space, save_name=_tmp_save_name)


def _embed_quality_visualize(vec_space: VectorSpace, best_embedding_by_n_cluster: Dict[int, Embedding]):
    _tmp_save_name = path.join("representative_clustering", vec_space.ref_feature_db.ref_img.exp_id, vec_space.name)
    # vec_space.prepare_embedding()
    # plot_clustering_evaluate.plot_labeling_quality_summary(
    #     vec_space, save_name=_tmp_save_name + f"_summary.png", embed_dict=vec_space.all_embed_by_labelling)

    for cluster_num, best_embed in best_embedding_by_n_cluster.items():
        plot_clustering.plot_one_beautiful_embedding(
            vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{cluster_num}clusters_cell_type.png",
            cell_type_flag=True, ellipse_flag=False, size=(2.5, 2.5))
        plot_clustering.plot_one_beautiful_embedding(
            vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{cluster_num}clusters_cell_type_ellipse.png",
            cell_type_flag=True, ellipse_flag=True, size=(2, 2), label_flag=False)
        plot_clustering.plot_one_beautiful_embedding(
            vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{cluster_num}clusters_cluster_id.png",
            cell_type_flag=False, ellipse_flag=False, size=(2.5, 2.5))
        plot_clustering.plot_one_beautiful_embedding(
            vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{cluster_num}clusters_cluster_id_small.png",
            cell_type_flag=False, ellipse_flag=False, size=(2, 2))


def _plasticity_complex_visualize_calb2sat(feature_db: FeatureDataBase, top_k: int,
                                           best_embedding_by_n_cluster: Dict[int, Embedding]):
    _tmp_save_name = path.join("main_figure", feature_db.ref_img.exp_id)
    bar_configs = {
        "single_day": ["ACC456", "SAT1", "SAT5", "SAT9"],
        "multiple_day": ["ACC456", "SAT123", "SAT456", "SAT789"],
    }
    basic_evoked_response_and_fold_change(feature_db)
    cell_types = reverse_dict(feature_db.cell_types)
    for bar_name, bar_config in bar_configs.items():
        plot_clustering_evaluate.plot_fold_change_bars(
            path.join(_tmp_save_name, f"fold_change_bars_GT_{bar_name}.png"),
            feature_db, bars_list=bar_config,
            group_of_cell_list={
                CELLTYPE2STR[CellType.Calb2_Neg]: cell_types[CellType.Calb2_Neg],
                CELLTYPE2STR[CellType.Calb2_Pos]: cell_types[CellType.Calb2_Pos],
            },
            group_colors=[
                CELLTYPE2COLOR[CellType.Calb2_Neg],
                CELLTYPE2COLOR[CellType.Calb2_Pos],
            ], size=(2.5, 1.5)
        )
    for n_cluster, representative_embedding in best_embedding_by_n_cluster.items():
        cluster_id_dict = reverse_dict(representative_embedding.label_by_cell)
        renamed_cluster_id_dict = {f"cluster {int(i+1)}": cluster_id_dict[i] for i in reversed(range(n_cluster))}
        plot_clustering_evaluate.plot_feature_summary(
            path.join(_tmp_save_name, f"feature_summary_{n_cluster}clusters.png"),
            feature_db, top25_feature_names[:top_k], selected_days=FEATURE_SELECTED_DAYS,
            group_of_cell_list={
                CELLTYPE2STR[CellType.Calb2_Neg]: cell_types[CellType.Calb2_Neg],
                CELLTYPE2STR[CellType.Calb2_Pos]: cell_types[CellType.Calb2_Pos],
                **renamed_cluster_id_dict
            },
            group_colors=[
                CELLTYPE2COLOR[CellType.Calb2_Neg],
                CELLTYPE2COLOR[CellType.Calb2_Pos],
                *[CLUSTER_COLORLIST[i] for i in reversed(range(n_cluster))]
            ], size=(0.7/1.5, 0.3)
        )
        if n_cluster == BEST_NUM_CLUSTERS:
            for bar_name, bar_config in bar_configs.items():
                plot_clustering_evaluate.plot_fold_change_bars(
                    path.join(_tmp_save_name, f"fold_change_bars_{n_cluster}clusters_major_{bar_name}.png"),
                    feature_db, bars_list=bar_config,
                    group_of_cell_list={
                        CELLTYPE2STR[CellType.Put_Calb2_Neg]:
                            list(chain.from_iterable([cluster_id_dict[i] for i in range(1, n_cluster)])),
                        CELLTYPE2STR[CellType.Put_Calb2_Pos]: cluster_id_dict[0],
                    },
                    group_colors=[
                        CELLTYPE2COLOR[CellType.Put_Calb2_Neg],
                        CELLTYPE2COLOR[CellType.Put_Calb2_Pos],
                    ], size=(2.5, 1.5)
                )
                plot_clustering_evaluate.plot_fold_change_bars(
                    path.join(_tmp_save_name, f"fold_change_bars_{n_cluster}clusters_{bar_name}.png"),
                    feature_db, bars_list=bar_config,
                    group_of_cell_list=renamed_cluster_id_dict,
                    group_colors=[CLUSTER_COLORLIST[i] for i in reversed(range(n_cluster))], size=(2.5, 1.5)
                )


def _plasticity_complex_visualize_ai148(feature_db: FeatureDataBase, top_k: int,
                                        best_embedding_by_n_cluster: Dict[int, Embedding]):
    _exp = "SAT" if feature_db.SAT_flag else "PSE"
    bar_configs = {
        "single_day": ["ACC456", f"{_exp}1", f"{_exp}5", f"{_exp}9"],
        "multiple_day": ["ACC456", f"{_exp}123", f"{_exp}456", f"{_exp}789"],
    }
    _tmp_save_name = path.join("main_figure", feature_db.ref_img.exp_id)
    basic_evoked_response_and_fold_change(feature_db)
    for n_cluster, representative_embedding in best_embedding_by_n_cluster.items():
        cluster_id_dict = reverse_dict(representative_embedding.label_by_cell)
        renamed_cluster_id_dict = {f"cluster {int(i+1)}": cluster_id_dict[i] for i in reversed(range(n_cluster))}
        plot_clustering_evaluate.plot_feature_summary(
            path.join(_tmp_save_name, f"feature_summary_{n_cluster}clusters.png"),
            feature_db, top25_feature_names[:top_k], selected_days=FEATURE_SELECTED_DAYS,
            group_of_cell_list=renamed_cluster_id_dict,
            group_colors=[CLUSTER_COLORLIST[i] for i in reversed(range(n_cluster))], size=(0.6, 0.3)
        )
        if n_cluster == BEST_NUM_CLUSTERS:
            for bar_name, bar_config in bar_configs.items():
                plot_clustering_evaluate.plot_fold_change_bars(
                    path.join(_tmp_save_name, f"fold_change_bars_{n_cluster}clusters_major_{bar_name}.png"),
                    feature_db, bars_list=bar_config,
                    group_of_cell_list={
                        CELLTYPE2STR[CellType.Put_Calb2_Neg]:
                            list(chain.from_iterable([cluster_id_dict[i] for i in range(1, n_cluster)])),
                        CELLTYPE2STR[CellType.Put_Calb2_Pos]: cluster_id_dict[0],
                    },
                    group_colors=[
                        CELLTYPE2COLOR[CellType.Put_Calb2_Neg],
                        CELLTYPE2COLOR[CellType.Put_Calb2_Pos],
                    ], size=(2.5, 2)
                )
                plot_clustering_evaluate.plot_fold_change_bars(
                    path.join(_tmp_save_name, f"fold_change_bars_{n_cluster}clusters_{bar_name}.png"),
                    feature_db, bars_list=bar_config,
                    group_of_cell_list=renamed_cluster_id_dict,
                    group_colors=[CLUSTER_COLORLIST[i] for i in reversed(range(n_cluster))], size=(2.5, 2)
                )


def _heatmap_by_clusters_calb2sat(single_image: Image, feature_db: FeatureDataBase,
                                  best_embedding_by_n_cluster: Dict[int, Embedding]):
    trials_config = {
        "stimulus_trial": {"trial_type": EventType.Puff},
    }
    col_config = {
        # "sortbymice": (["ACC4", "ACC5", "ACC6", f"SAT1", f"SAT5", f"SAT9"], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", "SAT1", "SAT2", "SAT3", "SAT4", "SAT5", "SAT9"], "ACC6"),
        "multiple_days": (["ACC456", 'SAT123', 'SAT456', ], "ACC456")
    }

    representative_embedding = best_embedding_by_n_cluster[BEST_NUM_CLUSTERS]
    cluster_id_dict = reverse_dict(representative_embedding.label_by_cell)
    for trials_name, trials_criteria in trials_config.items():
        for col_name, (col_criteria, sort_day) in col_config.items():
            _tmp_save_name = path.join("main_figure", f"{single_image.exp_id}",
                                       f"{single_image.exp_id}_{trials_name}_{col_name}")
            days_dict = {col_name: ADV_SAT[col_name] for col_name in col_criteria}
            for cluster_id, cells_uid_list in cluster_id_dict.items():
                plot_overview.plot_heatmap_overview_cellwise(
                    single_image.select(Calb2=1, cell_uid=tuple(cells_uid_list)),
                    feature_db, save_name=_tmp_save_name + f"_cluster{cluster_id}_Calb2_pos.png",
                    days_dict=days_dict, trials_criteria=trials_criteria,
                    sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Pos],
                    zscore_flag=True
                )
                plot_overview.plot_heatmap_overview_cellwise(
                    single_image.select(Calb2=0, cell_uid=tuple(cells_uid_list)),
                    feature_db, save_name=_tmp_save_name + f"_cluster{cluster_id}_Calb2_neg.png",
                    days_dict=days_dict, trials_criteria=trials_criteria,
                    sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Neg],
                    zscore_flag=True
                )
                plot_overview.plot_heatmap_overview_cellwise(
                    single_image.select(cell_uid=tuple(cells_uid_list)),
                    feature_db, save_name=_tmp_save_name + f"_cluster{cluster_id}_all_cells.png",
                    days_dict=days_dict, trials_criteria=trials_criteria,
                    sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Unknown],
                    zscore_flag=True
                )


def _heatmap_by_clusters_ai148(single_image: Image, feature_db: FeatureDataBase,
                               best_embedding_by_n_cluster: Dict[int, Embedding]):
    _exp = "SAT" if feature_db.SAT_flag else "PSE"
    trials_config = {
        "stimulus_trial": {"trial_type": EventType.Puff},
    }
    col_config = {
        # "sortbymice": (["ACC4", "ACC5", "ACC6", f"{_exp}1", f"{_exp}5", f"{_exp}9"], "mice"),
        "single_days": (["ACC4", "ACC5", "ACC6", f"{_exp}1", f"{_exp}2", f"{_exp}3", f"{_exp}4", f"{_exp}5", f"{_exp}9"], "ACC6"),
        "multiple_days": (["ACC456", f"{_exp}123", f"{_exp}456", ], "ACC456")
    }

    representative_embedding = best_embedding_by_n_cluster[BEST_NUM_CLUSTERS]
    cluster_id_dict = reverse_dict(representative_embedding.label_by_cell)
    for trials_name, trials_criteria in trials_config.items():
        for col_name, (col_criteria, sort_day) in col_config.items():
            _tmp_save_name = path.join("main_figure", f"{single_image.exp_id}",
                                       f"{single_image.exp_id}_{trials_name}_{col_name}")
            if feature_db.SAT_flag:
                days_dict = {col_name: ADV_SAT[col_name] for col_name in col_criteria}
            else:
                days_dict = {col_name: ADV_PSE[col_name] for col_name in col_criteria}

            for cluster_id, cells_uid_list in cluster_id_dict.items():
                plot_overview.plot_heatmap_overview_cellwise(
                    single_image.select(cell_uid=tuple(cells_uid_list)),
                    feature_db, save_name=_tmp_save_name + f"_cluster{cluster_id}_all_cell.png",
                    days_dict=days_dict, trials_criteria=trials_criteria,
                    sorting=(sort_day, brightness), theme_color='black', zscore_flag=True
                )


def tmp_plot(vec_space: VectorSpace):
    vec_space.prepare_embedding()
    candidate_labellings = []
    for single_labelling, embed_list in vec_space.all_embed_by_labelling.items():
        avg_score = np.mean([single_embed.score for single_embed in embed_list])
        occurrences = len(embed_list)
        if single_labelling.n_cluster in PLOTTING_CLUSTERS_OPTIONS:
            candidate_labellings.append({
                "labelling": single_labelling,
                "embed_list": embed_list,
                "avg_score": avg_score,
                "n_cluster": single_labelling.n_cluster,
                "occurrences": occurrences
            })
    # all_candidate_scores = [item["avg_score"] for item in candidate_labellings]
    # score_threshold = np.percentile(all_candidate_scores, q=TOP_LABELLING_THRESHOLD)
    all_candidate_occurrences = [item["occurrences"] for item in candidate_labellings]
    occurrence_threshold = np.percentile(all_candidate_occurrences, q=TOP_LABELLING_OCCURRENCE_THRESHOLD)

    # find the best embedding with above threshold score and largest occurrence
    best_labellings_by_cluster = defaultdict(list)
    for item in candidate_labellings:
        if item["occurrences"] > occurrence_threshold:
            current_n_cluster = item["n_cluster"]
            current_score = item["avg_score"]

            best_single_embedding = max(item["embed_list"], key=lambda embed: embed.score)
            best_labellings_by_cluster[current_n_cluster].append((best_single_embedding, current_score))

    _tmp_save_name = path.join("tmp_plot", vec_space.ref_feature_db.ref_img.exp_id, vec_space.name)
    for n_cluster, representative_embeddings in best_labellings_by_cluster.items():
        for top_id, (top_embed, tmp_score) in enumerate(sorted(representative_embeddings, key=lambda x: x[1], reverse=True)):
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, top_embed, save_name=_tmp_save_name + f"_best_embed_{n_cluster}clusters_cell_type_{top_id}.png",
                cell_type_flag=True, ellipse_flag=False, size=(2.5, 2.5))
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, top_embed, save_name=_tmp_save_name + f"_best_embed_{n_cluster}clusters_cluster_id_{top_id}.png",
                cell_type_flag=False, ellipse_flag=False, size=(2.5, 2.5))
            if top_id > 8:
                break


if __name__ == "__main__":
    top25_feature_names = json_load(SELECTED_CLUSTERING_FEATURE_JSON_PATH)
    for exp_id in ("Calb2_SAT", ):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        feature_prepare(mitten_feature)
        mitten_vector = get_feature_vector(mitten_feature, top25_feature_names,
                                           FEATURE_SELECTED_DAYS, "Top25_ACC456")
        # tmp_plot(mitten_vector)
        # exit()
        mitten_representative = prepare_representative_clustering(mitten_vector)
        print(mitten_representative)
        # _clustering_examples_visualization(mitten_vector, mitten_feature)
        _heatmap_by_clusters_calb2sat(mitten_data, mitten_feature, mitten_representative)
        # _embed_quality_visualize(mitten_vector, mitten_representative)
        # _plasticity_complex_visualize_calb2sat(mitten_feature, top_k=15,
        #                                        best_embedding_by_n_cluster=mitten_representative)

    for exp_id in ("Ai148_SAT", "Ai148_PSE"):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        feature_prepare(mitten_feature)
        mitten_vector = get_feature_vector(mitten_feature, top25_feature_names,
                                           "ACC456", "Top25_ACC456")
        mitten_representative = prepare_representative_clustering(mitten_vector)
        print(mitten_representative)
        # _clustering_examples_visualization(mitten_vector, mitten_feature)
        _heatmap_by_clusters_ai148(mitten_data, mitten_feature, mitten_representative)
        _embed_quality_visualize(mitten_vector, mitten_representative)
        _plasticity_complex_visualize_ai148(mitten_feature, top_k=10,
                                            best_embedding_by_n_cluster=mitten_representative)


