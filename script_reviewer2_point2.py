from src import *


if __name__ == "__main__":
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature_names = feature_prepare(mitten_feature)
        mitten_pvalues = mitten_feature.pvalue_ttest_ind_calb2(mitten_feature_names)
        top25_mitten_features = list(mitten_pvalues)[:25]
        print(top25_mitten_features)
        mitten_vector = get_feature_vector(mitten_feature, top25_mitten_features, "ACC456")

        import numpy as np
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        ordered_cell_ids = list(mitten_vector.keys())
        feature_matrix = np.array([mitten_vector[cid] for cid in ordered_cell_ids])
        colors_for_plot_original = [CELLTYPE2COLOR[mitten_feature.cell_types[cid]] for cid in ordered_cell_ids]

        umap_n_neighbors_options = np.arange(5, 15)
        umap_min_dist_options = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        dbscan_eps_options = [0.3, 0.5, 0.75, 1.0]  # DBSCAN eps, may need tuning based on UMAP output scale
        dbscan_min_samples = 4  # Fixed min_samples for DBSCAN

        best_silhouette_score = -1.1
        best_params = {}
        best_umap_embedding = None
        best_dbscan_labels = None
        best_n_clusters_found = 0

        print(f"Starting UMAP & DBSCAN parameter optimization...")
        print(f"Dataset: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        print(f"UMAP n_neighbors: {umap_n_neighbors_options}")
        print(f"UMAP min_dist: {umap_min_dist_options}")
        print(f"DBSCAN eps: {dbscan_eps_options}, min_samples: {dbscan_min_samples}\n")

        for n_n in umap_n_neighbors_options:
            for m_d in umap_min_dist_options:
                try:
                    reducer = umap.UMAP(
                        n_neighbors=n_n,
                        min_dist=m_d,
                        n_components=2,
                        random_state=42,
                    )
                    embedding = reducer.fit_transform(feature_matrix)

                    for eps_val in dbscan_eps_options:
                        print(f"Trying UMAP(nn={n_n}, md={m_d}), DBSCAN(eps={eps_val})")
                        clusterer = DBSCAN(eps=eps_val, min_samples=dbscan_min_samples)
                        cluster_labels = clusterer.fit_predict(embedding)

                        unique_labels = np.unique(cluster_labels)
                        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

                        print(f"  -> Found {n_clusters} clusters (excl. noise). Labels: {unique_labels}")

                        if n_clusters >= 2:
                            try:
                                score = silhouette_score(embedding, cluster_labels)
                                print(f"  -> Silhouette Score: {score:.4f}")
                                if score > best_silhouette_score:
                                    best_silhouette_score = score
                                    best_params = {
                                        'umap_n_neighbors': n_n,
                                        'umap_min_dist': m_d,
                                        'dbscan_eps': eps_val,
                                        'dbscan_min_samples': dbscan_min_samples
                                    }
                                    best_umap_embedding = embedding
                                    best_dbscan_labels = cluster_labels
                                    best_n_clusters_found = n_clusters
                            except ValueError as e_sil:
                                print(f"  -> Silhouette Score error: {e_sil}")
                        elif n_clusters == 1:
                            print(f"  -> Only 1 cluster found. Silhouette not applicable.")
                        else:
                            print(f"  -> All points classified as noise. Silhouette not applicable.")
                        print("-" * 20)
                except Exception as e:
                    print(f"  -> ERROR in UMAP/DBSCAN step for UMAP(nn={n_n}, md={m_d}): {e}")
                    print("-" * 20)

        print("\n--- Optimization Finished ---")
        if best_umap_embedding is not None:
            print(f"Best Parameters: {best_params}")
            print(f"Best Silhouette Score: {best_silhouette_score:.4f}")
            print(f"Clusters found (excl. noise): {best_n_clusters_found}")

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            plt.style.use('seaborn-v0_8-whitegrid')

            unique_cluster_labels = np.unique(best_dbscan_labels)
            n_actual_clusters = len([lbl for lbl in unique_cluster_labels if lbl != -1])

            palette = plt.cm.get_cmap('tab10', n_actual_clusters if n_actual_clusters > 0 else 1)
            cluster_color_map = {}
            cluster_idx = 0
            for label in sorted(unique_cluster_labels):
                if label == -1:
                    cluster_color_map[label] = (0.7, 0.7, 0.7, 0.6)
                else:
                    cluster_color_map[label] = palette(cluster_idx % palette.N)
                    cluster_idx += 1

            dbscan_point_colors = [cluster_color_map[label] for label in best_dbscan_labels]

            axes[0].scatter(
                best_umap_embedding[:, 0],
                best_umap_embedding[:, 1],
                c=dbscan_point_colors,
                s=60, alpha=0.85, edgecolors='w', linewidth=0.5
            )
            axes[0].set_title(
                f'DBSCAN Clustering (Best UMAP)\n'
                f'Score: {best_silhouette_score:.3f} ({best_n_clusters_found} clusters)',
                fontsize=12
            )

            legend_handles = []
            for label_val in sorted(unique_cluster_labels):
                label_text = f'Noise ({np.sum(best_dbscan_labels == -1)})' if label_val == -1 else f'Cluster {label_val} ({np.sum(best_dbscan_labels == label_val)})'
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                                 label=label_text,
                                                 markerfacecolor=cluster_color_map[label_val], markersize=9))
            if legend_handles:
                axes[0].legend(handles=legend_handles, title="DBSCAN Labels", fontsize=9, loc='best')

            axes[1].scatter(
                best_umap_embedding[:, 0],
                best_umap_embedding[:, 1],
                c=colors_for_plot_original,
                s=60, alpha=0.85, edgecolors='w', linewidth=0.5
            )
            axes[1].set_title('Original Assigned Colors (Best UMAP)', fontsize=12)

            # Create a simple legend for original colors if needed, assuming limited distinct original colors
            distinct_original_colors = sorted(list(set(colors_for_plot_original)))
            original_legend_handles = []
            for color_val in distinct_original_colors:
                # Attempt to find a representative label for this color, this is tricky without original labels
                # For this example, we'll just use the color itself.
                original_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                                          label=f'{mcolors.to_hex(color_val) if isinstance(color_val, tuple) else color_val}',
                                                          markerfacecolor=color_val, markersize=9))
            if original_legend_handles and len(original_legend_handles) <= 10:  # Avoid overly crowded legend
                axes[1].legend(handles=original_legend_handles, title="Original Colors", fontsize=9, loc='best')

            for ax in axes:
                ax.set_xlabel('UMAP Dimension 1', fontsize=10)
                ax.set_ylabel('UMAP Dimension 2', fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, linestyle='--', alpha=0.6)

            fig.suptitle(
                f'UMAP Projection & DBSCAN Clustering Results\nBest UMAP: nn={best_params.get("umap_n_neighbors", "N/A")}, '
                f'md={best_params.get("umap_min_dist", "N/A"):.2f} | '
                f'Best DBSCAN: eps={best_params.get("dbscan_eps", "N/A")}, '
                f'min_samples={best_params.get("dbscan_min_samples", "N/A")}',
                fontsize=14, fontweight='bold'
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        else:
            print("No suitable UMAP/DBSCAN parameters found that yielded >= 2 clusters for Silhouette score.")