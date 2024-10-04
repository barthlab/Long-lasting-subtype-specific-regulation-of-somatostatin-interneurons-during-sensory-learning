# This file contains the functions to generate the figures in the paper.
from __init__ import *
from tqdm import tqdm
from argparse import Namespace
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib as mpl
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats.mstats import ttest_ind, pearsonr
from scipy.stats import f_oneway

plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 24


############################################
# Unimportant plotting function for feature selection
############################################


def get_color_mappers(data_dict, **kwargs):
    color_mappers = []
    for feature_name, color_map_method in get_visualize_feature_label_names(data_dict, **kwargs):
        if color_map_method == "min max":
            norm = mpl.colors.Normalize(
                vmin=np.nanmin(data_dict[feature_name]), vmax=np.nanmax(data_dict[feature_name]))
            color_mappers.append((feature_name, cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)))
        elif color_map_method == "log":
            norm = mpl.colors.LogNorm(vmin=0.25, vmax=4)
            color_mappers.append((feature_name, cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)))
        elif color_map_method == "scalar":
            cmap = mpl.colors.ListedColormap([DotScheme['SST-O']['color'], DotScheme['SST-Calb2']['color']])
            norm = mpl.colors.BoundaryNorm([0, 0.5, 1], cmap.N)
            color_mappers.append((feature_name, cm.ScalarMappable(norm=norm, cmap=cmap)))
    return color_mappers


def plot_expression_distribution(data_dict, save_path, mean_values, criterion, criterion_value, value_name, **kwargs):
    sort_rank = np.argsort(mean_values)
    color_mappers = get_color_mappers(data_dict, **kwargs)
    xs = np.arange(len(mean_values))

    fig, ax = plt.subplots(len(color_mappers), 2, figsize=(6, 2 * len(color_mappers)),
                           gridspec_kw={'width_ratios': [1, 0.1]}, )
    for feature_id, (feature_name, color_mapper) in enumerate(tqdm(color_mappers)):
        tmp_data = data_dict[feature_name]
        ax1 = ax[feature_id, 0]
        ax2 = ax1.twinx()
        ax1.set_ylabel(value_name)
        # ax2.set_ylabel(feature_name)
        ax1.set_yscale('log')

        ax1.plot(xs, mean_values[sort_rank], color='black', lw=1, alpha=0.7)
        for sort_id, cell_id in enumerate(sort_rank):
            cell_color = color_mapper.to_rgba(tmp_data[cell_id]) if np.isfinite(tmp_data[cell_id]) else "black"
            ax2.scatter(sort_id, tmp_data[cell_id], color=cell_color, marker="^", s=15)
            if criterion[cell_id]:
                ax1.scatter(sort_id, criterion_value, color=DotScheme["SST-Calb2"]['color'], marker='_', s=10, zorder=3)
        fig.colorbar(color_mapper, cax=ax[feature_id, 1], orientation='vertical')
        ax[feature_id, 0].set_xlabel("rank")
        ax[feature_id, 1].set_xlabel(feature_name)
        ax[feature_id, 0].spines[['right', 'top']].set_visible(False)

    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def get_feature_f_statistic(data_dict, mean_values, criterion_value, **kwargs):
    color_mappers = get_color_mappers(data_dict, **kwargs)
    f_statistic = {}

    for feature_id, (feature_name, color_mapper) in enumerate(tqdm(color_mappers)):
        tmp_data = data_dict[feature_name]
        less = tmp_data[mean_values < criterion_value]
        more = tmp_data[mean_values >= criterion_value]
        stat_value, pvalue = ttest_ind(less, more)
        rvalue, r_pvalue = pearsonr(tmp_data, mean_values)
        f_statistic[feature_name] = (stat_value, pvalue, rvalue)
    return f_statistic


def plot_feature_correlation(data_dict, save_path, **kwargs):
    raw_keys = get_visualize_feature_label_names(data_dict, **kwargs)
    n = len(raw_keys)
    keys = [raw_keys[i][0] for i in range(n)]
    correlation_matrix = np.zeros((n, n))

    # Compute the correlation matrix
    for i in range(n):
        for j in range(n):
            if i <= j:
                array1 = np.ma.masked_invalid(data_dict[keys[i]])
                array2 = np.ma.masked_invalid(data_dict[keys[j]])
                corr = np.ma.corrcoef(array1, array2)[0, 1]
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

    # Create a heatmap
    plt.figure(figsize=(n / 2, n / 2))
    sns.set(font_scale=0.8)
    g = sns.heatmap(correlation_matrix, xticklabels=keys, yticklabels=keys,
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_feature_ttest(data_dict, save_path, **kwargs):
    f_statistics = get_feature_f_statistic(data_dict, data_dict["Calb2 Mean"], 200, **kwargs)
    raw_keys = get_visualize_feature_label_names(data_dict, **kwargs)
    n = len(raw_keys)
    keys = [raw_keys[i][0] for i in range(n)]
    sorted_rank = list(np.argsort([np.abs(f_statistics[feature_name][2]) for feature_name in keys]))[::-1]
    print(sorted_rank, len(f_statistics))

    plot_rank = sorted_rank[1:11] + [None for _ in range(2)] + sorted_rank[-10:]
    fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    ux, uy = [], []
    for col_id, feature_id in enumerate(plot_rank):
        if feature_id is not None:
            ax.barh(y=col_id, width=f_statistics[keys[feature_id]][2], height=0.25, color='black')
            print(keys[feature_id], f_statistics[keys[feature_id]])
            ux.append(col_id)
            uy.append(keys[feature_id])
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    # ax.set_xscale('log')
    ax.set_yticks(ux, uy)
    ax.spines[['right', 'bottom']].set_visible(False)
    plt.savefig(save_path, dpi=300)
    plt.close()


def color_decomposition(save_path, feature_name, data, dot_coordinates, calb2=None):
    data = np.abs(data)
    if np.nanmin(data) == 0:
        norm = mpl.colors.Normalize(vmin=np.nanmin(data, ), vmax=np.nanmax(data, ))
    else:
        zmax = max(1 / np.nanmin(data), np.nanmax(data))
        zmin = 1 / zmax
        norm = mpl.colors.LogNorm(vmin=zmin, vmax=zmax)
    color_mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [1, 0.05, ]})
    ax = axs[0]
    for cell_id in range(len(data)):
        cell_color = color_mapper.to_rgba(data[cell_id]) if np.isfinite(data[cell_id]) else "black"
        ax.scatter(dot_coordinates[cell_id, 0], dot_coordinates[cell_id, 1], color=cell_color, s=8, marker='o')
        if calb2 is not None and calb2[cell_id] and np.isfinite(data[cell_id]):
            ax.scatter(dot_coordinates[cell_id, 0], dot_coordinates[cell_id, 1], color='magenta', s=15, alpha=1,
                       marker='o')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    axs[1].set_xlabel(feature_name)
    fig.colorbar(color_mapper, cax=axs[1], orientation='vertical')
    fig.savefig(save_path, bbox_inches='tight', dpi=350)
    plt.close(fig)


def label_decomposition(save_path, data, dot_coordinates):
    n_group = len(tuple(data)) - 1
    id_dict = {data_name: i / n_group for i, data_name in enumerate(tuple(data))}
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    color_mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    color_dict = {data_name: color_mapper.to_rgba(id_dict[data_name]) for data_name in id_dict.keys()}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [1, 1, ]})
    ax = axs[0]
    for cell_id in range(len(data)):
        ax.scatter(dot_coordinates[cell_id, 0], dot_coordinates[cell_id, 1],
                   color=color_dict[data[cell_id]], s=8, marker='o')
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_dict.values()]
    axs[1].legend(markers, color_dict.keys(), numpoints=1, ncol=2)
    axs[1].spines[['right', 'top', "bottom", 'left']].set_visible(False)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    ax.set_aspect('equal')
    fig.savefig(save_path, bbox_inches='tight', dpi=350)
    plt.close(fig)


############################################
# Plotting clustering results for paper
############################################


def plot_cluster_features(save_path, normed_data, feature_names, calb2):
    """
    Plots the cluster features with their significance levels.

    Parameters:
    save_path (str): The path to save the plot.
    normed_data (numpy.ndarray): The normalized data array.
    feature_names (list): List of feature names.
    calb2 (numpy.ndarray): Boolean array indicating the presence of Calb2.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(30, 6))
    result_list = []
    new_names = FeatureNameConvertor(feature_names)

    # Calculate p-values for each feature and store them with feature names
    for feature_id in range(len(feature_names)):
        norm_values = normed_data[:, feature_id]
        pos_values, neg_values = norm_values[calb2], norm_values[~calb2]
        _, pvalue = ttest_ind(pos_values, neg_values)
        result_list.append((-np.log2(pvalue), new_names[feature_id]))

    # Sort the results by p-value in descending order
    result_list = sorted(result_list, key=lambda x: x[0], reverse=True)
    print(result_list)

    # Plot the features with their p-values
    for feature_id, (pvalue, feature_name) in enumerate(result_list):
        if "Spontaneous" in feature_name:
            bar_color = "#EFE7BC"  # yellow
        elif "Peak" in feature_name:
            bar_color = "#FFA384"  # salmon
        elif "Response Probability" in feature_name:
            bar_color = "#74BDCB"  # blue
        else:
            raise NotImplementedError
        ax.bar(x=feature_id, height=pvalue, width=0.5, color=bar_color)

    # Set x-ticks and labels
    ax.set_xticks(np.arange(len(feature_names)), [res[1] for res in result_list], fontsize=12, rotation=45, ha='right')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad=5)
    # ax.yaxis.set_tick_params(pad=5)
    ax.grid(axis='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
    # ax.invert_yaxis()

    # Annotate bars with p-values
    for i in ax.patches:
        ax.text(i.get_x() + 0.2, i.get_height() + 0.2, f"p={np.exp2(-i.get_height()):.2g}",
                fontsize=10, color='black', ha='center', va='center', )

    # Set y-ticks with p-value thresholds
    ax.set_yticks([-np.log2(0.5), -np.log2(0.05), -np.log2(0.005), ],
                  [f"p=0.5", f"p=0.05", f"p=0.005", ])

    # Remove spines and save the figure
    ax.spines[['right', 'top', "bottom", "left"]].set_visible(False)
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_decomposition_data(save_path, label_name, data_dict, decomposed_data):
    """
    Plots the decomposition data with different labels.

    Parameters:
    save_path (str): The path to save the plot.
    label_name (str): The name of the label in the data dictionary.
    data_dict (dict): Dictionary containing the data.
    decomposed_data (numpy.ndarray): The decomposed data array.

    Returns:
    None
    """
    labels = data_dict[label_name]
    decomposed_data -= np.max(decomposed_data, axis=0, keepdims=True)
    max_range = np.min(decomposed_data)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for dot_id in range(decomposed_data.shape[0]):
        if labels[dot_id] == 1:
            dot_type = "SST-Calb2"
        elif labels[dot_id] == 0:
            dot_type = "SST-O"
        elif labels[dot_id] == 2:
            dot_type = "SST-Calb2 (FP)"
        else:
            dot_type = "SST-O2"

        if label_name == "labels":
            dot_type += " (Put.)"

        tmp_scheme = DotScheme[dot_type]
        if tmp_scheme['type'] == 1:
            ax.scatter(decomposed_data[dot_id, 0], decomposed_data[dot_id, 1],
                       facecolors=tmp_scheme['color'], edgecolors='none', s=120, marker='o', lw=2,
                       zorder=3 if labels[dot_id] == 1 else 2)
        else:
            ax.scatter(decomposed_data[dot_id, 0], decomposed_data[dot_id, 1],
                       edgecolors=tmp_scheme['color'], facecolors='none', s=60, marker='o', lw=1.5)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(1.1 * max_range, -0.1 * max_range)
    ax.set_ylim(1.1 * max_range, -0.1 * max_range)
    ax.set_aspect('equal')
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_multiple_decomposition_data(save_path, label_name, data_dict, decomposition_dict):
    labels = data_dict[label_name]

    fig, axs = plt.subplots(2, 5, figsize=(6 * 2.5, 6))
    for d_id, decomposed_data in enumerate(decomposition_dict.values()):
        ax = axs[int(d_id / 5), d_id % 5]

        decomposed_data -= np.max(decomposed_data, axis=0, keepdims=True)
        max_range = np.min(decomposed_data)
        for dot_id in range(decomposed_data.shape[0]):
            if labels[dot_id] == 1:
                dot_type = "SST-Calb2"
            elif labels[dot_id] == 0:
                dot_type = "SST-O"
            elif labels[dot_id] == 2:
                dot_type = "SST-Calb2 (FP)"
            else:
                dot_type = "SST-O2"

            if label_name == "labels":
                dot_type += " (Put.)"

            tmp_scheme = DotScheme[dot_type]
            if tmp_scheme['type'] == 1:
                ax.scatter(decomposed_data[dot_id, 0], decomposed_data[dot_id, 1],
                           facecolors=tmp_scheme['color'], edgecolors='none', s=30, marker='o', lw=2,
                           zorder=3 if labels[dot_id] == 1 else 2)
            else:
                ax.scatter(decomposed_data[dot_id, 0], decomposed_data[dot_id, 1],
                           edgecolors=tmp_scheme['color'], facecolors='none', s=20, marker='o', lw=1.5)

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(1.1 * max_range, -0.1 * max_range)
        ax.set_ylim(1.1 * max_range, -0.1 * max_range)
        ax.set_aspect('equal')
        ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_histogram(save_path, labels, calb2, ):
    n_label = np.max(labels) + 1
    fig, ax = plt.subplots(1, 1, figsize=(n_label * 1, 4))
    label_order = (0, 2, 1)
    for label_id in range(n_label):
        calb2_pos, calb2_neg = 0, 0
        for cell_id in range(len(labels)):
            if labels[cell_id] == label_order[label_id]:
                if calb2[cell_id] == 1:
                    calb2_pos += 1
                else:
                    calb2_neg += 1
        ax.bar(x=label_id, height=calb2_pos, width=0.5, bottom=calb2_neg, color=DotScheme["SST-Calb2"]['color'])
        ax.bar(x=label_id, height=calb2_neg, width=0.5, bottom=0, color=DotScheme["SST-O"]['color'])
        ax.text(label_id, calb2_pos + calb2_neg, f"{calb2_pos + calb2_neg}", ha='center', weight='bold')
        ax.text(label_id, calb2_neg - 1.75, f"{calb2_neg}", ha='center', weight='bold', color='w', size=8)
        ax.text(label_id, calb2_pos + calb2_neg - 1.75, f"{calb2_pos}", ha='center', weight='bold', color='w', size=8)
    ax.set_xticks([])
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pie(save_path, labels, calb2, ):
    n_label = np.max(labels) + 1
    fig, ax = plt.subplots(1, n_label, figsize=(n_label * 2, 2))
    label_order = (0, 2, 1)
    for label_id in range(n_label):
        calb2_pos, calb2_neg = 0, 0
        for cell_id in range(len(labels)):
            if labels[cell_id] == label_order[label_id]:
                if calb2[cell_id] == 1:
                    calb2_pos += 1
                else:
                    calb2_neg += 1

        def foo(x):
            return int(round(x * (calb2_neg + calb2_pos) / 100))

        _, _, texts = ax[label_id].pie(
            [calb2_pos, calb2_neg], colors=[DotScheme["SST-Calb2"]['color'], DotScheme["SST-O"]['color']],
            autopct=foo, pctdistance=0.5, startangle=90, textprops={'color': "w"}, counterclock=False, )
    # ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def simple_beeswarm2(y, nbins=None, width=1.):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.

    Parameters:
    y (array-like): The y coordinates of the points.
    nbins (int, optional): The number of bins to use for the histogram. Defaults to None.
    width (float, optional): The width of the bins. Defaults to 1.

    Returns:
    numpy.ndarray: The x coordinates for the points in ``y``.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = 6
        while np.histogram(y, bins=nbins)[0].max() > 6 and nbins <= 50:
            nbins += 2

    x = np.zeros(len(y))

    nn, ybins = np.histogram(y, bins=nbins)
    nmax = nn.max()
    width = nmax * width / 5

    # Divide indices into bins
    ibs = [np.nonzero(y == ybins[0])[0], ]
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):
        i = np.nonzero((y > ymin) * (y <= ymax))[0]
        ibs.append(i)

    # Assign x indices
    dx = width / (nmax // 2)
    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(yy)]
            a = i[j::2]
            b = i[j + 1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def bar_plot_minus(save_dir, data, xnames, alphas=None):
    """
    Creates a bar plot with error bars and optional alpha transparency.

    Parameters:
    save_dir (str): The directory to save the plot.
    data (list): A list of data arrays for each bar.
    xnames (list): A list of names for each bar.
    alphas (list, optional): A list of alpha values for each bar. Defaults to None.

    Returns:
    None
    """
    alphas = [1. for _ in range(len(xnames))] if alphas is None else alphas

    fig, ax = plt.subplots(1, 1, figsize=(0.9 * len(xnames), 6), sharex='all', )
    x_bias = 0
    anova_list = []
    for x_id, x_name in enumerate(xnames):
        if x_name is not None:
            tmp_data = np.array(data[x_id])

            nan_free_tmp_data = []
            for element in tmp_data:
                if np.isfinite(element):
                    nan_free_tmp_data.append(element)
            if "Put." in x_name:
                anova_list.append(nan_free_tmp_data)
            data_bar = np.mean(nan_free_tmp_data)
            data_error = np.std(nan_free_tmp_data) / np.sqrt(len(nan_free_tmp_data))
            x_offset = simple_beeswarm2(nan_free_tmp_data, width=0.15) + x_id

            tmp_color = DotScheme[x_name]['color']
            # for cell_id in range(len(x_offset)):
            #     if DotScheme[x_name]['type'] == 1:
            #         ax.scatter(x_offset, nan_free_tmp_data, facecolors=tmp_color, edgecolors='none', alpha=alphas[x_id], s=8, clip_on=False)
            #     elif DotScheme[x_name]['type'] == 0:
            #         ax.scatter(x_offset, nan_free_tmp_data, edgecolors=tmp_color, facecolors='none', alpha=alphas[x_id], s=8, clip_on=False)
            #     else:
            #         raise NotImplementedError

            if DotScheme[x_name]['type'] == 1:
                patterns, tmp_alpha = None, 1
            elif DotScheme[x_name]['type'] == 0:
                patterns, tmp_alpha = '///', 0.3
            else:
                raise NotImplementedError
            print(f"Fold Change of {x_name}: {data_bar:.3f} \u00B1 {data_error:.3f}")
            ax.bar(
                x_id + x_bias - 0.35, data_bar, width=0.5, capsize=5, color=tmp_color, edgecolor=tmp_color,
                # hatch=patterns,
                linewidth=1., alpha=tmp_alpha,
                yerr=np.stack((np.zeros_like(data_error), data_error), axis=0).reshape(2, -1),
                error_kw={'elinewidth': 1, 'capthick': 0.5, "ecolor": tmp_color}
            )
        else:
            ax.bar(x_id + x_bias - 0.35, 0, width=0.5, alpha=0)
            x_bias -= 0.0

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(-1, None)
    ax.axhline(y=1, lw=1.5, color='gray', ls='--')
    if "PSE" in save_dir:
        ax.set_ylim(0, 3.5)
        ax.set_yticks([0, 1, 2, 3])
    else:
        ax.set_ylim(0, 2)
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticks([])

    fig.savefig(save_dir, bbox_inches='tight', dpi=200)
    plt.close()
    print(save_dir)
    # print(f"One-way ANOVA for three putative groups p={f_oneway(*anova_list)[1]}")


def feature_plot(save_dir, data, alphas=None):
    """
    Plots the features with their respective data points and density estimates.

    Parameters:
    save_dir (str): The directory to save the plot.
    data (dict): A dictionary where keys are feature names and values are arrays of data points.
    alphas (list, optional): A list of alpha transparency values for each feature. Defaults to None.

    Returns:
    None
    """
    xnames = list(data.keys())

    all_data = np.concatenate([data[key] for key in data.keys()])
    y_min, y_max = np.min(all_data[~np.isnan(all_data)]), np.max(all_data[~np.isnan(all_data)])
    delta = y_max - y_min
    y_min = y_min - 0.1 * delta
    y_max = y_max + 0.1 * delta
    y_median = np.median(all_data[~np.isnan(all_data)])
    ys = np.linspace(y_min, y_max, 100)
    alphas = [1. for _ in range(len(xnames))] if alphas is None else alphas
    x_bias = 0

    fig, ax = plt.subplots(1, 1, figsize=(2 * len(xnames), 6.))
    for x_id, x_name in enumerate(xnames):
        if x_name in data:
            if len(data[x_name]) > 0:
                tmp_data = np.array(data[x_name])

                nan_free_tmp_data = []
                for element in tmp_data:
                    if not np.isnan(element):
                        nan_free_tmp_data.append(element)

                tmp_color = DotScheme[x_name]['color']
                if DotScheme[x_name]['type'] == 1:
                    ax.scatter(simple_beeswarm2(nan_free_tmp_data, width=0.15) + x_id + x_bias,
                               nan_free_tmp_data, facecolors=tmp_color, edgecolors='none', alpha=alphas[x_id], s=45,
                               clip_on=False)
                elif DotScheme[x_name]['type'] == 0:
                    ax.scatter(simple_beeswarm2(nan_free_tmp_data, width=0.15) + x_id + x_bias,
                               nan_free_tmp_data, edgecolors=tmp_color, facecolors='none', alpha=alphas[x_id], s=45,
                               clip_on=False)
                else:
                    raise NotImplementedError

                if DotScheme[x_name]['type'] == 1:
                    patterns, tmp_alpha = None, 1
                elif DotScheme[x_name]['type'] == 0:
                    patterns, tmp_alpha = '///', 0.3
                else:
                    raise NotImplementedError

                kde = KernelDensity(kernel="epanechnikov", bandwidth=0.1 * delta).fit(
                    np.array(nan_free_tmp_data)[:, np.newaxis])
                density = np.exp(kde.score_samples(ys[:, np.newaxis]))
                density = 0.25 * density / np.max(density)
                ax.fill_betweenx(ys, x_id - 0.25 - density + x_bias, x_id - 0.25 + x_bias, color=tmp_color, lw=0,
                                 alpha=tmp_alpha)
            else:
                x_bias -= 0.25
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(-1, None)
    ax.set_xticks([])
    if "Spt" in save_dir:
        pass
    elif "Peak" in save_dir:
        ax.set_ylim(0, None)
    else:
        ax.set_ylim(0, 1)
    fig.savefig(save_dir, bbox_inches='tight', dpi=200)
    plt.close()


############################################
# Plotting whole session data for selected cells
############################################

example_traces_dir = path.join(path.dirname(__file__), "..", "figures", "example_traces")
os.makedirs(example_traces_dir, exist_ok=True)


def preprocess_vis(all_features, dataset_name):
    plotting_days = (3, 4, 5, 6, 10, 14)
    cell_order = 0
    selected_cell = None
    for mice_name in all_features.keys():
        for fov_name in all_features[mice_name].keys():
            fov_data = all_features[mice_name][fov_name]
            valid_days = list(fov_data.keys())
            cell_num = fov_data[valid_days[0]][0].cell_num

            for cell_id_withinfov in range(cell_num):
                if cell_order not in RemovedCellIdx[dataset_name]:
                    tmp_cell = Namespace(
                        cell_order=cell_order,
                        mice_name=mice_name,
                        fov_name=fov_name,
                        cell_id_withinfov=cell_id_withinfov,
                        days={plot_day: [] for plot_day in plotting_days},
                    )

                    for plot_day in plotting_days:
                        if plot_day in valid_days:
                            for session_data in fov_data[plot_day]:
                                tmp_cell.days[plot_day].append(
                                    Namespace(
                                        raw_trace=session_data.raw_trace[cell_id_withinfov],
                                        df_f0=session_data.df_f0[cell_id_withinfov],
                                        trend=session_data.trend[cell_id_withinfov],
                                        calcium=session_data.calcium[cell_id_withinfov],
                                        dropped_frames=session_data.dropped_frames,
                                        puff_types=session_data.puff_types,
                                        puff_times=session_data.puff_times,
                                        mad=session_data.mad[cell_id_withinfov],
                                    ))
                    if cell_order == 37:
                        selected_cell = tmp_cell
                cell_order += 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    session_data = selected_cell.days[3][0]
    vis_trace = session_data.calcium
    min_tip = np.max(vis_trace)

    # peaks_pos, properties = find_peaks(vis_trace, prominence=10 * session_data.mad, width=1, )
    # left_ips, right_ips = properties['left_ips'].astype(int), properties['right_ips'].astype(int)
    # trial_pos = [int(np.floor(session_length * trial_onset_time / session_duration))
    #              for trial_onset_time in session_data.puff_times]
    # for peak_id in range(len(peaks_pos)):
    #     spont_flag = np.all([peak_neighbor not in trial_pos for peak_neighbor in np.arange(-5, 5) + peaks_pos[peak_id]] )
    #     if spont_flag:
    #         ax.plot(np.arange(left_ips[peak_id], right_ips[peak_id]+1), vis_trace[left_ips[peak_id]: right_ips[peak_id]+1],
    #                 color="orange", lw=1.5)
    #         ax.scatter(peaks_pos[peak_id], vis_trace[peaks_pos[peak_id]] + 0.2, marker='v', color="orange", s=30)

    baseline_kwargs = {"lw": 0.4, "alpha": 0.1}
    prev_frame = 0
    for trial_id, (trial_onset_time, trial_type) in enumerate(
            zip(session_data.puff_times, session_data.puff_types)):
        trial_onset_frame = int(np.floor(session_length * trial_onset_time / session_duration))
        trial_flag = (trial_type == "real") or (trial_type == "Puff")
        if not trial_flag:
            continue
        trial_color = 'green' if trial_flag else "red"
        # trial_color = "black"

        response = session_data.calcium[trial_onset_frame: trial_onset_frame + 10]
        tip = max(0.2 * min_tip, np.max(vis_trace[trial_onset_frame: trial_onset_frame + 10]) * 1.1 + 0.1)
        if np.max(response) / session_data.mad < 5:
            continue
        #     ax.scatter(trial_onset_frame, tip, marker='v', color=trial_color, s=7)

        if trial_flag:
            ax.scatter(trial_onset_frame, -0.5, color=trial_color, marker='|', s=120, lw=2)
        else:
            ax.scatter(trial_onset_frame, -0.5, edgecolors=trial_color, facecolors='none',
                       marker='o', s=40, lw=1.5)

        ax.plot(np.arange(prev_frame, trial_onset_frame + 1), vis_trace[prev_frame:trial_onset_frame + 1],
                color='black', **baseline_kwargs)
        ax.plot(np.arange(trial_onset_frame, trial_onset_frame + 10),
                vis_trace[trial_onset_frame:trial_onset_frame + 10],
                # color=trial_color, **baseline_kwargs)
                color=trial_color, lw=1.5)
        prev_frame = trial_onset_frame + 9
    ax.plot(np.arange(prev_frame, len(vis_trace)), vis_trace[prev_frame:], color='black', **baseline_kwargs)

    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([0, 1, 2, ])
    # tmp_ax.set_ylabel(f"Day{plot_day}")
    ax.set_ylim(-0.7, 2.5)
    ax.set_xlim(-100, session_length)
    # ax.set_xlim(2000, 2100)

    fig.savefig(path.join(example_traces_dir, f"preprocess.jpg"), bbox_inches='tight', dpi=200)
    plt.close(fig)


def neurosis_analysis(all_features, dataset_name):
    from data_preprocess import get_data_main
    data_dict = get_data_main(dataset_name)
    code = dataset_name.split('_')[-1]

    cell_idx = {
        "Calb2_SAT": {
            "C1": [6, 7, 9, 22, 24, 25, 26, 44, 47, 48, 50, 51, 52, 56, 57, 59, 64, 65, 66, 68, 70, 72, 73],
            "C2": [4, 8, 11, 13, 18, 19, 23, 29, 31, 32, 35, 36, 40, 41, 42, 46, 61, 76],
            "C3": [0, 1, 2, 3, 5, 10, 12, 14, 15, 16, 17, 20, 21, 27, 28, 30, 33, 34, 37, 38, 39, 43, 45, 49, 53, 54,
                   55, 58, 60, 62, 63, 67, 69, 71, 74, 75, 77, 78],
        },
        "Ai148_SAT": {
            "C1": [1, 5, 11, 14, 20, 30, 33, 38, 67, 72, 74, 76, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 93, 94
                , 95, 97],
            "C2": [2, 3, 7, 9, 10, 13, 16, 17, 21, 22, 23, 25, 27, 28, 29, 31, 32, 34, 39, 42, 43, 44, 46, 47
                , 49, 50, 51, 52, 53, 54, 56, 57, 59, 61, 62, 91],
            "C3": [0, 4, 6, 8, 12, 15, 18, 19, 24, 26, 35, 36, 37, 40, 41, 45, 48, 55, 58, 60, 63, 64, 65, 66
                , 68, 69, 70, 71, 73, 75, 78, 79, 89, 90, 92, 96],
        },
        "Ai148_PSE": {
            "C1": [1, 10, 12, 17, 18, 27, 36, 43, 47, 50, 51, 52, 53, 58, 61, 63, 69, 70, 73, 74, 77, 80, 82, 83, 94,
                   100, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
            "C2": [0, 2, 8, 9, 11, 13, 14, 19, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 44,
                   45, 46, 56, 57, 60, 62, 65, 72, 81, 84, 85, 86, 87, 88, 90, 91, 92, 97, 99],
            "C3": [3, 4, 5, 6, 7, 15, 16, 20, 21, 30, 41, 48, 49, 54, 55, 59, 64, 66, 67, 68, 71, 75, 76, 78, 79, 89,
                   93, 95, 96, 98, 101, 102, 103, 104],
        }
    }
    tmp_cell_dict = cell_idx[dataset_name]

    plasticity_value = data_dict[f"{PlasticityFeature} ({code}5~{code}6 / ACC4~ACC6)"]
    # plotting_days = (3, 4, 5, 6, 7, 10, 11, 15)
    plotting_days = (3, 4, 5, 6, 10, 14)
    trial_days = (3, 4, 5)
    cell_cnt, cell_order = 0, 0
    all_cells = []
    for mice_name in all_features.keys():
        for fov_name in all_features[mice_name].keys():
            fov_data = all_features[mice_name][fov_name]
            valid_days = list(fov_data.keys())
            cell_num = fov_data[valid_days[0]][0].cell_num

            for cell_id_withinfov in range(cell_num):
                if cell_order not in RemovedCellIdx[dataset_name]:
                    tmp_cell = Namespace(
                        cell_id=cell_cnt,
                        cell_order=cell_order,
                        mice_name=mice_name,
                        fov_name=fov_name,
                        cell_id_withinfov=cell_id_withinfov,
                        days={plot_day: [] for plot_day in plotting_days},
                        plasticity=plasticity_value[cell_cnt],
                        calb2_flag=data_dict['Calb2'][cell_cnt] if dataset_name == "Calb2_SAT" else 0,
                    )

                    for plot_day in plotting_days:
                        if plot_day in valid_days:
                            for session_data in fov_data[plot_day]:
                                tmp_cell.days[plot_day].append(
                                    Namespace(
                                        df_f0=session_data.df_f0[cell_id_withinfov],
                                        trend=session_data.trend[cell_id_withinfov],
                                        calcium=session_data.calcium[cell_id_withinfov],
                                        dropped_frames=session_data.dropped_frames,
                                        puff_types=session_data.puff_types,
                                        puff_times=session_data.puff_times,
                                        mad=session_data.mad[cell_id_withinfov],
                                    ))

                    all_cells.append(tmp_cell)
                    cell_cnt += 1
                cell_order += 1

    all_cells = sorted(all_cells, key=lambda x: x.plasticity)
    df_f0_flag = True
    print(f"########### Plotting ......")
    for cells_names in tmp_cell_dict.keys():
        cells = [tmp_cell for tmp_cell in all_cells if tmp_cell.cell_id in tmp_cell_dict[cells_names]]
        fig, ax = plt.subplots(len(cells), len(plotting_days) + 2, sharey='row',
                               gridspec_kw={'width_ratios': [1 for _ in range(len(plotting_days))] + [0.5, 0.5],
                                            "wspace": 0.3},
                               figsize=(16 * (len(plotting_days)), 2 * len(cells)))
        for row_id, single_cell in enumerate(cells):
            puff_trials, blank_trials = [], []
            line_color = 'magenta' if single_cell.calb2_flag else "black"
            if df_f0_flag:
                min_tip = np.max([np.max(single_cell.days[plot_day][0].df_f0)
                                  for plot_day in plotting_days if len(single_cell.days[plot_day]) > 0])
            else:
                min_tip = np.max([np.max(single_cell.days[plot_day][0].calcium)
                                  for plot_day in plotting_days if len(single_cell.days[plot_day]) > 0])

            for col_id, plot_day in enumerate(plotting_days):
                if len(single_cell.days[plot_day]) == 0:
                    ax[row_id, col_id].remove()
                    continue
                session_data = single_cell.days[plot_day][0]
                vis_trace = session_data.df_f0 if df_f0_flag else session_data.calcium
                tmp_ax = ax[row_id, col_id]

                prev_frame = 0
                for trial_id, (trial_onset_time, trial_type) in enumerate(
                        zip(session_data.puff_times, session_data.puff_types)):
                    trial_onset_frame = int(np.floor(session_length * trial_onset_time / session_duration))
                    trial_flag = (trial_type == "real") or (trial_type == "Puff")
                    trial_color = 'green' if trial_flag else "red"

                    response = session_data.calcium[trial_onset_frame: trial_onset_frame + 10]
                    tip = max(0.2 * min_tip, np.max(vis_trace[trial_onset_frame: trial_onset_frame + 10]) * 1.1 + 0.1)
                    if np.max(response) / session_data.mad >= 5:
                        tmp_ax.scatter(trial_onset_frame, tip, marker='v', color=trial_color, s=7)

                    tmp_ax.axvline(x=trial_onset_frame, ls='-' if trial_flag else ':',
                                   color=trial_color, lw=1, ymin=0.95)
                    tmp_ax.plot(np.arange(prev_frame, trial_onset_frame + 1),
                                vis_trace[prev_frame:trial_onset_frame + 1],
                                color=line_color, lw=0.6, alpha=0.3)
                    tmp_ax.plot(np.arange(trial_onset_frame, trial_onset_frame + 10),
                                vis_trace[trial_onset_frame:trial_onset_frame + 10],
                                color=trial_color, lw=1.5)
                    if plot_day in trial_days:
                        if trial_flag:
                            puff_trials.append(session_data.calcium[trial_onset_frame - 15:trial_onset_frame + 25])
                        else:
                            blank_trials.append(session_data.calcium[trial_onset_frame - 15:trial_onset_frame + 25])
                    prev_frame = trial_onset_frame + 9
                tmp_ax.plot(np.arange(prev_frame, len(vis_trace)), vis_trace[prev_frame:],
                            color=line_color, lw=0.6, alpha=0.3)

                tmp_ax.spines[['right', 'top', 'bottom']].set_visible(False)
                tmp_ax.set_xticks([])
                # tmp_ax.set_ylabel(f"Day{plot_day}")
                tmp_ax.set_ylim(-0.49, np.clip(min_tip * 1.2, a_min=1, a_max=3))
                tmp_ax.set_xlim(0, session_length)
                tmp_ax.set_ylabel(f"Cell {single_cell.cell_order}")
            title = (f"{np.round(single_cell.plasticity, 2)} {single_cell.mice_name}\n "
                     f"{single_cell.fov_name} {single_cell.cell_id_withinfov} \n Cell {single_cell.cell_order}")

            puff_trials, blank_trials = np.stack(puff_trials, axis=0), np.stack(blank_trials, axis=0)
            ax[row_id, -2].plot(np.mean(puff_trials, 0), color='green')
            for trial_id in range(len(puff_trials)):
                ax[row_id, -2].plot(puff_trials[trial_id], color='green', alpha=0.3, lw=0.6)
            ax[row_id, -1].plot(np.mean(blank_trials, 0), color='red')
            for trial_id in range(len(blank_trials)):
                ax[row_id, -1].plot(blank_trials[trial_id], color='red', alpha=0.3, lw=0.6)
            ax[row_id, -1].set_ylabel(title)
            ax[row_id, -1].spines[['right', 'top', 'bottom']].set_visible(False)

            # ax[row_id, -1].yaxis.tick_right()
            # ax[row_id, -1].set_xticks([])

        fig.savefig(path.join(example_traces_dir, f"{dataset_name}_{cells_names}.jpg"), bbox_inches='tight', dpi=250)
        plt.close(fig)


def selective_plotting(all_features, dataset_name):
    from data_preprocess import get_data_main
    data_dict = get_data_main(dataset_name)
    code = dataset_name.split('_')[-1]

    cell_idx = {
        "Calb2_SAT": [19, 66, 23, 36, 11, 72, 68, 24],
        "Ai148_SAT": [58, 22, 6, 40, 42, 88, 38, 74],
        "Ai148_PSE": [99, 31, 111, 20, 115, 30, 19, 107]
    }
    window_idx = {
        "Calb2_SAT": [0, 5, 0, 9, 3, 6, 16, 7],
        "Ai148_SAT": [7, 4, 0, 2, 2, 0, 1, 4],
        "Ai148_PSE": [0, 12, 1, 10, 6, 1, 10, 1, ]
    }
    mini = True
    plasticity_value = data_dict[f"{PlasticityFeature} ({code}5~{code}6 / ACC4~ACC6)"]
    plotting_day = 4
    cell_cnt, cell_order = 0, 0
    all_cells = []
    for mice_name in all_features.keys():
        for fov_name in all_features[mice_name].keys():
            fov_data = all_features[mice_name][fov_name]
            valid_days = list(fov_data.keys())
            cell_num = fov_data[valid_days[0]][0].cell_num

            for cell_id_withinfov in range(cell_num):
                if cell_order not in RemovedCellIdx[dataset_name]:
                    tmp_cell = Namespace(
                        cell_id=cell_cnt,
                        cell_order=cell_order,
                        mice_name=mice_name,
                        fov_name=fov_name,
                        cell_id_withinfov=cell_id_withinfov,
                        days=[],
                        plasticity=plasticity_value[cell_cnt],
                        calb2_flag=data_dict['Calb2'][cell_cnt] if dataset_name == "Calb2_SAT" else 0,
                    )

                    if plotting_day in valid_days:
                        for session_data in fov_data[plotting_day]:
                            tmp_cell.days.append(
                                Namespace(
                                    df_f0=session_data.df_f0[cell_id_withinfov],
                                    trend=session_data.trend[cell_id_withinfov],
                                    calcium=session_data.calcium[cell_id_withinfov],
                                    dropped_frames=session_data.dropped_frames,
                                    puff_types=session_data.puff_types,
                                    puff_times=session_data.puff_times,
                                    mad=session_data.mad[cell_id_withinfov],
                                ))

                    all_cells.append(tmp_cell)
                    cell_cnt += 1
                cell_order += 1

    print(f"########### Plotting ......")
    cells = [tmp_cell for cell_id in cell_idx[dataset_name] for tmp_cell in all_cells if tmp_cell.cell_order == cell_id]
    y_offset = 0
    cell_names, offsets = [], []
    if mini:
        fig, axs = plt.subplots(1, 3, figsize=(6, 0.7 * len(cells)), sharey='row',
                                gridspec_kw={'width_ratios': [2, 0.5, 0.5], "wspace": 0.1})
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 0.7 * len(cells)), sharey='row',
                                gridspec_kw={'width_ratios': [1, 0.05, 0.05], "wspace": 0.1})

    min_tips = []
    for row_id, single_cell in enumerate(cells):
        session_data = single_cell.days[0]
        vis_trace = session_data.df_f0
        start_trial_idx = window_idx[dataset_name][row_id]
        start_frame = int(np.floor(session_length * session_data.puff_times[start_trial_idx] / session_duration))
        min_tips.append(np.max(vis_trace[start_frame - 18 * 5: start_frame + 5 * 20 * 5]))

    for row_id, single_cell in enumerate(cells):
        line_color = 'magenta' if single_cell.calb2_flag else "black"
        session_data = single_cell.days[0]
        vis_trace = session_data.df_f0

        prev_frame = 0
        start_frame = 0
        puff_trials, blank_trials = [], []

        if mini:
            start_trial_idx = window_idx[dataset_name][row_id]
            start_frame = int(np.floor(session_length * session_data.puff_times[start_trial_idx] / session_duration))
            prev_frame = start_frame - 18 * 5

        min_tip = max(min_tips[row_id], min_tips[(row_id + 4) % 8]) if mini else np.max(vis_trace)

        for trial_id, (trial_onset_time, trial_type) in enumerate(
                zip(session_data.puff_times, session_data.puff_types)):
            if mini and not (start_trial_idx <= trial_id <= start_trial_idx + 3):
                continue
            trial_onset_frame = int(np.floor(session_length * trial_onset_time / session_duration))
            trial_flag = (trial_type == "real") or (trial_type == "Puff")

            # axs[0].plot(np.arange(prev_frame, trial_onset_frame + 10),
            #             vis_trace[prev_frame: trial_onset_frame + 10] + y_offset, color=line_color, lw=0.6)
            axs[0].plot(np.arange(prev_frame, trial_onset_frame + 10) - start_frame,
                        vis_trace[prev_frame: trial_onset_frame + 10] + y_offset, color=line_color, lw=0.6)

            down_offset = np.min(vis_trace[trial_onset_frame - 15: trial_onset_frame + 25])
            if trial_flag:
                # axs[0].scatter(trial_onset_frame, y_offset + down_offset - 0.8, color='black', marker='|', s=15)
                axs[0].scatter(trial_onset_frame - start_frame, y_offset + down_offset - 0.8, color='black', marker='|',
                               s=30, lw=2)
                puff_trials.append(session_data.calcium[trial_onset_frame - 5: trial_onset_frame + 10])
            else:
                # axs[0].scatter(trial_onset_frame, y_offset + down_offset - 0.8, edgecolors='black', facecolors='none',
                #                marker='o', s=5)
                axs[0].scatter(trial_onset_frame - start_frame, y_offset + down_offset - 0.8, edgecolors='black',
                               facecolors='none',
                               marker='o', s=10, lw=1.5)
                blank_trials.append(session_data.calcium[trial_onset_frame - 5: trial_onset_frame + 10])
            prev_frame = trial_onset_frame + 9
        if mini:
            axs[0].plot(np.arange(prev_frame, prev_frame + 18 * 5) - start_frame,
                        vis_trace[prev_frame:prev_frame + 18 * 5] + y_offset,
                        color=line_color, lw=0.6)
        else:
            axs[0].plot(np.arange(prev_frame, len(vis_trace)) - start_frame, vis_trace[prev_frame:] + y_offset,
                        color=line_color, lw=0.6)

        puff_trials, blank_trials = np.stack(puff_trials, axis=0), np.stack(blank_trials, axis=0)

        mean_puff, sd_puff = np.mean(puff_trials, axis=0) + y_offset, np.std(puff_trials, axis=0) / np.sqrt(
            len(puff_trials))
        axs[1].plot(mean_puff, lw=0.6, color=line_color)
        axs[1].fill_between(np.arange(len(mean_puff)), mean_puff - sd_puff, mean_puff + sd_puff, color=line_color,
                            alpha=0.4, lw=0.)
        axs[1].scatter([4.5, ], [y_offset - 0.8, ], color='black', marker='|', s=30, lw=2)
        # axs[1].scatter([4.5, 5.5], [y_offset - 0.8, y_offset - 0.8], color='black', marker='|', s=15)

        mean_blank, sd_blank = np.mean(blank_trials, axis=0) + y_offset, np.std(blank_trials, axis=0) / np.sqrt(
            len(blank_trials))
        axs[2].plot(mean_blank, lw=0.6, color=line_color)
        axs[2].fill_between(np.arange(len(mean_puff)), mean_blank - sd_blank, mean_blank + sd_blank, color=line_color,
                            alpha=0.4, lw=0.)
        axs[2].scatter([4.5, ], [y_offset - 0.8, ], edgecolors='black', facecolors='none', marker='o',
                       s=10, lw=1.5)
        # axs[2].scatter([4.5, 5.5], [y_offset - 0.8, y_offset - 0.8], edgecolors='black', facecolors='none', marker='o',
        #                s=5)

        cell_names.append(f"Cell {single_cell.cell_order}")
        offsets.append(y_offset)
        y_offset += np.clip(min_tip * 1.2, a_min=1.5, a_max=None) + 2

    axs[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    axs[1].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    axs[2].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    axs[0].set_yticks(offsets + [2., ], cell_names + ["2", ])
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([])

    if mini:
        fig.savefig(path.join(example_traces_dir, f"{dataset_name}_select_mini.jpg"), bbox_inches='tight', dpi=200)
    else:
        fig.savefig(path.join(example_traces_dir, f"{dataset_name}_select.jpg"), bbox_inches='tight', dpi=200)
    plt.close(fig)
