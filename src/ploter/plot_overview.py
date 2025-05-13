import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict
from itertools import chain

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.ploter.plotting_params import *
from src.ploter.plotting_utils import *
from src.ploter.statistic_annotation import *


def plot_heatmap_overview_cellwise(
        single_image: Image, feature_db: FeatureDataBase, save_name: str,
        days_dict: Dict[str, Tuple[DayType, ...]],
        trials_criteria: dict = None, sorting: Tuple[str, Callable[TimeSeries, float]] = None, theme_color: str = "black"
):
    """
    [heatmap 1      ] [heatmap 2      ] ... [heatmap cols_names[-1]]
    [average curve 1] [average curve 2] ... [average curve cols_names[-1]]

    """
    cols_names = list(days_dict.keys())
    n_col = len(cols_names)
    trials_criteria = trials_criteria if trials_criteria is not None else {}

    extracted_data = extract_avg_df_f0(single_image=single_image, days_dict=days_dict, **trials_criteria)

    # sorting
    sort_col = "Record Date"
    if sorting is not None:
        sort_col, sort_func = sorting
        if sort_col == "mice":
            cells_uid_order = sorted(single_image.cells_uid,
                                     key=lambda x: (x.mice_id, x.fov_id, x.cell_id), reverse=True)
        else:
            assert sort_col in cols_names, f"Sorting name missing ({sort_col}) in {cols_names}"
            cells_uid_order = sorted(single_image.cells_uid, key=lambda x: sort_func(extracted_data[sort_col][x]),
                                     reverse=True)
    else:
        cells_uid_order = single_image.cells_uid

    # plotting
    fig, axs = plt.subplots(2, n_col+1, width_ratios=[1] * n_col + [0.08], height_ratios=[1, 0.15])
    axh, axc = axs[0, :], axs[1, :]
    for ax_id, col_name in enumerate(cols_names):
        grand_avg_df_f0, grand_sem_df_f0, (xs, grand_matrix) = sync_timeseries(
            [extracted_data[col_name][cell_uid] for cell_uid in cells_uid_order])
        """
        TODO: add extent and remove searchsorted
        """
        im = axh[ax_id].imshow(
            grand_matrix,
            aspect=3,
            origin='upper',
            vmin=DISPLAY_MIN_DF_F0_Ai148 if feature_db.Ai148_flag else DISPLAY_MIN_DF_F0_Calb2,
            vmax=DISPLAY_MAX_DF_F0_Ai148 if feature_db.Ai148_flag else DISPLAY_MAX_DF_F0_Calb2,
            cmap='viridis',
            interpolation='nearest'
        )
        axc[ax_id].plot(grand_avg_df_f0.v, lw=LW_SMALL_DF_F0, color=theme_color, alpha=ALPHA_DEEP_DF_F0)
        axc[ax_id].fill_between(np.arange(len(xs)), grand_avg_df_f0.v - grand_sem_df_f0.v,
                                grand_avg_df_f0.v + grand_sem_df_f0.v, lw=0, color=theme_color,
                                alpha=ALPHA_LIGHT_DF_F0)

        axc[ax_id].spines[['right', 'top']].set_visible(False)
        if ax_id == 0:
            axh[ax_id].set_ylabel(f"Cell ID (Ranked on {sort_col})" if sorting is not None else "Cell ID")
            axc[ax_id].set_ylabel(r'$\Delta F/F_0$')
            if sort_col == "mice":
                yticks, yticklabels = [0,], [cells_uid_order[0].mice_id,]
                for i in range(1, len(cells_uid_order)):
                    current_mice_id = cells_uid_order[i].mice_id
                    previous_mice_id = cells_uid_order[i - 1].mice_id
                    if current_mice_id != previous_mice_id:
                        yticks.append(i)
                        yticklabels.append(current_mice_id)

                axh[ax_id].set_yticks(yticks, yticklabels)
        else:
            axh[ax_id].set_yticklabels([])
            axh[ax_id].set_yticks([])
            axc[ax_id].set_yticklabels([])
            axc[ax_id].set_yticks([])

        x_tick_loc, x_tick_pos = [-1, 0, 1, 2], []
        for x_tick in x_tick_loc:
            x_tick_pos.append(np.searchsorted(xs, x_tick))
        axc[ax_id].set_ylim(DISPLAY_MIN_DF_F0_Ai148 if feature_db.Ai148_flag else DISPLAY_MIN_DF_F0_Calb2,
                            AVG_MAX_DF_F0_Ai148 if feature_db.Ai148_flag else AVG_MAX_DF_F0_Calb2)
        axh[ax_id].set_xlim(np.searchsorted(xs, -2), np.searchsorted(xs, 3))
        axc[ax_id].set_xlim(np.searchsorted(xs, -2), np.searchsorted(xs, 3))
        axh[ax_id].axvline(x=np.searchsorted(xs, 0), color='red', alpha=0.7, ls='--', lw=0.5)
        axc[ax_id].axvspan(np.searchsorted(xs, 0), np.searchsorted(xs, 0.5),
                           lw=0, color=OTHER_COLORS['puff'], alpha=0.4)
        axh[ax_id].set_xticks(x_tick_pos, x_tick_loc)
        axc[ax_id].set_xticks(x_tick_pos, x_tick_loc)
        axh[ax_id].set_xlabel("Time [s]")
        axc[ax_id].set_xlabel("Time [s]")
        axh[ax_id].set_title(col_name)

    cbar = fig.colorbar(im, cax=axh[-1])
    cbar.set_label(r'$\Delta F/F_0$')
    axc[-1].remove()

    fig.set_size_inches(7.5, 5)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_peak_complex(
        select_criteria_list: List[dict], feature_db: FeatureDataBase,
        bar_feature_name: str, curve_feature_name: str, save_name: str,
        days_ref: Dict[str, Tuple[DayType, ...]], bar_days: List[str], trace_days: List[str],
        image_color: List[str],
        kwargs_dict: List[Dict[str, Dict]] = None,  trials_criteria: dict = None, by_mouse_flag: bool = False
):
    """
    [avg trace 1] [avg trace 2] ... [avg trace n_img] [bars of feature] [curves of feature all days]

    """
    trials_criteria = trials_criteria if trials_criteria is not None else {}
    n_img = len(select_criteria_list)
    bar_offset, bar_width = 1/(n_img+1), 2*0.4/(n_img+1)
    kwargs_dict = kwargs_dict if kwargs_dict is not None else [{group_name: {} for group_name in trace_days}]*n_img
    assert len(image_color) == n_img

    fig, axs = plt.subplots(1, n_img+4, width_ratios=[1/n_img] * n_img + [0.35, 1, 0.35, 1.5],
                            gridspec_kw={"wspace": 0})
    axs[-4].set_visible(False)
    axs[-2].set_visible(False)
    axt, axb, axc = axs[:n_img], axs[-3], axs[-1]

    for image_id, single_select_criteria in enumerate(select_criteria_list):  # type: int, dict
        single_feature_db = feature_db.select(f"{image_id}", **single_select_criteria)
        extracted_data = extract_avg_df_f0(
            single_image=single_feature_db.ref_img,
            days_dict={group_name: days_ref[group_name] for group_name in trace_days}, **trials_criteria)

        statistic_dict = []
        for group_name in trace_days:
            kwargs = kwargs_dict[image_id][group_name]
            tmp_data = extracted_data[group_name] if not by_mouse_flag else by_cell2by_mice(extracted_data[group_name])
            oreo(axt[image_id], [single_ts for single_ts in tmp_data.values()],
                 mean_kwargs={"alpha": 0.7, "lw": 1, "color": image_color[image_id], **kwargs},
                 fill_kwargs={"alpha": 0.2, "lw": 0, "color": image_color[image_id], **kwargs})
            statistic_dict.append(tmp_data)

        # comparing first two set of the timeseries, every frame do paired
        timeseries_ttest_every_frame(axt[image_id], statistic_dict, expand=feature_db.Ai148_flag)

        statistic_dict = {}
        for bar_id, bar_name in enumerate(bar_days):
            tmp_feature = single_feature_db.get(feature_name=bar_feature_name, day_postfix=bar_name)
            tmp_data = tmp_feature.by_cells if not by_mouse_flag else by_cell2by_mice(tmp_feature.by_cells)
            oreo_bar(axb, [single_value for single_value in tmp_data.values()],
                     x_position=bar_id + image_id*bar_offset, width=bar_width, color=image_color[image_id])
            if bar_id == 0:
                axb.axhline(y=np.mean([single_value for single_value in tmp_data.values()]),
                            lw=1, c=image_color[image_id], ls=':', alpha=0.7)
            statistic_dict[bar_id + image_id*bar_offset] = tmp_data

        # comparing the later values to the first value
        # with paired t-test with Bonferroni correction
        paired_ttest_with_Bonferroni_correction(axb, statistic_dict)

        tmp_feature = single_feature_db.get(feature_name=curve_feature_name)
        xs = [day_id.value for day_id in single_feature_db.days]
        day_change_by_mice = invert_nested_dict({day_id: by_cell2by_mice(tmp_feature.get_day(day_id))
                                                 for day_id in single_feature_db.days})
        for mice_id, days_dict in day_change_by_mice.items():
            sorted_value = [days_dict[day_id] for day_id in single_feature_db.days]
            axc.plot(xs, sorted_value, lw=0.5, alpha=0.2, marker='.', markersize=2, color=image_color[image_id])

        if by_mouse_flag:
            grand_average = combine_dicts(*[days_dict for days_dict in day_change_by_mice.values()])
        else:
            grand_average = {day_id: list(tmp_feature.get_day(day_id).values())
                             for day_id in single_feature_db.days}
        grand_avg = np.array([nan_mean(grand_average[day_id]) for day_id in single_feature_db.days])
        grand_sem = np.array([nan_sem(grand_average[day_id]) for day_id in single_feature_db.days])
        axc.plot(xs, grand_avg, lw=1, alpha=0.7, marker='.', markersize=6, color=image_color[image_id])
        axc.fill_between(xs, grand_avg-grand_sem, grand_avg+grand_sem, lw=0, alpha=0.3, color=image_color[image_id])

        # comparing one-way repeated anova
        one_way_repeated_anova(
            axc,
            y_position=SIGNIFICANT_ANOVA_BAR_HEIGHT_SAT + 0.5*image_id if feature_db.SAT_flag else
            SIGNIFICANT_ANOVA_BAR_HEIGHT_PSE,
            data_dict={day_id: by_cell2by_mice(tmp_feature.get_day(day_id))
                       for day_id in single_feature_db.days} if by_mouse_flag else
            {day_id: tmp_feature.get_day(day_id) for day_id in single_feature_db.days}, color=image_color[image_id])

    for image_id in range(n_img):
        axt[image_id].set_ylim(DISPLAY_MIN_DF_F0_Ai148 if feature_db.Ai148_flag else DISPLAY_MIN_DF_F0_Calb2,
                               AVG_MAX_DF_F0_Ai148 if feature_db.Ai148_flag else AVG_MAX_DF_F0_Calb2)
        axt[image_id].spines[['right', 'top']].set_visible(False)
        axt[image_id].axvspan(0, 0.5, lw=0, color=OTHER_COLORS['puff'], alpha=0.4)
        axt[image_id].set_xticks([0, 2])
        axt[image_id].set_xlabel("Time [s]")
        if image_id > 0:
            axt[image_id].set_yticks([], [])
            axt[image_id].spines[['left', ]].set_visible(False)
        else:
            axt[image_id].set_ylabel(r'$\Delta F/F_0$')
    axb.set_ylim(0, AVG_MAX_DF_F0_Ai148 if feature_db.Ai148_flag else AVG_MAX_DF_F0_Calb2)
    axb.spines[['right', 'top']].set_visible(False)
    axb.axvspan(0.75, len(bar_days)-0.5, lw=0, alpha=0.4, zorder=0,
                color=OTHER_COLORS['SAT'] if feature_db.SAT_flag else OTHER_COLORS["PSE"], )
    axb.set_ylabel(r'Peak response ($\Delta F/F_0$)')
    axb.set_xticks(np.arange(len(bar_days)), bar_days, rotation=25)
    axc.set_ylim(0, AVG_MAX_FOLD_Ai148 if feature_db.Ai148_flag else AVG_MAX_FOLD_Calb2)
    axc.spines[['right', 'top']].set_visible(False)
    axc.set_xticks([day_id.value for day_id in feature_db.days],
                   [day_id.name for day_id in feature_db.days], rotation=45,)
    axc.set_ylabel(r'Normalized Peak')
    axc.axvspan(5.5, len(feature_db.days)-1, lw=0, alpha=0.4, zorder=0,
                color=OTHER_COLORS['SAT'] if feature_db.SAT_flag else OTHER_COLORS["PSE"], )
    axc.axhline(y=1., lw=1, c='red', ls=':', alpha=0.7)

    fig.set_size_inches(7.5, 1.7)
    fig.tight_layout()
    quick_save(fig, save_name)


