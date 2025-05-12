from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np
import os
import os.path as path
from collections import defaultdict

import statsmodels.api as sm
from scipy import stats
import pingouin as pg
from statsmodels.sandbox.stats.multicomp import multipletests

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.ploter.plotting_params import *
from src.ploter.plotting_utils import *


def get_asterisks(p_value, double_line_flag=True, simple_flag=False) -> Tuple[str, bool, dict]:
    if p_value < 0.001:
        asterisks = r"$\ast$"*3
    elif p_value < 0.01:
        asterisks = r"$\ast$"*2
    elif p_value < 0.05:
        asterisks = r"$\ast$"
    else:
        asterisks = "n.s."
    str_between = "\n" if double_line_flag else " "
    if not simple_flag:
        if p_value < 0.001:
            exponent = int(np.floor(np.log10(abs(p_value))))
            mantissa = p_value / 10 ** exponent
            p_text = f"{asterisks}{str_between}p = {mantissa:.1f} × 10$^{{{exponent}}}$"
        else:
            p_text = f"{asterisks}{str_between}p = {p_value:.2f}"
    else:
        p_text = asterisks

    significant_flag = p_value < 0.05
    if significant_flag:
        font_kwargs = {"alpha": 0.8, "color": 'black', "fontsize": STATISTICAL_FONTSIZE}
    else:
        font_kwargs = {"alpha": 0.3, "color": 'black', "fontsize": STATISTICAL_FONTSIZE}
    return p_text, significant_flag, font_kwargs


def statistic_bar(ax: matplotlib.axes.Axes, x1: float, x2: float, y: float, p_value: float, color: str = 'black'):
    p_text, significant_flag, font_kwargs = get_asterisks(p_value)
    if significant_flag:
        lw, alpha = 0.5, 0.8
    else:
        lw, alpha = 0.5, 0.3
    ax.plot([x1, x2], [y, y], color, lw=lw, alpha=alpha)
    ax.text((x1 + x2) / 2, y, p_text, **font_kwargs,
            horizontalalignment='center', verticalalignment='center')


def timeseries_ttest_every_frame(ax: matplotlib.axes.Axes, data_list: List[dict], expand: bool = False):
    assert len(data_list) == 2
    trace_set_1, trace_set_2 = data_list[0], data_list[1]
    total_timeseries = [trace_set_1[sort_key]
                        for sort_key in trace_set_1.keys()] + [trace_set_2[sort_key]
                                                               for sort_key in trace_set_1.keys()]
    _, _, (xs, interp_ts) = sync_timeseries(total_timeseries)
    assert interp_ts.shape == (len(trace_set_1) + len(trace_set_2), len(xs))
    significant_flags = np.full((len(xs),), fill_value=False, dtype=bool)
    for frame_id in range(len(xs)):
        value_1, value_2 = interp_ts[:len(trace_set_1), frame_id], interp_ts[len(trace_set_1):, frame_id]
        if stats.ttest_rel(value_1, value_2).pvalue < SIGNIFICANT_P:
            significant_flags[frame_id] = True
    pooled_data = pool_boolean_array(significant_flags, xs=xs)
    expand_ratio_low = DISPLAY_MIN_DF_F0_Ai148/DISPLAY_MIN_DF_F0_Calb2 if expand else 1
    expand_ratio_high = DISPLAY_MAX_DF_F0_Ai148/DISPLAY_MAX_DF_F0_Calb2 if expand else 1
    ax.imshow(pooled_data[np.newaxis, :], cmap=SIGNIFICANT_TRACE_COLORMAP, interpolation='nearest',
              extent=(xs[0], xs[-1], expand_ratio_low*SIGNIFICANT_TRACE_Y_EXTENT[0],
                      expand_ratio_high*SIGNIFICANT_TRACE_Y_EXTENT[1]), origin='lower', aspect='auto')


def paired_ttest_with_Bonferroni_correction_simple_version(ax: matplotlib.axes.Axes, data_dict: Dict[float, list]):
    keys = list(data_dict.keys())
    raw_p_values = [
        stats.ttest_ind(data_dict[keys[0]], data_dict[keys[i]], alternative='greater').pvalue
        for i in range(1, len(keys))
    ]
    reject, corrected_p_values, _, _ = multipletests(
        raw_p_values,
        alpha=SIGNIFICANT_ALPHA,
        method='bonferroni'  # Specify the correction method
    )

    print(f"\nBonferroni Corrected Results (alpha = {SIGNIFICANT_ALPHA}):")
    text_position = TEXT_OFFSET_SCALE * np.max([
        nan_max(dict_values) for dict_key, dict_values in data_dict.items() if dict_key != keys[0]])

    for comp, (p_corr, rej) in enumerate(zip(corrected_p_values, reject)):
        print(f" - {comp}: Corrected p-value = {p_corr:.4f}, Significant = {rej}")
        p_text, _, font_kwargs = get_asterisks(p_corr, simple_flag=True)
        ax.text(keys[comp+1], text_position, p_text, **font_kwargs,
                horizontalalignment='center', verticalalignment='center')


def paired_ttest_with_Bonferroni_correction(ax: matplotlib.axes.Axes, data_dict: Dict[float, dict]):
    keys = list(data_dict.keys())
    sorted_keys = list(data_dict[keys[0]].keys())
    raw_p_values = [
        stats.ttest_rel(
            [data_dict[keys[0]][sort_key] for sort_key in sorted_keys],
            [data_dict[keys[i]][sort_key] for sort_key in sorted_keys],
        ).pvalue
        for i in range(1, len(keys))
    ]
    # print("Raw Paired t-test Results:")
    # for comp, p_val in enumerate(raw_p_values):
    #     print(f" - {comp}: p-value = {p_val:.4f}")

    reject, corrected_p_values, _, _ = multipletests(
        raw_p_values,
        alpha=SIGNIFICANT_ALPHA,
        method='bonferroni'  # Specify the correction method
    )

    print(f"\nBonferroni Corrected Results (alpha = {SIGNIFICANT_ALPHA}):")
    text_position = TEXT_OFFSET_SCALE * np.max(
        [(nan_mean(list(dict_values.values())) + nan_sem(list(dict_values.values())))
         for dict_key, dict_values in data_dict.items() if dict_key != keys[0]])
    for comp, (p_corr, rej) in enumerate(zip(corrected_p_values, reject)):
        print(f" - {comp}: Corrected p-value = {p_corr:.4f}, Significant = {rej}")
        p_text, _, font_kwargs = get_asterisks(p_corr)
        ax.text(keys[comp+1], text_position, p_text, **font_kwargs,
                horizontalalignment='center', verticalalignment='center')


def one_way_repeated_anova(ax: matplotlib.axes.Axes, data_dict: Dict[DayType, dict], color: str, y_position: float,
                           start_day: int = 3, end_day: int = 10):
    subjects, days, values = [], [], []
    for day_id, subject_dict in data_dict.items():
        if start_day <= day_id.value <= end_day:
            for subject_id, single_value in subject_dict.items():
                subjects.append(subject_id.in_short())
                days.append(day_id.name)
                values.append(single_value)
    data_rm = pd.DataFrame({
        'subject': subjects,
        'day': days,
        'value': values
    })
    # Perform the Repeated Measures ANOVA
    # 'value' is the dependent variable
    # 'day' is the within-subject factor (repeated measure)
    # 'subject' is the subject identifier
    aov = pg.rm_anova(data=data_rm,
                      dv='value',
                      within='day',
                      subject='subject',
                      detailed=True)
    p_value = aov.loc[0, "p-unc"]
    print(p_value)
    print("\nRepeated Measures ANOVA Results:")
    print(aov)

    # post_hoc = pg.pairwise_ttests(data=data_rm,
    #                               dv='value',
    #                               within='day',
    #                               subject='subject',
    #                               padjust='sidak',
    #                               effsize='hedges')
    #
    # print("\nPost-hoc Pairwise Comparisons (Sidak-corrected):")
    # print(post_hoc[['A', 'B', 'p-unc', 'p-corr', 'hedges']])

    statistic_bar(ax, x1=start_day, x2=end_day, y=y_position, p_value=p_value, color=color)

