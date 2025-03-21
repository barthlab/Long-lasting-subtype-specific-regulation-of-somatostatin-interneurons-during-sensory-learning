from __init__ import *
import pandas as pd


def drop_frames(mat_array, offset_threshold=20, block_len_threshold=2):
    """
    Drops or interpolates frames in the given matrix array based on offset thresholds.

    Parameters:
    mat_array (dict): Dictionary containing the data arrays and operations.
    offset_threshold (int, optional): Threshold for the offset to consider a frame corrupted. Default is 20.
    block_len_threshold (int, optional): Maximum length of a block of corrupted frames to interpolate. Default is 2.

    Returns:
    tuple: A tuple containing the modified matrix array, an array indicating dropped frames, and the xy offset values.
    """
    ops = mat_array['ops'][0][0]
    F = mat_array['F']
    Fneu = mat_array['Fneu']
    num_neurites, total_frames = F.shape

    # Calculate xy offset values
    xy_off = np.sqrt(ops['xoff'] ** 2 + ops['yoff'] ** 2)[0]
    xy_off = (np.abs(xy_off - np.mean(xy_off)) / 10).astype(np.float32)

    # Reshape and normalize x and y offsets
    xoff, yoff = np.array(ops['xoff']).reshape(-1, session_length), np.array(ops['yoff']).reshape(-1, session_length)
    xoff = xoff - np.mean(xoff, axis=-1, keepdims=True)
    yoff = yoff - np.mean(yoff, axis=-1, keepdims=True)

    # Identify corrupted frames based on offset threshold
    x_off = np.argwhere(np.abs(xoff.reshape(-1)) > offset_threshold)
    y_off = np.argwhere(np.abs(yoff.reshape(-1)) > offset_threshold)
    corrupted_frames = np.unique(np.concatenate([x_off, y_off]))
    corrupted_frames = np.concatenate([corrupted_frames, [3 * total_frames]])

    block_start = 0
    dropped_frames = np.zeros(total_frames)

    # Process corrupted frames
    for i in range(len(corrupted_frames) - 1):
        cur_frame, nxt_frame = corrupted_frames[i], corrupted_frames[i + 1]
        if nxt_frame > cur_frame + 1:
            block_len = i - block_start + 1
            start_frame = corrupted_frames[block_start]

            # Interpolate or drop frames based on block length
            if (block_len <= block_len_threshold) and (start_frame + block_len < total_frames):
                interp = np.linspace(0, 1, block_len + 2)[np.newaxis, 1:-1]
                F[:, start_frame:start_frame + block_len] = (interp * F[:, start_frame + block_len, np.newaxis] +
                                                             (1 - interp) * F[:, start_frame - 1, np.newaxis])
                Fneu[:, start_frame:start_frame + block_len] = (interp * Fneu[:, start_frame + block_len, np.newaxis] +
                                                                (1 - interp) * Fneu[:, start_frame - 1, np.newaxis])
            else:
                dropped_frames[start_frame:start_frame + block_len] = 1
            block_start = i + 1

    mat_array['F'], mat_array['Fneu'] = F, Fneu
    return mat_array, dropped_frames, xy_off


def df_f_calculation(mat_array):
    """
    Calculates the delta F/F (fluorescence change over baseline fluorescence) for the given matrix array.

    Parameters:
    mat_array (dict): Dictionary containing the data arrays and operations.

    Returns:
    np.ndarray: The calibrated activity array.
    """
    F = mat_array['F']
    Fneu = mat_array['Fneu']
    is_cell = np.argwhere(mat_array['iscell'][:, 0] == 1)[:, 0]
    if DEBUG_FLAG:
        print(f"Cell Index : {is_cell}")
    central_activity, background_activity = F[is_cell], Fneu[is_cell]
    calibrated_activity = central_activity - 0.7 * background_activity
    return calibrated_activity


def read_xlsx(table_dir, header):
    """
    Reads an Excel file and converts each sheet to a numpy array.

    Parameters:
    table_dir (str): Directory of the Excel file.
    header (int): Row number to use as the column names.

    Returns:
    dict: A dictionary with sheet indices as keys and numpy arrays as values.
    """
    xl_file = pd.ExcelFile(table_dir)
    xlsx_dict = {sheet_id: xl_file.parse(sheet_name, header=header).to_numpy()
                 for sheet_id, sheet_name in enumerate(xl_file.sheet_names)}
    return xlsx_dict


#  useful func

def nanmean(a: np.ndarray | list):
    """
    Computes the mean of an array, ignoring NaN values.

    Parameters:
    a (np.ndarray or list): Input array or list.

    Returns:
    float: The mean of the array, ignoring NaN values.
    """
    if len(a) == 0:
        return np.nan
    else:
        return np.nanmean(a)


def nanmedian(a: np.ndarray | list):
    """
    Computes the median of an array, ignoring NaN values.

    Parameters:
    a (np.ndarray or list): Input array or list.

    Returns:
    float: The median of the array, ignoring NaN values.
    """
    if len(a) == 0:
        return np.nan
    else:
        return np.nanmedian(a)


def nanvar(a: np.ndarray | list):
    """
    Computes the variance of an array, ignoring NaN values.

    Parameters:
    a (np.ndarray or list): Input array or list.

    Returns:
    float: The variance of the array, ignoring NaN values.
    """
    if len(a) == 0:
        return np.nan
    else:
        return np.nanvar(a)


def nanstd(a: np.ndarray | list):
    """
    Computes the standard deviation of an array, ignoring NaN values.

    Parameters:
    a (np.ndarray or list): Input array or list.

    Returns:
    float: The standard deviation of the array, ignoring NaN values.
    """
    if len(a) == 0:
        return np.nan
    else:
        return np.nanstd(a)


def nanmax(a: np.ndarray | list):
    """
    Computes the maximum of an array, ignoring NaN values.

    Parameters:
    a (np.ndarray or list): Input array or list.

    Returns:
    float: The maximum of the array, ignoring NaN values.
    """
    if len(a) == 0:
        return np.nan
    else:
        return np.nanmax(a)


def nanmin(a: np.ndarray | list):
    """
    Computes the minimum of an array, ignoring NaN values.

    Parameters:
    a (np.ndarray or list): Input array or list.

    Returns:
    float: The minimum of the array, ignoring NaN values.
    """
    if len(a) == 0:
        return np.nan
    else:
        return np.nanmin(a)


def nansum(a: np.ndarray | list):
    """
    Computes the sum of an array, ignoring NaN values.

    Parameters:
    a (np.ndarray or list): Input array or list.

    Returns:
    float: The sum of the array, ignoring NaN values.
    """
    if len(a) == 0:
        return np.nan
    else:
        return np.nansum(a)
