"""Motion-border helpers vendored from ``aind-ophys-utils``.

This module is a copy of the minimal implementation used by lamf_analysis
from ``aind-ophys-utils`` version 0.2.0, copied on 2026-09-28. It is kept
locally to avoid installing that package's unrelated JAX/Torch dependencies.
"""

from collections import namedtuple

import numpy as np
import pandas as pd


MaxFrameShift = namedtuple("MaxFrameShift", ["left", "right", "up", "down"])


def get_max_correction_values(
    x_series: pd.Series, y_series: pd.Series, max_shift: float = 30.0
) -> MaxFrameShift:
    """
    Gets the max correction values in the cardinal directions from a series
    of correction values in the x and y directions

    Parameters
    ----------
    x_series: pd.Series:
        A series of movements in the x direction
    y_series: pd.Series:
        A series of movements in the y direction
    max_shift: float
        Maximum shift to allow when considering motion correction. Any
        larger shifts are considered outliers (only absolute value matters).

    For deprecated implementation see:
    allensdk.internal.brain_observatory.roi_filter_utils.calculate_max_border

    Returns
    -------
    MaxFrameShift
        A named tuple containing the maximum correction values found during
        motion correction workflow step. Saved with the following direction
        order [left, right, up, down].

    """
    # take abs of max shift as we are considering both positive and negative
    # directions
    max_shift = abs(max_shift)

    # filter based out analomies based on maximum_shift
    x_no_outliers = x_series[
        (x_series >= -max_shift) & (x_series <= max_shift)
    ]
    y_no_outliers = y_series[
        (y_series >= -max_shift) & (y_series <= max_shift)
    ]
    # calculate max border shifts
    right_shift = -1 * x_no_outliers.min()
    left_shift = x_no_outliers.max()
    down_shift = -1 * y_no_outliers.min()
    up_shift = y_no_outliers.max()

    max_correction = MaxFrameShift(
        left=left_shift, right=right_shift, up=up_shift, down=down_shift
    )

    # check if all exist
    if np.any(np.isnan(np.array(max_correction))):
        raise ValueError(
            "One or more motion correction shifts was found to be NaN, max shift found: "
            f"{max_correction}, with max_shift {max_shift}"
        )

    return max_correction


def get_max_correction_from_df(
    input_df: pd.DataFrame, max_shift: float = 30.0
) -> MaxFrameShift:
    """

    Parameters
    ----------
    input_df: pd.Dataframe
        Pandas dataframe in the following format
        ['framenumber','x','y','correlation','kalman_x', 'kalman_y'] or
        ['framenumber','x','y','correlation','input_x','input_y','kalman_x',
         'kalman_y','algorithm','type']
    max_shift: float
        Maximum shift to allow when considering motion correction. Any
        larger shifts are considered outliers.

    Returns
    -------
    max_shift
        A named tuple containing the maximum correction values found during
        motion correction workflow step. Saved with the following direction
        order [left, right, up, down].

    """
    max_shift = get_max_correction_values(
        x_series=input_df["x"].astype("float"),
        y_series=input_df["y"].astype("float"),
        max_shift=max_shift,
    )
    return max_shift
