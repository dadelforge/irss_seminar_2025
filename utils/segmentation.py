"""
date: 2025-02-04
author: Damien Delforge
email: damien.delforge@uclouvain.be
license: MIT License
"""
from enum import Enum
from typing import Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.ndimage import (label, binary_erosion, binary_dilation,
                           binary_closing, binary_opening)

TimeSeriesLike = Union[
    np.ndarray,
    pd.Series,
    pl.Series,
    Sequence[float]
]


# Define an Enum for the comparison operators
class ComparisonOperator(Enum):
    EQ = np.equal  # Equal (==)
    NE = np.not_equal  # Not equal (!=)
    LT = np.less  # Less than (<)
    LE = np.less_equal  # Less than or equal (<=)
    GT = np.greater  # Greater than (>)
    GE = np.greater_equal  # Greater than or equal (>=)

    @classmethod
    def from_string(
            cls,
            op: "ComparisonOperator | str"
    ) -> "ComparisonOperator":
        """
        Converts a string into the corresponding ComparisonOperator enum value.

        Parameters
        ----------
        op : ComparisonOperator or str
            Either a ComparisonOperator enum value or a string representing
            the desired operator (e.g., 'eq', 'ne', 'lt', etc.).

        Returns
        -------
        ComparisonOperator
            The corresponding ComparisonOperator enum value.
        """
        if isinstance(op, cls):
            return op  # If already a ComparisonOperator, return as is

        # Map strings to ComparisonOperator
        operator_map = {
            "eq": cls.EQ,
            "ne": cls.NE,
            "lt": cls.LT,
            "le": cls.LE,
            "gt": cls.GT,
            "ge": cls.GE,
        }

        try:
            # Convert string to enum (case-insensitive)
            return operator_map[op.lower()]
        except KeyError:
            raise ValueError(
                f"Invalid operator string '{op}'. Valid options are: "
                f"{list(operator_map.keys())}"
            )


# Define an Enum for the postprocessing morphology operators
class MorphologyOperator(Enum):
    DILATION = binary_dilation
    EROSION = binary_erosion
    OPENING = binary_opening
    CLOSING = binary_closing
    NONE = lambda x: None

    @classmethod
    def from_string(
            cls,
            op: "MorphologyOperator | str | None"
    ) -> "MorphologyOperator":
        """Creates an instance of MorphologyOperator from a string

        Parameters
        ----------
        op : MorphologyOperator or str or None
            The input object or string to be converted into a MorphologyOperator.
            If None is provided, an appropriate MorphologyOperator instance is created
            or returned based on the method's internal logic.

        Returns
        -------
        MorphologyOperator
            An instance of the MorphologyOperator class derived from the provided
            input.
        """
        if isinstance(op, cls):
            return op  # If already a MorphologyOperator, return as is

        # Map strings to MorphologyOperator
        operator_map = {
            "dilation": cls.DILATION,
            "erosion": cls.EROSION,
            "opening": cls.OPENING,
            "closing": cls.CLOSING,
        }

        try:
            # Convert string to enum (case-insensitive)
            return operator_map[op.lower()]
        except KeyError:
            raise ValueError(
                f"Invalid operator string '{op}'. Valid options are: "
                f"{list(operator_map.keys())}"
            )


def segment_timeseries(
        timeseries: TimeSeriesLike,
        baseline: TimeSeriesLike | int | float | callable,
        operator: ComparisonOperator | str,
        morphology_operator: MorphologyOperator | str | None = None,
        **morphology_kwargs: dict[str, any]
):
    """
    Segments a timeseries based on a comparison condition with a baseline value.

    Parameters
    ----------
    timeseries : array-like
        One-dimensional input data representing the timeseries to be segmented.
        Must be convertible to a NumPy array.

    baseline : int, float, array-like, or callable
        The baseline value or callable to compare against. Can be:
        - A scalar (int or float)
        - An array with the same length as the input timeseries
        - A callable function that takes the input timeseries and returns a
          scalar or an array.

    operator : ComparisonOperator or str
        The comparison operator defining the segmentation condition. Can be
        provided as an instance of the `ComparisonOperator` class or a string
        representing a valid operator (e.g., 'eq', 'ne', 'lt', 'le', 'gt', or
        'ge').

    morphology_operator : MorphologyOperator or str or None, optional
        Postprocessing operator to apply to the segmented regions for
        noise reduction, filtering or grouping.

    Returns
    -------
    segmented : numpy.ndarray
        One-dimensional array where each element is assigned a label
        corresponding to its segmented region. Elements that do not meet the
        condition are assigned a zero value.

    num_segments : int
        Total number of unique segments identified in the timeseries where the
        comparison condition evaluates to true.
    """
    # Validate and convert operator
    comparison_operator = ComparisonOperator.from_string(operator)
    if morphology_operator is not None:
        morphology_operator = MorphologyOperator.from_string(
            morphology_operator)

    # Ensure timeseries is a NumPy array
    timeseries = np.asarray(timeseries)

    baseline = baseline(timeseries) if callable(baseline) else baseline
    baseline = np.asarray(baseline) if not isinstance(baseline, (
        int, float)) else baseline

    # Check if timeseries is 1D
    if timeseries.ndim != 1:
        raise ValueError("timeseries must be one-dimensional.")

    # Apply the comparison operator
    binary_segmented = comparison_operator.value(timeseries, baseline)

    # Apply morphology
    if morphology_operator is not None:
        binary_segmented = morphology_operator(binary_segmented,
                                               **morphology_kwargs)

    # Segment connected regions
    segmented, num_segments = label(binary_segmented)

    return segmented, num_segments


def extract_segments(
        timeseries: TimeSeriesLike,
        label: np.ndarray,
        num_segments: int,
        use_funcs: tuple[callable, ...] = ()
):
    timeseries = np.asarray(timeseries)
    segment_list: list[dict] = []

    for segment_label in range(1, num_segments + 1):
        segment_indices = np.ravel(np.argwhere(label == segment_label))
        segment_properties = {
            'label': segment_label,
            'indices': segment_indices,
            'segment': timeseries[segment_indices],
            'length': segment_indices.shape[0],
            'start_index': segment_indices[0],
            'end_index': segment_indices[-1],
        }
        for func in use_funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else str(
                func)
            segment_properties[func_name] = func(segment_properties['segment'])
        segment_list.append(segment_properties)

    return segment_list


if __name__ == '__main__':
    # Example
    from functools import partial

    np.random.seed(0)
    timeseries = np.random.standard_normal(100)
    baseline = partial(np.percentile, q=90)
    segmented, num_segments = segment_timeseries(
        timeseries, baseline,
        operator='gt')
    print(segmented)
    print(num_segments)
    segment_list = extract_segments(timeseries, segmented, num_segments,
                                    use_funcs=(np.mean, np.std))
    print(pd.DataFrame(segment_list))
