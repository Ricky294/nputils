from __future__ import annotations

from typing import Iterable, Generator

import numpy as np
import pandas as pd


def to_array(x, /, dtype: object = None):
    """
    Converts parameter `x` to a numpy array.

    :return: numpy array

    :examples:
    >>> to_array(1)
    array([1])

    >>> to_array(1.5)
    array([1.5])

    >>> to_array([1, 2, 3])
    array([1, 2, 3])

    >>> to_array(np.array([1, 2, 3]))
    array([1, 2, 3])

    >>> to_array({"5": 10, "6": 12}, dtype=float)
    array([10., 12.])

    >>> to_array(pd.DataFrame({"x": [1, 3, 5], "y": [2, 4, 6]}), dtype=float)
    array([[1., 2.],
           [3., 4.],
           [5., 6.]])
    """

    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=dtype)
    elif isinstance(x, (list, tuple)):
        return np.array(x, dtype=dtype)
    elif isinstance(x, Generator):
        return np.array(list(x))
    elif isinstance(x, dict):
        return np.array(list(x.values()), dtype=dtype)

    return np.array([x], dtype=dtype)


def min_over_period(x: np.ndarray, /, period: int):
    """
    Returns the minimum values on `x` with lookback `period`.

    :param x: 1D iterable
    :param period: lookback period
    :return: numpy array

    :examples:
    >>> min_over_period(np.array([1, 2, 3, 4, 2, 1, 5, 3]), period=3)
    array([1, 1, 1, 2, 2, 1, 1, 1])
    """
    window = np.lib.stride_tricks.sliding_window_view(x, period)
    min_in_period = np.min(window, axis=-1)
    first_mins = window[0, :period - 1]
    return np.concatenate(
        ([first_mins[:i + 1].min() for i in range(len(first_mins))], min_in_period)
    )


def max_over_period(x: np.ndarray, /, period: int):
    """
    Returns the maximum values on `x` with lookback `period`.

    :param x: 1D iterable
    :param period: lookback period
    :return: numpy array

    :examples:
    >>> max_over_period(np.array([1, 2, 3, 4, 2, 1, 5, 3]), period=3)
    array([1, 2, 3, 4, 4, 4, 5, 5])
    """

    window = np.lib.stride_tricks.sliding_window_view(x, period)
    max_in_period = np.max(window, axis=-1)
    first_maxes = window[0, :period - 1]
    return np.concatenate(
        ([first_maxes[:i + 1].max() for i in range(len(first_maxes))], max_in_period)
    )


def absolute_change(x: np.ndarray, /) -> np.ndarray | float | int:
    """
    Returns the array's absolute change.

    :param x: numpy array with any dimension
    :return: constant (x has 1 dim) or numpy array (x has multiple dims)

    :examples:
    >>> absolute_change(np.array([1, 2, 3, -1]))
    6

    >>> absolute_change(np.array([[1, 2, 3, -1], [2, 1, 4, 1]]))
    array([6, 7])
    """
    abs_changes = np.abs(x[..., :-1] - x[..., 1:])
    return np.sum(abs_changes, axis=-1)


def sum_increase(x: np.ndarray, /) -> np.ndarray | float | int:
    """
    Sums differences between elements where value is greater than previous.

    :param x: numpy array with any number of dims
    :return: constant (x has 1 dim) or numpy array (x has multiple dims)

    :examples:
    >>> sum_increase(np.array([2, 3, 4, 3, 1]))
    2
    >>> sum_increase(np.array([[2, 3, 4, 3, 1], [-2, 2, 0, 1, -1]]))
    array([2, 5])
    """
    change = x[..., 1:] - x[..., :-1]
    inc = np.where(change > 0, change, 0)
    return np.sum(inc, axis=-1)


def sum_decrease(x: np.ndarray, /):
    """
    Sums differences between elements where value is less than previous.

    :param x: numpy array with any number of dims
    :return: constant (x has 1 dim) or numpy array (x has multiple dims)

    :examples:
    >>> sum_decrease(np.array([2, 3, 4, 3, 1]))
    3
    >>> sum_decrease(np.array([[2, 3, 4, 3, 1], [-2, 2, 0, 1, -1]]))
    array([3, 4])
    """
    change = x[..., 1:] - x[..., :-1]
    dec = np.where(change < 0, change, 0)
    return np.abs(np.sum(dec, axis=-1))


def bottoms(x: np.ndarray, /, n=1):
    """
    Returns a boolean numpy array.
    True where at least `n` consecutive value decreases before, and increases after a value.

    :param x: 1d iterable
    :param n: lookback and lookahead period
    :return: boolean numpy array

    :examples:
    >>> a = np.array([1, 2, 3, 2, 1])
    >>> bottoms(a, n=1)
    array([False, False, False, False, False])

    >>> b = np.array([2, 1, 3, 4, 3, 2])
    >>> bottoms(b, n=1)
    array([False,  True, False, False, False, False])

    >>> bottoms(b, n=2)
    array([False, False, False, False, False, False])
    """

    inc_arr = increase(x, n)
    dec_arr = decrease(x, n)

    dec_inc_arr = np.all(np.vstack((dec_arr[:-n], inc_arr[n:])), axis=0)
    return np.concatenate((dec_inc_arr, np.full(n, False)))


def peaks(x: np.ndarray, /, n=1):
    """
    Returns a boolean numpy array.
    True where at least `n` consecutive value increases before, and decreases after a value.

    :param x: 1d iterable
    :param n: lookback and lookahead period
    :return: boolean numpy array

    :examples:
    >>> a = np.array([1, 2, 1, 2, 3])
    >>> peaks(a, n=1)
    array([False,  True, False, False, False])

    >>> a = np.array([1, 2, 1, 2, 3])
    >>> peaks(a, n=2)
    array([False, False, False, False, False])

    >>> b = np.array([1, 2, 3, 2, 1])
    >>> peaks(b, n=2)
    array([False, False,  True, False, False])
    """

    inc_arr = increase(x, n)
    dec_arr = decrease(x, n)

    dec_inc_arr = np.all(np.vstack((dec_arr[n:], inc_arr[:-n])), axis=0)
    return np.concatenate((dec_inc_arr, np.full(n, False)))


def normalize(x: np.ndarray, /):
    """
    Normalizes values along all axes between 0 and 1.

    :param x: numpy array with any dimension
    :return: normalized array (values between 0 and 1)

    :examples:
    >>> a = np.array([1, 2, 3])
    >>> normalize(a)
    array([0. , 0.5, 1. ])

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> normalize(b)
    array([[0. , 0.2, 0.4],
           [0.6, 0.8, 1. ]])
    """
    return (x - np.min(x)) / np.ptp(x)


def replace_where_not_found(x1: np.ndarray, x2: np.ndarray, /, replace):
    """
    Replaces values with `replace` in `x1` where no equal element is present in `x2`.

    `x1` and `x2` can have different lengths.

    Note: Parameter assignment of `x1` and `x2` is not interchangeable.

    :param x1: 1D iterable
    :param x2: 1D iterable
    :param replace: Replacement value where condition is True.
    :return: numpy array

    :examples:
    >>> arr = np.array([5, 10, 11, 3, 1, 2])
    >>> replace_where_not_found(arr, np.array([5, 11, 2]), 0)
    array([ 5,  0, 11,  0,  0,  2])

    >>> replace_where_not_found(np.array([5, 11, 2]), arr, 0)
    array([ 5, 11,  2])

    >>> replace_where_not_found(arr, np.array([5, 5, 2, 2]), [1, 2, 3, 4])
    array([5, 1, 2, 3, 4, 2])
    """
    x1_copy = x1.copy()
    x2_copy = x2.copy()
    arr_matrix = x2_copy.reshape(x2_copy.size, 1)
    mask = np.all(x1_copy != arr_matrix, axis=0)

    return replace_where(x1_copy, mask, replace=replace)


def replace_where(x: np.ndarray, condition: np.ndarray, /, replace) -> np.ndarray:
    """
    Replaces values in `arr` where `condition` evaluates to True with `replace`.

    :param x: Numpy array in which elements get replaced based on condition.
    :param condition: Replacement condition.
    :param replace: Contains the replacement value(s).
    :return: numpy array
    :raises ValueError: replace must be constant or its length must equal with the condition mask True count.

    :examples:
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> replace_where(arr, arr > 3, 0)
    array([1, 2, 3, 0, 0])

    >>> replace_where(arr, arr > 3, [0, 1])
    array([1, 2, 3, 0, 1])

    >>> replace_where(arr, arr > 3, [0, 1, 2])
    ValueError: Cannot assign 3 input values to the 2 output values where the mask is true.
    """
    x_copy = x.copy()
    if isinstance(replace, Iterable):
        replace = to_array(replace)

    x_copy[condition] = replace
    return x_copy


def replace_with_previous_where(x: np.ndarray, condition: np.ndarray, /) -> np.ndarray:
    """
    Replaces values in `x` with its previous value where `condition` is True.

    :param x: 1D iterable
    :param condition: boolean 1D numpy array
    :return: numpy array

    :examples:
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> replace_with_previous_where(arr, arr > 2)
    array([1, 2, 2, 2, 2])

    >>> arr2 = np.array([1, 0, 0, 1, 0])
    >>> replace_with_previous_where(arr2, arr2 == 0)
    array([1, 1, 1, 1, 1])
    """
    x_copy = x.copy()
    prev = np.arange(len(x_copy))
    prev[condition] = 0
    prev = np.maximum.accumulate(prev)
    return x_copy[prev]


def increase(x: np.ndarray, n=1) -> np.ndarray:
    """
    Returns True where at least `n` consecutive value is increasing in `x`.

    :param x: numpy array with any number of dimensions
    :param n: Required number of increasing values to get True
    :return: boolean numpy array

    :examples:
    >>> a = np.array([1, 2, 1, 0, 1, 2])
    >>> increase(a, 1)
    array([False,  True, False, False,  True,  True])

    >>> increase(a, 2)
    array([False, False, False, False, False,  True])

    >>> b = np.array([[1, 2, 1, 0, 1, 2],
    ...               [1, 2, 3, 4, 3, 4]])
    >>> increase(b, 2)
    array([[False, False, False, False, False,  True],
           [False, False,  True,  True, False, False]])
    """

    shape = list(x.shape[0:-1])
    shape.append(n)
    padding = np.full(shape, False)
    inc = np.concatenate((padding, x[..., :-1] < x[..., 1:]), axis=-1)

    if n == 1:
        return inc

    inc_window = np.lib.stride_tricks.sliding_window_view(inc, n, axis=-1)
    return np.all(inc_window, axis=-1)


def decrease(x: np.ndarray, n=1) -> np.ndarray:
    """
    Returns True where at least `n` consecutive value is decreasing in `x`.

    :param x: numpy array with any number of dimensions
    :param n: Required number of decreasing values to get True
    :return: boolean numpy array

    :examples:
    >>> a = np.array([1, 2, 1, 0, 1, 2])
    >>> decrease(a, 1)
    array([False, False,  True,  True, False, False])

    >>> decrease(a, 2)
    array([False, False, False,  True, False, False])

    >>> b = np.array([[1, 2, 1, 0, 1, 2],
    ...               [4, 3, 4, 3, 2, 1]])
    >>> decrease(b, 2)
    array([[False, False, False,  True, False, False],
           [False, False, False, False,  True,  True]])
    """

    shape = list(x.shape[0:-1])
    shape.append(n)
    padding = np.full(shape, False)
    dec = np.concatenate((padding, x[..., :-1] > x[..., 1:]), axis=-1)

    if n == 1:
        return dec

    dec_window = np.lib.stride_tricks.sliding_window_view(dec, n, axis=-1)
    return np.all(dec_window, axis=-1)


def change(x: np.ndarray, /):
    """
    Substracts adjacent values in `x`.

    :param x: numeric numpy array
    :return: numeric numpy array

    :examples:
    >>> change(np.array([5, 10, 9, 7]))
    array([ 0,  5, -1, -2])
    """

    return np.concatenate(([0], x[1:] - x[:-1]))


def shift(x: np.ndarray | Iterable | int | float, num: int, /, fill_value=np.nan, dtype=float):
    """
    Left or right shifts `x` by `num` and fills the shifter values with `fill_values`.

    :param x: Numpy array to shift.
    :param num: Left shifts (if less than 0) or right shifts (if greater than 0) x, this number of times.
    :param fill_value: Shifted values are replaced by this.
    :param dtype: Numpy data type
    :return: numpy array

    :examples:
    >>> x = [1, 2, 3, 4]
    >>> shift(x, 2)
    array([nan, nan,  1.,  2.])

    >>> shift(x, -2, fill_value=0)
    array([3., 4., 0., 0.])
    """

    result = np.empty_like(x, dtype=dtype)
    if num > 0:
        result[:num] = fill_value
        result[num:] = x[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = x[-num:]
    else:
        result[:] = x
    return result
