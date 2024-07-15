#!/usr/bin/env python

from typing import Any, List, Sequence, TypeVar, Union

import matplotlib.colors as mcolors
import numpy as np
from numpy.typing import NDArray
from std_msgs.msg import ColorRGBA, MultiArrayDimension

# 各種描画に使う色を定義
# list(mcolors.TABELAU_COLORS.values())も使用可能
color_list = ["r", "g", "b", "m", "c", "y"]


def padding(
    original_array: Union[Sequence, NDArray],
    return_list_length: int = 3,
    padding_value: float = 0.0,
) -> List[float]:
    """1~3次元それぞれへの対応として，後半部分を0 or 指定した値で埋める

    Args:
        original_array (Sequence): 元の配列
        return_list_length (int): 埋めた後のリストのサイズ．Defaults to 3.
        padding_value (float): 埋める際に用いる値. Defaults to 0.

    Returns:
        List[float]: 埋めた結果のリスト

    Note:
        結果を渡す先にPoint, Vector3を想定しているのでreturn_list_lengthのデフォルト値を3に設定している．
    """
    original_array_length = len(original_array)
    return [(original_array[i] if i < original_array_length else padding_value) for i in range(return_list_length)]


def get_color_rgba(color: str, alpha: float = 1.0) -> ColorRGBA:
    return ColorRGBA(**dict(zip(["r", "g", "b", "a"], mcolors.to_rgba(color, alpha))))


def get_random_color_rgba(alpha: float = 1.0) -> ColorRGBA:
    return ColorRGBA(**dict(zip(["r", "g", "b", "a"], [*np.random.random(3), alpha])))


MultiArray = TypeVar("MultiArray")


def ndarray_to_multiarray(multiarray_type: MultiArray, ndarray: NDArray) -> MultiArray:
    """Convert numpy.ndarray to multiarray"""
    multiarray = multiarray_type()  # type: ignore
    multiarray.layout.dim = [
        MultiArrayDimension(label=f"dim{i}", size=ndarray.shape[i], stride=ndarray.shape[i] * ndarray.dtype.itemsize)
        for i in range(ndarray.ndim)
    ]
    multiarray.data: List[float] = ndarray.reshape(1, -1)[0].tolist()  # type: ignore
    return multiarray  # type: ignore


def multiarray_to_ndarray(pytype: Any, dtype: Any, multiarray: MultiArray) -> NDArray:
    """Convert multiarray to numpy.ndarray"""
    dims = [multiarray.layout.dim[i].size for i in range(len(multiarray.layout.dim))]  # type: ignore
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)  # type: ignore


def smooth_ramp(x: Union[float, NDArray]) -> NDArray:
    """連続なランプ関数

    Note:
        f(x) = \exp{-SR(x^2-R^2)^2}で山が平らに潰れた凹型の関数が得られる
        ____      ____
            \____/
    """
    return x * (np.arctan(x) / np.pi + 1 / 2)
