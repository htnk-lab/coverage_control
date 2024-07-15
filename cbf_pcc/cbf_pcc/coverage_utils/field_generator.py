#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray


@dataclass
class FieldGenerator:
    """フィールド生成器

    Attributes:
        grid_accuracy (NDArray): フィールド離散化の精度を規定．limitで軸ごとに定めた範囲での格子点の数を示す．
        limit (NDArray): 軸ごとのフィールド範囲．[[x_min, x_max], ...]
        linspace (List[NDArray]): フィールド生成用の各軸ごとの座標集合
        grid_span (NDArray): 離散点1つの担当する格子幅

    Note:
        np.meshgridのindexing引数をijと設定することで，grid_mapはx, y, zの順で返る
    """

    grid_accuracy: NDArray
    limit: NDArray
    linspace: List[NDArray] = field(init=False)
    grid_span: NDArray = field(init=False)

    def __post_init__(self) -> None:
        dim = len(self.grid_accuracy)

        self.grid_span = (self.limit[0:dim, 1] - self.limit[0:dim, 0]) / self.grid_accuracy
        # フィールドを隙間なく埋めるため，limit値にオフセットをかける
        self.linspace = [
            np.linspace(
                start=self.limit[i][0] + self.grid_span[i] / 2,
                stop=self.limit[i][1] - self.grid_span[i] / 2,
                num=self.grid_accuracy[i],
            )
            for i in range(dim)
        ]

    def generate_phi(self) -> NDArray:
        """重要度分布を生成"""
        return np.ones(self.grid_accuracy)

    def generate_grid_map(self) -> List[NDArray]:
        """フィールド格子点を生成

        Returns:
            List[NDArray]: 格子点の各座標．[x_grid_map, y_grid_map, ...]
        """
        return np.meshgrid(*self.linspace, indexing="ij")


def test() -> None:
    limit = np.array([[-1, 1], [-2, 2], [-3, 3]])
    grid_accuracy = np.array([100, 200, 300])
    generator = FieldGenerator(grid_accuracy=grid_accuracy, limit=limit)
    x_grid_map, y_grid_map, z_grid_map = generator.generate_grid_map()
    print(f"x.shape: {x_grid_map.shape}, y.shape: {y_grid_map.shape}, z.shape: {z_grid_map.shape}")


if __name__ == "__main__":
    test()
