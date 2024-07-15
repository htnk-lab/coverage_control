#!/usr/bin/env python

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Voronoi:
    """ボロノイ領域を管理

    Attributes:
        p (float): p-norm
        radius (float): r-limited voronoi計算における半径．
                        FOVの規定に用いる．Defaults to float("inf")

    Note:
        各領域内の離散点に相当する位置にはTrue，それ以外にはFalseを割り当てる．
        記号表記はH.Dan et al. 2020に倣う

        voronoi_region (NDArray): ボロノイ領域（V_{i}）
        fov_region (NDArray): 円形センサーモデルによる視野（B_{i}）．
                              デフォルトではボロノイ領域のみを考慮するよう半径を無限大でとっている．
        sensing_region (NDArray): センシング領域（S_{i}=V_{i} \bigcap B_{i}）
    """

    p: float = 2
    radius: float = float("inf")

    def calc_tesselation(
        self,
        agent_position: NDArray,
        neighbor_agent_position_list: List[NDArray],
        phi: NDArray,
        grid_map: List[NDArray],
        point_density: float,
    ) -> Tuple[NDArray, List[NDArray], NDArray]:
        """ボロノイ分割を計算

        Args:
            agent_position (NDArray): ボロノイ領域の母点
            neighbor_agent_position_list (List[NDArray]): 近隣の母点．少なくとも近隣が全て含まれていれば良いため，
                                                          その他の母点が含まれていても，計算上は特に問題はない．
            phi (NDArray): 重要度マップ
            grid_map (List[NDArray]): フィールド離散点の各座標．[x_grid_map, y_grid_map, ...]
            point_density (float): 離散点の密度．離散点1つの担当する面積/体積に相当．

        Returns:
            Tuple[NDArray, List[NDArray], NDArray]:
            ボロノイ重心，ボロノイ領域を示す離散点の座標集合（grid_mapと合わせて座標軸ごとにまとめてリストの要素とする），ボロノイ領域内外を1/0で示したマップ
        """

        dim = len(grid_map)

        # ボロノイ領域を初期化
        voronoi_region: NDArray = np.ones_like(grid_map[0], dtype=np.bool_)

        assert self.p >= 1 and self.radius > 0
        distance_from_agent = sum([abs(grid_map[i] - agent_position[i]) ** self.p for i in range(dim)]) ** (1 / self.p)
        fov_region = distance_from_agent < self.radius

        for neighbor_agent_position in neighbor_agent_position_list:
            distance_from_neighbor_agent = sum(
                [abs(grid_map[i] - neighbor_agent_position[i]) ** self.p for i in range(dim)]
            ) ** (1 / self.p)

            # 近隣エージェント（neighbor_agent_position）より自分（agent_position）からの方が近い点を抽出
            near_region = distance_from_neighbor_agent > distance_from_agent
            # near_regionの積算により自分が最も近いエージェントとなる離散点を抽出
            voronoi_region *= near_region

        # ボロノイ領域と視野の積集合
        sensing_region = voronoi_region * fov_region

        # センシング領域の重心計算
        centroid_position = self.calc_centroid_position(grid_map, phi, sensing_region, point_density, dim)
        sensing_region_grid_points = [grid_map[i][sensing_region] for i in range(dim)]
        return centroid_position, sensing_region_grid_points, sensing_region

    @staticmethod
    def calc_centroid_position(
        grid_map: List[NDArray], phi: NDArray, region: NDArray, point_density: float, dim: int
    ) -> NDArray:
        weighted_grid_map = grid_map * phi * region
        mass = np.sum(phi * region) * point_density
        assert mass > 0

        centroid_position: NDArray = np.array([weighted_grid_map[i].sum() for i in range(dim)]) * point_density / mass
        return centroid_position
