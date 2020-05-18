"""
车辆模块
"""

import networkx as nx
import numpy as np


class Truck:
    """
    车辆类
    """
    def __init__(self, name: str, d_max: float, w_max: float):
        """
        :param name: 车辆名称
        :param d_max: 车辆距离上限
        :param w_max: 车辆载重上限
        """
        self.name = name
        self.d_max = d_max
        self.w_max = w_max
        self.d = 0  # 当前载重
        self.w = 0  # 当前距离
        self.gene = 0  # 对应图中的顶点（遗传算法染色体）
        self.coverage = set()  # 车辆覆盖范围
        self.subgraph = nx.Graph()  # 车辆覆盖范围生成的子图
        self.path = []  # 车辆在子图中的路径

    def __repr__(self):
        return "[%s] d_max=%.2f w_max=%.2f d=%.2f w=%.2f path=" % (
            self.name, self.d_max, self.w_max, self.d, self.w) + str(self.path) + " coverage=" + str(self.coverage)

    def __copy__(self):
        cp = Truck(self.name, self.d_max, self.w_max)
        cp.d = self.d
        cp.w = self.w
        cp.coverage = self.coverage.copy()
        cp.path = self.path.copy()
        return cp

    def adaptive(self) -> float:
        """
        求单个车辆的适应度值
        :return: float 适应度值
        """
        # todo 优化适应度函数，使之既能在越界情况下值很小，又能反映满载率
        return Truck.adaptive_function(self.w, self.w_max) + Truck.adaptive_function(self.d, self.d_max)

    @staticmethod
    def adaptive_function(index: float, threshold: float) -> float:
        """
        某个指标的适应度函数：低于阈值时，正数指数衰减；高于阈值时，负数指数增长
        :param index: 指标值
        :param threshold: 指标上限阈值
        :return: float 单个指标适应度值
        """
        # todo 优化适应度函数，使之既能在越界情况下值很小，又能反映满载率
        if index > threshold:
            return -np.exp(index - threshold)
        else:
            return np.exp(index - threshold)
