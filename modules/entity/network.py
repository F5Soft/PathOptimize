import math
import random
from typing import List

import networkx as nx
import numpy as np

from .truck import Truck


class Network:
    """
    配送网络类
    Author: bsy, pq
    Date: 2020-04-27
    """

    def __init__(self, dists: np.ndarray, trucks: List[Truck]):
        """
        :param dists: 初始化的邻接矩阵
        :param trucks: 配送网络具有的车辆列表
        """
        self.graph = nx.from_numpy_array(dists)
        self.trucks = trucks
        self.graph.add_node(0, name='配送中心')

    def coverage_allocate(self):
        """
        分配各个卡车的配送范围
        :return: None
        """
        trucks_len = len(self.trucks)
        for u in self.graph:
            if u != 0 and not math.isinf(self.graph[0][u]['weight']):
                self.graph.nodes[u]['label'] = random.randrange(trucks_len)
        partition = nx.algorithms.node_classification.harmonic_function(self.graph)
        for t in self.trucks:
            t.coverage.clear()
            t.coverage.add(0)
        for u, t in enumerate(partition):
            self.trucks[t].coverage.add(u)
        for t in self.trucks:
            t.subgraph = self.graph.subgraph(t.coverage)

    def path_generate(self, truck: Truck):
        predecessors, dists = nx.floyd_warshall_predecessor_and_distance(truck.subgraph)
        n = len(truck.subgraph)
        dp = np.zeros((n, 2 ** n))
        for i in range(n):
            pass

        # todo 状态压缩动态规划算法求单个卡车的可重复TSP问题
        """
        数学模型：dp[i][v]表示在状态v下，到达城市i所用的最小花费。其中状态v用二进制数来表示
        右起第i位表示第i个城市的状态，0为未拜访，1为已拜访。则状态转移方程为：
            dp[i][v]=min(dp[i][v],dp[k][v^(1<<(i-1)]+map[k][i])
        其中k从1到n依次取值来得到最小值，map[k][i]表示城市k到城市i的路程，^为异或运算，v^(1<<i-1)将v中第i位置0
        即dp[k][v^(1<<(i-1)]表示未访问城市i的状态下，到达城市k的花费。
        时间复杂度：Floyd预处理为O(n^3)，枚举各种状态的时间复杂度为O(2^n*n^2)，总的时间复杂度即为O(2^n*n^2)
        """

    def coverage_mutation(self, mutation_rate=1):
        """
        按照一定概率交换两个卡车的部分配送范围
        :param mutation_rate: 交换概率
        :return: None
        """
        pass

    def gexf_summary(self):
        nx.write_gexf(self.graph, "../../static/gexf/network.gexf")
