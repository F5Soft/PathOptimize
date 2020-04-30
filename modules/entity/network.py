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
        """node_classification：顶点分类，属于图神经网络的内容，可以搜相关资料写进文档，这里直接调个包"""
        for t in self.trucks:
            t.coverage.clear()
            t.coverage.add(0)
        for u, t in enumerate(partition):
            self.trucks[t].coverage.add(u)
        for t in self.trucks:
            t.subgraph = self.graph.subgraph(t.coverage)

    def path_generate(self, truck: Truck):
        """
        生成一个卡车的回路
        :param truck: 待生成回路的卡车
        :return: None

        数学模型：dp[i][v]表示在状态v下，到达城市i所用的最小花费。其中状态v用二进制数来表示
        右起第i位表示第i个城市的状态，0为未拜访，1为已拜访。则状态转移方程为：
            dp[i][v]=min(dp[i][v],dp[k][v^(1<<(i-1)]+dist[k][i])
        其中k从1到n依次取值来得到最小值，dist[k][i]表示城市k到城市i的路程，^为异或运算，v^(1<<i-1)将v中第i位置0
        即dp[k][v^(1<<(i-1)]表示未访问城市i的状态下，到达城市k的花费。
        时间复杂度：Floyd预处理为O(n^3)，枚举各种状态的时间复杂度为O(2^n*n^2)，总的时间复杂度即为O(2^n*n^2)
        """

        predecessors, dists = nx.floyd_warshall_predecessor_and_distance(truck.subgraph)
        n = len(truck.subgraph)
        mapping = np.zeros(n)

        for i, u in enumerate(truck.subgraph):
            mapping[i] = u
        dp = np.zeros((n, 1 << (n - 1)))
        path = dict()
        for i in range(1, n):
            u = int(mapping[i])
            dp[i, 0] = dists[u][0]
            path[u] = {0: u}
        path[0] = dict()

        for status in range(1 << (n - 1) - 1):
            for i in range(1, n):
                u = int(mapping[i])
                dist_opt = np.inf
                path_opt = []
                for v in truck.subgraph[u]:
                    if v != 0 and (1 << (v - 1)) | status == status:
                        dist = dp[v, status ^ (1 << (v - 1))] + dists[u][v]
                        if dist < dist_opt:
                            dist_opt = dist
                            path_opt = nx.reconstruct_path(u, v, predecessors).pop() + path[v][status ^ (1 << (v - 1))]
                if not np.isinf(dist_opt):
                    dp[i, status] = dist_opt
                    path[u][status] = path_opt

        i = 0
        status = 1 << (n - 1) - 1
        u = int(mapping[i])
        dist_opt = np.inf
        path_opt = []
        for v in truck.subgraph[u]:
            if v != 0 and (1 << (v - 1)) | status == status:
                dist = dp[v, status ^ (1 << (v - 1))] + dists[u][v]
                if dist < dist_opt:
                    dist_opt = dist
                    path_opt = nx.reconstruct_path(u, v, predecessors) + path[v][status ^ (1 << (v - 1))]
        if not np.isinf(dist_opt):
            dp[i, status] = dist_opt
            path[u][status] = path_opt
        else:
            raise nx.NetworkXError
        print(dp[i, 0])
        print(path[u][0])

        # todo TSP算法Debug

    def coverage_mutation(self, mutation_rate=1):
        """
        按照一定概率交换两个卡车的部分配送范围
        :param mutation_rate: 交换概率
        :return: None
        """
        # todo

        pass

    def gexf_summary(self):
        nx.write_gexf(self.graph, "../../static/gexf/network.gexf")
