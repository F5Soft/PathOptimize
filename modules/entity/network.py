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

        # todo 顶点分类中考虑载重上限的情况

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
        subgraph = truck.subgraph.copy()
        predecessors, dists = nx.floyd_warshall_predecessor_and_distance(subgraph)
        subgraph.remove_node(0)  # 去除原图中的顶点0

        n = len(subgraph)
        status_max = 1 << n
        dp = [{} for i in range(n)]  # dp[i][status]表示从编号i城市出发，经过status中为1的顶点并回到配送中心的最小花费
        path = [{} for i in range(n)]  # path[i][status]表示上述最小花费所走的路径
        mapping = {u: i for i, u in enumerate(subgraph)}  # 顶点到顶点编号的映射
        for u in subgraph:
            dp[mapping[u]][0] = dists[u][0]  # 状态为0说明直接返回配送中心，因此距离即为当前顶点到配送中心的距离
            path[mapping[u]][0] = nx.reconstruct_path(u, 0, predecessors)  # 并且上述距离对应的路径即为u直接到0

        for status in range(status_max):
            for u in subgraph:
                dist_opt = np.inf
                path_opt = []
                for v in subgraph[u]:
                    v_map = mapping[v]
                    status_v = 1 << v_map
                    if status_v | status == status:  # 如果v属于当前的状态中
                        dist = dists[u][v] + dp[v_map][status ^ status_v]  # 那么可将v去除
                        if dist < dist_opt:
                            dist_opt = dist
                            path_opt = nx.reconstruct_path(u, v, predecessors)[:-1] + path[v_map][status ^ status_v]
                if not np.isinf(dist_opt):
                    u_map = mapping[u]
                    dp[u_map][status] = dist_opt
                    path[u_map][status] = path_opt

        status = status_max - 1
        dist_opt = np.inf
        path_opt = []
        for v in truck.subgraph[0]:
            v_map = mapping[v]
            status_v = 1 << v_map
            dist = dists[0][v] + dp[v_map][status ^ status_v]
            if dist < dist_opt:
                dist_opt = dist
                path_opt = nx.reconstruct_path(0, v, predecessors)[:-1] + path[v_map][status ^ status_v]

        print(path_opt)
        print(dist_opt)
        truck.d = dist_opt
        truck.path = path_opt

    def coverage_mutation(self, mutation_rate=1):
        """
        按照一定概率交换两个卡车的部分配送范围
        :param mutation_rate: 交换概率
        :return: None
        """
        # todo

        pass

    def gexf_summary(self):
        nx.write_gexf(self.graph, "static/gexf/test.gexf")
