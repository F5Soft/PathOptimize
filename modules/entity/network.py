"""
配送网络模块
Author: bsy, pq
Date: 2020-05-03
"""

import copy
import random
from typing import List, Dict

import networkx as nx
import numpy as np

from modules.entity.truck import Truck


class Network:
    """
    配送网络类
    """
    def __init__(self, nodes: List, edges: List, trucks: List[Truck]):
        """
        :param dists: 初始化的邻接矩阵
        :param trucks: 配送网络具有的车辆列表
        """
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        self.trucks = trucks

    def __repr__(self):
        pr = ""
        d = 0.0
        for t in self.trucks:
            pr += str(t) + '\n'
            d += t.d
        return pr + "Total distance: %.2fkm" % d

    def __copy__(self):
        nodes = self.graph.nodes(data=True)
        edges = self.graph.edges(data=True)
        trucks = [copy.copy(t) for t in self.trucks]
        cp = Network(nodes, edges, trucks)
        return cp

    def coverage_init(self):
        """
        初始化网络中各个卡车的配送范围
        :return: None
        """
        n = len(self.graph)
        for t in self.trucks:
            t.gene = random.randrange(1, n)
        self.coverage_allocate()

    def coverage_mutation(self, mutation_rate=0.1):
        """
        使所有卡车的基因按照给定的概率突变
        :param mutation_rate: 突变概率
        :return: None
        """
        n = len(self.graph)
        for t in self.trucks:
            if random.random() < mutation_rate:
                t.gene = random.randrange(n)

    def coverage_recombination(self, network2):
        """

        :param network2:
        :return:
        """
        n = len(self.trucks)
        a = random.randrange(n)
        b = random.randrange(n)
        low = min(a, b)
        high = max(a, b)
        for i in range(low, high + 1):
            self.trucks[i].gene, network2.trucks[i].gene = network2.trucks[i].gene, self.trucks[i].gene

    def coverage_allocate(self):
        """
        通过各个卡车的基因情况分配卡车的配送范围
        :return: None
        """
        # 统计基因信息
        gene_count = np.zeros(len(self.graph), np.int)
        for t in self.trucks:
            gene_count[t.gene] += 1
        # 生成无配送中心的图，并将距离转为邻接程度
        graph = self.graph.copy()
        for u, v in graph.edges:
            graph[u][v]['weight'] **= -1
        # 根据基因给顶点标号
        for i, t in enumerate(self.trucks):
            if 'label' in graph.nodes[t.gene] and random.random() < 1 / gene_count[t.gene]:
                graph.nodes[t.gene]['label'] = i
            else:
                graph.nodes[t.gene]['label'] = i
        # 根据顶点标号进行顶点分类，属于图神经网络的内容，可以搜相关资料写进文档，这里直接调个包
        partition = nx.algorithms.node_classification.harmonic_function(graph)
        # 更新卡车范围
        for t in self.trucks:
            t.coverage.clear()
            t.coverage.add(0)
            t.w = 0
        for u, t in enumerate(partition):
            self.trucks[t].coverage.add(u)
            self.trucks[t].w += self.graph.nodes[u]['demand']
        for t in self.trucks:
            t.subgraph = self.graph.subgraph(t.coverage)

    def path_generate(self):
        """
        生成所有卡车在各自配送范围内的回路
        :return: None
        """
        for t in self.trucks:
            t.path = []
            if len(t.coverage) > 1:
                self.path_generate_for_truck(t)

    def path_generate_for_truck(self, truck: Truck):
        """
        生成一个卡车的回路
        数学模型：dp[i][status]表示从编号i城市出发，经过status中为1的顶点并回到配送中心的最小花费。其中状态status用二进制数来表示
        右起第i位表示第i个城市的状态，0为未拜访，1为已拜访。设i, j为顶点映射到整数的编号，u, v为对应的真实的顶点编号，则状态转移方程为：
        ```dp[i][status] = min(dists[u][v] + dp[i][status], dp[v][status^(1<<j)])```
        其中v是u的所有邻接顶点，依次取值来得到最小值。dists[u][v]表示城市到城市u到v的最短路程。
        时间复杂度：Floyd预处理为O(n^3)，枚举各种状态的时间复杂度为O(2^n*n^2)，总的时间复杂度即为O(2^n*n^2)
        :param truck: 待生成回路的卡车
        :return: None
        """
        # todo 如果卡车子图顶点数较多，该算法会GG，考虑增加贪心算法求次优TSP
        # 取子图，求Floyd
        subgraph = truck.subgraph.copy()
        predecessors, dists = nx.floyd_warshall_predecessor_and_distance(subgraph)
        subgraph.remove_node(0)  # 去除原图中的顶点0
        # 初始化动态规划数组及顶点映射
        n = len(subgraph)
        status_max = 1 << n
        dp = [{} for i in range(n)]  # dp[i][status]表示从编号i城市出发，经过status中为1的顶点并回到配送中心的最小花费
        path = [{} for i in range(n)]  # path[i][status]表示上述最小花费所走的路径
        mapping = {u: i for i, u in enumerate(subgraph)}  # 顶点到顶点编号的映射
        for u in subgraph:
            dp[mapping[u]][0] = dists[u][0]  # 状态为0说明直接返回配送中心，因此距离即为当前顶点到配送中心的距离
            path[mapping[u]][0] = nx.reconstruct_path(u, 0, predecessors)  # 并且上述距离对应的路径即为u直接到0
        # 动态规划配送网点
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
        # 动态规划配送中心
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
        # 更新卡车路径
        truck.d = dist_opt
        truck.path = path_opt

    def adaptive(self):
        """
        求该配送网络的适应度值，用于自然选择，为该网络的平均卡车适应度值
        :return: float 适应度值
        """
        # todo 优化适应度函数，使之既能反映单个卡车的适应度，又能反映总距离d_tot
        adaptive_mean = np.mean([t.adaptive() for t in self.trucks if len(t.coverage) > 1])  # 各个有任务的卡车适应度平均
        d_tot = np.sum([t.d for t in self.trucks if len(t.coverage) > 1])  # 总路程
        return adaptive_mean / d_tot

    def gexf_summary(self, filename: str):
        """
        导出图像的gexf信息，用于echarts插件读取
        :param filename: 文件名
        :return: None
        """
        nx.write_gexf(self.graph, "static/gexf/" + filename)
