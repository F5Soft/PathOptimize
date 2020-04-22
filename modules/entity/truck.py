import copy
import random
from networkx import Graph


class Truck:
    """
    卡车类
    Author: bsy
    Date: 2020-04-22
    """

    def __init__(self, base_graph: Graph, d_max: float, w_max: float):
        """
        :param base_graph: 卡车所在的基图
        :param d_max: 卡车距离上限
        :param w_max: 卡车载重上限
        """
        self.G = base_graph
        self.d_max = d_max
        self.w_max = w_max
        self.d = 0
        self.w = 0
        self.range = set()  # 卡车覆盖范围
        self.path = []  # 卡车路径
        self.len = 0  # 卡车路径长度

    def __repr__(self):
        return "Dist: {:.2f}, Weight: {:.2f}, Path: ".format(self.d, self.w) + str(self.path)

    def __copy__(self):
        cp = Truck(self.G, self.d_max, self.w_max)
        cp.d = self.d
        cp.w = self.w
        cp.range = self.range.copy()
        cp.path = self.path.copy()
        cp.len = self.len
        return cp

    def caculate_w(self):
        """
        重新计算当前载重并返回
        :return: 当前载重：float
        """
        self.w = sum([self.G.nodes[i]['w'] for i in self.path])
        return self.w

    def caculate_d(self):
        """
        重新计算当前距离并返回
        :return: 当前距离：float
        """
        self.d = sum([self.G[self.path[i]][self.path[i + 1]]['d'] for i in range(self.len - 1)]
                     ) + self.G[0][self.path[0]]['d'] + self.G[self.path[-1]][0]['d']
        return self.d

    def path_allocate(self):
        """
        在给定range下，分配一个路径path并计算d
        :return: None
        """
        self.path = list(self.range)
        self.len = len(self.path)
        random.shuffle(self.path)
        self.caculate_w()
        self.caculate_d()

    def path_mutation(self):
        """
        返回一个range不变，而路径突变的Truck对象。如果突变后的路径满足约束，则返回新的Truck对象，否则仍然返回自己
        :return: 突变后的卡车：Truck
        """
        if self.len < 2:
            return self
        cp = copy.copy(self)
        [x, y] = random.sample(range(cp.len), 2)
        cp.path[x], cp.path[y] = cp.path[y], cp.path[x]
        cp.caculate_d()
        if cp.d > cp.d_max:
            return self
        return cp
