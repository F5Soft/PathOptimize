import random
from networkx import Graph
from .truck import Truck


class Network(Graph):
    """
    配送网络类
    Author: bsy, pq
    Date: 2020-04-24
    """

    def __init__(self, trucks, **attr):
        """
        :param base_graph: 配送网络的基图
        :param trucks: 配送网络具有的车辆列表
        :param attr: 图的初始化参数（点、边）
        """
        super().__init__(**attr)
        self.trucks = trucks

    def range_allocate(self):
        """
        分配各个卡车的配送范围
        :return: None
        """
        ranges = {i for i in range(1, len(self))}
        pass

    def range_mutation(self, mutation_rate=1):
        """
        按照一定概率交换两个卡车的部分配送范围
        :param mutation_rate: 交换概率
        :return: None
        """
        pass
