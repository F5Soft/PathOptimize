from networkx import Graph
from .truck import Truck


class Network(Graph):
    """
    配送网络类
    Author: bsy, pq
    Date: 2020-04-22
    """

    def __init__(self, trucks: Truck, **attr):
        """
        :param base_graph: 配送网络的基图
        :param trucks: 配送网络具有的车辆
        :param attr: 父类的初始化参数
        """
        super().__init__(**attr)
        self.trucks = trucks