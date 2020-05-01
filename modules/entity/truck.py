import networkx as nx


class Truck:
    """
    卡车类
    Author: bsy
    Date: 2020-04-27
    """

    def __init__(self, name: str, d_max: float, w_max: float):
        """
        :param name: 卡车名称
        :param d_max: 卡车距离上限
        :param w_max: 卡车载重上限
        """
        self.name = name
        self.d_max = d_max
        self.w_max = w_max
        self.d = 0  # 当前载重
        self.w = 0  # 当前距离
        self.coverage = set()  # 卡车覆盖范围
        self.subgraph = nx.Graph()  # 卡车覆盖范围生成的子图
        self.path = []  # 卡车在子图中的路径

    def __repr__(self):
        return "[%s] d_max=%.2f w_max=%.2f d=%.2f w=%.2f path=" % (
            self.name, self.d_max, self.w_max, self.d, self.w) + str(self.path)

    def __copy__(self):
        cp = Truck(self.name, self.d_max, self.w_max)
        cp.d = self.d
        cp.w = self.w
        cp.coverage = self.coverage.copy()
        cp.path = self.path.copy()
        return cp
