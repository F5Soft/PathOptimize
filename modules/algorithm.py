from .entity import Truck, Network

trucks = []
network = None


def set_trucks(data):
    """
    配置货车
    :param data: 元组列表，列表长度为卡车数，元组的两项项代表最大距离和最大载重
    :return: None
    """
    trucks[0] = Truck(0, 0, 0)
    pass


def set_network(adj_matrix):
    """
    配置运送网络
    :param adj_matrix: 网络的邻接矩阵二维列表，（元素double型, inf为不可达）
    :return: None
    """
    networks = Network(trucks)
    pass


def optimize():
    pass
