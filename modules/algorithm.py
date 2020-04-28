import numpy as np

from .entity import Truck, Network

trucks = []
network = None


def set_trucks():
    """
    配置货车
    :param data: 元组列表，列表长度为卡车数，元组的两项项代表最大距离和最大载重
    :return: None
    """
    global trucks
    trucks = [Truck("蓝翔挖掘机", 3, 10), Truck("顺丰快递车", 6, 8)]


def set_network():
    """
    配置运送网络
    :param adj_matrix: 网络的邻接矩阵二维列表，（元素double型, inf为不可达）
    :return: None
    """
    global trucks, network
    dists = np.array([[0, 7.4, 12.1, 5.3], [7.4, 0, 5.8, 9], [12.1, 5.8, 0, 12.2], [5.3, 9, 12.2, 0]])
    network = Network(dists, trucks)


def optimize():
    pass
