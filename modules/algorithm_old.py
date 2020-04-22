import math
import random

import matplotlib.pyplot as plt
import networkx as nx

# 先建一个图测试，w表示节点的所需的货物重量，d表示两个节点的距离，0是配送中心
G = nx.Graph()
G.add_node(0)
G.add_node(1, w=3)
G.add_node(2, w=7)
G.add_node(3, w=5)
G.add_node(4, w=2)
G.add_edge(0, 1, d=0.3)
G.add_edge(0, 2, d=0.5)
G.add_edge(0, 3, d=0.9)
G.add_edge(0, 4, d=0.6)
G.add_edge(1, 2, d=3)
G.add_edge(2, 3, d=5)
G.add_edge(3, 4, d=3)
G.add_edge(1, 4, d=3)
G.add_edge(2, 4, d=1)
G.add_edge(1, 3, d=2)


class Truck:
    '''
    卡车类，一个卡车对象包括出发前重量w，距离d，载重上限w_max，距离上限d_max，
    对接的配送点range，路径path(range的一个排列)，路径长度len
    '''

    def __init__(self, d_max, w_max):
        self.w_max = w_max
        self.d_max = d_max
        self.w = 0
        self.d = 0
        self.range = set()
        self.path = []
        self.len = 0

    def __repr__(self):
        return "Dist: {:.2f}, Weight: {:.2f}, Path: ".format(self.d, self.w)+str(self.path)

    def copy(self):
        '''
        复制自己，返回一份新的，避免由于传递的是对象地址而修改本身
        '''
        cp = Truck(self.d_max, self.w_max)
        cp.w = self.w
        cp.d = self.d
        cp.range = self.range.copy()
        cp.path = self.path.copy()
        cp.len = self.len
        return cp

    def path_allocate(self):
        '''
        在给定range下，分配一个路径path并计算d
        '''
        self.path = list(self.range)
        self.len = len(self.path)
        random.shuffle(self.path)
        self.w = sum([G.nodes[i]['w'] for i in self.path])
        self.d = sum([G[self.path[i]][self.path[i+1]]['d'] for i in range(self.len-1)]
                     ) + G[0][self.path[0]]['d'] + G[self.path[-1]][0]['d']

    def path_mutation(self):
        '''
        返回一个range不变，而路径突变的Truck对象。
        如果突变后的路径满足约束，则返回新的Truck对象，否则仍然返回自己
        '''
        if self.len < 2:
            return self
        cp = self.copy()
        [x, y] = random.sample(range(cp.len), 2)
        cp.path[x], cp.path[y] = cp.path[y], cp.path[x]
        cp.d = sum([G[cp.path[i]][cp.path[i+1]]['d'] for i in range(cp.len-1)]
                   ) + G[0][self.path[0]]['d'] + G[self.path[-1]][0]['d']
        if cp.d > cp.d_max:
            return self
        return cp


trucks = [Truck(5, 12) for i in range(2)]  # 模拟2个卡车，每个卡车距离上限5，载中上限12
ranges = {i for i in range(1, len(G))}  # {1,2,3,4}


def range_allocate():
    '''
    ranges确定时，给每个卡车分配互相不重叠的range，并且满足载重量和距离约束
    这个函数性能不佳，且数值给小了会死循环，需要想办法优化！！！！！！！！！！！！
    '''
    r = ranges.copy()
    while r:
        for t in trucks:
            if r:
                i = random.choice(list(r))
                if G.nodes[i]['w']+t.w > t.w_max:
                    break
                t.range.add(i)
                t.path_allocate()
                if t.d > t.d_max:
                    t.range.remove(i)
                    t.path_allocate()
                    break
                r.remove(i)


def range_mutation():
    '''
    取两个卡车，这两个卡车交换各自的一部分range，进行突变
    待实现！！！！！！！！！！！！
    '''
    mutation_rate = 0.6
        if random.random() < mutation_rate:
             i = random.randint(0, len(Network.trucks))
              j = random.randint(0, len(Network.trucks))
               while i != j:
                    j = random.randint(0, len(Network.trucks))
                point1 = random.randint(0, len(Network.trucks[i].ranges[i]))
                point2 = random.randint(0, len(Network.trucks[j].ranges[j]))
                t = Network.trucks[i].ranges[point1]
                Network.trucks[i].ranges[point1] = Network.trucks[j].ranges[point2]
                Network.trucks[j].ranges[point2] = t
    pass


if __name__ == '__main__':

    T = 10000  # 温度10000
    range_allocate()
    while T:  # 直到温度为0
        i = random.randrange(0, len(trucks))
        if trucks[i].range:
            t = trucks[i].path_mutation()
            if t.d <= trucks[i].d:  # 如果突变后的距离小于原先距离，接受
                trucks[i] = t
            # 否则按一定概率接受
            elif random.random() < 1/(1+math.exp(-(t.d-trucks[i].d)/T)):
                trucks[i] = t
        T -= 1

    # 打印结果
    for t in trucks:
        print(t)
    print("Total dist: {:.2f}".format(sum([t.d for t in trucks])))

    # 画图
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, font_color='#FFFFFF')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): G[u][v]['d'] for (u, v) in G.edges})
    plt.show()
