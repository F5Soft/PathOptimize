from flask import *

from modules import algorithm as alg
from modules.entity import Network
from modules.entity import Truck

app = Flask(__name__)

trucks_data = [("顺丰快递车", 8, 50, 8)]  # 默认有1种卡车，距离上限50km，载重上限8t，共8辆
nodes_data = [(0, {'name': '配送中心', 'demand': 0}), (1, {'name': '配送点1', 'demand': 2}),
              (2, {'name': '配送点2', 'demand': 1.5}), (3, {'name': '配送点3', 'demand': 4.5}),
              (4, {'name': '配送点4', 'demand': 3}), (5, {'name': '配送点5', 'demand': 1.5}),
              (6, {'name': '配送点6', 'demand': 4}), (7, {'name': '配送点7', 'demand': 2.5}),
              (8, {'name': '配送点8', 'demand': 3})]
edges_data = [(0, 1, {'weight': 7.4}), (0, 3, {'weight': 5.3}), (0, 2, {'weight': 12.1}), (0, 4, {'weight': 6.6}),
              (0, 5, {'weight': 8.2}), (0, 6, {'weight': 11.9}), (0, 7, {'weight': 11.2}), (0, 8, {'weight': 10.8}),
              (1, 2, {'weight': 5.8}), (1, 3, {'weight': 9.0}), (1, 4, {'weight': 7.1}), (1, 5, {'weight': 11.4}),
              (1, 6, {'weight': 12.8}), (1, 7, {'weight': 6.9}), (1, 8, {'weight': 10.7}), (2, 3, {'weight': 12.2}),
              (2, 4, {'weight': 9.4}), (2, 5, {'weight': 10.1}), (2, 6, {'weight': 13.7}), (2, 7, {'weight': 2.8}),
              (2, 8, {'weight': 9.0}), (3, 4, {'weight': 4.9}), (3, 5, {'weight': 4.1}), (3, 6, {'weight': 7.8}),
              (3, 7, {'weight': 11.9}), (3, 8, {'weight': 8.5}), (4, 5, {'weight': 4.0}), (4, 6, {'weight': 6.0}),
              (4, 7, {'weight': 6.6}), (4, 8, {'weight': 4.0}), (5, 6, {'weight': 3.9}), (5, 7, {'weight': 10.0}),
              (5, 8, {'weight': 4.4}), (6, 7, {'weight': 10.8}), (6, 8, {'weight': 5.7}), (7, 8, {'weight': 5.5})]


@app.route('/', methods=['GET'])
def index():
    """
    获取首页
    :return: 渲染后的首页网页
    """
    categories = []
    for t in trucks_data:
        for i in range(t[1]):
            categories.append(t[0] + str((i + 1)))
    return render_template("index.html", categories=categories)


@app.route('/trucks', methods=['GET'])
def get_trucks():
    """
    获取卡车配置页面
    :return: 渲染后的卡车配置页面
    """
    global trucks_data
    return render_template("trucks.html", trucks_data=trucks_data)


@app.route('/network', methods=['GET'])
def get_network():
    """
    获取网络配置页面
    :return: 渲染后的网络配置页面
    """
    return render_template("network.html", nodes_data=nodes_data, edges_data=edges_data)


@app.route('/trucks', methods=['POST'])
def set_trucks():
    """
    卡车配置接口
    :return: 如果配置成功，返回success，否则返回fail
    """
    global trucks_data
    names = request.form.getlist('name[]')
    nums = request.form.getlist('nums[]')
    d_maxs = request.form.getlist('d_max[]')
    w_maxs = request.form.getlist('w_max[]')
    n = len(names)
    trucks_data = [(names[i], int(nums[i]), float(d_maxs[i]), float(w_maxs[i])) for i in range(n)]
    print(trucks_data)
    return "success"


@app.route('/network/nodes', methods=['POST'])
def set_network_nodes():
    """
    网络顶点配置接口
    :return: 如果配置成功，返回success，否则返回fail
    """
    global nodes_data
    names = request.form.getlist('name[]')
    demands = request.form.getlist('demand[]')
    n = len(names)
    nodes_data = [(i, {'name': names[i], 'demand': float(demands[i])}) for i in range(n)]
    print(nodes_data)
    return "success"


@app.route('/network/edges', methods=['POST'])
def set_network_edges():
    """
    网络边配置接口
    :return: 如果配置成功，返回success，否则返回fail
    """
    global edges_data
    u = request.form.getlist('u[]')
    v = request.form.getlist('v[]')
    weights = request.form.getlist('weight[]')
    n = len(weights)
    edges_data = [(int(u[i]), int(v[i]), {'weight': float(weights[i])}) for i in range(n)]
    print(edges_data)
    return "success"


@app.route('/', methods=['POST'])
def apply_algorithm():
    """
    首页应用算法接口
    :return: 线路优化算法的输出结果
    """
    global trucks_data
    # 生成初始卡车列表
    trucks_initial = []
    for t in trucks_data:
        for i in range(int(t[1])):
            trucks_initial.append(Truck(t[0], float(t[2]), float(t[3])))
    # 生成初始网络。网络数据来自第一个测试数据集
    network_initial = Network(nodes_data, edges_data, trucks_initial)
    # 初始化种群并开始遗传算法
    # todo 调整遗传算法参数：种群数，突变率等，让算法收敛的快些
    population = alg.population_init(network_initial, 10)
    network_best = alg.genetic_algorithm(population, mutation_rate=0.1, recombination_rate=0.6)
    print(network_best.adaptive())
    print(network_best)
    network_best.gexf_summary("network.gexf")
    return str(network_best)


@app.route('/gexf', methods=['GET'])
def get_gexf():
    """
    获取配送网络可视化信息
    :return: 配送网络的gexf文件内容
    """
    return render_template("gexf/network.gexf")


if __name__ == '__main__':
    app.run()
