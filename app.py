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
    """主页"""
    return render_template("index.html")


@app.route('/trucks', methods=['GET'])
def get_trucks():
    global trucks_data
    return render_template("trucks.html", trucks_data=trucks_data)


@app.route('/network', methods=['GET'])
def get_network():
    """配置网络接口"""
    return render_template("network_demo.html", nodes_data=nodes_data, edges_data=edges_data)


@app.route('/trucks', methods=['POST'])
def set_trucks():
    """配置卡车接口"""
    global trucks_data
    trucks_data = list(zip(request.form.getlist('name[]'), request.form.getlist('nums[]'),
                           request.form.getlist('d_max[]'), request.form.getlist('w_max[]')))
    print(trucks_data)
    return "success"


@app.route('/network/nodes', methods=['POST'])
def set_network_nodes():
    global nodes_data
    names = request.form.getlist('name[]')
    demands = request.form.getlist('demand[]')
    n = len(names)
    nodes_data = [(i, {'name': names[i], 'demand': float(demands[i])}) for i in range(n)]
    print(nodes_data)
    return "success"


@app.route('/network/edges', methods=['POST'])
def set_network_edges():
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
    return str(network_best)


if __name__ == '__main__':
    app.run()
