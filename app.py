import numpy as np
from flask import *

from modules import algorithm as alg
from modules.entity import Network
from modules.entity import Truck

app = Flask(__name__)

trucks_data = [("顺丰快递车", 8, 50, 8)]  # 默认有1种卡车，距离上限50km，载重上限8t，共8辆
network_data = []


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
    return render_template("network_demo.html")


@app.route('/trucks', methods=['POST'])
def set_trucks():
    """配置卡车接口"""
    global trucks_data
    trucks_data = list(zip(request.form.getlist('name[]'), request.form.getlist('nums[]'),
                       request.form.getlist('d_max[]'), request.form.getlist('w_max[]')))
    print(trucks_data)
    return "success"


@app.route('/network', methods=['POST'])
def set_network():
    global network_data
    network_data = request.form
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
    dists = np.array([[0, 7.4, 12.1, 5.3, 6.6, 8.2, 11.9, 11.2, 10.8],
                      [7.4, 0, 5.8, 9, 7.1, 11.4, 12.8, 6.9, 10.7],
                      [12.1, 5.8, 0, 12.2, 9.4, 10.1, 13.7, 2.8, 9],
                      [5.3, 9, 12.2, 0, 4.9, 4.1, 7.8, 11.9, 8.5],
                      [6.6, 7.1, 9.4, 4.9, 0, 4, 6, 6.6, 4],
                      [8.2, 11.4, 10.1, 4.1, 4, 0, 3.9, 10, 4.4],
                      [11.9, 12.8, 13.7, 7.8, 6, 3.9, 0, 10.8, 5.7],
                      [11.2, 6.9, 2.8, 11.9, 6.6, 10, 10.8, 0, 5.5],
                      [10.8, 10.7, 9, 8.5, 4, 4.4, 5.7, 5.5, 0]]
                     )
    names = ["配送中心0", "配送点1", "配送点2", "配送点3", "配送点4", "配送点5", "配送点6", "配送点7", "配送点8"]
    demands = [0, 2, 1.5, 4.5, 3, 1.5, 4, 2.5, 3]
    network_initial = Network(dists, names, demands, trucks_initial)
    # 初始化种群并开始遗传算法
    # todo 调整遗传算法参数：种群数，突变率等，让算法收敛的快些
    population = alg.population_init(network_initial, 10)
    network_best = alg.genetic_algorithm(population, mutation_rate=0.1, recombination_rate=0.6)
    print(network_best.adaptive())
    print(network_best)
    return "success"


if __name__ == '__main__':
    app.run()
