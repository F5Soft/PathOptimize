from flask import *

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    """主页"""
    trucks = get_trucks()
    return render_template("index.html", trucks_data=trucks)


@app.route('/trucks', methods=['GET'])
def get_trucks():
    trucks = [{"i": 0, "name": "蓝翔挖掘机", "nums": 12, "d_max": 3, "w_max": 9},
              {"i": 1, "name": "顺丰快递车", "nums": 8, "d_max": 12, "w_max": 6}]
    return render_template("trucks.html", trucks_data=trucks)


@app.route('/network', methods=['GET'])
def get_network():
    """配置网络接口"""
    return render_template("network.html")


@app.route('/trucks', methods=['POST'])
def set_trucks():
    """配置卡车接口"""
    trucks_data = request.form.getlist('name[]')
    print(trucks_data)
    return "Hello"
    # if request.method == 'POST':
    #    set_trucks(request.data)


@app.route('/network', methods=['POST'])
def set_network():
    pass


if __name__ == '__main__':
    app.run()
