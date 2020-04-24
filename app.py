import json
from flask import *
from .modules.algorithm import *

app = Flask(__name__)


@app.route('/')
def index():
    """主页"""
    return render_template("index.html")


@app.route('/trucks')
def get_trucks():
    """配置卡车接口"""
    if request.method == 'POST':
        set_trucks(request.data)


@app.route('/network')
def get_network():
    """配置网络接口"""
    if request.method == 'POST':
        set_network(request.data)


if __name__ == '__main__':
    app.run()
