# 71623 参赛作品源代码
基于GCN和GA的物流网络优化算法
## 目录结构
```
├─data/        程序运行相关数据
├─modules/     算法模块
│  ├─entity/   算法所需的实体类
├─static/      静态文件
│  ├─css/      前端界面css文件
│  └─js/       前端界面js文件
├─templates/   前端界面网页
│  └─gexf/     物流网络数据xml文件
├─app.py       主程序
```

## 安装及运行环境配置

### 1. 安装Python
要求Pyton版本号大于等于3.6.8。Windows可在python官网 https://www.python.org/ 下载，Linux可用如下命令安装：
```shell script
sudo apt install python3 python3-pip
```
### 2. 安装依赖的Python库
在Windows下，打开命令提示符(cmd)，输入
```shell script
pip install numpy networkx flask
```
如果是在Linux环境下，输入
```shell script
pip3 install numpy networkx flask
```
### 3. 运行主程序
在Windows下，打开命令提示符(cmd)，切换到当前目录后输入
```shell script
python app.py
```
如果是在在Linux环境下，同样切换到当前目录，输入
```shell script
python3 app.py
```
即可运行。
### 4. 查看及使用
打开浏览器，访问 http://127.0.0.1:5000 即可进入作品界面。