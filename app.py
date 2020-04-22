from flask import Flask

app = Flask(__name__)

# 准备用网页做用户界面，flask作为框架
@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()