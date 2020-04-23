from flask import *

app = Flask(__name__)

# 准备用网页做用户界面，flask作为框架
@app.route('/')
def hello_world():
    return render_template("index.html")


if __name__ == '__main__':
    app.run()