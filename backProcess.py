from modeluse import *
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#通过接口进行调用
'''
@app.route('/<path:path>') 是一个装饰器
它告诉 Flask 在处理指定路径的 HTTP 请求时应该调用下面的函数
在这个例子中，路径是 <path:path>，意味着它可以匹配任何路径，并将路径作为参数传递给下面的函数
'''

@app.route('/process', methods=['POST'])
def submit_form():
    # 获取前端传来的urls参数
    data = request.json
    message = data.get('urls', [])
    string = str(message[0])
    urls = string.split("\n")

    goods = []
    bads = []

    for i in range(0,len(urls),1):
        X_predict = [urls[i]]
        model, vector = loadModel()
        x = vector.transform(X_predict)
        y_predict = model.predict(x)
        if y_predict[0] == 'good':
            goods.append(urls[i])
        else:
            bads.append(urls[i])
    print(goods)
    print(bads)

    processed_data = {'good': goods, 'bad': bads}
    print(processed_data)

    # 将处理后的数据以JSON格式返回给前端
    return jsonify(processed_data)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'false'
    file = request.files['file']
    if file.filename == '':
        return 'false'
    if file:
        filename = file.filename
        file.save(os.path.join("data", filename))
        with open(os.path.join("data", filename), 'r') as f:
            file_content = f.read()
            print(file_content)
        return 'true'


# 定义根路由，提供index.html页面
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def returnIndex():
    return render_template('index.html')

@app.route('/getData.html')
def getData():
    return render_template('getData.html')

@app.route('/dataAnalysis.html')
def dataAnalysis():
    return render_template('dataAnalysis.html')

@app.route('/dataOutput.html')
def dataOutput():
    return render_template('dataOutput.html')

@app.route('/modelAnalysis.html')
def modelAnalysis():
    return render_template('modelAnalysis.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)

'''
这是 Python 的惯用法，用来检查当前模块是否是被直接执行的
如果是，则执行 app.run()，启动 Flask 应用的服务器
'''
