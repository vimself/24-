from modeluse import *
from DataUtils import *
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import random

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

goods = []
bads = []
filename = ''
is_upload = False

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
    print(message)
    string = str(message[0])
    urls = string.split("\n")

    global goods
    global bads
    if goods:
        goods = []
    if bads:
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

    # processed_data = {'good': goods, 'bad': bads}
    # print(processed_data)

    # 将处理后的数据以JSON格式返回给前端
    # return jsonify(processed_data)
    return 'true'


@app.route('/upload', methods=['POST'])
def upload_file():
    global filename
    global is_upload
    global goods
    global bads
    if goods:
        goods = []
    if bads:
        bads = []
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
            urls = file_content.split('\n')
            for i in range(0, len(urls), 1):
                X_predict = [urls[i]]
                model, vector = loadModel()
                x = vector.transform(X_predict)
                y_predict = model.predict(x)
                if y_predict[0] == 'good':
                    goods.append(urls[i])
                else:
                    bads.append(urls[i])

        return 'true'

@app.route('/get_MinganData')
def get_MinganData():
    global goods
    global bads
    data = {"good url":0,"bad url":0}
    if goods or bads:
        for item in goods:
            dict = data_count(item)
            for key in dict[0]:
                data["good url"] += dict[0].get(key,"default_value")
        for item in bads:
            dict = data_count(item)
            for key in dict[0]:
                data["bad url"] += dict[0].get(key,"default_value")
        return data
    else:
        return {}

@app.route('/get_url_length')
def get_url_length():
    data = {}
    for i in range(1, 21):
        # 随机生成恶意URL和良性URL的频率
        malicious_freq = random.randint(1, 100)
        benign_freq = random.randint(1, 100)
        data[i] = {"malicious": malicious_freq, "benign": benign_freq}
    return jsonify(data)

@app.route('/get_goodUrl_data')
def get_goodUrl_data():
    global goods
    dict = {"/":0,"?":0,"&":0,"<br/>":0,"&amp;":0,".":0,"-":0,"_":0}
    if goods:
        for item in goods:
            d = data_count(item)
            for key in d[1]:
                dict[key] += d[1].get(key,"default_value")
    return jsonify(dict)

@app.route('/get_badUrl_data')
def get_badUrl_data():
    global bads
    dict = {"/":0,"?":0,"&":0,"<br/>":0,"&amp;":0,".":0,"-":0,"_":0}
    if bads:
        for item in bads:
            d = data_count(item)
            for key in d[1]:
                dict[key] += d[1].get(key,"default_value")
    return jsonify(dict)

@app.route('/get_badWord_data')
def get_badWord_data():
    dict = {"@": 0, "eval": 0, "exec": 0, "<": 0, ">": 0, "system": 0, "echo": 0, "script": 0, "javascript": 0,
             "order": 0, "select": 0, "perl": 0, "php": 0, "sleep": 0, "alert": 0, "union": 0, "group": 0}
    global bads
    if bads:
        for item in bads:
            d = data_count(item)
            for key in d[0]:
                dict[key] += d[0].get(key, "default_value")
    return jsonify(dict)


@app.route('/result', methods=['GET'])
def get_result():
    global goods
    global bads

    processed_data = {'good': goods, 'bad': bads}
    return jsonify(processed_data)

@app.route('/getData_bad_to_good')
def getData_bad_to_good():
    global goods
    global bads
    data = {"good": 0, "bad": 0}
    data["good"] = len(goods)
    data["bad"] = len(bads)
    return jsonify(data)

@app.route('/getData_bad_to_bad')
def getData_bad_to_bad():
    global bads
    data = {"badest": 0, "bad": 0}
    for item in bads:
        data["bad"] += 1
        dict = data_count(item)
        for key in dict[0]:
            if dict[0].get(key) != 0:
                data["badest"] += 1
                data["bad"] -= 1
                break

    return jsonify(data)

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
