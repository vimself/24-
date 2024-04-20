from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/process', methods=['POST'])
def submit_form():
    # 获取前端传来的urls参数
    data = request.json
    urls = data.get('urls', [])
    # print(str(urls[0]))
    # 在这里处理接收到的urls参数，生成需要返回的数据responseData
    # 这里简单地将收到的urls参数返回，你可以根据实际情况处理数据
    processed_data = {'processed_urls': str(urls[0])}

    # 将处理后的数据以JSON格式返回给前端
    return jsonify(processed_data)

# 定义根路由，提供index.html页面
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
