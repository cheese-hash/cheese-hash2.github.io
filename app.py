import base64

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np
import requests
from io import BytesIO
import os
import re

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 解决跨域问题

# 配置：害虫类别和数据集路径
CLASS_NAMES = ["稻飞虱", "二化螟", "稻纵卷叶螟", "稻蓟马", "稻象甲", "稻螟蛉"]
NUM_CLASSES = len(CLASS_NAMES)

# 数据集路径配置
DATASET_PATHS = {
    "稻飞虱": r"D:\PythonProject2\data\稻飞虱",
    "二化螟": r"D:\PythonProject2\data\二化螟",
    "稻纵卷叶螟": r"D:\PythonProject2\data\稻纵卷叶螟",
    "稻蓟马": r"D:\PythonProject2\data\稻蓟马",
    "稻象甲": r"D:\PythonProject2\data\稻象甲",
    "稻螟蛉": r"D:\PythonProject2\data\稻螟蛉"  # 假设您有这个类别的数据
}

# 检查数据集路径是否存在
for cls, path in DATASET_PATHS.items():
    if not os.path.exists(path):
        app.logger.warning(f"警告: 类别 '{cls}' 的数据集路径不存在 - {path}")


# 模型定义
class PestModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 加载预训练的EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=True)
        # 替换分类头以适应我们的害虫类别
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PestModel(NUM_CLASSES).to(device)
model.eval()  # 设为评估模式

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 预测函数 - 只返回置信度最高的结果
def predict(image):
    # 预处理图片
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # 模型预测
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # 获取置信度最高的类别
    top_index = np.argmax(probabilities)

    return {
        "predicted_class": CLASS_NAMES[top_index],
        "confidence": round(float(probabilities[top_index]) * 100, 2)
    }


# 从URL下载图片
def load_image_from_url(url):
    try:
        # 处理data URL
        if url.startswith('data:image/'):
            # 提取base64数据
            base64_data = re.sub('^data:image/.+;base64,', '', url)
            image_data = BytesIO(base64.b64decode(base64_data))
            return Image.open(image_data).convert('RGB')

        # 处理普通URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"无法加载图片: {str(e)}")


# 自动识别接口 - 智能判断输入类型
@app.route('/api/auto-predict', methods=['POST'])
def auto_predict():
    try:
        # 检查是否是文件上传
        if 'image' in request.files and request.files['image'].filename:
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            result = predict(image)
            return jsonify(result)

        # 检查是否是JSON数据（URL或data URL）
        data = request.get_json()
        if data and 'input' in data:
            input_str = data['input'].strip()

            # 判断是否是图片URL或data URL
            if input_str.startswith(('http://', 'https://', 'data:image/')):
                image = load_image_from_url(input_str)
                result = predict(image)
                return jsonify(result)

        return jsonify({"error": "请上传图片或提供有效的图片URL"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 获取数据集信息
@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    info = {}
    for cls, path in DATASET_PATHS.items():
        if os.path.exists(path):
            # 统计每个类别下的图片数量
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
            count = sum(1 for f in os.listdir(path)
                        if f.lower().endswith(image_extensions))
            info[cls] = {
                "path": path,
                "image_count": count,
                "exists": True
            }
        else:
            info[cls] = {
                "path": path,
                "image_count": 0,
                "exists": False
            }
    return jsonify(info)


# 启动服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)