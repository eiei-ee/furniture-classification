"""
家具识别系统 — Flask Web 前端
运行: python app.py
访问: http://127.0.0.1:5000
"""
import os, io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify

# 配置
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "furniture_model.pth")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["bed", "cabinet", "chair", "sofa", "table"]
CLASS_NAMES_CN = {"bed": "床", "cabinet": "柜子", "chair": "椅子", "sofa": "沙发", "table": "桌子"}

app = Flask(__name__)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def load_model():
    global model
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASS_NAMES)),
    )
    model = model.to(device)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"模型已加载 (Val Acc: {checkpoint.get('val_acc', 'N/A')})")
        return True
    else:
        print(f"警告: 模型文件不存在 ({MODEL_PATH})")
        print("请先运行 furniture_classification.ipynb 训练模型。")
        return False


def predict_image(image_bytes):
    """对上传的图片进行推理"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    results = []
    for i, cls_name in enumerate(CLASS_NAMES):
        results.append({
            "class_en": cls_name,
            "class_cn": CLASS_NAMES_CN.get(cls_name, cls_name),
            "probability": float(probs[i]),
        })
    results.sort(key=lambda x: x["probability"], reverse=True)

    top = results[0]
    return {
        "prediction": top["class_cn"],
        "prediction_en": top["class_en"],
        "confidence": f"{top['probability']:.2%}",
        "all_probs": results,
    }


@app.route("/")
def index():
    model_ready = os.path.exists(MODEL_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    try:
        image_bytes = file.read()
        result = predict_image(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


if __name__ == "__main__":
    model_loaded = load_model()
    if model_loaded:
        print(f"设备: {device}")
        print(f"类别: {CLASS_NAMES}")
    print("启动 Flask 服务...")
    app.run(host="0.0.0.0", port=5000, debug=True)
