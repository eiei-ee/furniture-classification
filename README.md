# 家具识别系统 — 基于深度学习的图像分类

基于 PyTorch + MobileNetV2 迁移学习的五类家具自动识别系统。武汉软件工程职业学院《深度学习应用开发》课程设计项目。

支持识别：**床 (bed)、柜子 (cabinet)、椅子 (chair)、沙发 (sofa)、桌子 (table)**。

## 效果概览

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 89.33% |
| 最佳验证准确率 | 90.22% (Epoch 18) |
| 测试集 Loss | 0.4598 |
| 模型参数量 | 2,882,309 |
| 模型体积 | ~11MB |

## 项目结构

```
├── app.py                        # Flask Web 推理服务
├── download_dataset.py           # 从 HuggingFace 下载 CIFAR-100 并提取 5 类家具
├── generate_results.py           # 完整训练 + 评估 + 图表生成
├── eval_only.py                  # 仅评估已有模型
├── generate_report.py            # 生成课程设计 .docx 报告
├── furniture_classification.ipynb # Jupyter Notebook 实验记录
├── requirements.txt              # Python 依赖
├── model_metrics.txt             # 模型评估指标
├── templates/
│   └── index.html                # Web 前端页面
├── training_curves.png           # 训练 Loss/Accuracy 曲线
├── confusion_matrix.png          # 混淆矩阵
├── per_class_accuracy.png        # 各类别准确率柱状图
├── class_distribution.png        # 数据集类别分布
└── .gitignore
```

## 快速体验（推荐）

下载预训练模型，跳过训练直接启动 Web 演示：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 从 Release 下载预训练模型
# 访问 https://github.com/eiei-ee/furniture-classification/releases
# 下载 furniture_model.pth，放到项目根目录

# 3. 下载数据集（仅用于 Web 演示的示例图片）
python download_dataset.py

# 4. 启动 Web 演示
python app.py
# 浏览器访问 http://127.0.0.1:5000，上传家具图片即可识别
```

## 完整训练（从零开始）

```bash
pip install -r requirements.txt
python download_dataset.py
python generate_results.py   # 训练 + 评估 + 图表
python app.py
```

## 数据集

从 [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) 中筛选 5 个家具类别，通过 HuggingFace 镜像下载 Parquet 格式数据，按 70/15/15 划分：

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| 训练集 | 2,100 | 模型参数学习 |
| 验证集 | 450 | 超参数调优、早停判断 |
| 测试集 | 450 | 最终泛化能力评估 |

5 个类别各 600 张，分布均匀。

### 数据增强

训练集使用 5 种增强：RandomHorizontalFlip、RandomRotation(±20°)、ColorJitter、RandomAffine、ImageNet 标准化。

## 模型架构

**骨干网络**: MobileNetV2 (ImageNet 预训练权重，全模型微调)

**分类头**:
```
Dropout(0.5) → Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→5)
```

### 训练配置

| 超参数 | 设定 |
|--------|------|
| 优化器 | Adam (lr=0.0005, weight_decay=1e-4) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=3) |
| 早停 | patience=7，监控验证准确率 |
| 最大 Epoch | 30 |
| Batch Size | 32 |
| 输入尺寸 | 224×224 |

## 分类报告

```
              precision    recall  f1-score   support
         bed     0.8400    0.9333    0.8842        90
     cabinet     0.9773    0.9556    0.9663        90
       chair     0.9149    0.9556    0.9348        90
        sofa     0.8625    0.7667    0.8118        90
       table     0.8750    0.8556    0.8652        90

    accuracy                         0.8933       450
```

- **cabinet** 表现最优 (F1=96.6%)，**sofa** 最易混淆 (F1=81.2%)
- sofa ↔ bed、sofa ↔ table 为主要误分类对

## Web 演示

Flask + HTML5 前端，支持拖拽上传图片，实时返回 Top-1 预测类别、置信度及各类别概率分布。

## 技术栈

Python · PyTorch · TorchVision · Flask · NumPy · Matplotlib · Seaborn · scikit-learn · Pillow · python-docx
