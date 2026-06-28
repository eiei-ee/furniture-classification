# Furniture Classification

深度学习家具识别系统，面向家居图片的五分类识别任务，覆盖 `bed`、`cabinet`、`chair`、`sofa`、`table` 五类家具。

本项目包含数据处理、模型训练、测试评估、结果可视化和 Flask 演示页面，适合作为深度学习应用开发课程设计或 AI 应用落地练习项目。

## 项目亮点

- 完成五类家具图像分类模型训练、验证与测试评估。
- 输出训练曲线、混淆矩阵、类别准确率、预测样例等可视化结果。
- 测试集准确率：**89.33%**
- 最佳验证准确率：**89.11%**
- 支持从产品视角分析模型能力边界，为后续数据优化和场景适配提供依据。

## 结果指标

| 指标 | 数值 |
| --- | --- |
| Test Accuracy | 89.33% |
| Test Loss | 0.3876 |
| Best Val Accuracy | 89.11% |
| Best Epoch | 11 |
| Total Epochs | 19 |
| Parameters | 2,882,309 |

## 类别表现

| 类别 | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| bed | 0.8605 | 0.8222 | 0.8409 |
| cabinet | 0.9667 | 0.9667 | 0.9667 |
| chair | 0.9348 | 0.9556 | 0.9451 |
| sofa | 0.8471 | 0.8000 | 0.8229 |
| table | 0.8557 | 0.9222 | 0.8877 |

## 项目结构

```text
.
├── app.py                         # Flask 演示应用
├── download_dataset.py             # 数据集下载/处理脚本
├── generate_results.py             # 训练与结果生成脚本
├── eval_only.py                    # 模型评估脚本
├── generate_report.py              # 课程报告生成脚本
├── furniture_classification.ipynb   # Notebook 实验记录
├── model_metrics.txt               # 模型评估指标
├── templates/
│   └── index.html                  # Web 页面模板
├── test_images/                    # 示例测试图片
├── training_curves.png             # 训练曲线
├── confusion_matrix.png            # 混淆矩阵
├── per_class_accuracy.png          # 类别准确率
├── class_distribution.png          # 类别分布
├── sample_images.png               # 数据样例
└── prediction_samples.png          # 预测样例
```

## 说明

为了避免仓库体积过大，以下内容未纳入 GitHub：

- 原始数据集 `dataset/`
- 模型权重 `*.pth`
- 离线依赖包 `*.whl`
- Python 缓存、IDE 配置和临时文件

如果需要运行完整训练流程，请按脚本说明重新准备数据集并训练模型。

## 技术栈

- Python
- PyTorch / TorchVision
- Flask
- NumPy
- Matplotlib / Seaborn
- scikit-learn
- Pillow

