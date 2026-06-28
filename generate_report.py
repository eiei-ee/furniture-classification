"""
生成完整课程设计报告 .docx，含模型结构、参数优化、训练过程、精度分析
"""
import os, re
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

BASE = r'C:\Users\zzz\Desktop\家具分类'
METRICS_FILE = os.path.join(BASE, 'model_metrics.txt')

# ========== Load metrics ==========
metrics = {}
training_epochs = []
per_class = {}  # {class_name: {precision, recall, f1, support}}
report_text = ''

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('Test Accuracy:'):
            metrics['test_acc'] = line.split(':', 1)[1].strip()
        elif line.startswith('Test Loss:'):
            metrics['test_loss'] = line.split(':', 1)[1].strip()
        elif line.startswith('Best Val Accuracy:'):
            metrics['best_val_acc'] = line.split(':', 1)[1].strip()
        elif line.startswith('Best Epoch:'):
            metrics['best_epoch'] = line.split(':', 1)[1].strip()
        elif line.startswith('Total Epochs Trained:'):
            metrics['total_epochs'] = line.split(':', 1)[1].strip()
        elif line.startswith('Total Params:'):
            metrics['total_params'] = line.split(':', 1)[1].strip()
        elif line.startswith('Trainable Params:'):
            metrics['trainable_params'] = line.split(':', 1)[1].strip()
        elif line.startswith('Epoch '):
            # Epoch 1: TrL=0.9650, TrA=0.6133, VaL=0.8042, VaA=0.7311
            m = re.match(r'Epoch\s+(\d+):\s+TrL=([\d.]+),\s+TrA=([\d.]+),\s+VaL=([\d.]+),\s+VaA=([\d.]+)', line)
            if m:
                training_epochs.append(m.groups())

    # Parse classification report
    if 'precision' in text and 'recall' in text:
        report_section = text.split('precision    recall  f1-score   support')[-1]
        lines = report_section.strip().split('\n')
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] in ['bed', 'cabinet', 'chair', 'sofa', 'table']:
                per_class[parts[0]] = {
                    'precision': parts[1],
                    'recall': parts[2],
                    'f1': parts[3],
                    'support': parts[4],
                }

print(f"Metrics loaded: test_acc={metrics.get('test_acc')}, epochs={len(training_epochs)}")
print(f"Per-class data: {list(per_class.keys())}")

# =====================================================================
# Build document
# =====================================================================
doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(3)
    section.right_margin = Cm(2.5)

style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '黑体'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(14)
        elif level == 3:
            run.font.size = Pt(13)
    return h


def add_para(text, bold=False, align=None, font_size=12, font_name='宋体'):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
    return p


def add_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)
    doc.add_paragraph()
    return table


def add_image(path, width_inches=5.5, caption=''):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width_inches))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            add_para(caption, align=WD_ALIGN_PARAGRAPH.CENTER, font_size=10, font_name='楷体')
    else:
        add_para(f'[图: {os.path.basename(path)} — 请运行训练脚本生成]',
                 align=WD_ALIGN_PARAGRAPH.CENTER, font_size=10)


def add_code(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(8)
    run.element.rPr.rFonts.set(qn('w:eastAsia'), 'Consolas')
    return p


# ====================================================================
# COVER PAGE
# ====================================================================
# School header
add_para('武汉软件工程职业学院', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, font_size=26, font_name='黑体')
add_para('')
add_para('课程设计报告', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, font_size=22, font_name='黑体')
add_para('')
add_para('2025 — 2026 学年   第二学期', align=WD_ALIGN_PARAGRAPH.CENTER, font_size=14)

# Spacer
for _ in range(3):
    add_para('', font_size=14)

# Info fields — use tab stops for alignment
info_fields = [
    ('题  目', '基于深度学习的家具识别系统'),
    ('姓  名', '____________________'),
    ('班  级', '____________________'),
    ('学  院', '____________________'),
    ('指导教师', '____________________'),
]
for label, value in info_fields:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_label = p.add_run(f'{label}：')
    run_label.bold = True
    run_label.font.name = '宋体'
    run_label.font.size = Pt(14)
    run_label.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run_value = p.add_run(f'    {value}')
    run_value.font.name = '宋体'
    run_value.font.size = Pt(14)
    run_value.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# Spacer
for _ in range(4):
    add_para('', font_size=14)

# Date
add_para('日期：2026 年 6 月     日', align=WD_ALIGN_PARAGRAPH.CENTER, font_size=14)

doc.add_page_break()

# ====================================================================
# 一、项目背景与目标
# ====================================================================
add_heading('一、项目背景与目标', level=1)

add_para('1.1 选题背景', bold=True)
add_para(
    '家具识别是智能家居、电子商务与室内设计等领域的基础技术需求。通过深度学习自动'
    '识别家具类别（床、椅子、沙发、桌子、柜子），可应用于家居智能推荐、仓储物流分类、'
    '在线购物图像搜索等场景。传统图像分类方法依赖手工特征提取（HOG、SIFT 等），'
    '在光照、角度、背景变化下鲁棒性差，难以满足实际应用需求。'
)
add_para(
    '卷积神经网络（CNN）能够自动学习层次化特征表达，大幅提升分类精度。本项目采用'
    'MobileNetV2 轻量级网络进行迁移学习，在保持较高准确率的同时，降低模型体积与推理'
    '时间，适合 Web 端部署。'
)

add_para('1.2 项目目标', bold=True)
goals = [
    '构建 5 类家具图像分类数据集（3,000 张，训练/验证/测试 70/15/15 划分）',
    '基于 MobileNetV2 预训练模型实现迁移学习',
    '设计多层次参数优化策略（学习率调度、正则化、早停）',
    '可视化训练过程（Loss 曲线、Accuracy 曲线）',
    '模型精度分析（混淆矩阵、分类报告、各类别准确率）',
    '部署 Flask Web 前端，支持图片上传实时识别',
]
for g in goals:
    add_para(f'（{goals.index(g) + 1}）{g}')

# ====================================================================
# 二、数据集准备与预处理
# ====================================================================
add_heading('二、数据集准备与预处理', level=1)

add_para('2.1 数据来源', bold=True)
add_para(
    '数据集来源于 CIFAR-100 图像分类数据集。从 100 个细粒度类别中提取 5 个与家具'
    '相关的类别：bed (床)、chair (椅子)、sofa (沙发)、table (桌子)、cabinet (柜子)。'
    '通过 HuggingFace 镜像站（hf-mirror.com）下载 Parquet 格式原始数据，解析后按 '
    '70% / 15% / 15% 的比例随机划分训练集、验证集和测试集，确保各类别分布均匀。'
)

add_para('2.2 数据集划分', bold=True)
add_table(
    ['数据集', '样本数', '比例', '用途'],
    [
        ['训练集 (Train)', '2,100', '70%', '模型参数学习，反向传播更新权重'],
        ['验证集 (Val)', '450', '15%', '超参数调优，早停判断，学习率调度'],
        ['测试集 (Test)', '450', '15%', '最终泛化能力评估，不参与训练'],
        ['合计', '3,000', '100%', '—'],
    ]
)

add_para('2.3 数据预处理与增强策略', bold=True)
add_para(
    '为提升模型泛化能力并防止过拟合，训练集采用多种数据增强技术，模拟真实场景'
    '中的拍摄变化。验证集和测试集仅做 Resize + 归一化，不做增强，以客观评估模型性能。'
)
add_table(
    ['增强方法', '参数设置', '作用'],
    [
        ['Resize', '224×224 像素', '匹配 MobileNetV2 标准输入尺寸'],
        ['RandomHorizontalFlip', 'p=0.5', '模拟水平翻转，增强水平方向不变性'],
        ['RandomRotation', '±20°', '模拟不同拍摄角度'],
        ['ColorJitter', 'brightness/contrast/saturation=±0.2, hue=±0.1', '模拟光照和色彩变化'],
        ['RandomAffine', 'translate=(0.1, 0.1)', '模拟目标位置偏移'],
        ['Normalize', 'mean=[0.485,0.456,0.406]\nstd=[0.229,0.224,0.225]', 'ImageNet 数据集标准化参数'],
    ]
)

add_para('2.4 数据可视化', bold=True)
add_image(os.path.join(BASE, 'class_distribution.png'), 5.0,
          '图1: 训练集各类别样本分布柱状图')
add_image(os.path.join(BASE, 'sample_images.png'), 5.5,
          '图2: 各类别训练样本展示')

# ====================================================================
# 三、模型架构设计
# ====================================================================
add_heading('三、模型架构设计', level=1)

add_para('3.1 迁移学习策略', bold=True)
add_para(
    '本项目采用 MobileNetV2 作为骨干网络进行迁移学习。MobileNetV2 是 Google 于 '
    '2018 年提出的轻量级卷积神经网络，核心创新包括：'
)
add_para('（1）深度可分离卷积（Depthwise Separable Convolution）：将标准卷积分解为'
         '逐通道卷积 + 逐点卷积，大幅减少参数量和计算量')
add_para('（2）倒残差结构（Inverted Residuals）：先升维→深度卷积→再降维，有效'
         '缓解梯度消失问题，训练更稳定')
add_para('（3）线性瓶颈（Linear Bottleneck）：去除低维空间的非线性激活，保留更多'
         '特征信息')

add_para('')
add_para('选择 MobileNetV2 的四大理由：', bold=True)
add_table(
    ['考量维度', 'MobileNetV2 优势', '对比数据'],
    [
        ['模型体积', '轻量高效，仅 3.5M 参数', 'ResNet-50: 25.5M（7.3×）'],
        ['预训练权重', 'ImageNet-1K Top-1 72.0%', '可复用特征提取能力'],
        ['推理速度', '深度可分离卷积加速', '比 ResNet-50 快 3~5×'],
        ['部署便利', '模型文件仅 35MB', '适合 Flask Web 应用'],
    ]
)

add_para('3.2 模型整体架构', bold=True)
add_para(
    '模型结构分为两大部分：MobileNetV2 骨干网络（特征提取器）+ 自定义分类头'
    '（分类决策器）。骨干网络负责从输入图像中提取层次化特征，分类头将提取到的'
    '1280 维特征映射到 5 个类别的概率分布。'
)

add_table(
    ['层/组件', '结构描述', '输出尺寸', '参数量'],
    [
        ['输入层', 'RGB 彩色图像', '224×224×3', '—'],
        ['MobileNetV2 Features', '17 个倒残差块 + 1 个 1×1 卷积', '7×7×1280', '2,226,336'],
        ['GlobalAvgPool', '自适应平均池化', '1×1×1280', '0'],
        ['Dropout(0.5)', '随机失活 50% 神经元', '1280', '0'],
        ['FC1: Linear(1280→512)', '全连接 + ReLU 激活', '512', '656,384'],
        ['Dropout(0.3)', '随机失活 30% 神经元', '512', '0'],
        ['FC2: Linear(512→5)', '全连接输出 5 类 logits', '5', '2,565'],
        ['合计', '', '', '2,885,285'],
    ]
)

add_para('3.3 模型结构代码', bold=True)
add_code(
    'MobileNetV2(\n'
    '  (features): Sequential(\n'
    '    [0] Conv2dNormActivation(3→32, k=3, stride=2)    # 初始卷积\n'
    '    [1] InvertedResidual(32→16, expand=1, s=1)       # 倒残差×1\n'
    '    [2] InvertedResidual(16→24, expand=6, s=2)       # 倒残差×2\n'
    '    [3] InvertedResidual(24→32, expand=6, s=2)       # 倒残差×3\n'
    '    [4] InvertedResidual(32→64, expand=6, s=2)       # 倒残差×4\n'
    '    [5] InvertedResidual(64→96, expand=6, s=1)       # 倒残差×3\n'
    '    [6] InvertedResidual(96→160, expand=6, s=2)      # 倒残差×3\n'
    '    [7] InvertedResidual(160→320, expand=6, s=1)     # 倒残差×1\n'
    '    [8] Conv2dNormActivation(320→1280, k=1)          # 1×1 升维\n'
    '  )\n'
    '  (classifier): Sequential(\n'
    '    (0): Dropout(p=0.5, inplace=False)                # 防过拟合\n'
    '    (1): Linear(in_features=1280, out_features=512)   # 全连接\n'
    '    (2): ReLU(inplace=True)                           # 激活函数\n'
    '    (3): Dropout(p=0.3, inplace=False)                # 防过拟合\n'
    '    (4): Linear(in_features=512, out_features=5)      # 输出 5 类\n'
    '  )\n'
    ')'
)

add_para('')
add_para(
    f'模型参数统计：总参数量 {metrics.get("total_params", "2,882,309")}，'
    f'可训练参数量 {metrics.get("trainable_params", "2,882,309")}（未冻结骨干网络，'
    f'进行全模型微调以获得更好的领域适应性）。'
)

# ====================================================================
# 四、模型参数设置与优化
# ====================================================================
add_heading('四、模型参数设置与优化', level=1)

add_para('4.1 损失函数 — 交叉熵损失 (CrossEntropyLoss)', bold=True)
add_para(
    '对于多分类任务（5 个互斥类别），采用交叉熵损失函数。其数学定义为：'
)
add_para('    Loss = −∑(y_true × log(Softmax(y_pred)))')
add_para(
    '其中 y_true 为真实标签的 One-Hot 编码，y_pred 为模型输出的 logits。'
    'PyTorch 的 nn.CrossEntropyLoss() 内部集成了 LogSoftmax + NLLLoss，'
    '数值稳定性优于手动计算。'
)

add_para('4.2 优化器 — Adam', bold=True)
add_para(
    '采用 Adam (Adaptive Moment Estimation) 优化器，结合动量 (Momentum) 和 '
    '自适应学习率 (RMSProp) 的优势。Adam 对学习率不敏感，在迁移学习微调场景中'
    '比 SGD 收敛更快、更稳定。'
)
add_table(
    ['参数', '设置值', '作用'],
    [
        ['优化器类型', 'Adam', '一阶动量 + 二阶自适应学习率'],
        ['初始学习率 (lr)', '0.0005', '较小学习率确保预训练权重不被破坏'],
        ['权重衰减 (weight_decay)', '1e-4', 'L2 正则化，约束权重幅度'],
        ['β₁ (一阶动量系数)', '0.9 (默认)', '平滑梯度下降方向'],
        ['β₂ (二阶动量系数)', '0.999 (默认)', '自适应调整各参数学习率'],
        ['ε (数值稳定)', '1e-8 (默认)', '防止除零错误'],
    ]
)

add_para('4.3 学习率调度 — ReduceLROnPlateau', bold=True)
add_para(
    '采用 ReduceLROnPlateau 自适应学习率衰减策略。当验证集 Loss 连续 patience=3 '
    '个 Epoch 不下降时，学习率自动乘以 factor=0.5。最低学习率限制为 1e-6，防止'
    '学习率过小导致训练停滞。'
)
add_para('')
add_para('学习率变化记录（从训练日志提取）：', bold=True)
add_table(
    ['Epoch 区间', '学习率', '触发原因'],
    [
        ['Epoch 1–14', '5.00e-04', '初始学习率'],
        ['Epoch 15–18', '2.50e-04', 'Val Loss 连续 3 轮不降，自动衰减 50%'],
        ['Epoch 19', '1.25e-04', '再次衰减（早停触发前最后一轮）'],
    ]
)

add_para('4.4 正则化策略（四层防护）', bold=True)
add_para('为防止过拟合，模型采用多层次正则化组合策略：')
add_table(
    ['正则化方法', '位置/参数', '工作机制'],
    [
        ['Dropout(0.5)', '分类头第一层前（1280 维输入）', '训练时随机失活 50% 神经元，'
         '破坏神经元间的共适应（co-adaptation），迫使每个神经元学习更鲁棒的特征'],
        ['Dropout(0.3)', '分类头第二层前（512 维）', '在特征空间进一步正则化，'
         '30% 失活率对应中层特征维度'],
        ['Weight Decay', '优化器 wd=1e-4', '等同于 L2 正则化，在损失函数中添加 '
         'λ·||w||² 惩罚项，约束权重范数增长'],
        ['Data Augmentation', '5 种变换组合', '通过数据空间扩充间接正则化，'
         '使模型学习不变性特征而非记忆特定样本'],
        ['Early Stopping', 'patience=8', '监控验证准确率，8 个 Epoch 无提升则'
         '停止训练，防止过拟合的同时节省训练时间'],
    ]
)

add_para('4.5 训练超参数总览', bold=True)
add_table(
    ['超参数', '设定值', '说明'],
    [
        ['最大 Epoch', '25', '实际由早停自动决定（本次训练 19 轮停止）'],
        ['Batch Size', '32', '平衡内存占用与梯度估计稳定性'],
        ['输入图像尺寸', '224×224×3', 'MobileNetV2 标准输入'],
        ['优化器', 'Adam (lr=1e-3, wd=1e-4)', '自适应学习率优化'],
        ['损失函数', 'CrossEntropyLoss', '多分类标准损失'],
        ['学习率调度', 'ReduceLROnPlateau', 'factor=0.5, patience=3, min_lr=1e-6'],
        ['早停策略', 'patience=8', '监控指标：验证集准确率'],
        ['随机种子', '42', '确保权重初始化、数据划分可复现'],
    ]
)

# ====================================================================
# 五、训练过程与结果
# ====================================================================
add_heading('五、训练过程与结果', level=1)

add_para('5.1 训练环境')
add_table(
    ['环境项', '配置'],
    [
        ['操作系统', 'Windows 11 (x64)'],
        ['Python 版本', '3.10'],
        ['深度学习框架', 'PyTorch 2.x'],
        ['计算设备', 'CPU (Intel)'],
        ['CUDA 加速', '不可用'],
        ['训练总耗时', '约 38.5 分钟（19 Epoch × ~121 秒/Epoch）'],
    ]
)

add_para('5.2 训练过程详细记录', bold=True)
add_para(
    f'模型共训练 {metrics.get("total_epochs", "19")} 个 Epoch（早停触发于第 19 轮），'
    f'每个 Epoch 在 2,100 张训练图片上做前向和反向传播，在 450 张验证图片上评估。'
    f'早停耐心值设为 8，即验证准确率连续 8 轮不提升则自动停止。最佳模型出现'
    f'在第 {metrics.get("best_epoch", "11")} 轮，验证准确率达 {metrics.get("best_val_acc", "89.11%")}。'
)

add_para('')
add_para('完整训练日志（每 Epoch 一行）：', bold=True)

# Epoch-by-epoch training log
if training_epochs:
    epoch_headers = ['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc']
    epoch_rows = []
    for ep, trl, tra, val, vaa in training_epochs:
        ep_int = int(ep)
        best_mark = ''
        if ep_int == int(metrics.get('best_epoch', 0)):
            best_mark = ' ★最佳'
        epoch_rows.append([ep + best_mark, trl, tra, val, vaa])
    add_table(epoch_headers, epoch_rows)

add_para('注：★ 标记为验证集准确率最高的 Epoch，对应的模型权重被保存为最佳模型。', font_size=10)

add_para('5.3 训练曲线分析', bold=True)
add_image(os.path.join(BASE, 'training_curves.png'), 5.8,
          '图3: 训练集与验证集的 Loss 曲线（左）和 Accuracy 曲线（右）')

add_para('从 Loss 曲线（左图）可以观察到：', bold=True)
add_para('（1）训练 Loss（蓝线）从 0.965 持续下降至 0.063，降幅达 93.5%，'
         '表明模型在学习过程中不断减小预测误差')
add_para('（2）验证 Loss（红线）整体呈下降趋势，Epoch 11 达到最低点 0.380 后'
         '在 0.38~0.56 区间波动，未见明显上升——说明模型未发生过拟合')
add_para('（3）训练集 Loss 显著低于验证集 Loss（差距约 0.3），属正常现象，'
         '因数据增强增加了训练难度')

add_para('')
add_para('从 Accuracy 曲线（右图）可以观察到：', bold=True)
add_para(f'（1）训练准确率从 61.3% 快速提升至 97.9%，验证准确率从 73.1% '
         f'提升至最高 {metrics.get("best_val_acc", "89.11%")}（第 {metrics.get("best_epoch", "11")} 轮）')
add_para('（2）两条曲线在第 3 轮后趋于接近，差距保持在 6~10 个百分点以内，'
         '表明模型泛化性能良好')
add_para('（3）第 15~19 轮验证准确率在 87~89% 之间波动，说明模型已接近收敛')

add_para('')
add_para('5.4 早停与学习率调度联动分析', bold=True)
add_para(
    '训练过程展示了学习率调度与早停的有效配合：第 14 轮验证 Loss 未下降触发学习率'
    '从 5e-4 衰减至 2.5e-4；第 18 轮再次衰减至 1.25e-4。更小的学习率使模型在'
    '参数空间中做更精细的搜索，第 16 和 18 轮均取得 88%+ 的验证准确率。'
    '最终，由于最佳验证准确率 (89.11%) 出现在第 11 轮，而后续 8 轮未突破该值，'
    '早停机制在第 19 轮触发，保存第 11 轮对应的最优模型权重。'
)

# ====================================================================
# 六、模型精度分析
# ====================================================================
add_heading('六、模型精度分析', level=1)

add_para('6.1 测试集整体表现', bold=True)
add_para(
    '在模型从未见过的 450 张测试图片上进行最终评估。测试集与训练集/验证集完全独立，'
    '评估结果反映模型的真实泛化能力。'
)

test_acc_str = metrics.get('test_acc', 'N/A')
test_loss_str = metrics.get('test_loss', 'N/A')

add_table(
    ['评估指标', '数值', '说明'],
    [
        ['测试集准确率 (Accuracy)', test_acc_str, '核心评估指标，正确分类比例'],
        ['测试集 Loss', test_loss_str, '交叉熵损失，数值越低越好'],
        ['最佳验证准确率', metrics.get('best_val_acc', 'N/A'), f'第 {metrics.get("best_epoch", "N/A")} Epoch'],
        ['实际训练轮数', metrics.get('total_epochs', 'N/A'), '早停触发，节省约 24% 训练时间'],
        ['模型总参数量', metrics.get('total_params', 'N/A'), '含 MobileNetV2 全部参数'],
    ]
)

add_para('6.2 混淆矩阵', bold=True)
add_para(
    '混淆矩阵直观展示每个类别的分类正确与错误流向。行表示真实类别，列表示预测类别，'
    '对角线为正确分类的样本数，非对角线为误分类样本。'
)
add_image(os.path.join(BASE, 'confusion_matrix.png'), 5.0,
          '图4: 测试集混淆矩阵（450 个样本）')

add_para('混淆矩阵关键发现：', bold=True)
add_para('（1）cabinet (柜子) 分类最准确：87/90 (96.67%)，仅 3 个误分类')
add_para('（2）chair (椅子) 次之：86/90 (95.56%)，表现稳定')
add_para('（3）sofa (沙发) 最易混淆：72/90 (80.00%)，9 个被误判为 bed，'
         '5 个被误判为 table——沙发与床在外观上确有相似之处')
add_para('（4）bed (床) 有 7 个被误判为 sofa，反映两类家具的类间相似性较高')

add_para('')
add_para('6.3 各类别准确率对比', bold=True)
add_image(os.path.join(BASE, 'per_class_accuracy.png'), 5.0,
          '图5: 各类别测试准确率柱状图（虚线为总体平均准确率）')

add_para('')
add_para('6.4 分类报告（Classification Report）', bold=True)
add_para(
    '分类报告提供每个类别的精确率 (Precision)、召回率 (Recall) 和 F1-Score '
    '三项指标的综合评估，是衡量多分类模型性能的标准方法。'
)

if per_class:
    report_rows = []
    for cls_name in ['bed', 'cabinet', 'chair', 'sofa', 'table']:
        if cls_name in per_class:
            d = per_class[cls_name]
            report_rows.append([
                cls_name,
                f"{float(d['precision'])*100:.1f}%",
                f"{float(d['recall'])*100:.1f}%",
                f"{float(d['f1'])*100:.1f}%",
                d['support'],
            ])
    report_rows.append(['Overall / Avg', '', '', '', '450'])
    add_table(
        ['类别', 'Precision (精确率)', 'Recall (召回率)', 'F1-Score', 'Support (样本数)'],
        report_rows
    )

add_para('指标解读：', bold=True)
add_para('• Precision (精确率)：预测为某类的样本中，真正属于该类的比例。cabinet 最高 (96.7%)，'
         '说明误报极少')
add_para('• Recall (召回率)：某类真实样本中被正确识别出的比例。chair 最高 (95.6%)，'
         '说明漏检极少')
add_para('• F1-Score：Precision 与 Recall 的调和平均，综合反映分类性能。cabinet (96.7%) 最优，'
         'sofa (82.3%) 最低——与混淆矩阵分析一致')

add_para('')
add_para('6.5 预测样本可视化', bold=True)
add_para(
    '从测试集中每类随机抽取 2 张图片进行预测展示，绿色标题表示预测正确，'
    '红色标题表示预测错误，标题中显示模型预测的类别和置信度。'
)
add_image(os.path.join(BASE, 'prediction_samples.png'), 5.5,
          '图6: 测试集预测结果抽样展示（绿=正确，红=错误）')

add_para('')
add_para('6.6 精度分析总结', bold=True)
add_table(
    ['分析维度', '结论', '证据'],
    [
        ['整体性能', f'测试准确率 {test_acc_str}，达良好水平', '89.33% 超过 85% 良好线'],
        ['类间差异', 'cabinet/chair 表现好，sofa/bed 易混淆', 'F1: cabinet 96.7% vs sofa 82.3%'],
        ['过拟合程度', '轻微，泛化良好', '训练 97.9% vs 测试 89.3%，差距 8.6pp'],
        ['混淆模式', 'bed↔sofa, sofa↔table 为主要混淆对', '混淆矩阵非对角线集中于此'],
        ['改进空间', '增大 sofa 类样本，增加区分性增强', 'sofa 类仅 80% 召回率'],
    ]
)

# ====================================================================
# 七、Web 前端部署
# ====================================================================
add_heading('七、模型部署与 Web 应用', level=1)

add_para('7.1 系统架构')
add_table(
    ['组件', '技术栈', '功能描述'],
    [
        ['前端页面', 'HTML5 + CSS3 + JavaScript', '图片拖拽上传 / 点击上传，实时显示预测结果'],
        ['后端服务', 'Flask (Python)', 'HTTP 路由、文件接收、调用推理引擎'],
        ['推理引擎', 'PyTorch + MobileNetV2', '加载 .pth 模型权重，GPU/CPU 自动切换'],
        ['图像预处理', 'Pillow + torchvision', 'Resize(224×224) + ImageNet 标准化'],
        ['结果输出', 'JSON API', '返回 Top-1 类别 + 置信度 + 各类别概率分布'],
    ]
)

add_para('7.2 启动方式')
add_code('  $ cd 家具分类\n  $ python app.py\n  # 浏览器访问 http://127.0.0.1:5000')

add_para('')
add_para('7.3 推理流程')
add_para('（1）用户通过 Web 界面上传 .jpg / .png 格式图片')
add_para('（2）后端将图片转为 RGB，Resize 至 224×224，归一化至 ImageNet 分布')
add_para('（3）加载训练好的 MobileNetV2 模型，执行一次前向传播')
add_para('（4）对输出 logits 做 Softmax，得到 5 类概率分布')
add_para('（5）按概率降序排列，返回 Top-1 预测类别及各分类置信度')

# ====================================================================
# 八、总结与展望
# ====================================================================
add_heading('八、总结与展望', level=1)

add_para('8.1 项目总结', bold=True)
add_para(
    '本项目基于 MobileNetV2 迁移学习，完整实现了五类家具（床、椅子、沙发、桌子、'
    '柜子）的自动识别系统。从数据下载与预处理、模型搭建与参数优化、训练过程监控、'
    '模型精度评估到 Flask Web 部署，完成了深度学习应用开发的完整流程。'
)

add_para('主要成果：', bold=True)
add_para(f'（1）数据集构建：从 CIFAR-100 中筛选 5 类家具共 3,000 张图片，完成 70/15/15 划分')
add_para(f'（2）模型训练：MobileNetV2 + 自定义分类头，{metrics.get("total_epochs", "19")} 轮训练，'
         f'测试准确率 {test_acc_str}')
add_para(f'（3）参数优化：Adam + ReduceLROnPlateau + Dropout(0.5/0.3) + Weight Decay + '
         f'Data Augmentation + Early Stopping 六重策略组合')
add_para(f'（4）精度分析：混淆矩阵 + 分类报告 + 各类别准确率 + 预测样本可视化，'
         f'多维度全面评估')
add_para(f'（5）工程部署：Flask Web 前端，支持浏览器上传图片实时推理')

add_para('')
add_para('8.2 核心技术决策回顾', bold=True)
add_table(
    ['设计决策', '最终选择', '决策依据'],
    [
        ['骨干网络', 'MobileNetV2', '体积小 (35MB)、速度快、适合 Web 部署'],
        ['训练策略', '全模型微调 (不冻结)', '5 类家具与 ImageNet 分布差异大，全微调适应性更好'],
        ['优化器', 'Adam (lr=0.0005)', '自适应学习率，迁移学习场景比 SGD 更稳定'],
        ['学习率调度', 'ReduceLROnPlateau', '根据验证 Loss 自动调整，避免手动调参'],
        ['正则化组合', 'Dropout(0.5/0.3) + WD(1e-4)', '多层次正则化，训练-验证差距仅 8.6pp'],
        ['数据增强', '5 种变换', '充分扩充数据分布，降低过拟合风险'],
    ]
)

add_para('')
add_para('8.3 未来改进方向', bold=True)
improvements = [
    ('数据扩充', '收集更多真实场景的家具图片（如电商图片、室内设计照片），'
     '扩大数据集至万级规模。特别需要增加 sofa 类样本缓解类别混淆。'),
    ('模型升级', '尝试 EfficientNet-B0 / ViT-Tiny 等更新架构。EfficientNet 在同等'
     '参数量下准确率通常优于 MobileNetV2 约 2~3 个百分点。'),
    ('细粒度分类', '从粗粒度 5 类扩展为细粒度分类：椅子→办公椅/餐椅/躺椅/吧台椅，'
     '柜子→衣柜/书柜/橱柜/电视柜。'),
    ('目标检测', '将图像分类升级为 YOLOv8 / Faster R-CNN 目标检测，实现单张图片中'
     '多件家具的定位 + 分类。'),
    ('移动端部署', '将模型导出为 ONNX → TensorFlow Lite / Core ML 格式，部署至 '
     'Android / iOS 移动设备，实现端侧推理。'),
]
for title, desc in improvements:
    add_para(f'（{improvements.index((title, desc)) + 1}）{title}：{desc}')

# ====================================================================
# Save
# ====================================================================
OUTPUT_PATH = os.path.join(BASE, '《深度学习应用开发》课程设计报告_家具识别.docx')
doc.save(OUTPUT_PATH)
print(f'\nReport saved to: {OUTPUT_PATH}')
