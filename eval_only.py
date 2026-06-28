"""仅加载已有模型评估，快速生成指标和图表"""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os, random

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'C:\Users\zzz\Desktop\家具分类'
device = torch.device('cpu')
IMG_SIZE = 224
BATCH_SIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["bed", "cabinet", "chair", "sofa", "table"]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Load test data
test_dataset = datasets.ImageFolder(os.path.join(BASE, 'dataset', 'test'), transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'Test samples: {len(test_dataset)}')

# Build & load model
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5), nn.Linear(in_features, 512), nn.ReLU(inplace=True),
    nn.Dropout(0.3), nn.Linear(512, 5)
)
model = model.to(device)

ckpt = torch.load(os.path.join(BASE, 'furniture_model.pth'), map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded. Epoch: {ckpt['epoch']}, Val Acc: {ckpt['val_acc']:.4f}")

# Evaluate
criterion = nn.CrossEntropyLoss()
running_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

test_loss = running_loss / total
test_acc = correct / total
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
print('Saved: confusion_matrix.png')

# Per-class accuracy
fig, ax = plt.subplots(figsize=(10, 5))
per_class_acc = [cm[i, i] / cm[i].sum() * 100 for i in range(5)]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
bars = ax.bar(CLASS_NAMES, per_class_acc, color=colors)
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%',
            ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(per_class_acc) * 1.2)
ax.set_title('Per-Class Accuracy (Test Set)', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.axhline(y=test_acc*100, color='gray', linestyle='--', alpha=0.5, label=f'Overall ({test_acc*100:.1f}%)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
print('Saved: per_class_accuracy.png')

# Classification report
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
print('\n' + report)

# Save metrics
with open(os.path.join(BASE, 'model_metrics.txt'), 'w') as f:
    f.write(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n')
    f.write(f'Test Loss: {test_loss:.4f}\n')
    f.write(f'Val Accuracy (from ckpt): {ckpt["val_acc"]:.4f}\n')
    f.write(f'Best Epoch: {ckpt["epoch"]}\n')
    f.write(f'\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)}\n')
    for i in range(5):
        f.write(f'Per-class {CLASS_NAMES[i]}: {per_class_acc[i]:.2f}%\n')

# Prediction samples
def denormalize(tensor):
    img = tensor.clone()
    for t, m, s in zip(img, MEAN, STD):
        t.mul_(s).add_(m)
    return img.clamp(0, 1)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()
samples_per_class = 2
shown = []
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_images = [(img, label) for img, label in test_dataset if label == cls_idx]
    random.shuffle(cls_images)
    for j in range(samples_per_class):
        if j < len(cls_images):
            shown.append((cls_images[j][0], cls_idx, cls_name))

for i, (img, true_idx, true_name) in enumerate(shown[:10]):
    model.eval()
    with torch.no_grad():
        img_tensor = img.unsqueeze(0).to(device)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_idx = int(torch.argmax(outputs, 1).item())
    pred_name = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    img_display = denormalize(img)
    ax = axes[i]
    ax.imshow(img_display.permute(1, 2, 0))
    color = 'green' if pred_idx == true_idx else 'red'
    ax.set_title(f'True: {true_name}\nPred: {pred_name} ({confidence:.1%})', color=color, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('Prediction Samples (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'prediction_samples.png'), dpi=150, bbox_inches='tight')
print('Saved: prediction_samples.png')
print('\nDone!')
