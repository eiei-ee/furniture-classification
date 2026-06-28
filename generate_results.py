"""
生成训练曲线、评估指标、混淆矩阵等所有报告所需图表
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import os, random, time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Config
DATA_DIR = 'dataset'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_EPOCHS = 25

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Load datasets
train_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset/val', transform=eval_transform)
test_dataset = datasets.ImageFolder('dataset/test', transform=eval_transform)
class_names = train_dataset.classes
class_to_idx = {name: i for i, name in enumerate(class_names)}
print(f'Classes: {class_names}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

# Build model
def build_model(num_classes=5, freeze_backbone=False):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

model = build_model(num_classes=NUM_CLASSES, freeze_backbone=False)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total_params:,}')
print(f'Trainable params: {trainable_params:,}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

# Training
print(f'\nTraining {NUM_EPOCHS} epochs...')
print('=' * 70)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0
best_epoch = 0
patience = 0
EARLY_STOP = 8

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    scheduler.step(val_loss)

    elapsed = time.time() - epoch_start
    print(f'Epoch {epoch:3d}/{NUM_EPOCHS} | '
          f'TrL: {train_loss:.4f} | TrA: {train_acc:.4f} | '
          f'VaL: {val_loss:.4f} | VaA: {val_acc:.4f} | '
          f'LR: {optimizer.param_groups[0]["lr"]:.2e} | {elapsed:.1f}s')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        patience = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names,
        }, 'furniture_model_best.pth')
    else:
        patience += 1

    if patience >= EARLY_STOP:
        print(f'\nEarly stopping at epoch {epoch}')
        break

total_time = time.time() - start_time
print(f'\nTraining done! Time: {total_time:.0f}s ({total_time/60:.1f}min)')
print(f'Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}')

# =====================================================
# Plot 1: Training curves (Loss + Accuracy)
# =====================================================
epochs_range = range(1, len(history['train_loss']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax1.scatter(best_epoch, history['val_loss'][best_epoch - 1],
            color='red', s=100, zorder=5, label=f'Best (Epoch {best_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
ax2.scatter(best_epoch, best_val_acc, color='red', s=100, zorder=5,
            label=f'Best ({best_val_acc:.2%})')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print('Saved: training_curves.png')

# =====================================================
# Load best model and evaluate on test set
# =====================================================
checkpoint = torch.load('furniture_model_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'\nLoaded best model (Epoch {checkpoint["epoch"]}, Val Acc: {checkpoint["val_acc"]:.4f})')

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f'\n[Test Set Results]')
print(f'  Loss:     {test_loss:.4f}')
print(f'  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

# =====================================================
# Collect all predictions for confusion matrix
# =====================================================
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# =====================================================
# Plot 2: Confusion Matrix
# =====================================================
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print('Saved: confusion_matrix.png')

# =====================================================
# Classification Report
# =====================================================
print('\n' + '=' * 60)
print('Classification Report')
print('=' * 60)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

# =====================================================
# Per-class accuracy
# =====================================================
print('Per-class Accuracy:')
for i, cls_name in enumerate(class_names):
    cls_correct = cm[i, i]
    cls_total = cm[i].sum()
    print(f'  {cls_name}: {cls_correct}/{cls_total} ({cls_correct/cls_total*100:.2f}%)')

# =====================================================
# Plot 3: Prediction samples
# =====================================================
def denormalize(tensor):
    img = tensor.clone()
    for t, m, s in zip(img, MEAN, STD):
        t.mul_(s).add_(m)
    return img.clamp(0, 1)

def predict_single_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    return pred.item(), probs.squeeze().cpu().numpy()

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()

samples_per_class = 2
shown = []
for cls_idx, cls_name in enumerate(class_names):
    cls_images = [(img, label) for img, label in test_dataset if label == cls_idx]
    random.shuffle(cls_images)
    for j in range(samples_per_class):
        if j < len(cls_images):
            shown.append((cls_images[j][0], cls_idx, cls_name))

for i, (img, true_idx, true_name) in enumerate(shown[:10]):
    pred_idx, probs = predict_single_image(model, img, device)
    pred_name = class_names[pred_idx]
    confidence = probs[pred_idx]

    img_display = denormalize(img)
    ax = axes[i]
    ax.imshow(img_display.permute(1, 2, 0))
    color = 'green' if pred_idx == true_idx else 'red'
    ax.set_title(f'True: {true_name}\nPred: {pred_name} ({confidence:.1%})',
                 color=color, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('Prediction Samples (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('prediction_samples.png', dpi=150, bbox_inches='tight')
print('Saved: prediction_samples.png')

# =====================================================
# Plot 4: Per-class accuracy bar chart
# =====================================================
fig, ax = plt.subplots(figsize=(10, 5))
per_class_acc = [cm[i, i] / cm[i].sum() * 100 for i in range(len(class_names))]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
bars = ax.bar(class_names, per_class_acc, color=colors)
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(per_class_acc) * 1.2)
ax.set_title('Per-Class Accuracy (Test Set)', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.axhline(y=test_acc*100, color='gray', linestyle='--', alpha=0.5, label=f'Overall ({test_acc*100:.1f}%)')
ax.legend()
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
print('Saved: per_class_accuracy.png')

# Save metrics to text file for report
with open('model_metrics.txt', 'w') as f:
    f.write(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n')
    f.write(f'Test Loss: {test_loss:.4f}\n')
    f.write(f'Best Val Accuracy: {best_val_acc:.4f}\n')
    f.write(f'Best Epoch: {best_epoch}\n')
    f.write(f'Total Epochs Trained: {len(history["train_loss"])}\n')
    f.write(f'Total Params: {total_params:,}\n')
    f.write(f'Trainable Params: {trainable_params:,}\n')
    f.write(f'\nFinal Training History:\n')
    for i in range(len(history['train_loss'])):
        f.write(f'Epoch {i+1}: TrL={history["train_loss"][i]:.4f}, TrA={history["train_acc"][i]:.4f}, '
                f'VaL={history["val_loss"][i]:.4f}, VaA={history["val_acc"][i]:.4f}\n')
    f.write(f'\n{classification_report(all_labels, all_preds, target_names=class_names, digits=4)}\n')

print('\nDone! All results saved.')
