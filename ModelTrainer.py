import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import torch
import torchvision
import torch.nn as nn

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

desired_classes = ['Class']
class_names = desired_classes

# Load dataset to check available classes
temp_dataset = torchvision.datasets.ImageFolder(root=train_dir)
available_classes = temp_dataset.classes
print(f"Available classes in dataset: {available_classes}")
print(f"Training on classes: {desired_classes}")

for cls in desired_classes:
    if cls not in available_classes:
        raise ValueError(f"Class '{cls}' not found in dataset. Available classes: {available_classes}")

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False
set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
pretrained_vit_transforms = pretrained_vit_weights.transforms()
print(pretrained_vit_transforms)

k_folds = 5
batch_size = 64
num_epochs = 3
learning_rate = 1e-3
max_images_per_class = 10000

data_transform = pretrained_vit_weights.transforms()
full_dataset = ImageFolder(root=train_dir, transform=data_transform)

def filter_dataset_by_classes(dataset, desired_classes, max_per_class=None):
    desired_indices = [dataset.class_to_idx[cls] for cls in desired_classes]
    class_samples = {idx: [] for idx in desired_indices}
    for path, label in dataset.samples:
        if label in desired_indices:
            class_samples[label].append((path, label))
    if max_per_class is not None:
        for class_idx in class_samples:
            class_samples[class_idx] = class_samples[class_idx][:max_per_class]
    filtered_samples = []
    for class_idx in desired_indices:
        new_label = desired_indices.index(class_idx)
        for path, _ in class_samples[class_idx]:
            filtered_samples.append((path, new_label))
    dataset.samples = filtered_samples
    dataset.targets = [label for _, label in filtered_samples]
    dataset.classes = desired_classes
    dataset.class_to_idx = {cls: i for i, cls in enumerate(desired_classes)}
    return dataset

full_dataset = filter_dataset_by_classes(full_dataset, desired_classes, max_images_per_class)

class_counts = {}
for _, label in full_dataset.samples:
    class_name = class_names[label]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
targets = [label for _, label in full_dataset.samples]

fold_accuracies = []
class_stats = {class_name: {"TP": [], "TN": [], "FP": [], "FN": [], "recall": [], "precision": [], "f1": []} for class_name in class_names}

best_model_state = None
best_fold_accuracy = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"Fold {fold + 1}/{k_folds}")

    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)

    print(f"  Train size: {len(train_sub)}")
    print(f"  Validation size: {len(val_sub)}")

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size)

    model = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.heads.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_loader_with_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (X, y) in enumerate(train_loader_with_progress):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 == 0:
                train_loader_with_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f"  Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            preds = torch.argmax(y_pred, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    acc = np.trace(cm) / np.sum(cm)
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    fold_accuracies.append(acc)

    if acc > best_fold_accuracy:
        best_fold_accuracy = acc
        best_model_state = model.state_dict()

    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_stats[class_name]["TP"].append(TP)
        class_stats[class_name]["TN"].append(TN)
        class_stats[class_name]["FP"].append(FP)
        class_stats[class_name]["FN"].append(FN)
        class_stats[class_name]["recall"].append(recall)
        class_stats[class_name]["precision"].append(precision)
        class_stats[class_name]["f1"].append(f1)

        print(f"  Class '{class_name}' Metrics:")
        print(f"    TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"    Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}\n")

torch.save(best_model_state, "best_vit_model.pth")

summary_data = {"Metric": ["TP", "TN", "FP", "FN", "Recall", "Precision", "F1-Score"]}
for class_name in class_names:
    summary_data[class_name] = [
        int(round(np.mean(class_stats[class_name]["TP"]))),
        int(round(np.mean(class_stats[class_name]["TN"]))),
        int(round(np.mean(class_stats[class_name]["FP"]))),
        int(round(np.mean(class_stats[class_name]["FN"]))),
        round(np.mean(class_stats[class_name]["recall"]), 4),
        round(np.mean(class_stats[class_name]["precision"]), 4),
        round(np.mean(class_stats[class_name]["f1"]), 4)
    ]
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

rows = ["Recall", "Precision", "F1-Score"]
metric_key_map = {"Recall": "recall", "Precision": "precision", "F1-Score": "f1"}
columns = []
data = []

for fold in range(k_folds):
    for class_name in class_names:
        columns.append((f"Fold {fold+1}", class_name))

for metric in rows:
    row = []
    key = metric_key_map[metric]
    for fold in range(k_folds):
        for class_name in class_names:
            value = class_stats[class_name][key][fold]
            row.append(round(value, 4))
    data.append(row)

multi_columns = pd.MultiIndex.from_tuples(columns, names=["Fold", "Class"])
formatted_df = pd.DataFrame(data, index=rows, columns=multi_columns)
csv_path = "./fold_metrics_summary.csv"
formatted_df.to_csv(csv_path)

# Load and predict functions
def load_best_model():
    model = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
    model.heads = nn.Linear(in_features=768, out_features=len(class_names))
    model.load_state_dict(torch.load("best_vit_model.pth"))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model):
    transform = pretrained_vit_weights.transforms()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return class_names[pred]
