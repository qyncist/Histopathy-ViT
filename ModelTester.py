import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os
import re

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder class labels
test_class_names = ['class']
model_class_names = ['class']

# Load pretrained ViT weights
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# Load trained ViT model
def load_trained_model(model_path="best_vit_model.pth"):
    model = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
    model.heads = nn.Linear(in_features=768, out_features=len(model_class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_image_number(filename):
    match = re.match(r'(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return None

def filter_test_dataset_exclude_training(dataset, desired_classes, training_limit=10000):
    available_classes = [cls for cls in desired_classes if cls in dataset.class_to_idx]
    desired_indices = [dataset.class_to_idx[cls] for cls in available_classes]
    class_samples = {idx: [] for idx in desired_indices}

    for path, label in dataset.samples:
        if label in desired_indices:
            filename = os.path.basename(path)
            img_number = extract_image_number(filename)
            if img_number is not None and img_number > training_limit:
                class_samples[label].append((path, label))

    filtered_samples = []
    for class_idx in desired_indices:
        new_label = desired_indices.index(class_idx)
        for path, _ in class_samples[class_idx]:
            filtered_samples.append((path, new_label))

    dataset.samples = filtered_samples
    dataset.targets = [label for _, label in filtered_samples]
    dataset.classes = available_classes
    dataset.class_to_idx = {cls: i for i, cls in enumerate(available_classes)}

    return dataset

def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            raw_predictions = torch.argmax(outputs, dim=1)
            mapped_predictions = [2 if pred.item() == 3 else pred.item() for pred in raw_predictions]
            all_predictions.extend(mapped_predictions)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_predictions), np.array(all_labels)

def calculate_metrics(y_true, y_pred, class_names):
    accuracy = np.mean(y_true == y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def save_results_to_csv(metrics, class_names, save_path="results.csv"):
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': metrics['precision_per_class'],
        'Recall': metrics['recall_per_class'],
        'F1_Score': metrics['f1_per_class']
    })
    summary = pd.DataFrame([{
        'Class': 'Overall',
        'Precision': metrics['precision_macro'],
        'Recall': metrics['recall_macro'],
        'F1_Score': metrics['f1_macro']
    }])
    df = pd.concat([df, summary], ignore_index=True)
    df.to_csv(save_path, index=False)

def run_test_evaluation():
    model = load_trained_model("best_vit_model.pth")
    test_transform = pretrained_vit_weights.transforms()
    test_dataset = ImageFolder(root="/path/to/test_data", transform=test_transform)
    test_dataset = filter_test_dataset_exclude_training(test_dataset, test_class_names, training_limit=10000)

    if len(test_dataset) == 0:
        print("No test images found.")
        return None

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    predictions, true_labels = evaluate_model(model, test_loader)
    metrics = calculate_metrics(true_labels, predictions, test_class_names)
    plot_confusion_matrix(metrics['confusion_matrix'], test_class_names)
    save_results_to_csv(metrics, test_class_names)
    return metrics

if __name__ == "__main__":
    run_test_evaluation()
