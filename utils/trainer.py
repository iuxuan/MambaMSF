import torch
import numpy as np
from tqdm import tqdm

def train_epoch(net, train_loader, optimizer, criterion, device):
    net.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    for batch_patches, batch_labels in tqdm(train_loader, desc='Training'):
        batch_patches = batch_patches.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(batch_patches)
        
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
        total_loss += loss.item()
        total += batch_labels.size(0)
        correct += predicted.eq(batch_labels).sum().item()
    
    # 打印训练集的预测分布
    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    print("\nTraining set prediction distribution:")
    for label, count in zip(unique_preds, pred_counts):
        print(f"Class {label}: {count} predictions")
    
    # 打印训练集的真实标签分布
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print("\nTraining set true label distribution:")
    for label, count in zip(unique_labels, label_counts):
        print(f"Class {label}: {count} samples")
    
    return total_loss / len(train_loader), correct / total

def evaluate_full_image(net, data, gt, patch_size, device, transform, batch_size=2048):
    net.eval()
    height, width, channels = data.shape
    
    print(f"Target gt before conversion: {np.unique(gt)}")
    gt_converted = np.where(gt == 0, -1, gt - 1)
    print(f"Target gt after conversion: {np.unique(gt_converted)}")
    
    predictions = np.zeros((height, width), dtype=np.int64)
    mask = gt_converted != -1
    
    half_patch = patch_size // 2
    padded_data = np.pad(data,
                        ((half_patch, half_patch),
                         (half_patch, half_patch),
                         (0, 0)),
                        mode='reflect')
    
    patches = []
    positions = []
    
    with torch.no_grad():
        for i in range(height):
            for j in range(width):
                if gt_converted[i, j] != -1:
                    patch = padded_data[i:i+patch_size, j:j+patch_size, :]
                    patch = patch.astype(np.float32)
                    patch = transform(patch).float()
                    patches.append(patch)
                    positions.append((i,j))
                    
                    if len(patches) == batch_size or (i == height-1 and j == width-1):
                        if patches:
                            batch_patches = torch.stack(patches).to(device)
                            outputs = net(batch_patches)
                            _, pred = outputs.max(1)
                            
                            for pos, p in zip(positions, pred.cpu().numpy()):
                                predictions[pos[0], pos[1]] = p
                            
                            patches = []
                            positions = []
    
    correct = (predictions[mask] == gt_converted[mask]).sum()
    total = mask.sum()
    accuracy = correct / total
    
    class_acc = []
    unique_classes = np.unique(gt_converted[mask])
    for c in unique_classes:
        class_mask = gt_converted == c
        class_correct = (predictions[class_mask] == gt_converted[class_mask]).sum()
        class_total = class_mask.sum()
        class_acc.append(float(class_correct) / class_total)
    
    print(f"Predictions unique values: {np.unique(predictions[mask])}")
    print(f"Number of valid pixels: {total}")
    print(f"Number of correct predictions: {correct}")
    
    predictions[~mask] = -1
    print(f"Final predictions range: {np.unique(predictions[predictions != -1])}")
    
    return accuracy, predictions, class_acc