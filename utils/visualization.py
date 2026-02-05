import matplotlib.pyplot as plt
import numpy as np
from .visual_predict import visualize_predict

def plot_accuracy(accuracy_list1, accuracy_list2, save_path, dataset1_name, dataset2_name):
    plt.figure(figsize=(10, 6))
    epochs = range(10, len(accuracy_list1) * 10 + 1, 10)
    plt.plot(epochs, accuracy_list1, marker='o', label=f'{dataset1_name} (Train)', linewidth=2)
    plt.plot(epochs, accuracy_list2, marker='s', label=f'{dataset2_name} (Target domain)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Target domain')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.savefig(save_path)
    plt.close()

def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    print(f"Input ground truth unique labels: {np.unique(gt_vis)}")
    print(f"Input prediction unique labels: {np.unique(pred_vis)}")
    
    gt_mask = gt_vis != -1
    pred_mask = pred_vis != -1
    
    gt_processed = np.copy(gt_vis)
    pred_processed = np.copy(pred_vis)
    
    valid_labels = sorted(list(set(np.unique(gt_vis[gt_mask]))))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
    
    for old_label, new_label in label_mapping.items():
        gt_processed[gt_vis == old_label] = new_label
        pred_processed[pred_vis == old_label] = new_label
    
    gt_processed[~gt_mask] = -1
    pred_processed[~pred_mask] = -1
    
    print(f"Processed ground truth unique labels: {np.unique(gt_processed)}")
    print(f"Processed prediction unique labels: {np.unique(pred_processed)}")
    
    visualize_predict(gt_processed, pred_processed, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    
    if not only_vis_label:
        visualize_predict(gt_processed, pred_processed, 
                        save_single_predict_path.replace('.png', '_mask.png'), 
                        save_single_gt_path, 
                        only_vis_label=True)