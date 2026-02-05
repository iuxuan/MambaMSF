import torch
import numpy as np
import torch.utils.data as data

class HSIDataset(data.Dataset):
    """HSI数据集的patch方式加载"""
    def __init__(self, data, gt, patch_size, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        
        # 将0标签设为背景(-1)，其他标签减1
        self.gt = np.where(gt == 0, -1, gt - 1)
        
        # 预处理: 填充图像边缘
        half_patch = patch_size // 2
        self.padded_data = np.pad(data,
                                 ((half_patch, half_patch),
                                  (half_patch, half_patch),
                                  (0, 0)),
                                 mode='reflect')
        
        # 获取有效位置和对应的patch
        self.patches = []
        self.labels = []
        
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                if self.gt[i, j] != -1:  # 排除背景
                    # 预先切分patch
                    patch = self.padded_data[
                        i:i + self.patch_size,
                        j:j + self.patch_size,
                        :]
                    self.patches.append(patch.astype(np.float32))
                    self.labels.append(self.gt[i, j])
        
        self.patches = np.array(self.patches)
        self.labels = np.array(self.labels)
        print(f"Dataset created with {len(self.labels)} valid samples")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch.float(), torch.tensor(int(label), dtype=torch.long)

def create_patch_dataset(data, gt, patch_size, train_indices, transform=None):
    """创建基于patch的数据集"""
    height, width = gt.shape
    patches = []
    labels = []
    
    for idx in train_indices:
        i = idx // width
        j = idx % width
        if gt[i, j] != -1:  # 只处理有标签的元素
            half_patch = patch_size // 2
            padded_data = np.pad(data, 
                                ((half_patch, half_patch), 
                                 (half_patch, half_patch), 
                                 (0, 0)), 
                                mode='reflect')
            
            patch = padded_data[
                i:i + patch_size,
                j:j + patch_size,
                :
            ]
            
            if transform:
                patch = transform(patch)
            patches.append(patch)
            labels.append(gt[i, j])
    
    return patches, labels