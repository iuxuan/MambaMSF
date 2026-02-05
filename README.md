# MambaMSF: A Mamba-based Multi-scale Feature Fusion Method for Hyperspectral Image Classification

This repository contains the implementation of **MambaMSF**, a hyperspectral image (HSI) classification method that integrates multi-scale feature fusion with the Mamba architecture.

## 🛠️ Environment Setup

This project requires Python 3.8+ and PyTorch.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Note on `mamba-ssm`**
   `mamba-ssm` requires a GPU and matching CUDA version. If you encounter issues, please refer to the [official installation guide](https://github.com/state-spaces/mamba).

## 📂 Dataset Preparation

The project expects the following directory structure in `./data`:

```
data/
├── UP/
│   ├── PaviaU.mat          
│   └── PaviaU_gt.mat       
├── HanChuan/
│   ├── WHU_Hi_HanChuan.mat
│   └── WHU_Hi_HanChuan_gt.mat
└── HongHu/
    ├── WHU_Hi_HongHu.mat   
    └── WHU_Hi_HongHu_gt.mat
```

> **Note**: Due to file size limits and copyright, the original `.mat` data files are not included in this repository. Please download them from their respective sources and place them in the `data/` directory as shown above.

## Dataset Download
You can download the processed datasets (UP, HanChuan, HongHu) from Google Drive:
**https://drive.google.com/drive/folders/1lsj3U7SAv4JkFMiWYKq_MS0nvFqQ0rfu?usp=drive_link**

## Usage

To train and evaluate the model, you can select the target dataset in `config.py`.

### Running on Pavia University (Default)
```bash
python train_MambaMSF.py
```

### Running on HanChuan or HongHu
1. Open `config.py`.
2. Change `dataset_index`:
   - `0`: UP (Pavia University)
   - `1`: HanChuan
   - `2`: HongHu
3. Run the training script.

## Results

### Pavia University (UP)
- **Overall Accuracy (OA)**: 96.74%
- **Average Accuracy (AA)**: 97.11%
- **Kappa Coefficient**: 96.63%

## Reference

If you use this code for your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11065432, 
   author={Xuan, Junyan and Ren, Zhenzhen and Deng, Bin and Zhai, Yikui}, 
   booktitle={2025 IEEE 7th International Conference on Communications, Information System and Computer Engineering (CISCE)}, 
   title={MambaMSF: A Mamba-based Multi-scale Feature Fusion Method for Hyperspectral Image Classification}, 
   year={2025}, 
   volume={}, 
   number={}, 
   pages={388-392}, 
   keywords={Learning systems;Adaptation models;Computer architecture;Feature extraction;Data models;Spatial databases;Faces;Hyperspectral imaging;Image classification;Information systems;Hyperspectral image classification;Multi-scale feature fusion;Adaptive fusion mechanism;Mamba}, 
   doi={10.1109/CISCE65916.2025.11065432}}
```
