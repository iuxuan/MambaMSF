# MambaMSF: A Mamba-based Multi-scale Feature Fusion Method for Hyperspectral Image Classification

This repository contains the implementation of **MambaMSF**, a hyperspectral image (HSI) classification method that integrates multi-scale feature fusion with the Mamba architecture.

## 📄 Reference
**MambaMSF: A Mamba-based Multi-scale Feature Fusion Method for Hyperspectral Image Classification**
*Junyan Xuan, Zhenzhen Ren, Bin Deng, Yikui Zhai*

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
│   ├── PaviaU.mat          (Required: Image Data)
│   └── PaviaU_gt.mat       (Required: Ground Truth)
├── HanChuan/
│   ├── WHU_Hi_HanChuan.mat (Required: Image Data)
│   └── WHU_Hi_HanChuan_gt.mat
└── HongHu/
    ├── WHU_Hi_HongHu.mat   (Required: Image Data)
    └── WHU_Hi_HongHu_gt.mat
```

> **Note**: Due to file size limits and copyright, the original `.mat` data files are not included in this repository. Please download them from their respective sources and place them in the `data/` directory as shown above.

## 🚀 Usage

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

### Configuration
You can modify `config.py` to change training parameters:
- `dataset_index`: 0 for UP (Pavia University)
- `max_epoch`: Number of training epochs (Default: 200)
- `device`: GPU device (e.g., "cuda:0")

## 📊 Results

### Pavia University (UP)
- **Overall Accuracy (OA)**: ~96.27% (Our Run) / ~96.65% (Paper)
- **Average Accuracy (AA)**: ~97.12%
- **Kappa Coefficient**: ~0.95

## 🔧 Troubleshooting
- **`torch.load` warning**: The code has been updated to handle `weights_only=True` for safer checkpoint loading.
- **`calflops` / `spectral` errors**: Ensure these packages are installed via `pip install calflops spectral`.
