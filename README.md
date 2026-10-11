# CIFAR-10 Small-Data Robustness (500 training images)

Train a small CNN with an Inception-style module **from scratch** on only 500 CIFAR-10 images (400 train / 100 validation) and evaluate it on a shifted (rotated, darkened/brightened, noisy, occluded) test set.

## Results (mean ± std over 3 seeds: 42, 123, 2026)

| Metric | Mean | Std |
|---|---|---|
| Validation accuracy | 20.67% | ± 3.21% |
| Clean-test accuracy | 18.99% | ± 3.92% |
| Shifted-test accuracy (local approximation) | 17.41% | ± 2.94% |

- Test-time augmentation (original + horizontal flip): 17.94% → 18.15% (+0.21 points), about 2x inference time.
- Grayscale proxy for the hidden corruption: 11.63% (vs. 18.68% clean). This is a proxy, not the official hidden corruption.

**Note:** results are only modestly above chance (10%). The shifted test set here is a local approximation built in the notebook, not the official hidden pipeline.

## Model and training setup

- Framework: TensorFlow 2.21 (Keras), trained from scratch (no pretrained weights)
- Architecture: CNN with an Inception-style module, 28,158 trainable parameters (limit: 1,000,000)
- Augmentation (on the fly, training only): random crop, horizontal flip, rotation up to ±25°, brightness/contrast jitter, blur/noise, Cutout, Mixup
- Regularization: BatchNorm, Dropout 0.35 before the classifier, AdamW (weight decay 1e-4), label smoothing 0.05, ReduceLROnPlateau, early stopping (patience 5)
- All hyperparameters were chosen using the 100-image validation split only. Test data was used once at the end.

## Repository contents

- `YOUR_NOTEBOOK_NAME.ipynb` - the complete notebook (Parts A to F) with outputs
- `Summary.docx` - one-page summary
- `README.md` - this file
- `requirements.txt` - Python dependencies
- Plots (optional): learning curves, per-class accuracy, confusion matrix

## How to reproduce

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```
2. **Install dependencies** (Python 3.10+ recommended)
   ```bash
   pip install -r requirements.txt
   ```
3. **Open the notebook** in Jupyter, Google Colab, or Kaggle:
   ```bash
   jupyter notebook YOUR_NOTEBOOK_NAME.ipynb
   ```
4. **Run all cells from top to bottom** (Kernel > Restart & Run All).
   - CIFAR-10 downloads automatically via `keras.datasets` on the first run (about 170 MB, internet required).
   - The 500-image subset (50 per class) and the 400/100 train/validation split are created by the notebook with a fixed seed.
   - The shifted test set is generated inside the notebook (local approximation).
5. **Seeds:** the notebook runs 3 seeds (42, 123, 2026) and reports mean ± std. Results can vary slightly across hardware and TensorFlow versions.

Expected runtime: a few minutes on CPU for the small model; faster with a GPU.

## Rules followed

- No pretrained weights; at most 1,000,000 trainable parameters
- Inception-style module included
- No tuning on shifted or clean test data
- Augmentation applied on the fly, training data only
- Maximum 60 epochs, fixed seeds
