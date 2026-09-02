# Computer-Vision
# Computer Vision Coursework, Classical and Deep Learning

Project: Υπολογιστική Όραση (Computer Vision), 2025-2026

Two projects: Project #1 covers classical image processing and feature extraction with NumPy/scikit-image, and Project #2 covers deep-learning-based vision with PyTorch (semantic segmentation and a CNN-versus-Transformer comparison).

## Contents

| Folder | Project  |
|---|---|
| [`CV-project-1/CV-project-1/CV-Project#1/`](<CV-project-1/CV-project-1/CV-Project#1>) | Project #1: classical image processing, 8 exercises |
| [`project-2/CV-project2/`](<project-2/CV-project2>) | Project #2: deep learning, 3 exercises (segmentation x2, CNN vs ViT) |

Each folder also holds the corresponding written report as a PDF; both reports are effectively a rendered write-up of the same code and results described below.

## Project #1: classical image processing

A single Jupyter notebook, [`cv_project1.ipynb`](<CV-project-1/CV-project-1/CV-Project#1/cv_project1.ipynb>) (60 cells), working through 8 classical computer-vision exercises with `numpy`, `scikit-image`, and `matplotlib`. Every exercise ends with a quantitative comparison table (PSNR, SSIM, or MSE against the original image), as the assignment brief requires.

1. **Brightness transforms and histograms** (`lenna.bmp`): normalized histogram, global histogram equalization, and power-law (gamma) transforms at gamma in {0.4, 0.7, 1.0, 1.5, 2.2}, scored by PSNR/SSIM. Gamma = 0.7 gives the best contrast-versus-structure trade-off (highest SSIM among the non-identity transforms); histogram equalization overshoots into visible oversaturation.
2. **Spatial-domain filtering** (`lenna.bmp`): salt-and-pepper noise (p=0.08) and Gaussian noise (variance 0.01) are injected, then removed with median filtering (3x3/5x5/7x7) and with mean/Gaussian filtering (kernel size and sigma swept), scored by PSNR/SSIM. Median 3x3 wins for salt-and-pepper noise (edge-preserving), Gaussian sigma=1.0 wins for Gaussian noise (matches the noise's own statistics): filter choice has to match the noise type.
3. **Frequency-domain filtering** (`butterfly.jpg`): 2D FFT amplitude and phase spectra, then ideal and Gaussian low-pass filters at cutoffs D0 in {0.1, 0.3, 0.5}, scored by MSE. Ideal filters cause Gibbs ringing from their sharp cutoff; Gaussian filters roll off smoothly and reach lower MSE at the same cutoff.
4. **Thresholding segmentation** (`coins.jpg`): global Otsu, multi-level Otsu (3 classes), and local adaptive thresholding. Local thresholding (block size 51) is the most robust to uneven illumination; global Otsu is simplest and works well on the bimodal, evenly-lit histogram here.
5. **Watershed segmentation** (`coins.jpg`): distance-transform watershed, with a grid search over smoothing sigma, minimum peak distance, and relative threshold. The minimum peak distance should roughly track half a coin's radius; too small over-segments, too large merges coins together.
6. **Edge detection** (`crowd.bmp`): Sobel, Roberts, and Prewitt gradients with Otsu thresholding, Laplacian-of-Gaussian, and Canny, plus a quantitative edge-density and connected-component comparison. Ranking on accuracy and edge continuity: Canny > LoG > Sobel/Prewitt > Roberts; Roberts is the most noise-sensitive.
7. **Texture and shape features** (`butterfly.jpg`, `girlface.bmp`, `boats.bmp`): Local Binary Patterns at (P=8, R=1) and (P=16, R=2), and HOG descriptors. LBP captures texture (feather texture, skin smoothness, geometric boat lines), HOG captures gradient and edge geometry (useful for facial structure); the two descriptors are complementary.
8. **SIFT keypoints and matching** (`lighthouse.bmp`): an affine transform (30 degree rotation, 0.7 scale) is applied to a copy of the image, SIFT keypoints are detected on both versions, and matched with Lowe's ratio test (threshold 0.75) plus cross-checking. SIFT keeps a high match rate across the rotation and scale change, demonstrating its scale/rotation invariance; the false matches that do occur concentrate in the transform's zero-padded regions.

## Project #2: deep learning for vision

Three standalone PyTorch scripts, one per exercise ([`ex1_sbd.py`](<project-2/CV-project2/ex1_sbd.py>), [`ex2_pet.py`](<project-2/CV-project2/ex2_pet.py>), [`ex3_cifar.py`](<project-2/CV-project2/ex3_cifar.py>)), each training 3 hyperparameter variants on CPU-sized subsets of the full datasets, and each writing training curves, sample predictions, a `results.json`, and a comparison bar chart to `results/exN/`.

Exercises 1 and 2 share the same lightweight U-Net: 3 downsampling stages (conv block plus max-pool, channels doubling from a configurable base), a bottleneck, and 3 upsampling stages (transposed convolution, skip-connection concatenation, conv block), a 1x1 convolution head, trained with Adam and cosine learning-rate annealing, evaluated on pixel accuracy and mean IoU (mIoU).

### Exercise 1: SBD semantic segmentation

- Semantic Boundaries Dataset (via `torchvision`), 21 classes (background plus 20 PASCAL VOC categories), resized to 128x128, 600 training and 200 validation images (a subset of the full roughly 8,500/2,800, chosen for CPU feasibility).
- Three experiments: baseline (5 epochs, lr=1e-3, base channels=16), more epochs (10 epochs, lr=5e-4), larger model (base channels=32).
- Results: baseline mIoU 4.13% / pixel accuracy 69.8%; more-epochs mIoU 4.37% / pixel accuracy 70.6% (best); larger-model mIoU 4.08% / pixel accuracy 69.9%.
- Reading: pixel accuracy looks reasonable (about 70%) but mIoU is very low (about 4%), because most pixels are background in a 21-class problem trained on only 600 images for 5 to 10 epochs. A bigger model did not help under this data and compute budget; more training did.

### Exercise 2: Oxford-IIIT Pet segmentation

- Oxford-IIIT Pet trimaps (foreground, background, boundary, remapped to 3 classes), resized to 128x128, 1,200 training and 300 validation images (a subset of about 3,680 total).
- The same three-experiment design as Exercise 1.
- Results: baseline mIoU 61.28% / pixel accuracy 83.7%; more-epochs mIoU 64.63% / pixel accuracy 85.7% (best); larger-model mIoU 58.8% / pixel accuracy 82.75%.
- Reading: with only 3 well-separated classes instead of 21, the same architecture and a comparable amount of data reach about 65% mIoU versus about 4% for SBD, underlining how much class count and imbalance, not just pixel accuracy, drive segmentation difficulty.

### Exercise 3: CNN versus Vision Transformer on CIFAR-10

- CIFAR-10, 5,000 training and 1,000 validation images (a subset of 50,000/10,000).
- `SmallCNN`: 3 convolutional blocks (32, 64, 128 channels) with BatchNorm and max-pooling, then a dropout fully-connected head.
- `TinyViT`: patch size 4 (64 patches for a 32x32 image), embedding dimension 64, 4 attention heads, 4 transformer encoder layers, a CLS token, pre-normalization.
- Three experiments per architecture: baseline, more epochs, and a third variant (high dropout for the CNN, higher learning rate for the ViT).
- Results: CNN baseline 57.7%, CNN more-epochs 66.1% (best overall), CNN high-dropout 52.5%; ViT baseline 40.1%, ViT more-epochs 47.1% (best ViT), ViT higher-lr 40.7%.
- Reading: `SmallCNN` beats `TinyViT` by about 19 points at its best, while also training roughly 2 to 4 times faster per experiment. The report attributes this to a CNN's built-in inductive biases (locality, weight sharing) being far more data-efficient than a transformer's global self-attention, which typically only pulls ahead of CNNs after large-scale pretraining well beyond this project's 5,000-image subset.

## Cross-project conclusions

From Project #2's final summary, which reads across all three deep-learning exercises:

- The "more epochs, lower learning rate" variant was consistently the best of the three in every exercise, while a larger model or a more aggressive hyperparameter did not help under the same small-data, CPU-limited budget, suggesting these models were still undertrained rather than needing more capacity.
- Pixel accuracy alone is a misleading segmentation metric under class imbalance: SBD's roughly 70% pixel accuracy corresponds to only about 4% mIoU (21 imbalanced classes, a small training set), while Pet's roughly 84 to 86% pixel accuracy corresponds to a much more meaningful 61 to 65% mIoU (3 balanced classes).
- With a small dataset and no pretraining, a small CNN's inductive bias clearly outperforms a small Vision Transformer; the report notes that transformer-style architectures need much larger-scale pretraining data to close that gap.
- All three exercises were deliberately run on small subsets and short schedules for CPU feasibility; the report's suggested next step is GPU training on the full datasets at higher resolution.

## How to run

### Project #1

Open `cv_project1.ipynb` in Jupyter. Requires `numpy`, `matplotlib`, and `scikit-image` (SIFT and feature matching come from `skimage.feature`). It expects the seven source images (`lenna.bmp`, `boats.bmp`, `butterfly.jpg`, `coins.jpg`, `crowd.bmp`, `girlface.bmp`, `lighthouse.bmp`) in an `images-project-1/` folder alongside the notebook.

### Project #2

Each exercise is a standalone script, run from inside `project-2/CV-project2/`:

```bash
python ex1_sbd.py   # downloads SBD (about 1.5 GB) automatically via torchvision on first run
python ex2_pet.py   # expects oxford-iiit-pet/images/images and oxford-iiit-pet/annotations/annotations already extracted
python ex3_cifar.py # downloads CIFAR-10 automatically via torchvision
```

Each script needs `torch`, `torchvision`, `matplotlib`, `numpy`, and `Pillow`, runs on CPU or CUDA automatically, and writes its curves, sample-prediction plots, comparison chart, and `results.json` under `results/exN/`.

## Repository notes

Both project folders have sibling "- Copy" directories and earlier draft scripts or notebooks alongside the versions described above (for example `patch_notebook*.py`, `cv_project1_save.py`, `ex1_sbd.ipynb`, `CV_Project2_ReportV1.docx`). This README describes the two folders specifically in scope, `CV-Project#1` and `CV-project2`, which hold the final notebook or scripts and the report referenced throughout.
