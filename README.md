
# 🧠 Brain Tumor Detection using Deep Learning

An end-to-end **deep learning and computer vision pipeline** for automated **brain tumor classification, segmentation, and detection** from MRI scans.  
This project explores multiple architectures and tasks to assist radiologists by improving diagnostic accuracy and visual interpretability.

---

## 📌 Project Overview

Brain tumor analysis from MRI images is challenging due to:
- Low contrast between tumor and healthy tissue
- Variations in tumor shape, size, and location
- Time-consuming and error-prone manual diagnosis

This project addresses these challenges using **deep learning and transfer learning** across three major tasks:
- **Classification** – Identify tumor presence and type
- **Segmentation** – Precisely outline tumor regions
- **Detection** – Localize tumors using bounding boxes

---

## 🧠 Tasks & Models

### 🔹 Classification
Transfer learning models used:
- VGG16
- VGG19
- ResNet50
- EfficientNet-B0

**Task:** Binary and multi-class classification of brain tumors from MRI images.

---

### 🔹 Segmentation
Models implemented:
- U-Net
- Attention U-Net

**Task:** Pixel-level tumor segmentation to highlight exact tumor boundaries.

---

### 🔹 Detection
Model used:
- YOLO (binary and multi-class)

**Task:** Tumor localization using bounding box detection.

---

## 📊 Results (Best Performance)

| Task | Metric | Score |
|-----|------|------|
| Classification | Accuracy | **96.35%** |
| Segmentation | Dice Coefficient | **0.899** |
| Segmentation | IoU | **0.6866** |
| Detection | mAP | **0.9166** |

> Results may vary depending on dataset split, preprocessing, and hyperparameters.

---

## 🗂️ Repository Structure

```

Brain-Tumor-detection/
├── notebooks/          # Jupyter notebooks (training & experiments)
├── assets/
│   └── plots/         # Training curves & prediction visualizations   
├── README.md
├── requirements.txt
└── .gitignore

```

---

## 📁 Notebooks

The `notebooks/` folder contains:
- Binary and multi-class classification experiments
- U-Net and Attention U-Net segmentation
- YOLO-based tumor detection

Each notebook includes:
- Data preprocessing
- Model architecture
- Training and evaluation
- Visual results

---

## 🖼️ Visual Results

Representative training curves and prediction results are available in:
```

assets/plots/

````

Plots are organized by model and task (classification, segmentation, detection).

---

## 🧠 Datasets

Due to size and licensing constraints, datasets are **not included** in this repository.

Datasets used:
- Figshare Brain Tumor Dataset
- Brain Tumor MRI Dataset (Kaggle)

Please download the datasets separately and update dataset paths in the notebooks.

---

## ⚙️ Setup & Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
````

Main libraries used:

* Python
* TensorFlow / Keras
* PyTorch
* OpenCV
* NumPy, Matplotlib, scikit-learn

---

## ⚠️ Notes

* Notebooks were primarily developed using **Google Colab**
* Colab-specific paths such as `/content/` should be updated when running locally
* Large files (datasets, model weights, videos) are intentionally excluded

---

## 🚀 Future Work

* 3D MRI volume processing (3D U-Net / nnU-Net)
* Multi-modal MRI fusion (T1, T2, FLAIR)
* Lightweight models for real-time clinical deployment
* Uncertainty estimation for reliable medical predictions

---

## 👤 Author

**Kiran Viswanadhapalli**

GitHub: [https://github.com/kiran-viswanadhapalli](https://github.com/kiran-viswanadhapalli)


---

## ⭐ Acknowledgments

This project was developed for academic and research exploration in **medical image analysis and deep learning**.

```
```

