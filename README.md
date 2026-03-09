# 🔥 FlameVision — Fire Detection with CNN

A binary image classification model that detects **fire vs. no fire** in images using a custom Convolutional Neural Network built with TensorFlow/Keras.

---

## 📌 Overview

FlameVision trains a lightweight 4-block CNN on the [FlameVision dataset](https://www.kaggle.com/datasets/) to classify images as either `FIRE` or `NOFIRE`. The model is designed to balance accuracy and speed, making it suitable for real-time or edge deployment scenarios.

| Metric | Value |
|--------|-------|
| Task | Binary Image Classification |
| Input | RGB images (128×128) |
| Output | FIRE / NOFIRE |
| Framework | TensorFlow / Keras |
| Optimizer | Adam (lr=0.001) |

---

## 🏗️ Model Architecture

```
Input (128×128×3)
    │
    ├── Conv2D(32) → BatchNorm → MaxPool → Dropout(0.1)
    ├── Conv2D(64) → BatchNorm → MaxPool → Dropout(0.2)
    ├── Conv2D(128) → BatchNorm → MaxPool → Dropout(0.3)
    ├── Conv2D(128) → BatchNorm → GlobalAvgPool
    │
    ├── Dense(64, relu) → Dropout(0.4)
    └── Dense(1, sigmoid)  →  P(FIRE)
```

Key design choices:
- **GlobalAveragePooling** instead of Flatten — reduces parameters and overfitting
- **BatchNormalization** after each convolutional block — stabilizes and accelerates training
- **Progressive Dropout** (0.1 → 0.4) — stronger regularization in deeper layers

---

## 📂 Project Structure

```
flamevision/
├── src/
│   └── train.py          # Model definition, training, and evaluation
├── notebooks/
│   └── FlameVision.ipynb # Original Google Colab notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/flamevision.git
cd flamevision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the **FlameVision** dataset from Kaggle and place it at:

```
dataset/FlameVision/archive/Classification/
    ├── train/
    │   ├── fire/
    │   └── nofire/
    ├── valid/
    │   ├── fire/
    │   └── nofire/
    └── test/
        ├── fire/
        └── nofire/
```

### 4. Train the model

```bash
python src/train.py
```

> **Google Colab users:** uncomment the `load_dataset_from_drive(...)` line in `main()` and set the correct path to your ZIP file on Google Drive.

---

## ⚙️ Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image size | 128×128 | Reduced from 224×224 for speed |
| Batch size | 128 | Larger batch for faster iteration |
| Max epochs | 10 | With EarlyStopping |
| Learning rate | 0.001 | Reduced automatically by ReduceLROnPlateau |

### Data Augmentation (training only)

- Rotation ±15°
- Width/height shift ±10%
- Horizontal flip
- Zoom ±10%

---

## 📊 Callbacks

| Callback | Config |
|----------|--------|
| `EarlyStopping` | monitors `val_accuracy`, patience=3 |
| `ReduceLROnPlateau` | monitors `val_loss`, factor=0.5, patience=2 |
| `ModelCheckpoint` | saves best model by `val_accuracy` |

---

## 🔍 Inference

To predict a single image programmatically:

```python
from src.train import predict_image, build_model
import tensorflow as tf

model = tf.keras.models.load_model("fire_detection_model.keras")
label, confidence = predict_image("path/to/image.jpg", model)
print(f"{label} ({confidence:.2%})")
```

---

## 🧰 Tech Stack

- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/) — metrics
- [Google Colab](https://colab.research.google.com/) — training environment

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
