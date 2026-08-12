# 🌿 KnowLeaf – Plant Identifier

**KnowLeaf** is an end‑to‑end deep learning system that identifies plants from leaf/flower images, classifies them as **Edible (Safe)** or **Poisonous**, and retrieves detailed botanical profiles – including medicinal uses, toxicity details, soil preferences, and first‑aid precautions.

> 🚀 **Live Demo**: The application is built with Streamlit for a clean, interactive user interface.

---

## 📌 Key Features

- 🌱 **Dual‑Model Ensemble** – Combines **ResNet50** and **DenseNet121** for robust, accurate classification.
- 🧪 **Test‑Time Augmentation (TTA)** – Applies horizontal flip + 90° rotation during inference to improve generalization.
- 🎯 **Safety‑Aware Thresholding** – Uses an optimal decision threshold (τ = 0.58) to balance sensitivity and specificity.
- 🖼️ **Visual Similarity Matching** – Employs HSV histogram comparison to match the uploaded plant to its closest species in the dataset.
- 📖 **Rich Plant Profiles** – Retrieves information from structured JSON databases (`Poisonous_profiles.json` for poisonous, `Non_poisonous_profiles.json` for edible).
- ⚡ **Real‑Time Inference** – Lightweight enough to run on standard CPUs with fast inference times.

---

## 📊 Dataset

The dataset contains **4,558 curated, background‑removed images** across **239 species** (120 Non‑Poisonous, 119 Poisonous).

| Split | Non‑Poisonous | Poisonous | Total |
|-------|---------------|-----------|-------|
| **Train** | 1,823 | 1,823 | **3,646** |
| **Validation** | 456 | 456 | **912** |
| **Total** | **2,279** | **2,279** | **4,558** |

> 📥 **Download the processed dataset**: [KnowLeaf – Edible and Poisonous Plant Images](https://www.kaggle.com/datasets/YOUR_KAGGLE_USERNAME/knowleaf-processed-plant-dataset) 

---

## 🧠 Model Architecture & Training

We use an **ensemble** of two pre‑trained CNNs with transfer learning from ImageNet:

| Model | Parameters | Inference Time (ms) |
|-------|------------|---------------------|
| **ResNet50** | 25.6M | 45 |
| **DenseNet121** | 8.1M | 52 |
| **Ensemble** | 33.7M | 97 |

### Training Strategy
1. **Phase 1 (Head Training)** – Base layers frozen, train only the classification head.
2. **Phase 2 (Fine‑Tuning)** – Unfreeze base layers, freeze BatchNormalization layers for stability.
3. **Ensemble Averaging** – Weighted average of model outputs (α = 0.5).
4. **Threshold Optimization** – Sweep τ from 0.35 to 0.65; selected **τ = 0.58** for best balance.

---

## 📈 Results

### Performance Metrics (Validation Set)

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1‑Score |
|-------|--------------|-----------------|-----------------|----------|
| ResNet50 (TTA) | 90.13 | 92.32 | 87.94 | 0.903 |
| DenseNet121 (TTA) | 89.36 | 91.45 | 87.28 | 0.896 |
| **Ensemble (TTA – Argmax)** | 90.35 | 93.20 | 87.50 | 0.906 |
| **Ensemble (TTA – τ=0.58)** | **90.79** | 91.01 | **90.57** | **0.908** |

> 🛡️ **Safety‑Critical Insight**:  
> The thresholded ensemble (τ = 0.58) reduces **False Positives** (safe plants flagged as poisonous) from 57 → 43, while only slightly increasing **False Negatives** (missed poisonous plants) from 31 → 41. This yields the best **overall safety balance** for real‑world deployment.

### Confusion Matrix – Best Model (Ensemble + τ=0.58)

| | Predicted: Safe | Predicted: Poisonous |
|---|---|---|
| **Actual: Safe** | **413** (True Negatives) | **43** (False Positives) |
| **Actual: Poisonous** | **41** (False Negatives) | **415** (True Positives) |

- **Sensitivity** (catching poisonous plants): **91.01%**
- **Specificity** (correctly identifying safe plants): **90.57%**

---

## 📁 Project Structure

```
KnowLeaf-Plant-Identifier/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── profiles/
│   │   ├── Poisonous_profiles.json       # Poisonous plant info
│   │   └── Non_poisonous_profiles.json   # Non‑poisonous plant info
│   └── species_lists/
│       ├── poisonous_species.txt        # 119 species names
│       └── non_poisonous_species.txt    # 120 species names
│
├── models/
│   └── class_indices.json               # {"Non_Poisonous": 0, "Poisonous": 1}
│   # (Trained .keras models → Google Drive – see download section below)
│
├── scripts/
│   ├── data_acquisition/
│   │   └── plant_image_scraper.py
│   ├── data_preprocessing/
│   │   ├── check_duplicates.py
│   │   ├── remove_bg.py
│   │   └── split_final.py
│   ├── model_training/
│   │   └── train_model_ensemble.py
│   └── data_evaluation/                 # Optional (evaluation scripts)
│       ├── evaluation.py
│       ├── final_results_comprehensive.json
│       └── optimal_parameters_final.json
│
├── UI/
│   └── app_code.py                      # Streamlit web application
│
└── figures/                             # Evaluation visualisations
    ├── performance_metrics_table.png
    ├── accuracy_inference_tradeoff.png
    ├── precision_recall_curve.png
    ├── cm_resnet.png
    ├── cm_densenet.png
    ├── cm_ensemble.png
    ├── cm_ensemble_tau.png
    ├── dataset_distribution.png
    └── error_reduction_ensemble.png
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```
git clone https://github.com/shreya170411/KnowLeaf-Plant-Identifier.git
cd KnowLeaf-Plant-Identifier
```
### 2. Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Download the trained models
The trained models (resnet50_best.keras and densenet121_best.keras) are too large for GitHub. Download them from Google Drive:
- [ResNet50 Model](https://drive.google.com/file/d/1nnpBEs08l0xV31RjbPNaR-HBs04ctnvB/view?usp=drive_link)
- [DenseNet121 Model](https://drive.google.com/file/d/1Llmn9f1IwK8kXcLORe-kkthRb4efHoO-/view?usp=drive_link)

Place both files in the models/ folder.

### 5. Run the Streamlit app
```
streamlit run UI/app_code.py
```
---
## 🧪 Running the Full Pipeline (Reproduction)

If you want to reproduce the dataset, training, and evaluation from scratch:

- **Scrape images** – `scripts/data_acquisition/plant_image_scraper.py`
- **Preprocess** – Run `check_duplicates.py` → `remove_bg.py` → `split_final.py` (in that order)
- **Train the ensemble** – `scripts/model_training/train_model_ensemble.py`
- **Evaluate the models** – `scripts/data_evaluation/evaluation.py` (generates confusion matrices, PR curves, and performance tables)
- **Launch the UI** – `streamlit run UI/app_code.py` (starts the web application for plant identification)

> 📦 **Note**: The Kaggle dataset already contains the final, cleaned `final_train/` and `final_val/` folders, so you can skip the scraping and preprocessing steps and proceed directly to training if desired.

---
## 🙏 Acknowledgements
- TensorFlow / Keras for deep learning framework
- Streamlit for the web interface
- DuckDuckGo for image scraping
- The open‑source community for all the libraries that made this possible
---
## ⚠️ Disclaimer
- The model achieves strong performance on the evaluated dataset, but it is not infallible. Classification accuracy may vary when applied to images captured in real‑world conditions (different lighting, angles, backgrounds, or plant growth stages) or to plant species not present in the training set. Misclassifications can and will occur.
- This tool is intended for educational and informational purposes only. It is not a substitute for professional botanical expertise. Never consume, handle, or use any plant for medicinal purposes based solely on the output of this system. Always verify identifications with a qualified expert before making any decisions that could affect health or safety.
  
🌿 Always verify plant identifications with a botanical expert before consumption or medicinal use.
