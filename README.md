# ml-methods-algorithms-for-ebay-products
# 🤖 Machine Learning: Methods & Algorithms

> Comparative study of classical ML algorithms on **eBay Auctions** and **CIFAR-10** datasets

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [About](#-about)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Tasks & Results](#-tasks--results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Key Findings](#-key-findings)
- [References](#-references)
- [Author](#-author)

---

## 🎯 About

This project explores the development and evaluation of Machine Learning models, covering both **regression** and **classification** problems. Using real-world data from **eBay auctions**, we predict auction outcomes and final prices, identifying the most important features that influence results.

Additionally, the project examines **image classification** on the CIFAR-10 dataset, testing the limits of classical algorithms on image data.

> **Course:** Machine Learning: Methods and Algorithms  
> **Program:** MSc in Information Systems & Services — Big Data and Analytics  
> **Institution:** University of Piraeus — Department of Digital Systems  
> **Academic Year:** 2025–2026

---

## 📊 Datasets

### 🛒 eBay Auctions Dataset
- **Full set:** ~296k records, 28 columns *(TrainingSet.csv + TestSet.csv)*
- **Sales subset:** Only auctions that resulted in a sale *(TrainingSubset.csv + TestSubset.csv)*

### 🖼️ CIFAR-10
- **60,000 color images** at 32×32 resolution across 10 classes
- **Split:** 50,000 train / 10,000 test
- Loaded from official `data_batch_1..5` and `test_batch`

---

## 📁 Project Structure

```
.
├── data/
│   ├── eBay/
│   │   ├── TrainingSet.csv
│   │   ├── TestSet.csv
│   │   ├── TrainingSubset.csv
│   │   └── TestSubset.csv
│   └── cifar-10-batches-py/
│       ├── data_batch_1..5
│       └── test_batch
├── notebooks/
│   ├── (a)_regression_price.ipynb
│   ├── (b)_classification_quantitysold.ipynb
│   ├── (c)_classification_highprice.ipynb
│   └── cifar10_classification.ipynb
├── report/
│   └── me25028_texniki_anafora.pdf
└── README.md
```

---

## 🧪 Methodology

A consistent, reproducible approach was used throughout all tasks:

| Choice | Value |
|--------|-------|
| Random state | `42` |
| Train/test split | `test_size=0.2` |
| Stratification | Enabled for classification |
| Pipeline | `sklearn.pipeline.Pipeline` (avoids data leakage) |
| Numerical features | `SimpleImputer(median)` + `StandardScaler` |
| Categorical features | `SimpleImputer(most_frequent)` + `OneHotEncoder(min_frequency=5)` |

> 💡 **Why pipelines?** Preprocessing is fitted **only on the training set** to ensure no information leaks into the test set.

---

## 🚀 Tasks & Results

### 🔵 (α) Regression on `Price`

Two linear baseline models were compared on a continuous target.

| Model | MAE | RMSE | R² |
|-------|------|------|------|
| **Ridge** *(α=1.0)* | **1.220** | **1.448** | **0.999** |
| SGDRegressor | 4.983 | 8.857 | 0.962 |

✅ **Winner: Ridge** — predictions almost perfectly aligned with actual values; residuals randomly distributed around zero.

---

### 🟢 (β) Classification on `QuantitySold` (binary)

Class distribution: **69.89% (0)** vs **30.11% (1)** — handled with `class_weight='balanced'`.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|------|
| **LogisticRegression** | **0.886** | 0.793 | **0.840** | **0.816** |
| LinearSVC | 0.884 | 0.799 | 0.819 | 0.809 |

✅ **Winner: LogisticRegression** — slightly better at recovering true sales, with marginally more false positives as a tradeoff.

---

### 🟣 (γ) Classification on `HighPrice`

Custom target: `HighPrice = 1` if `Price ≥ AvgPrice`, else `0`. Leakage-prone columns removed.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|------|
| **LinearSVC** | **0.993** | **0.979** | 0.999 | **0.989** |
| LogisticRegression | 0.988 | 0.963 | 0.999 | 0.981 |

✅ **Winner: LinearSVC** — fewer false positives. Both models near-perfect — features clearly separate the classes.

---

### 🖼️ CIFAR-10 Image Classification

**Pipeline:** Flatten 32×32×3 → `StandardScaler` → `PCA(n_components=100)` → Classifier

#### Full 10 classes
| Model | Accuracy |
|-------|----------|
| **LinearSVC** | **0.396** |
| KNN (k=7) | 0.381 |

#### Subset of 4 classes *(airplane, automobile, ship, truck)*
| Model | Accuracy |
|-------|----------|
| **LinearSVC** | **0.578** |
| KNN (k=7) | 0.558 |

> 📈 Explained variance from PCA: **89.68%**

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — Pipeline, ColumnTransformer, OneHotEncoder, LogisticRegression, LinearSVC, Ridge, SGDRegressor, PCA, KNeighborsClassifier
- **NumPy** — array operations & image reconstruction
- **pandas** — data manipulation
- **Matplotlib / Seaborn** — visualization

---

## 🏁 Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
# Clone the repository
git clone https://github.com/pantoine31/ml-methods-algorithms-for-ebay-products.git
cd pantoine31

# Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the notebooks

```bash
jupyter notebook
```

Then navigate to the `notebooks/` directory and open the notebook of interest.

---

## 💡 Key Findings

- 📈 On the **eBay dataset**, properly preprocessed **linear models perform very well**:
  - Task (α): **Ridge** clearly outperforms SGD
  - Task (β): Both models close — small edge to Logistic Regression
  - Task (γ): Near-perfect performance from both
- 🖼️ On **CIFAR-10**, **PCA** allows classical classifiers to handle high-dimensional image data, but the full 10-class problem remains **challenging** without deep-learning approaches (CNNs).
- ⚖️ **Class imbalance** matters: using `class_weight='balanced'` materially improves recall on minority classes.
- 🔒 **Pipelines + stratification** are essential for fair, reproducible comparisons.

---

## 📚 References

1. Pedregosa et al., *"Scikit-learn: Machine Learning in Python"*, **JMLR 12**, 2011.
2. **Scikit-learn Documentation** — Pipeline, ColumnTransformer, OneHotEncoder, LogisticRegression, LinearSVC, Ridge, SGDRegressor, PCA, KNeighborsClassifier.
3. A. Krizhevsky, *"Learning Multiple Layers of Features from Tiny Images"*, Technical Report, University of Toronto, 2009.
4. M. Mohri, A. M. Medina. *"Learning Algorithms for Second-Price Auctions with Reserve"*, **JMLR 17** (2016): 1–25.
5. A. Krizhevsky, V. Nair, G. Hinton. **CIFAR-10 and CIFAR-100 Datasets** — [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 👤 Author

**Antonios Papakonstantinou**  
🎓 MSc Student — Big Data & Analytics  
🏛️ University of Piraeus — Department of Digital Systems  
🆔 Student ID: `me25028`  
📧 [antonhspap@icloud.com](mailto:antonhspap@icloud.com)

> **Supervisor:** Prof. Orestis Telelis

---

<div align="center">

⭐ **If you find this project useful, please consider giving it a star!** ⭐

</div>
