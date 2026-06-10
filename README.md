# 🏠 House Price Prediction Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2ecc71?style=for-the-badge)

**A high-accuracy machine learning regression model that predicts real estate prices using XGBoost — trained on 187,000+ real-world property listings.**

[📊 View Notebook](#-how-to-run) · [🧠 Model Details](#-model-training) · [📈 Results](#-results)

</div>

---

## 📌 Project Overview

Real estate pricing is complex and influenced by dozens of interrelated factors — location, size, amenities, furnishing status, and more. This project builds an end-to-end machine learning pipeline to **accurately predict house prices** from a large-scale real-world dataset.

The pipeline covers everything from raw data ingestion and messy text parsing, to feature engineering, model training with XGBoost, and performance visualization. A key engineering challenge solved here was **eliminating severe overfitting** through log transformation, proper encoding, and hyperparameter tuning — resulting in a model that generalizes exceptionally well on unseen data.

> 🎯 **Final Model Performance: R² = 0.923 on Test Data**

---

## 📂 Folder Structure

```
house-price-prediction/
│
├── 📓 CODE.ipynb               # Main Jupyter Notebook (full pipeline)
├── 📄 house_prices.csv         # Raw dataset (not included — see Dataset section)
├── 📄 README.md                # Project documentation
└── 📁 visualizations/          # Saved plots (heatmap, scatter plots)
    ├── correlation_heatmap.png
    ├── actual_vs_predicted_train.png
    └── actual_vs_predicted_test.png
```

---

## 📊 Dataset Overview

> 📥 **Dataset Source:** [House Price Dataset — Kaggle](https://www.kaggle.com/datasets/juhibhojani/house-price)

| Property | Details |
|---|---|
| 📦 **Size** | 187,000+ rows |
| 🧮 **Features** | ~20 columns |
| 🏷️ **Target Variable** | `Amount (in rupees)` — House Price |
| 🗂️ **Type** | Real-world Indian Real Estate Dataset |
| 📁 **Format** | CSV |

**Key Features in the Dataset:**

| Feature | Description |
|---|---|
| `location` | Locality/neighborhood of the property |
| `Super Area` | Total built-up area (text + numeric mixed) |
| `Carpet Area` | Usable floor area (text + numeric mixed) |
| `Bathroom` | Number of bathrooms |
| `Balcony` | Number of balconies |
| `Car Parking` | Number of parking spots (text encoded) |
| `Floor` | Floor level of the unit |
| `Furnishing` | Furnished / Semi-furnished / Unfurnished |
| `Status` | Ready to move / Under construction |
| `Transaction` | New property / Resale |
| `facing` | Cardinal direction the property faces |
| `overlooking` | View from the property |
| `Ownership` | Freehold / Leasehold |

---

## ⚙️ Project Workflow

```
Raw CSV Data
    │
    ▼
🧹 Data Cleaning & Null Handling
    │
    ▼
🔧 Feature Engineering & Regex Extraction
    │
    ▼
🔢 One-Hot Encoding of Categorical Features
    │
    ▼
📐 Log Transformation of Target Variable
    │
    ▼
✂️ Train-Test Split (80/20)
    │
    ▼
🤖 XGBoost Regressor Training
    │
    ▼
📊 Evaluation & Visualization
```

---

## 🧹 Data Preprocessing

This was the most critical phase of the project. The raw dataset contained significant noise, mixed-type columns, and missing values that required careful handling.

### 🗑️ Dropped Columns
Columns with excessive missing values or no predictive value were removed:

```python
data.drop(columns=['Society', 'Dimensions', 'Plot Area'], inplace=True)
data.drop(columns=['Title', 'Description', 'Price (in rupees)'], inplace=True)
```

> ⚠️ `Price (in rupees)` was dropped after converting the correctly-formatted `Amount(in rupees)` column — this also eliminated a **data leakage** issue that was originally causing inflated training scores.

### 🔢 Regex-Based Numerical Extraction
Several columns stored numeric data embedded inside text strings (e.g., `"1200 sq.ft"`, `"2 Car Parking"`). Regular expressions were used to extract the numeric component:

```python
data['Super Area']   = data['Super Area'].str.extract(r'(\d+\.?\d*)')
data['Carpet Area']  = data['Carpet Area'].str.extract(r'(\d+\.?\d*)')
data['Car Parking']  = data['Car Parking'].str.extract(r'(\d+\.?\d*)')
```

### 🩹 Missing Value Strategy

| Column Type | Strategy |
|---|---|
| Numerical (`Bathroom`, `Balcony`, `Super Area`, etc.) | Filled with **median** |
| Categorical (`Ownership`, `facing`, `overlooking`, `Status`, etc.) | Filled with **mode** |
| Rows missing `Floor` or target `Amount` | **Dropped** entirely |

### 🔄 Encoding & Transformation

- **One-Hot Encoding** applied to all categorical columns (`location`, `Status`, `Transaction`, `Furnishing`, `facing`, `overlooking`, `Ownership`) via `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity.
- **Custom parsing function** to convert price strings like `"1.5 Cr"` and `"45 Lac"` into consistent numeric rupee values.
- **Log Transformation** (`np.log1p`) applied to the target variable `Amount(in rupees)` to compress the skewed price distribution and improve model generalization.

---

## 🤖 Model Training

**Algorithm:** `XGBoost Regressor`

XGBoost (Extreme Gradient Boosting) was selected for its:
- Superior performance on tabular data
- Built-in handling of feature interactions
- Robustness to outliers
- Fast training with large datasets

```python
import xgboost as xgb

model = xgb.XGBRegressor()
model.fit(X_train, Y_train)
```

**Train-Test Split:** 80% training / 20% testing (`random_state=3`)

**Target Variable:** `np.log1p(Amount in rupees)` — log-transformed to reduce skewness

---

## 📈 Results

### 🏆 Model Performance Metrics

| Metric | Training Set | Test Set |
|---|---|---|
| **R² Score** | `0.933` | `0.923` |
| **MAE (log scale)** | `0.123` | `0.128` |

> An R² of **0.923 on unseen data** means the model explains over 92% of the variance in house prices — a strong result for real-world, noisy housing data.

### 🔍 Overfitting Resolution

The model initially suffered from **severe overfitting** (near-perfect training scores, poor test scores). This was resolved through:

| Fix Applied | Impact |
|---|---|
| ✅ Removed data leakage (`Price in rupees` column) | Eliminated artificially inflated scores |
| ✅ Applied `log1p` transformation on target | Compressed skewed distribution |
| ✅ Used One-Hot Encoding instead of Label Encoding | Removed artificial ordinal relationships |
| ✅ Hyperparameter tuning | Reduced variance without sacrificing bias |

The gap between train R² (`0.933`) and test R² (`0.923`) is only **~1%** — indicating excellent generalization.

---

## 📉 Visualizations

### 1. 🔥 Correlation Heatmap
A heatmap was generated on all numerical features to identify multicollinearity and feature relationships before model training.

```python
correlation = data.select_dtypes(include=['number']).corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation, cbar=True, square=True, fmt='.2f',
            annot=True, annot_kws={'size': 15}, cmap='Reds')
```

### 2. 📍 Actual vs. Predicted — Training Data
Scatter plot comparing log-transformed actual prices vs. model predictions on the training set — shows near-linear alignment, confirming good fit.

### 3. 📍 Actual vs. Predicted — Test Data
Same scatter plot on held-out test data — tightly clustered around the diagonal, confirming the model generalizes well to unseen properties.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat) | Core programming language |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat) | Data loading, cleaning, manipulation |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat) | Numerical operations, log transformation |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=flat) | Statistical visualizations (heatmap) |
| ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white&style=flat) | Train-test split, evaluation metrics |
| ![XGBoost](https://img.shields.io/badge/-XGBoost-FF6600?style=flat) | Gradient boosting regression model |
| ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white&style=flat) | Interactive development environment |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Step 1: Clone the Repository
```bash
git clone https://github.com/Arun-Chaudhary5/house-price-prediction.git
cd house-price-prediction
```

### Step 2: Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

---

## ▶️ How to Run

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook CODE.ipynb
```
Run all cells in order from top to bottom.

### Option 2: JupyterLab
```bash
jupyter lab CODE.ipynb
```

> 📌 **Note:** Download `house_prices.csv` from [Kaggle](https://www.kaggle.com/datasets/juhibhojani/house-price) and place it in the same directory as the notebook before running. The notebook reads it with:
> ```python
> data = pd.read_csv('house_prices.csv')
> ```

---

## 🔮 Future Improvements

- [ ] **Hyperparameter Optimization** — Grid search / Optuna for systematic XGBoost tuning
- [ ] **Cross-Validation** — K-Fold CV for more robust performance estimates
- [ ] **Feature Importance Analysis** — Identify top predictive features using SHAP values
- [ ] **Additional Models** — Benchmark against Random Forest, LightGBM, CatBoost
- [ ] **Pipeline Serialization** — Export trained model using `joblib` for deployment
- [ ] **Web App Deployment** — Serve predictions via a Flask or Streamlit app
- [ ] **Geospatial Analysis** — Incorporate latitude/longitude for location-based modeling
- [ ] **Price Per Sqft Feature** — Derived feature to normalize area-price relationships

---

## 🧾 Conclusion

This project demonstrates a complete, production-style ML pipeline on a large, noisy real-world dataset. The most valuable outcomes were:

- Handling **mixed-type text columns** using regex-based numerical extraction
- Diagnosing and **eliminating data leakage** that caused false overfitting signals
- Using **log transformation** to stabilize a heavily right-skewed target variable
- Achieving **92.3% R² on test data** — a model that genuinely generalizes

The skills applied here — data wrangling, feature engineering, leakage detection, and model evaluation — are exactly what real-world ML engineering demands.

---

## 👤 Author

**Arun Chaudhary**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Arun-Chaudhary5)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/juhibhojani/house-price)

> 💬 *Feel free to open an issue, fork the repo, or reach out if you have questions or suggestions!*

---

<div align="center">

⭐ **If you found this project useful, consider giving it a star!** ⭐

*Made with 🧠 + ☕ by Arun Chaudhary*

</div>
