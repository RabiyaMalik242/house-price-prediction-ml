# 🏠 House Price Prediction

A machine learning project that predicts house prices based on property features such as size, number of bedrooms, location preference, and amenities. Built as part of an AI/ML Internship.

---

## 📌 Objective

Build and evaluate regression models to predict house prices using the House Price Prediction Dataset, and identify the most important property features driving price.

---

## 📂 Dataset

- **Source:** [Housing Prices Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- **File:** `Housing.csv`
- **Target:** `price` — Sale price of the house

**Key Features:**

| Feature | Description |
|---|---|
| `area` | Total area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of floors |
| `airconditioning` | Has air conditioning (yes/no) |
| `prefarea` | Located in preferred area (yes/no) |
| `furnishingstatus` | Furnished / semi-furnished / unfurnished |

---

## 🛠️ Tools & Libraries

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## ⚙️ Project Pipeline

1. **Data Loading & Inspection** — shape, dtypes, missing values
2. **Exploratory Data Analysis (EDA)**
   - Price distribution (raw + log + box plot)
   - Scatter plots with trend lines
   - Categorical feature box plots
   - Correlation heatmap
   - Outlier detection
3. **Preprocessing**
   - Categorical encoding (yes/no → 1/0, furnishing → ordinal) with before vs after comparison
   - Train/test split (80/20)
   - StandardScaler with before vs after comparison
4. **Model Training**
   - Linear Regression
   - Gradient Boosting Regressor
5. **Evaluation**
   - MAE, RMSE, R²
   - Actual vs Predicted scatter plots
   - Residual plots and residual distribution
6. **Feature Importance Analysis**
   - Linear Regression coefficients
   - Gradient Boosting feature importance
7. **Predictions**
   - Test set predictions with error % analysis
   - Sorted actual vs predicted line chart
   - Single house price estimate with visual
   - Batch predictions for multiple houses
8. **Model Saving** — joblib (.pkl)

---

## 📊 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 979,680  | 1,331,071 | 0.6495 |
| Gradient Boosting | 1,007,535 | 1,341,598 | 0.6439 |

> Best Model by R²: Linear Regression.

---

## 🔑 Key Findings

- `area` is the single strongest predictor of house price
- Properties with air conditioning and in preferred areas command a significant price premium
- Gradient Boosting captures non-linear feature interactions better than Linear Regression
- Furnishing status and number of bathrooms are also notable price drivers

---

## 💾 Models

Run the notebook to train and save the models locally. The following files will be generated:

```
house_price_lr_model_BEST.pkl  ← recommended for deployment
house_price_gb_model.pkl
house_price_scaler.pkl
```

> ⚠️ The scaler must be loaded alongside Linear Regression — new inputs must be scaled before prediction. Gradient Boosting does not require scaling.

---

## 📁 Repository Structure

```
house-price-prediction-ml/
│
├── house_price_prediction.ipynb   # Main notebook
├── Housing.csv                    # Dataset
├── images/                        # Saved plot images
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository
```bash
   git clone https://github.com/RabiyaMalik242/house-price-prediction-ml.git
```
2. Install dependencies
```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
```
3. Download `Housing.csv` from [Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) and place it in the project folder
4. Open and run `house_price_prediction.ipynb` top to bottom

---

## 📝 Note on Metrics

This is a **regression task** — classification metrics like accuracy, precision, and recall do not apply. The correct evaluation metrics for regression are:

| Metric | What it measures |
|---|---|
| **MAE** | Average absolute difference between predicted and actual price |
| **RMSE** | Like MAE but penalizes large errors more heavily |
| **R²** | % of price variation explained by the model (closer to 1.0 = better) |

---

## 👩‍💻 Author

Rabiya Malik BS Software Engineering — AI/ML Internship Project
