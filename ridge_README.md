# Ridge Regression on Housing Data

Predicting median house values using **Ridge Regression** with automated hyperparameter tuning via **GridSearchCV** and **RepeatedKFold** cross-validation.

---

## Overview

Ridge Regression adds L2 regularization to ordinary least squares, penalizing large coefficients to reduce overfitting. This notebook:

1. Fits a baseline Ridge model (`alpha=1`)
2. Tunes `alpha` over a grid using cross-validated MAE
3. Evaluates the best model with RMSE and R²
4. Inspects feature coefficients

---

## Project Structure

```
ridge-regression-housing/
│
├── ridge_regression.ipynb   # Main notebook
├── housing.csv              # Boston Housing dataset
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Dataset

The `housing.csv` file contains the classic Boston Housing dataset:

| Column  | Description                                      |
|---------|--------------------------------------------------|
| CRIM    | Per capita crime rate by town                    |
| ZN      | Proportion of residential land zoned             |
| INDUS   | Proportion of non-retail business acres          |
| CHAS    | Charles River dummy variable                     |
| NOX     | Nitric oxide concentration                       |
| RM      | Average number of rooms per dwelling             |
| AGE     | Proportion of owner-occupied units built pre-1940|
| DIS     | Weighted distances to employment centres         |
| RAD     | Index of accessibility to radial highways        |
| TAX     | Full-value property tax rate                     |
| PTRATIO | Pupil-teacher ratio by town                      |
| B       | Proportion of Black residents                    |
| LSTAT   | % lower-status population                        |
| **MEDV**| **Median value of homes (target)**               |

---

## Workflow

```
Load Data → Train/Test Split → Baseline Ridge (alpha=1)
    → GridSearchCV (alpha 0.0–0.9) → Tuned Model → Evaluation
```

---

## Results

| Model              | RMSE  | R²   |
|--------------------|-------|------|
| Ridge (alpha=1)    | ~4.72 | ~0.73|
| Ridge (best alpha) | ~4.68 | ~0.74|

---

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/ridge-regression-housing.git
cd ridge-regression-housing
pip install -r requirements.txt
jupyter notebook ridge_regression.ipynb
```

---

## Requirements

```
numpy
pandas
scikit-learn
jupyter
```

---

## Key Fixes from Original Code

| Bug | Fix |
|-----|-----|
| `scoringm'neg_mean_absolute_error'` (missing `=`) | `scoring='neg_mean_absolute_error'` |
| `cv scv` (missing `=`, wrong variable name) | `cv=cv` |
| `n_jobs =- 1` (space breaks negation) | `n_jobs=-1` |
| `'MAE: .3f'` (missing `%`) | `'MAE: %.3f'` |
| `'Config: s'` (missing `%`) | `'Config: %s'` |
| `[Ridge_model.coef_](https://...)` (markdown link syntax in code) | `pd.Series(ridge_model.coef_, ...)` |
| `Ridge(alpha=0.7,)` hardcoded after tuning | Uses `results.best_params_['alpha']` dynamically |
| Inconsistent variable casing (`Ridge_model`) | Standardized to `ridge_model` (PEP 8) |

---

## License

MIT License
