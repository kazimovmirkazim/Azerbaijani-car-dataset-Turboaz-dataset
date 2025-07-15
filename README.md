# Turbo.az Car Price Prediction

## Project Overview

This project is focused on predicting car prices using real-world listings collected from the Turbo.az platform. The goal is to transform unstructured, inconsistent data into a clean and reliable format, and then apply machine learning models to generate accurate price predictions.

The approach combines thorough data preparation with robust model building to ensure practical results.

---

## Technical Approach

### Data Preparation
- Applied regular expressions to extract and clean fields such as engine size, price, and mileage.
- Handled missing values using `SimpleImputer`.
- Used `OneHotEncoder` for categorical variables, and scaled numerical features using `StandardScaler` and `MinMaxScaler`.
- All preprocessing steps were wrapped in a `scikit-learn` pipeline for modularity and reusability.

### Modeling
- A `StackingRegressor` ensemble was built using the following base models:
  - `RidgeCV` – for regularized linear modeling.
  - `RandomForestRegressor` – to capture nonlinear relationships.
  - `SVR` – for high-precision local patterns.
- Final predictions are made using a `LinearRegression` model that combines the base outputs.

### Evaluation
- Model performance is measured using:
  - `mean_squared_error` (MSE)
  - `mean_absolute_error` (MAE)

---

## How to Use

### Environment Setup

Make sure you have Python 3.7+ and the following libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib
