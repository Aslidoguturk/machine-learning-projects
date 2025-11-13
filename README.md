# Machine Learning Projects – Regression & Classification

This repository contains two supervised machine learning projects completed in Python using Google Colab:

1. 🏡 **Housing Price Prediction (Regression)**
2. 🚢 **Titanic Survival Prediction (Classification)**

These projects show end-to-end workflows: data cleaning, feature engineering, model training, hyperparameter tuning, and evaluation with visualisations.

---

## 🏡 Project 1 – Housing Price Prediction (Regression)

**Goal:**  
Predict the median house value in California using numerical, categorical, and geospatial features.

**Dataset:**  
California Housing dataset (course-provided CSV).

**Main Steps:**
- Handle missing values for numerical and categorical features.
- Drop non-informative columns (e.g. ID column).
- Create an interaction feature: `income_rooms_interaction = median_income * total_rooms`.
- One-Hot Encode the categorical feature `ocean_proximity`.
- Standardise numerical features and generate polynomial features (degree = 2).
- Split data into training (first 810 rows) and test set (last 190 rows).

**Models compared:**
- 🌲 **Random Forest Regressor** (main model, tuned with `RandomizedSearchCV`)
- 🌳 **XGBoost Regressor** (baseline)
- 📈 **Linear Regression** (baseline)

**Techniques used:**
- `ColumnTransformer` + `Pipeline` for preprocessing  
- `RandomizedSearchCV` for hyperparameter tuning  
- Evaluation with:
  - Mean Squared Error (MSE)
  - R² and Adjusted R²
- Plots of **Actual vs Predicted** values for each model.

**Key Result:**  
Random Forest achieved the best performance (highest R², lowest MSE), outperforming both Linear Regression and XGBoost on the test set.

Notebook: [`regressionnnn.ipynb`](./regressionnnn.ipynb)

---

## 🚢 Project 2 – Titanic Survival Prediction (Classification)

**Goal:**  
Predict whether a Titanic passenger survived (0/1) based on demographic and ticket information.

**Dataset:**  
Titanic passenger dataset (course-provided CSV).

**Main Steps:**
- Inspect and handle missing values (`Age`, `Fare`, `Embarked`, etc.).
- Impute numerical features with median values.
- Fill categorical missing values with the mode.
- Drop low-information columns such as `Name` and `Ticket No.`.
- Encode categorical features:
  - `Sex` encoded as 0/1
  - `Embarked` label-encoded
- Standardise numerical features for models sensitive to scale.
- Use last 140 rows as test set, the rest as training.

**Models compared (all tuned):**
- ⚙️ **SVM (RBF kernel)** – main model
- 🌲 **Random Forest Classifier**
- 👥 **K-Nearest Neighbours (KNN)**

**Techniques used:**
- `GridSearchCV` for hyperparameter tuning of all three models
- Evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Feature importance plot from Random Forest
- ROC–AUC curves
- Confusion matrices for each model.

**Key Result:**  
Random Forest achieved the best overall performance (accuracy and F1-score), while SVM provided competitive results with a tuned RBF kernel, and KNN performed reasonably but was more sensitive to feature scaling and noise.

Notebook: [`classification.ipynb`](./classification.ipynb)

---

## 🧠 Skills & Libraries Demonstrated

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**: preprocessing, pipelines, models, metrics, GridSearchCV, RandomizedSearchCV, cross-validation
- **XGBoost**
- Model evaluation and comparison
- Data visualisation with **Matplotlib** and **Seaborn**

---

## 🔧 How to Use These Notebooks

These projects were developed in **Google Colab**, but they can also be run locally.

1. Clone the repository:
   ```bash
   git clone https://github.com/Aslidoguturk/machine-learning-projects.git
   cd machine-learning-projects
