# Bank Customer Churn Prediction - ANN

## Overview
This project uses an Artificial Neural Network (ANN) built with TensorFlow/Keras
to predict whether a bank customer will churn (leave the bank), based on their
demographic and account information. This is a **binary classification** problem.

---

## Dataset
- **Source:** `Bank Customer Churn Prediction.csv`
- **Rows:** 10,000
- **Original Columns:** 12
- **Dropped Column:** `customer_id` (not predictive)
- **Final Input Features:** 10
- **Target Variable:** `churn`
  - `0` = Customer stays
  - `1` = Customer churns

### Features Used

| Feature | Description |
|---|---|
| credit_score | Customer's credit score |
| country | Customer's country (France, Spain, Germany) |
| gender | Male / Female |
| age | Customer's age |
| tenure | Number of years with the bank |
| balance | Account balance |
| products_number | Number of bank products used |
| credit_card | Whether the customer has a credit card (0/1) |
| active_member | Whether the customer is an active member (0/1) |
| estimated_salary | Estimated annual salary |

---

## Data Preprocessing
1. **Inspection:** Checked for missing values (`isnull().sum()` → none found)
   and duplicate rows (`duplicated().sum()` → none found).
2. **Dropped** `customer_id` since it carries no predictive information.
3. **Defined X and y:**
   - `X` = all columns except `churn`
   - `y` = `churn`
4. **Identified column types:**
   - Numerical columns: `credit_score`, `age`, `tenure`, `balance`,
     `products_number`, `credit_card`, `active_member`, `estimated_salary`
   - Categorical columns: `country`, `gender`
5. **Built preprocessing pipelines** using `ColumnTransformer`:
   - **Numerical pipeline:** `SimpleImputer(strategy='mean')` →
     `StandardScaler()`
   - **Categorical pipeline:** `SimpleImputer(strategy='most_frequent')` →
     `OneHotEncoder(drop='first')`
6. **Train-Test Split:** 80% train / 20% test (`random_state=42`)
7. Applied `preprocessor.fit_transform()` on training data and
   `preprocessor.transform()` on test data.

---

## Handling Class Imbalance
The target variable is imbalanced — roughly **80% Stay** vs **20% Churn**.

To prevent the model from simply predicting "Stay" for everyone, class
weights were computed using `compute_class_weight(class_weight='balanced')`
and passed into `model.fit()`.
Class Weights: {0: 0.629, 1: 2.433}
---

## Model Architecture (ANN)

Input Layer (11 features after one-hot encoding)

↓

Dense(64, activation='relu')

↓

Dropout(0.3)

↓

Dense(32, activation='relu')

↓

Dense(1, activation='sigmoid')   →  Output probability (0–1)
- **relu** activations in hidden layers allow the network to learn
  non-linear patterns.
- **Dropout(0.3)** randomly disables 30% of neurons during training to
  reduce overfitting.
- **sigmoid** output activation produces a probability between 0 and 1,
  which is thresholded at 0.5 to get the final 0/1 prediction.

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 20 |
| Batch Size | 32 |
| Validation Split | 10% |
| Class Weights | Applied (balanced) |

---

## Evaluation Results
precision    recall  f1-score   support

       0       0.93      0.80      0.86      1607
       1       0.48      0.75      0.58       393

accuracy                           0.79      2000
macro avg       0.70      0.78      0.72      2000

weighted avg       0.84      0.79      0.81      2000
| Metric | Class 0 (Stay) | Class 1 (Churn) |
|---|---|---|
| Precision | 0.93 | 0.48 |
| Recall | 0.80 | 0.75 |
| F1-Score | 0.86 | 0.58 |

**Overall Accuracy: 79%**

---

## Observations
- **Before class weighting**, the model achieved ~86% accuracy but only
  47% recall on the churn class — it was missing more than half of the
  customers who actually churned.
- **After applying `class_weight='balanced'`**, recall on the churn class
  improved to 75%, at the cost of overall accuracy dropping to 79% and
  precision on the churn class falling to 48%.
- This trade-off is generally **acceptable for churn prediction**: missing
  a customer who is about to churn (false negative) is more costly to a
  business than mistakenly flagging a loyal customer (false positive),
  since retention offers can be sent to a slightly broader group at low cost.
- **Possible future improvements:**
  - Try SMOTE (oversampling) as an alternative to class weighting
  - Tune the classification threshold (instead of the default 0.5)
  - Experiment with deeper networks, different layer sizes, or learning rates
  - Use k-fold cross-validation for more reliable performance estimates

---

## Tech Stack
- Python 3.13
- NumPy, Pandas
- Scikit-learn (preprocessing, pipelines, metrics)
- TensorFlow / Keras (ANN model)

---

## How to Run
1. Install dependencies:
```bash
   pip install numpy pandas scikit-learn tensorflow
```
2. Place `Bank Customer Churn Prediction.csv` in the project directory.
3. Open and run the notebook (`Untitled.ipynb`) cell by cell from top to bottom.

---

## Repository Structure
.

├── Bank Customer Churn Prediction.csv

├── Untitled.ipynb

└── README.md
