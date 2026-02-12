# Logistic Regression from Scratch

This project implements logistic regression from scratch using NumPy.
The goal is to understand binary classification, log-loss optimization,
class imbalance handling, and threshold tuning — and to compare the results
with scikit-learn’s implementation.

## Dataset

The project uses the **Default of Credit Card Clients** dataset (UCI, Taiwan, 2005).

- Each row represents a credit card client with demographic data and repayment history
- The target variable is **default payment next month**
  - 1 — default
  - 0 — no default
- The dataset is imbalanced (~80% non-default, ~20% default)

## Project structure

├── data/
│   └── default of credit card clients.csv
├── notebooks/
│   └── 01_logistic_regression_from_scratch.ipynb
├── src/
│   ├── model.py          # Logistic regression implementation from scratch
│   ├── preprocessing.py  # Train/val/test split and normalization
│   └── plotting.py       # Visualization utilities
├── requirements.txt
└── README.md

## Methodology

The project follows a structured machine learning workflow:

1. Data loading and inspection
2. One-hot encoding for categorical features
3. Train / validation / test split
4. Z-score normalization (computed on training set only)
5. Training a baseline logistic regression model using batch gradient descent
6. Handling class imbalance using balanced class weights
7. Feature engineering (repayment ratios, delays, bill dynamics)
8. Hyperparameter tuning (learning rate and L2 regularization)
9. Threshold tuning based on F1-score
10. Model evaluation using ROC-AUC, Precision–Recall, and Confusion Matrix
11. Comparison with scikit-learn implementation

## Feature engineering

Several engineered features were introduced to better capture repayment behavior:

- Mean and max debt-to-limit ratios
- Average payment ratio
- Average and maximum repayment delay
- Bill growth dynamics
- Maximum bill over six months

All features were computed row-wise to avoid data leakage.

## Class imbalance handling

Because the dataset is imbalanced, balanced class weighting was applied:

- Higher weight assigned to the minority class (default = 1)
- This improves recall and encourages the model to detect defaulters

Threshold tuning was performed using F1-score to balance precision and recall.

## Regularization and tuning

- Learning rate (alpha) selected using ROC-AUC on validation set
- L2 regularization strength (lambda) tuned separately
- Threshold selected based on F1-score

Validation results show:
- Learning rate significantly affects convergence and ROC-AUC
- L2 regularization has minor but measurable influence
- Threshold tuning improves precision–recall balance

## Results

The final scratch implementation achieved:

- ROC-AUC ≈ 0.73 (validation)
- Balanced precision and recall (~0.5–0.53 range after tuning)

Performance is very close to scikit-learn’s LogisticRegression,
confirming correctness of the implementation.

## Visualizations

The notebook includes:
- Training loss curve (log-loss vs iterations)
- ROC curve
- Precision–Recall curve
- Confusion matrix

## Conclusion

The project demonstrates a complete end-to-end binary classification pipeline
implemented from scratch.

Despite being a linear model, logistic regression captures meaningful signal
in the credit dataset when combined with proper preprocessing, feature engineering,
class weighting, and threshold tuning.

While scikit-learn provides faster and more robust optimization (LBFGS),
the scratch implementation offers transparency and strong educational value.

## How to run

pip install -r requirements.txt