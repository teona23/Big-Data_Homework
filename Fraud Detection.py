# Fraud Detection with Logistic Regression and Cross-Validation
# Author: [Your Name]
# Date: [Submission Date]
# Description: Predicts financial fraud using financial statement data

import pandas as pd
import numpy as np
import sklearn

# === Step 1: Load Data ===
input_df = pd.read_csv("A-shared Financial Statement Data.cvs")
output_df = pd.read_csv("A-shared Financial Fraud Data.cvs")

# === Step 2: Clean and Merge ===
input_df = input_df.drop(columns=['Unnamed: 0', 'ShortName', 'Accper', 'DeclareDate'], errors='ignore')
output_df = output_df.drop(columns=['Unnamed: 0'], errors='ignore')
merged_df = pd.merge(input_df, output_df, on=['Stkcd', 'year'], how='inner')

# === Step 3: Prepare Features and Target ===
X = merged_df.drop(columns=['fraud', 'fraudnum', 'violation', 'IfCorrect'], errors='ignore')
X = X.select_dtypes(include='number')  # Keep only numeric features
y = merged_df['fraud']                 # Target: 1 = Fraud, 0 = No Fraud

# === Step 4: Define Model Pipeline ===
pipeline = sklearn.pipeline.Pipeline([
    ('imputer', sklearn.impute.SimpleImputer(strategy='mean')),
    ('scaler', sklearn.preprocessing.StandardScaler()),
    ('classifier', sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=42))
])

# === Step 5: Cross-Validation Evaluation ===
f1 = sklearn.metrics.make_scorer(sklearn.metrics.f1_score)
scores = sklearn.model_selection.cross_val_score(pipeline, X, y, cv=5, scoring=f1)

print("F1 Scores (5-Fold CV):", scores)
print("Mean F1 Score:", np.mean(scores))
print("Standard Deviation:", np.std(scores))

# === Step 6: Train Final Model ===
pipeline.fit(X, y)

# === Step 7: Predict on a New Example ===
new_sample = X.iloc[[0]]  # Use first row for demo
pred = pipeline.predict(new_sample)

print("\nPrediction for sample input (1 = Fraud, 0 = No Fraud):", pred[0])