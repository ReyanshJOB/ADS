# Experiment 8: Feature Selection — Chi-Square, RFE, Logistic Regression
# Install dependencies: pip install pandas scikit-learn numpy

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression

# ----- Load Data -----
df = pd.read_csv('Admission_St.csv')
print("Original Data:")
print(df.head())

# ----- Scale Numeric Features -----
scaler = StandardScaler()
df[['GRE', 'GPA']] = scaler.fit_transform(df[['GRE', 'GPA']])

X = df[['GRE', 'GPA', 'RANK']]
y = df['Admit']

# ----- Feature Selection: Chi-Square -----
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(abs(X), y)
print("\nSelected Features using Chi-Square:")
print(selector.get_support())

# ----- Feature Selection: RFE -----
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
fit = rfe.fit(X, y)
print("\nSelected Features using RFE:")
print(fit.support_)

# ----- Train / Test Split & Model Evaluation -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)
