import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ----- Load Data -----
df = pd.read_csv('exp2_CountryAgeSalary.csv')
print("Original Dataset:\n", df)

# ----- Inspect -----
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# ----- Impute Missing Values (Mean Strategy) -----
num_imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = num_imputer.fit_transform(df[['Age', 'Salary']])
print("\nAfter Imputing Missing Values:\n", df)

# ----- Label Encoding -----
le = LabelEncoder()
df['Country_encoded'] = le.fit_transform(df['Country'])
df['Purchased_encoded'] = le.fit_transform(df['Purchased'])
print("\nAfter Label Encoding:\n", df)

# ----- One-Hot Encoding -----
df_ohe = pd.get_dummies(df[['Country', 'Purchased']], dtype=int)
print("\nOne-Hot Encoding:\n", df_ohe)

# ----- Outlier Detection (IQR) -----
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Salary'] < Q1 - 1.5 * IQR) | (df['Salary'] > Q3 + 1.5 * IQR)]
print("\nOutliers in Salary:\n", outliers)

# ----- Round & Display Cleaned Dataset -----
df['Age'] = df['Age'].round(2)
df['Salary'] = df['Salary'].round(2)
print("\nCleaned Dataset:\n", df)
