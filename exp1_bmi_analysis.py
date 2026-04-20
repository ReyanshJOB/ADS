import pandas as pd
import matplotlib.pyplot as plt

# ----- Load Data -----
df = pd.read_csv('exp1_bmi_1.csv')
num_cols = ['Height', 'Weight', 'bmi', 'Age']

print(df)
print("\n", df.describe())

# ----- Descriptive Statistics -----
print("\nMean:\n", df[num_cols].mean())
print("\nMedian:\n", df[num_cols].median())
print("\nMode:\n", df[num_cols].mode().iloc[0])

print("\nVariance:\n", df[num_cols].var())
print("\nStd Dev:\n", df[num_cols].std())
print("\nRange:\n", df[num_cols].max() - df[num_cols].min())
print("\nIQR:\n", df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25))

print("\nSkewness:\n", df[num_cols].skew())
print("\nKurtosis:\n", df[num_cols].kurt())

print("\nGender-wise Mean:\n", df.groupby('Gender')[num_cols].mean())

# ----- Visualisations -----
df[num_cols].plot(kind='box', figsize=(10, 5), title='Boxplots')
plt.tight_layout()
plt.savefig('exp1_boxplots.png', dpi=150)
plt.show()

df[num_cols].hist(figsize=(10, 6), bins=5)
plt.suptitle('Histograms')
plt.tight_layout()
plt.savefig('exp1_histograms.png', dpi=150)
plt.show()
