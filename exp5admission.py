# Experiment 5: Handling Class Imbalance using SMOTE

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_excel("exp5.xlsx")

print("First 5 Rows:\n", df.head())
print("\nShape:", df.shape)

# Features and Target
X = df[['GRE', 'GPA', 'RANK']]
y = df['Admit']

print("\nBefore SMOTE (Training Data)")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(y_train.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE")
print(pd.Series(y_resampled).value_counts())

# Model Before SMOTE
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
y_pred_before = model1.predict(X_test)

# Model After SMOTE
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_resampled, y_resampled)
y_pred_after = model2.predict(X_test)

# Performance
print("\n--- BEFORE SMOTE ---")
print(classification_report(y_test, y_pred_before))

print("\n--- AFTER SMOTE ---")
print(classification_report(y_test, y_pred_after))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_after)

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Before SMOTE
axes[0].bar(
    ['Class 0', 'Class 1'],
    y_train.value_counts().sort_index()
)
axes[0].set_title("Before SMOTE")

# After SMOTE
axes[1].bar(
    ['Class 0', 'Class 1'],
    pd.Series(y_resampled).value_counts().sort_index()
)
axes[1].set_title("After SMOTE")

# Confusion Matrix
axes[2].imshow(cm, cmap="Blues")
axes[2].set_title("Confusion Matrix")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()