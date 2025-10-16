import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('features_dataset.csv')

# Data Preparation
X = df.drop(columns=['filename', 'label', 'original_path'])
y_labels = df['label']

# Drop the COLUMNS with missing data.
X.dropna(axis='columns', inplace=True)
print(f"Using {X.shape[1]} complete features to train the model.")

le = LabelEncoder()
y = le.fit_transform(y_labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training Random Forest model to get feature importances
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Feature Importances
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Visualization
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for Parkinson\'s Detection', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Audio Features', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=300)
plt.show()
