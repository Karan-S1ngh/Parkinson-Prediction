import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('features_dataset_40mfcc.csv')

# Data Preparation
df.dropna(subset=['label'], inplace=True)

X = df.drop(columns=['filename', 'label', 'original_path'])
y_labels = df['label']

X.dropna(axis='columns', inplace=True)
print(f"Using {X.shape[1]} complete features to train the model.")

le = LabelEncoder()
y = le.fit_transform(y_labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training CatBoost model to get feature importances
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_scaled, y)

# Feature Importances
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

total_importance = importance_df['importance'].sum()
importance_df['importance_percentage'] = (importance_df['importance'] / total_importance) * 100

print(importance_df[['feature', 'importance_percentage']].to_string())
importance_df.sort_values(by='importance', ascending=False, inplace=True)

print("\nTop 10 Features by Importance:")
print(importance_df.head(10).to_string())

plt.figure(figsize=(12, 8))
sns.barplot(x='importance_percentage', y='feature', data=importance_df.head(20), palette='viridis_r')
plt.title('Top 20 Feature Importances for Parkinson\'s Detection', fontsize=16)
plt.xlabel('Importance (%)', fontsize=12) 
plt.ylabel('Audio Features', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_plot_percentage.png', dpi=300)
plt.show()
