import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('features_dataset.csv')
    print("--- Successfully loaded features_dataset.csv ---")
except FileNotFoundError:
    print("ERROR: 'features_dataset.csv' not found.")
    

# Statistical Comparison
key_features = [
    'rms_energy', 'mfcc_1', 'mean_pitch', 'jitter_local',
    'shimmer_local', 'spectral_bandwidth', 'hnr'
]
existing_key_features = [f for f in key_features if f in df.columns and df[f].notna().any()]

print("Sample Counts by Diagnosis")
print(df['label'].value_counts())
print("="*30 + "\n")

stats_summary = df.groupby('label')[existing_key_features].agg(['mean', 'std']).round(4)
print("Comparative Statistics for Key Features:")
print(stats_summary)

# Class Distribution
ax = sns.countplot(x='label', data=df, palette='viridis', order=sorted(df['label'].dropna().unique()))
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=12)
ax.set_title('Distribution of All Samples by Diagnosis', fontsize=16)
ax.set_xlabel('Diagnosis', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
plt.show()

# Distribution of Key Features
for feature in existing_key_features:
    if df[feature].notna().sum() > 1:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x=feature, hue='label', fill=True, palette='viridis', common_norm=False)
        plt.title(f'Distribution of {feature} by Diagnosis', fontsize=16)
        plt.xlabel(f'{feature} Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.show()

# Box Plots for Key Features
for feature in existing_key_features:
    if df[feature].notna().sum() > 1:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='label', y=feature, palette='viridis', order=sorted(df['label'].dropna().unique()))
        plt.title(f'Box Plot of {feature} by Diagnosis', fontsize=16)
        plt.xlabel('Diagnosis', fontsize=12)
        plt.ylabel(f'{feature} Value', fontsize=12)
        plt.show()
