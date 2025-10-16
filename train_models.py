import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')

FEATURES_FILE = 'features_dataset_40mfcc.csv'
MODEL_DIR = 'model_assets'

# Load Data
try:
    df = pd.read_csv(FEATURES_FILE)
    print(f"--- Full dataset loaded with {df.shape[0]} samples. ---")
except FileNotFoundError:
    print(f"ERROR: '{FEATURES_FILE}' not found. Please run the feature extraction script first.")
    exit()

# Data Cleaning and Preparation
#  Remove duplicates to prevent data leakage
initial_count = len(df)
df.drop_duplicates(subset=['filename'], keep='first', inplace=True, ignore_index=True)
if initial_count > len(df):
    print(f"--- Dropped {initial_count - len(df)} duplicate file entries. Working with {len(df)} unique samples. ---")

# Define features (X) and labels (y)
X_raw = df.drop(columns=['filename', 'label', 'original_path']).copy()
y_labels = df['label']
feature_names = X_raw.columns.tolist()

# Ensure all feature columns are numeric before processing
for col in X_raw.columns:
    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_raw)
X = pd.DataFrame(X_imputed, columns=feature_names) # X is now a clean DataFrame
print(f"--- Missing values imputed. Using {X.shape[1]} features for training. ---")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_labels)
print(f"--- Classes found and encoded: {list(le.classes_)} ---")

# Split Data for Model Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n--- Data split into {len(X_train)} training and {len(X_test)} testing samples for evaluation. ---")

# Feature Scaling for evaluation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define, Train, and Evaluate Models
# Define base classifiers for the Voting Classifier
clf1 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = SVC(kernel='rbf', probability=True, random_state=42)

models = {
    "Random Forest": clf2,
    "SVM": clf3,
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": clf1,
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "MLP Neural Network": MLPClassifier(random_state=42, max_iter=1000),
    "Voting Classifier (XGB+RF+SVM)": VotingClassifier(
        estimators=[('xgb', clf1), ('rf', clf2), ('svm', clf3)], voting='soft'
    )
}

best_model_name = ""
best_model_instance = None
best_score = 0.0

for name, model in models.items():
    print("\n" + "="*50)
    print(f"  Training and Evaluating: {name}")
    print("="*50)
    
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    score = f1_score(y_test, predictions, average='weighted')
    
    if score > best_score:
        best_score = score
        best_model_name = name
        best_model_instance = model

    print(f"\n{name} Performance (Weighted F1-Score: {score:.4f})")
    print(classification_report(y_test, predictions, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Final Model Training and Saving for Deployment
print("\n" + "="*50)
print("  Finalizing Best Model and Assets for Deployment")
print("="*50)

print(f"Best performing model is: '{best_model_name}' with a weighted F1-score of {best_score:.4f}")

# Re-initialize the winning model to ensure it's a fresh instance for final training
final_model = best_model_instance

# CORRECTED FINAL FITTING: Fit preprocessing tools on the ENTIRE dataset
print("\nRe-fitting imputer on all raw data...")
final_imputer = SimpleImputer(strategy='median').fit(X_raw)

print("Re-fitting scaler on all imputed data...")
X_imputed_full = final_imputer.transform(X_raw)
final_scaler = StandardScaler().fit(X_imputed_full)

# Transform the full dataset with the newly fitted tools
X_scaled_full = final_scaler.transform(X_imputed_full)

print(f"Training final '{best_model_name}' model on the entire dataset ({len(X_scaled_full)} samples)...")
final_model.fit(X_scaled_full, y)
print("Final model training complete.")

# Create directory to save assets
os.makedirs(MODEL_DIR, exist_ok=True)

# Save all necessary assets
joblib.dump(final_model, os.path.join(MODEL_DIR, 'best_model.joblib'))
joblib.dump(final_scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
joblib.dump(final_imputer, os.path.join(MODEL_DIR, 'imputer.joblib'))
joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.joblib'))

print(f"\nSuccess! Best model ('{best_model_name}') and all assets saved to '{MODEL_DIR}'. Ready for deployment.")
