import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Load Data
try:
    df = pd.read_csv('features_dataset.csv')
    print(f" Successfully loaded features_dataset.csv ")
    print(f"Full dataset contains {df.shape[0]} samples.")
except FileNotFoundError:
    print("ERROR: 'features_dataset.csv' not found. Please run the preparation and feature extraction scripts first.")


# Data Cleaning and Preparation
X = df.drop(columns=['filename', 'label', 'original_path'])
y_labels = df['label']

print(f"Original number of features: {X.shape[1]}")
X.dropna(axis='columns', inplace=True)
print(f"Using {X.shape[1]} complete features for training.")

if X.shape[1] == 0:
    print("No complete feature columns were found.")


le = LabelEncoder()
y = le.fit_transform(y_labels)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and Evaluate Models

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    print("\n" + "="*40)
    print(f"  Training {name} Model")
    print("="*40)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test_scaled)

    # Performance Evaluation
    print(f"\n {name} Performance ")
    print(classification_report(y_test, predictions, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
