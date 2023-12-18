import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Load your dataset
file_path = r"C:\Users\sajja\OneDrive - UCB-O365\3rd Manuscript\JoeyMohammed.xlsx"
data = pd.read_excel(file_path, sheet_name='nooutliernew2')

# Data preprocessing
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf/-inf with NaN
data = data.dropna(axis=0)  # Removing rows with NaN

# Define the target column and columns you want to include as features
target_column = 'Condition'
features_to_include = ['toeoutangle', 'toeoffangle', 'lumbarcoronalROM', 'lumbarsagittalROM', 'gaitspeed']

# Select features (X) and the target (y)
X = data[features_to_include]
y = data[target_column]

# Encode the target variable - For classification models
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Set up the SVM model
svm_model = make_pipeline(StandardScaler(), SVC(random_state=42))  # Adjust hyperparameters for your own desire

# Train the model on the training set
svm_model.fit(X_train, y_train)

# Predict on the testing set
y_pred_test_svm = svm_model.predict(X_test)

# Report the Test accuracy
test_accuracy_svm = accuracy_score(y_test, y_pred_test_svm)
print(f'SVM Test Accuracy: {test_accuracy_svm}')

# Stratified KFold Cross-validation with SVM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Adjust hyperparameters
groups = data['Subject']
cv_scores_svm = cross_val_score(svm_model, X, y_encoded, groups=groups, cv=skf, scoring='accuracy')
print(f'SVM Cross-validation scores: {cv_scores_svm}')
print(f'SVM Mean cross-validation accuracy: {cv_scores_svm.mean()}')
