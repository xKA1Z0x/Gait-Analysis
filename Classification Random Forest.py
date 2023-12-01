import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, label_binarize


#Load your pd friendly dataset here
file_path = r"C:\your\dataset\directory\here"
data = pd.read_excel(file_path, sheet_name='yourdataset')

# Data preprocessing
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf/-inf with NaN
data.dropna(axis=0) #Removing rows with NaN

# Define the target column and columns you don't want to have as your features
target_column = 'Targert'
features_to_drop = ['Non feature columns', target_column]

# Select features (X) and the target (y)
X = data.drop(columns=features_to_drop)
y = data[target_column]

# Encode the target variable - For classification models
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded) #Adjust hyperparameters for your own desire

# Set up the classification Random Forest model
model = RandomForestClassifier(n_estimators=1500, random_state=42, max_depth=5, 
                            min_samples_split=6, min_samples_leaf=8, max_features='sqrt') #Adjust hyperparameters for your own desire

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on the testing set
y_pred_test = model.predict(X_test)
y_pred_probs_test = model.predict_proba(X_test)[:, 1]

#Report the Train-test accuracy and confusion matrix
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Test Accuracy: {test_accuracy}')


#Stratified KFold Crossvalidation
rf = RandomForestClassifier(n_estimators=500, random_state=42) #Adjust hyperparameters
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #Adjust hyperparameters
groups = data['Subject']
cv_scores = cross_val_score(rf, X, y_encoded, groups=groups, cv=skf, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')




