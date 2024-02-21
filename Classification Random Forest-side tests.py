import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix



#Load your pd friendly dataset here
file_path = r"yourdatasethere.xlsx"
data = pd.read_excel(file_path, sheet_name='yourdatasheetname')

# Data preprocessing
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf/-inf with NaN
data = data.dropna(axis=0) #Removing rows with NaN

# Define the target column and columns you do or don't want to have as your features
target_column = 'your target'
features_to_include = ['your features']
#features_to_drop = ['your features to drop']
# Select features (X) and the target (y)
X = data[features_to_include]
y = data[target_column]
#uncomment below in case you want to drop features instead of selecting features
#X = data.drop(columns=features_to_drop)

# Encode the target variable - For classification models
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
print(label_mapping)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded) #Adjust hyperparameters for your own desire

# Set up the classification Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42) #Adjust hyperparameters for your own desire

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on the testing set
y_pred_test = model.predict(X_test)
y_pred_probs_test = model.predict_proba(X_test)[:, 1]

#Report the Train-test accuracy and confusion matrix
test_accuracy = accuracy_score(y_test, y_pred_test)
kappa_score = cohen_kappa_score(y_test, y_pred_test)
print(f'Cohen Kappa Score: {kappa_score}')
print(f'Test Accuracy: {test_accuracy}')
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Create a DataFrame for better visualization and mapping
cm_df = pd.DataFrame(cm, index=label_mapping.keys(), columns=label_mapping.keys())

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

#Stratified KFold Crossvalidation
#tune the hyperparameters
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) #Adjust hyperparameters
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #Adjust hyperparameters
groups = data['Subject']
cv_scores = cross_val_score(rf, X, y_encoded, groups=groups, cv=skf, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')
#Fit the data into the model and show the important features
rf.fit(X, y_encoded)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns[indices]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances[indices]})
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(feature_importances['Feature'], feature_importances['Importance'], color='b', align='center')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()




#conducting SHAP analysis - it usually takes more time
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X, y_encoded)
# Create a SHAP explainer object
explainer = shap.TreeExplainer(rf_full)
# Calculate SHAP values for the training data
shap_values = explainer.shap_values(X)
# Summarize the SHAP values in a plot
shap.summary_plot(shap_values, X, plot_type="bar")

# Assuming shap_values is a list of 2D arrays, where each array corresponds to a class
n_classes = len(shap_values)
feature_names = X.columns
class_names = ['BTENS', 'Baseline', 'CTENS'] # Replace with your actual class names
# Iterate over each class and create a DataFrame
for i in range(n_classes):
    # Calculate mean SHAP values for the current class
    mean_shap_values = np.mean(shap_values[i], axis=0)
    shap_summary = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': mean_shap_values
    })
    shap_summary = shap_summary.sort_values(by='Mean SHAP Value', ascending=False)
    # Print the DataFrame for each class
    print(f"SHAP Summary for {class_names[i]}")
    print(shap_summary)
    print("\n")
#plotting dependence plot - make sure to select the array of values you want to look (depends on the number of classes)
shap.dependence_plot('your feature', shap_values[2], X, interaction_index=None, cmap='YlOrRd')
# Show the plot
plt.show()

#in case you want to save your SHAP values
import pickle
with open('shap_values.pkl', 'wb') as file:
    pickle.dump(shap_values, file)
#in case you want to reload your SHAP values from directory
with open('shap_values.pkl', 'rb') as file:
    shap_values = pickle.load(file)

#Please email me your questions: sada8020@colorado.edu
