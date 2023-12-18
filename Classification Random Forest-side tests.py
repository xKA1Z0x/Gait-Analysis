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
file_path = r"C:\Users\sajja\OneDrive - UCB-O365\3rd Manuscript\JoeyMohammed.xlsx"
data = pd.read_excel(file_path, sheet_name='nooutliernew2')

# Data preprocessing
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf/-inf with NaN
data = data.dropna(axis=0) #Removing rows with NaN

# Define the target column and columns you don't want to have as your features
target_column = 'Condition'
features_to_include = ['toeoutangle','toeoffangle','lumbarcoronalROM','lumbarsagittalROM','gaitspeed']
#features_to_drop = ['Subject', 'Condition']
# Select features (X) and the target (y)
X = data[features_to_include]
y = data[target_column]
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
# Assuming label_mapping is a dictionary mapping encoded labels to original labels
cm_df = pd.DataFrame(cm, index=label_mapping.keys(), columns=label_mapping.keys())

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

#Stratified KFold Crossvalidation
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) #Adjust hyperparameters
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #Adjust hyperparameters
groups = data['Subject']
cv_scores = cross_val_score(rf, X, y_encoded, groups=groups, cv=skf, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')

rf.fit(X, y_encoded)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns[indices]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances[indices]})
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(feature_importances['Feature'], feature_importances['Importance'], color='b', align='center')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()





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

shap.dependence_plot('lumbarsagittalROM', shap_values[2], X, interaction_index=None, cmap='YlOrRd')  # Assuming 'gaitspeed' is the feature name

# Show the plot
plt.show()


class_labels = label_encoder.classes_
class_mapping = {index: label for index, label in enumerate(class_labels)}

print("Class Mapping:", class_mapping)






import pickle
# Save SHAP values
with open('shap_values.pkl', 'wb') as file:
    pickle.dump(shap_values, file)
# To load SHAP values later
with open('shap_values.pkl', 'rb') as file:
    shap_values = pickle.load(file)






condition_specific_data = data[data['Condition'] == 'Baseline']
sns.histplot(condition_specific_data['armROM'], kde=True)
plt.title('Distribution of Lumbar Coronal Range of Motion')
plt.xlabel('lumbarCoronalROM')
plt.ylabel('Frequency')
plt.show()
# Q-Q Plot
stats.probplot(condition_specific_data['toeoutangle'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot for Lumbar Sagittal Range of Motion at Baseline')
plt.show()
# Shapiro-Wilk Test
stat, p = stats.shapiro(condition_specific_data['toeoutangle'].dropna())
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))









# Select the 'armROM' column
armROM = data['gaitspeed']
# Calculate IQR
Q1 = armROM.quantile(0.25)
Q3 = armROM.quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = armROM[(armROM < lower_bound) | (armROM > upper_bound)]

# Count the number of outliers
num_outliers = outliers.count()

print(f"Number of outliers: {num_outliers}")
print("Outlier values:", outliers.values)
