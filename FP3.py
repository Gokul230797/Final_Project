# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load the dataset (update the file path as needed)
data = pd.read_csv('/Users/user1/Documents/Final Project/Maternal Health Risk Dataset.csv')

# Initial data exploration
print("Dataset Overview:\n", data.info())
print("\nFirst Five Rows:\n", data.head())

# Data Cleaning
# Check for missing values and duplicates
print("\nMissing Values:\n", data.isnull().sum())
print(f"\nDuplicate Rows: {data.duplicated().sum()}")

# Drop duplicates if any
data = data.drop_duplicates()

# Statistical Summary
print("\nSummary Statistics:\n", data.describe())

# Encode RiskLevel for numerical operations
label_encoder = LabelEncoder()
data['RiskLevel'] = label_encoder.fit_transform(data['RiskLevel'])

# Display the mapping for verification
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"\nRisk Level Encoding: {label_mapping}")

# --- Exploratory Data Analysis (EDA) ---
# 1. Age Distribution with Risk Level Overlay
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Age', hue='RiskLevel', multiple='stack', palette='Set2', kde=True)
plt.title('Age Distribution by Risk Level')
plt.xlabel('Age (Years)')
plt.ylabel('Count')
plt.show()

# 2. Blood Pressure Distribution
plt.figure(figsize=(8, 6))
sns.violinplot(data=data, x='RiskLevel', y='SystolicBP', palette='Set1', inner='quartile')
plt.title('Systolic Blood Pressure Distribution by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Systolic BP (mm Hg)')
plt.show()

# 3. Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Feature Engineering: Add Derived Features
data['PulsePressure'] = data['SystolicBP'] - data['DiastolicBP']
data['RiskAgeRatio'] = data['Age'] / (data['HeartRate'] + 1)

# --- Addressing Class Imbalance ---
# Check initial class distribution
print("\nInitial Class Distribution:")
print(data['RiskLevel'].value_counts())

# Prepare data for resampling
X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'PulsePressure', 'RiskAgeRatio']]
y = data['RiskLevel']

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution after SMOTE
print("\nClass Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# --- Model Training and Evaluation ---
# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Cross-Validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

# Predictions and Evaluation
y_pred = rf_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- ROC Curve ---
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


from keras.models import Sequential
from keras.layers import Dense

ann_model = Sequential()
ann_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Assuming multiclass classification

ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)
print("ANN Classification Report:\n", classification_report(y_test, y_pred_ann))
