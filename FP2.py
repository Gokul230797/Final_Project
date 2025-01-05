# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from xgboost import XGBClassifier

# Load the dataset (update the file path as needed)
data = pd.read_csv('/Users/user1/Documents/Final Project/Maternal Health Risk Dataset.csv')

# Data Cleaning
data = data.drop_duplicates()
label_encoder = LabelEncoder()
data['RiskLevel'] = label_encoder.fit_transform(data['RiskLevel'])

# Feature Engineering
data['PulsePressure'] = data['SystolicBP'] - data['DiastolicBP']
data['RiskAgeRatio'] = data['Age'] / (data['HeartRate'] + 1)
data['BP_BS_Interaction'] = data['SystolicBP'] * data['BS']

# Address Class Imbalance
X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'PulsePressure', 'RiskAgeRatio', 'BP_BS_Interaction']]
y = data['RiskLevel']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scaling and Splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# --- Random Forest ---
rf_params = {
    'n_estimators': [200, 300],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# --- ANN ---
ann_model = Sequential()
ann_model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(64, activation='relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(len(np.unique(y_train)), activation='softmax'))
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
ann_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)
print("\nANN Classification Report:\n", classification_report(y_test, y_pred_ann))

# --- XGBoost ---
xgb_params = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'n_estimators': [150, 200],
    'subsample': [0.8, 1.0]
}
xgb_grid_search = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), xgb_params, cv=3, n_jobs=-1, scoring='accuracy')
xgb_grid_search.fit(X_train, y_train)
best_xgb_model = xgb_grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Evaluate Models
print("\nModel Comparison:")
print("Random Forest Accuracy:", np.round(best_rf_model.score(X_test, y_test) * 100, 2), "%")
print("ANN Accuracy:", np.round(ann_model.evaluate(X_test, y_test, verbose=0)[1] * 100, 2), "%")
print("XGBoost Accuracy:", np.round(best_xgb_model.score(X_test, y_test) * 100, 2), "%")
