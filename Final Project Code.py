# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the dataset (update the file path as needed)
data = pd.read_csv('/Users/user1/Documents/Final Project/Maternal Health Risk Dataset.csv')

# Display the first few records of the data
print(data.head())

# Cleaning the data
# Checking for any missing values
print("Missing values:\n", data.isnull().sum())

# Checking for duplicate rows
print(f"Duplicate rows: {data.duplicated().sum()}")

# Summary statistics for numeric columns
print("\nSummary Statistics:\n", data.describe())

# Visualizing the distribution of Age
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True, color='blue')
plt.title('Age Distribution for Maternal Health')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Visualizing the distribution of Blood Pressure (Systolic and Diastolic)
plt.figure(figsize=(8, 6))
sns.histplot(data['SystolicBP'], kde=True, color='blue', label='Systolic BP')
sns.histplot(data['DiastolicBP'], kde=True, color='red', label='Diastolic BP')
plt.title('Distribution of Blood Pressure')
plt.xlabel('Blood Pressure (mm Hg)')
plt.ylabel('Count')
plt.legend()
plt.show()

# Visualizing the distribution of Heart Rate and Blood Sugar
plt.figure(figsize=(8, 6))
sns.histplot(data['HeartRate'], kde=True, color='green', label='Heart Rate')
sns.histplot(data['BS'], kde=True, color='purple', label='Blood Sugar')
plt.title('Heart Rate and Blood Sugar Distribution')
plt.xlabel('Value')
plt.ylabel('Count')
plt.legend()
plt.show()

# Visualizing the Risk Level distribution (Categorical data)
plt.figure(figsize=(8, 6))
sns.countplot(x='RiskLevel', data=data, palette='Set2')
plt.title('Risk Level Distribution')
plt.xlabel('Risk Level')
plt.ylabel('Frequency')
plt.show()

# Creating a correlation matrix for numerical variables
correlation_matrix = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Maternal Health Features')
plt.show()

# Visualizing how features relate to Risk Level
plt.figure(figsize=(8, 6))
sns.boxplot(x='RiskLevel', y='Age', data=data, palette='Set3')
plt.title('Age vs Risk Level')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='RiskLevel', y='SystolicBP', data=data, palette='Set3')
sns.boxplot(x='RiskLevel', y='DiastolicBP', data=data, palette='Set3')
plt.title('Blood Pressure vs Risk Level')
plt.show()

# Boxplots to detect outliers in numerical variables
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['SystolicBP'])
plt.title('Systolic Blood Pressure Outliers')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=data['DiastolicBP'])
plt.title('Diastolic Blood Pressure Outliers')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=data['HeartRate'])
plt.title('Heart Rate Outliers')
plt.show()

# Pairplot to observe relationships between various features
sns.pairplot(data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel']], hue='RiskLevel')
plt.title('Pairwise Feature Relationships')
plt.show()

# Addressing class imbalance (if necessary)
# Analyzing the distribution of RiskLevel
print("\nClass Distribution for Risk Level:")
print(data['RiskLevel'].value_counts())

# If necessary, apply SMOTE to balance the classes
X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = data['RiskLevel']

# Mapping categorical target values to numeric (optional for machine learning)
y = y.map({'low risk': 0, 'mid risk': 1, 'high risk': 2})

# Applying SMOTE to balance the dataset
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Checking the distribution of classes after applying SMOTE
print("\nUpdated Class Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Saving the cleaned and resampled data for future use
resampled_data = pd.DataFrame(X_resampled, columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
resampled_data['RiskLevel'] = y_resampled
resampled_data.to_csv('resampled_maternal_health.csv', index=False)
