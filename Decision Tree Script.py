# Complete Python Script: From EDA to Feature Engineering to Decision Tree Classifier

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 2: Load the dataset
file_path = '/mnt/data/Loan_Modelling (1).csv'  # Replace this with the correct file path in Colab
data = pd.read_csv(file_path)

# Step 3: Exploratory Data Analysis (EDA)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Univariate Analysis
def univariate_analysis(data, column, chart_type='histogram'):
    if chart_type == 'histogram':
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.show()
    elif chart_type == 'boxplot':
        plt.figure(figsize=(8, 6))
        sns.boxplot(data[column])
        plt.title(f"Boxplot of {column}")
        plt.show()

# Plot for numeric variables
numerical_columns = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
for col in numerical_columns:
    univariate_analysis(data, col)

# Multivariate Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots for visualizing relationships
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Income', y='CCAvg', hue='Personal_Loan')
plt.title('Income vs. Credit Card Spending by Loan Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Income', y='Mortgage', hue='Personal_Loan')
plt.title('Income vs. Mortgage by Loan Status')
plt.show()

# Step 4: Data Preprocessing and Feature Engineering

# Drop unnecessary columns
data = data.drop(['ID', 'ZIPCode'], axis=1)

# Handle missing values
data.fillna(data.median(), inplace=True)

# Convert numeric columns to appropriate types
numeric_columns = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.fillna(data.median(), inplace=True)  # Handle resulting NaNs

# Feature Engineering
data['Income_to_Mortgage_Ratio'] = data['Mortgage'] / (data['Income'] + 1e-5)
data['Total_Accounts'] = data['Securities_Account'] + data['CD_Account'] + data['CreditCard']
data['Spending_to_Income_Ratio'] = data['CCAvg'] / (data['Income'] + 1e-5)

# Binning income into brackets
income_bins = [data['Income'].min(), data['Income'].quantile(0.33), data['Income'].quantile(0.66), data['Income'].max()]
income_labels = ['Low', 'Medium', 'High']
data['Income_Bracket'] = pd.cut(data['Income'], bins=income_bins, labels=income_labels)

# Binning age into groups
age_bins = [data['Age'].min(), 35, 55, data['Age'].max()]
age_labels = ['Young', 'Middle Aged', 'Senior']
data['Age_Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# Scale continuous features
scaler = StandardScaler()
scaled_features = ['Income', 'CCAvg', 'Mortgage', 'Income_to_Mortgage_Ratio', 'Spending_to_Income_Ratio']
data[scaled_features] = scaler.fit_transform(data[scaled_features])

# Encode categorical variables
categorical_features = ['Education', 'Family', 'Income_Bracket', 'Age_Group']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Step 5: Prepare Data for Modeling
X = data.drop('Personal_Loan', axis=1)
y = data['Personal_Loan']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 6: Decision Tree Classifier with Weighted Classes
dt_classifier_weighted = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,
    class_weight={0: 1, 1: 5},
    random_state=42
)
dt_classifier_weighted.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred_weighted = dt_classifier_weighted.predict(X_test)

# Confusion Matrix
conf_matrix_weighted = confusion_matrix(y_test, y_pred_weighted)
print("Confusion Matrix (Weighted):\n", conf_matrix_weighted)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_weighted, annot=True, fmt='d', cmap='Blues', xticklabels=['No Loan', 'Loan'], yticklabels=['No Loan', 'Loan'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Weighted)')
plt.show()

# Classification Report
class_report_weighted = classification_report(y_test, y_pred_weighted)
print("Classification Report (Weighted):\n", class_report_weighted)

# Accuracy Score
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
print("Accuracy of the Weighted Decision Tree model:", accuracy_weighted)

# Feature Importance
feature_importances_weighted = pd.Series(dt_classifier_weighted.feature_importances_, index=X.columns)
feature_importances_weighted = feature_importances_weighted.sort_values(ascending=False)

print("Feature Importances (Weighted):\n", feature_importances_weighted)

# Visualize Feature Importances
plt.figure(figsize=(12, 6))
feature_importances_weighted.plot(kind='bar')
plt.title('Feature Importances (Weighted)')
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    dt_classifier_weighted,
    feature_names=X.columns,
    class_names=['No Loan', 'Loan'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
