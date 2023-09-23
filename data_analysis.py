import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset (replace 'dataset.csv' with your dataset file)
df = pd.read_csv("dataset.csv")
print(df.head())

# Summary statistics
print(df.describe())

# Visualizations (e.g., histogram)
sns.histplot(df['age'], bins=20)
plt.title('Age Distribution')
plt.show()


# Check for missing values
print(df.isnull().sum())

# Calculate the mean of the column
mean_value = df['age'].mean()

# Fill missing values with the mean
df['column_name'].fillna(mean_value, inplace=True)


# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=['categorical_column'], drop_first=True)

# Define features (X) and target variable (y)
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split the data (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a classifier (choose an appropriate algorithm)
classifier = LogisticRegression()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate and print relevant metrics (e.g., accuracy)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
print("Confusion Matrix:")
print(conf_matrix)

# Create a regressor (choose an appropriate algorithm)
regressor = LinearRegression()

# Train the regressor
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate and print relevant metrics (e.g., RMSE and R-squared)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
