import pandas as pd

# Load dataset
data = pd.read_csv("../dataset/heart.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Dataset shape
print("\nDataset shape (rows, columns):")
print(data.shape)

# Column names
print("\nColumn names:")
print(data.columns)

# Check missing values
print("\nMissing values in each column:")
print(data.isnull().sum())
