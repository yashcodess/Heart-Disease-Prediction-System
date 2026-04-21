import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("../dataset/heart.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (for consistency in future models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest (best model)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(rf_model, "../model/heart_disease_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("Model and scaler saved successfully.")
