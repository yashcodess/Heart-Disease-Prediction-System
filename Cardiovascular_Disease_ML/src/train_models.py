import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("../dataset/heart.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------
# Model 1: Logistic Regression
# ------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# ------------------------
# Model 2: Random Forest
# ------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)  # RF does not require scaling
rf_pred = rf.predict(X_test)

# ------------------------
# Model 3: Support Vector Machine
# ------------------------
svm = SVC(kernel="rbf")
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# ------------------------
# Evaluation Function
# ------------------------
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Performance:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))

# ------------------------
# Results
# ------------------------
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("SVM", y_test, svm_pred)
