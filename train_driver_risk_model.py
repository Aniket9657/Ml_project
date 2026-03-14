import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("driver_risk_dataset_5000.csv")

# Encode categorical columns
label_encoder = LabelEncoder()

df["car_type"] = label_encoder.fit_transform(df["car_type"])
df["locality"] = label_encoder.fit_transform(df["locality"])
df["profession"] = label_encoder.fit_transform(df["profession"])

# Features and target
X = df.drop("risk_label", axis=1)
y = df["risk_label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "driver_risk_model.pkl")

print("\nModel saved as driver_risk_model.pkl")