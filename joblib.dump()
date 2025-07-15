import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("diamonds.csv")
df["size"] = df["x"] * df["y"] * df["z"]

# Define features and target
X = df[["carat", "cut", "color", "clarity", "depth", "table", "size"]]
y = df["price"]

# Define column types
numerical_features = ["carat", "depth", "table", "size"]
categorical_features = ["cut", "color", "clarity"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Model pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "diamond_price_model.pkl")
