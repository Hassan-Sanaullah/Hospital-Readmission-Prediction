from preprocess import load_and_preprocess_data, prepare_features
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths
data_file = "data/diabetic_data.csv"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Load and preprocess
df = load_and_preprocess_data(data_file)
X, y = prepare_features(df)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic_Regression": LogisticRegression(solver='liblinear'),
    "Decision_Tree": DecisionTreeClassifier(max_depth=28),
    "Random_Forest": RandomForestClassifier(n_estimators=10, max_depth=25)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{model_dir}/{name}.pkl")
    print(f"{name} saved.")
