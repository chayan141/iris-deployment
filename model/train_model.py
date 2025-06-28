from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .preprocessing import DataPreprocessor
import pickle

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
preprocessor = DataPreprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Save the preprocessor and model
preprocessor.save("iris_preprocessor.pkl")
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Preprocessor saved as iris_preprocessor.pkl")
print("Model saved as iris_model.pkl")