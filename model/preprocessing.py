from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        """Fit the scaler to the data."""
        self.scaler.fit(data)
        return self

    def transform(self, data):
        """Transform the input data using the fitted scaler."""
        return self.scaler.transform(data)

    def fit_transform(self, data):
        """Fit the scaler and transform the data."""
        return self.scaler.fit_transform(data)

    def save(self, filepath):
        """Save the preprocessor to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load the preprocessor from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)