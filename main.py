from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from model.preprocessing import DataPreprocessor


# Initialize FastAPI app
app = FastAPI(title="Iris Classification API", description="API for predicting Iris species using a pre-trained model")

# Define input data model using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the preprocessor and model
preprocessor = DataPreprocessor.load("iris_preprocessor.pkl")
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define class names for Iris species
class_names = ["setosa", "versicolor", "virginica"]

@app.get("/")
async def root():
    return {"message": "Welcome to the Iris Classification API"}

@app.post("/predict")
async def predict(data: IrisInput):
    try:
        # Convert input data to numpy array
        input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        
        # Preprocess the input data
        input_data_scaled = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled).max()
        
        # Get predicted class name
        predicted_class = class_names[int(prediction[0])]
        
        return {
            "prediction": predicted_class,
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}