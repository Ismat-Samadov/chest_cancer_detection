# app.py
import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf
import uvicorn

app = FastAPI(title="Chest Cancer Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model on startup
MODEL_PATH = "models/chest_ct_binary_classifier.keras"
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Initialize with a dummy model
        model = None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    # Resize and normalize the image
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict cancer from a chest CT scan"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PNG and JPEG images are supported"}
        )
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = float(model.predict(processed_image)[0][0])
        
        # Use threshold from your model (0.7416)
        threshold = 0.7416
        diagnosis = "Cancer" if prediction > threshold else "Normal"
        confidence = prediction if diagnosis == "Cancer" else 1 - prediction
        
        return {
            "filename": file.filename,
            "diagnosis": diagnosis,
            "cancer_probability": float(prediction),
            "confidence": float(confidence),
            "threshold_used": threshold
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during prediction: {str(e)}"}
        )

@app.get("/")
async def redirect_to_docs():
    """Redirect root to documentation"""
    return {"message": "API Running", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)