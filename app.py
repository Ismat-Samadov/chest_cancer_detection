# app.py
import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(
    title="Chest CT Cancer Detection API",
    description="API for detecting cancer in chest CT scans using deep learning",
    version="1.0.0"
)

# Create necessary directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)  # Create static directory to avoid the error

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files only if the directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Constants
IMG_SIZE = 224  # Same as in training
MODEL_PATH = "models/chest_ct_binary_classifier.keras"  # Path to your saved model

# Initialize model
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Add a route to serve the HTML page
@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    """Serve the web UI for image upload and analysis"""
    return templates.TemplateResponse("index.html", {"request": request})

# Preprocessing functions (same as in your training script)
def apply_clahe(img, chance=0.0):  # Set chance to 0 for inference (deterministic)
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        img = ImageOps.equalize(img)
        return np.array(img) / 255.0
    return img

def apply_random_contrast(img, chance=0.0, factor_range=(0.5, 1.5)):
    """Apply contrast adjustment"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def apply_random_sharpness(img, chance=0.0, factor_range=(0.5, 2.0)):
    """Apply sharpness adjustment"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def preprocess_image(img):
    """
    Preprocess a PIL Image for inference
    
    Args:
        img: PIL Image object
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Resize to match model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to RGB (in case it's grayscale)
    img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Apply minimal preprocessing (no augmentation during inference)
    img_array = apply_clahe(img_array, chance=0.0)
    img_array = apply_random_contrast(img_array, chance=0.0)
    img_array = apply_random_sharpness(img_array, chance=0.0)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Pydantic models for request/response validation
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    cancer_probability: float
    classification_threshold: float

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chest CT Cancer Detection API",
        "model": "DenseNet121-based binary classifier",
        "endpoints": {
            "/predict": "Upload and analyze chest CT scan images",
            "/ui": "Web interface for image upload and analysis",
            "/health": "Health check endpoint"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), threshold: Optional[float] = 0.5):
    """
    Predict cancer in chest CT scan
    
    Args:
        file: Uploaded image file
        threshold: Classification threshold (0.0-1.0, default=0.5)
        
    Returns:
        Prediction results including classification and confidence
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate threshold
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image")
    
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        img_array = preprocess_image(img)
        
        # Make prediction
        prediction = float(model.predict(img_array)[0][0])
        predicted_class = int(prediction > threshold)
        
        # Class names
        class_names = ['Normal', 'Cancer']
        
        # Create response
        result = {
            "prediction": class_names[predicted_class],
            "confidence": float(prediction if predicted_class == 1 else 1 - prediction),
            "cancer_probability": float(prediction),
            "classification_threshold": float(threshold)
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": model is not None}

# Run the app with uvicorn if executed as a script
if __name__ == "__main__":
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)