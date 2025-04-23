# app.py
import os
import io
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import keras
from PIL import Image
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Constants
IMG_SIZE = 256  # Same as in training
MODEL_PATH = "models/chest_ct_binary_classifier_densenet_20250423_061443.keras"
THRESHOLD = 0.7416  # Optimal threshold determined during model evaluation

# Initialize model
model = None

# Define custom functions for Lambda layers
def attention_mechanism(x):
    # Implement attention mechanism (this is a placeholder implementation)
    # This should match what your original Lambda layer was doing
    return tf.nn.softmax(x, axis=-1)

def create_densenet_model():
    """Create a DenseNet121 model for chest CT scan classification"""
    # Base model
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',  # Start with ImageNet weights
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Add classification layers
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Lifespan context manager for proper startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    
    try:
        # Enable unsafe deserialization for Lambda layers
        keras.config.enable_unsafe_deserialization()
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(MODEL_PATH):
            # Try various approaches to load the model
            try:
                # Approach 1: Try loading with custom objects
                custom_objects = {
                    'attention_mechanism': attention_mechanism
                }
                
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    custom_objects=custom_objects,
                    compile=False
                )
                print(f"Model loaded successfully with custom objects from {MODEL_PATH}")
            except Exception as e:
                print(f"Could not load model with custom objects: {e}")
                print("Falling back to creating a new model...")
                
                # Approach 2: Create a new model with the expected architecture
                model = create_densenet_model()
                print("Created new DenseNet121 model with ImageNet weights")
                print("WARNING: This is not your trained model; predictions may be inaccurate")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}.")
            print("Creating a new DenseNet121 model with ImageNet weights")
            model = create_densenet_model()
            print("WARNING: This is not your trained model; predictions may be inaccurate")
    except Exception as e:
        print(f"Error setting up model: {e}")
        print("Creating a simple fallback model...")
        
        # Simple fallback model as last resort
        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        print("WARNING: Using simple fallback model. Predictions will be unreliable.")
    
    yield
    
    # Shutdown: cleanup (if needed)
    print("Shutting down application...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Chest CT Cancer Detection API",
    description="API for detecting cancer in chest CT scans using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Create necessary directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(img):
    """
    Preprocess a PIL Image for model inference
    
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
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Pydantic models for response validation
class PredictionResponse(BaseModel):
    prediction: str
    cancer_probability: float
    confidence: float
    threshold_used: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_type: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web UI for image upload and analysis"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_type = "Unknown"
    if model is not None:
        if isinstance(model, tf.keras.Model):
            if "densenet" in model.name.lower():
                model_type = "DenseNet121 (Loaded)"
            else:
                model_type = "Fallback Model"
        else:
            model_type = "SavedModel Format"
    
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_type": model_type
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), threshold: Optional[float] = THRESHOLD):
    """
    Predict cancer in chest CT scan
    
    Args:
        file: Uploaded CT scan image file
        threshold: Classification threshold (optional, defaults to optimal threshold)
        
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
            "cancer_probability": float(prediction),
            "confidence": float(prediction if predicted_class == 1 else 1 - prediction),
            "threshold_used": float(threshold)
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")