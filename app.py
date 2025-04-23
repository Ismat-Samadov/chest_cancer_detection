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
MODEL_PATH = "models/chest_ct_binary_classifier_densenet_20250423_104845.keras"
THRESHOLD = 0.7416  # Optimal threshold determined during model evaluation

# Initialize model
model = None

# Custom layer classes to match the trainer.py
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        channel = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(channel // self.ratio, activation='relu', 
                          kernel_initializer='he_normal', use_bias=False)
        self.dense2 = tf.keras.layers.Dense(channel, activation='sigmoid', 
                          kernel_initializer='he_normal', use_bias=False)
        self.reshape = tf.keras.layers.Reshape((1, 1, channel))
        self.multiply = tf.keras.layers.Multiply()
        
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, inputs):
        gap = self.gap(inputs)
        dense1 = self.dense1(gap)
        dense2 = self.dense2(dense1)
        reshape = self.reshape(dense2)
        
        return self.multiply([inputs, reshape])
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'ratio': self.ratio
        })
        return config

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, padding='same', activation='sigmoid')
        self.multiply = tf.keras.layers.Multiply()
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Max pool along channel dimension
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        # Average pool along channel dimension
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        # Concatenate
        concat = tf.concat([max_pool, avg_pool], axis=3)
        # Apply convolution
        attention_map = self.conv(concat)
        # Apply attention
        return self.multiply([inputs, attention_map])
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

# Custom focal loss function
def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Custom implementation of Focal Loss"""
    # Get binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Convert y_true to float32 if needed
    y_true = tf.cast(y_true, dtype=tf.float32)
    
    # Calculate the modulating factor
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = tf.pow((1.0 - p_t), gamma)
    
    # Apply the factors and return
    return alpha_factor * modulating_factor * bce

# Lifespan context manager for proper startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    
    try:
        # Enable unsafe deserialization for compatibility with complex models
        keras.config.enable_unsafe_deserialization()
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(MODEL_PATH):
            try:
                # Try to load the model with custom objects
                custom_objects = {
                    'ChannelAttention': ChannelAttention,
                    'SpatialAttention': SpatialAttention,
                    'sigmoid_focal_crossentropy': sigmoid_focal_crossentropy
                }
                
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    custom_objects=custom_objects,
                    compile=False
                )
                print(f"Model loaded successfully from {MODEL_PATH}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Creating DenseNet121 model with ImageNet weights")
                # Create a simple model as fallback
                base_model = tf.keras.applications.DenseNet121(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3)
                )
                
                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(256, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(64, activation='relu')(x)
                predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                print("WARNING: Using untrained model with ImageNet weights. Predictions will be unreliable.")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}.")
            print("Creating DenseNet121 model with ImageNet weights")
            # Create a simple model as fallback
            base_model = tf.keras.applications.DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            print("WARNING: Using untrained model with ImageNet weights. Predictions will be unreliable.")
    except Exception as e:
        print(f"Error setting up model: {e}")
        print("Creating a simple fallback model...")
        
        # Simple fallback model as last resort
        inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
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
            # Get model architecture information
            for layer in model.layers:
                if isinstance(layer, tf.keras.applications.densenet.DenseNet121):
                    model_type = "DenseNet121 with Custom Layers"
                    break
            if model_type == "Unknown" and hasattr(model, 'name'):
                if "densenet" in model.name.lower():
                    model_type = "DenseNet Model"
                elif "resnet" in model.name.lower():
                    model_type = "ResNet Model"
                else:
                    model_type = "Custom Model"
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
        
        # Make prediction with reduced verbosity
        prediction = float(model.predict(img_array, verbose=0)[0][0])
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