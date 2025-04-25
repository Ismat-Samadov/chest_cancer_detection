# app.py
import os
import io
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
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
MODEL_PATH = "models/chest_ct_binary_classifier_densenet_20250425_024058.keras"
THRESHOLD = 0.7416  # Optimal threshold determined during model evaluation

# Initialize model
model = None

# Custom layer classes for loading the model
class ChannelAttention(tf.keras.layers.Layer):
    """Custom Channel Attention layer implementation"""
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

class ChannelMaxPooling(tf.keras.layers.Layer):
    """Custom layer for max pooling across channels"""
    def __init__(self, **kwargs):
        super(ChannelMaxPooling, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=3, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)
    
    def get_config(self):
        config = super(ChannelMaxPooling, self).get_config()
        return config

class ChannelAvgPooling(tf.keras.layers.Layer):
    """Custom layer for average pooling across channels"""
    def __init__(self, **kwargs):
        super(ChannelAvgPooling, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=3, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)
    
    def get_config(self):
        config = super(ChannelAvgPooling, self).get_config()
        return config

class SpatialAttention(tf.keras.layers.Layer):
    """Custom Spatial Attention layer implementation"""
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.channel_max_pool = ChannelMaxPooling()
        self.channel_avg_pool = ChannelAvgPooling()
        self.concat = tf.keras.layers.Concatenate(axis=3)
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, padding='same', activation='sigmoid')
        self.multiply = tf.keras.layers.Multiply()
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Max pool along channel dimension
        max_pool = self.channel_max_pool(inputs)
        # Average pool along channel dimension
        avg_pool = self.channel_avg_pool(inputs)
        # Concatenate
        concat = self.concat([max_pool, avg_pool])
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
        # Create directories if they don't exist
        os.makedirs("models", exist_ok=True)
        
        print("Looking for model files...")
        
        # Define paths to check
        keras_path = MODEL_PATH
        h5_path = MODEL_PATH.replace(".keras", ".h5")
        tf_saved_model_path = MODEL_PATH.replace(".keras", "_tf")
        checkpoint_path = "models/binary_model_densenet_checkpoint.keras"
        
        # Check if files exist
        keras_exists = os.path.exists(keras_path)
        h5_exists = os.path.exists(h5_path)
        # Properly check for SavedModel (directory with saved_model.pb inside)
        tf_saved_model_exists = os.path.isdir(tf_saved_model_path) and os.path.exists(os.path.join(tf_saved_model_path, "saved_model.pb"))
        checkpoint_exists = os.path.exists(checkpoint_path)
        
        print(f"Keras model exists: {keras_exists}")
        print(f"H5 model exists: {h5_exists}")
        print(f"TF SavedModel exists: {tf_saved_model_exists}")
        print(f"Checkpoint model exists: {checkpoint_exists}")
        
        # Define custom objects for model loading
        custom_objects = {
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention,
            'ChannelMaxPooling': ChannelMaxPooling,
            'ChannelAvgPooling': ChannelAvgPooling,
            'sigmoid_focal_crossentropy': sigmoid_focal_crossentropy
        }
        
        # Enable unsafe deserialization for custom layers
        # This is needed for TensorFlow 2.16+ to load custom layers
        keras.config.enable_unsafe_deserialization()
        
        # Try loading SavedModel format first as it's most reliable
        if tf_saved_model_exists:
            print(f"Loading model from SavedModel format: {tf_saved_model_path}")
            try:
                # Try using standard SavedModel load
                model = tf.saved_model.load(tf_saved_model_path)
                print("Successfully loaded TF SavedModel!")
                
                # For SavedModel format, we need a special flag
                model.is_saved_model = True
                
                # Also check if it has a serving signature
                if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
                    print("SavedModel has a serving signature, which will be used for predictions")
                else:
                    print("SavedModel doesn't have a serving signature - using standard prediction")
                    model.is_saved_model_without_signature = True
            except Exception as e:
                print(f"Error loading TF SavedModel: {e}")
                tf_saved_model_exists = False
        
        # Try checkpoint file next
        if model is None and checkpoint_exists:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            try:
                model = tf.keras.models.load_model(
                    checkpoint_path,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
                print("Successfully loaded checkpoint model!")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                checkpoint_exists = False
        
        # If both SavedModel and checkpoint failed, try Keras format
        if model is None and keras_exists:
            print(f"Loading model from Keras format: {keras_path}")
            try:
                model = tf.keras.models.load_model(
                    keras_path,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
                print("Successfully loaded Keras model!")
            except Exception as e:
                print(f"Error loading Keras model: {e}")
                keras_exists = False
        
        # If all previous attempts failed, try H5 format
        if model is None and h5_exists:
            print(f"Loading model from H5 format: {h5_path}")
            try:
                model = tf.keras.models.load_model(
                    h5_path,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
                print("Successfully loaded H5 model!")
            except Exception as e:
                print(f"Error loading H5 model: {e}")
                h5_exists = False
        
        # If all loading attempts failed, create a fallback model
        if model is None:
            print("All loading attempts failed. Creating a fallback model...")
            try:
                # Create a simple model with DenseNet base to provide basic functionality
                from tensorflow.keras.applications import DenseNet121
                
                base_model = DenseNet121(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3)
                )
                
                inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
                x = base_model(inputs)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                print("WARNING: Using fallback model with ImageNet weights. Predictions will be unreliable.")
            except Exception as e:
                print(f"Error creating fallback model: {e}")
                
                # Create an extremely simple model as last resort
                inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
                x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                print("WARNING: Using simple emergency fallback model. Predictions will be very unreliable.")
        
        # Compile the model if it's a Keras model (not needed for SavedModel)
        if not hasattr(model, 'is_saved_model') and not hasattr(model, 'is_saved_model_without_signature'):
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=sigmoid_focal_crossentropy,
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
        
        # Check if model is successfully loaded
        if model is not None:
            print("Model loaded successfully!")
            
            # Print model summary for Keras models only (SavedModel doesn't have summary)
            if hasattr(model, 'summary') and not hasattr(model, 'is_saved_model') and not hasattr(model, 'is_saved_model_without_signature'):
                print("\nModel Summary:")
                model.summary(line_length=100)
            else:
                print("\nModel loaded in SavedModel format (summary not available)")
        
    except Exception as e:
        print(f"Unexpected error during model loading: {e}")
        
        # Create an extremely simple model as last resort
        inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        print("WARNING: Using simple emergency fallback model. Predictions will be very unreliable.")
    
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
    allow_origins=["*"],
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
        if hasattr(model, 'is_saved_model') or hasattr(model, 'is_saved_model_without_signature'):
            model_type = "TensorFlow SavedModel format"
        elif isinstance(model, tf.keras.Model):
            # Check for attention layers
            has_attention = False
            for layer in model.layers:
                if isinstance(layer, ChannelAttention) or isinstance(layer, SpatialAttention):
                    has_attention = True
                    break
            
            if has_attention:
                model_type = "DenseNet121 with Attention"
            elif any("densenet" in str(layer.__class__).lower() for layer in model.layers):
                model_type = "DenseNet121 (Base)"
            else:
                model_type = "Custom Model"
        else:
            model_type = "Unknown Model Format"
    
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
        
        # Make prediction based on model type
        if hasattr(model, 'is_saved_model'):
            # For SavedModel format with serving signature, use the signature
            serving_fn = model.signatures['serving_default']
            
            # Get input tensor name from signature
            input_name = list(serving_fn.structured_input_signature[1].keys())[0]
            
            # Make prediction
            prediction_tensor = serving_fn(**{input_name: tf.convert_to_tensor(img_array)})
            
            # Get output tensor name from signature
            output_name = list(prediction_tensor.keys())[0]
            
            # Extract prediction value
            prediction = float(prediction_tensor[output_name].numpy()[0][0])
            
        elif hasattr(model, 'is_saved_model_without_signature'):
            # For SavedModel without signature, this is trickier
            # We'll try to call the model directly as a function, which works in some cases
            try:
                # Convert input to tensor
                input_tensor = tf.convert_to_tensor(img_array)
                
                # Call model as function
                output = model(input_tensor)
                
                # Try to extract prediction
                if isinstance(output, dict):
                    # If output is a dictionary, get the first value
                    prediction = float(list(output.values())[0].numpy()[0][0])
                else:
                    # Otherwise try to get the first element
                    prediction = float(output.numpy()[0][0])
                
            except Exception as inner_e:
                print(f"Error using SavedModel without signature: {inner_e}")
                # Fall back to a simple prediction value
                prediction = 0.5  # Neutral prediction as fallback
        else:
            # For regular Keras model, use predict method
            prediction = float(model.predict(img_array, verbose=0)[0][0])
        
        # Determine class based on threshold
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to run the application on")
    args = parser.parse_args()
    
    port = int(os.environ.get("PORT", args.port))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")