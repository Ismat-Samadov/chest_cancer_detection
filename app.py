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
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Global variables for model
model = None
model_type = None
threshold = 0.7416  # Default optimal threshold from your model

@app.on_event("startup")
async def startup_event():
    global model, model_type, threshold
    
    # Define model paths based on your directory structure
    models_dir = "models"
    keras_models = glob.glob(os.path.join(models_dir, "*.keras"))
    h5_models = glob.glob(os.path.join(models_dir, "*.h5"))
    tf_models = glob.glob(os.path.join(models_dir, "*_tf_*"))
    tflite_models = glob.glob(os.path.join(models_dir, "*.tflite"))
    
    logger.info(f"Found models - Keras: {len(keras_models)}, H5: {len(h5_models)}, TF: {len(tf_models)}, TFLite: {len(tflite_models)}")
    
    # Try loading model in order: Keras, H5, SavedModel, TFLite
    # 1. Try Keras format first
    if not model and keras_models:
        try:
            newest_keras = max(keras_models, key=os.path.getctime)
            logger.info(f"Attempting to load Keras model: {newest_keras}")
            model = tf.keras.models.load_model(newest_keras)
            model_type = "keras"
            logger.info(f"Successfully loaded Keras model: {newest_keras}")
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
    
    # 2. Try H5 format if Keras failed
    if not model and h5_models:
        try:
            newest_h5 = max(h5_models, key=os.path.getctime)
            logger.info(f"Attempting to load H5 model: {newest_h5}")
            model = tf.keras.models.load_model(newest_h5)
            model_type = "h5"
            logger.info(f"Successfully loaded H5 model: {newest_h5}")
        except Exception as e:
            logger.error(f"Failed to load H5 model: {e}")
    
    # 3. Try TensorFlow SavedModel format if previous attempts failed
    if not model and tf_models:
        try:
            newest_tf = max(tf_models, key=os.path.getctime)
            logger.info(f"Attempting to load TF SavedModel: {newest_tf}")
            model = tf.saved_model.load(newest_tf)
            model_type = "tf_saved_model"
            logger.info(f"Successfully loaded TF SavedModel: {newest_tf}")
            
            # Try to get the serving function
            try:
                model.signatures['serving_default']
                logger.info("Successfully found serving_default signature")
            except Exception as e:
                logger.warning(f"Model loaded but no serving_default signature found: {e}")
        except Exception as e:
            logger.error(f"Failed to load TF SavedModel: {e}")
    
    # 4. Try TFLite format as last resort
    if not model and tflite_models:
        try:
            newest_tflite = max(tflite_models, key=os.path.getctime)
            logger.info(f"Attempting to load TFLite model: {newest_tflite}")
            interpreter = tf.lite.Interpreter(model_path=newest_tflite)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Store them as attributes of the interpreter
            interpreter.input_details = input_details
            interpreter.output_details = output_details
            
            model = interpreter
            model_type = "tflite"
            logger.info(f"Successfully loaded TFLite model: {newest_tflite}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
    
    if model:
        logger.info(f"Model loaded successfully with type: {model_type}")
    else:
        logger.error("Failed to load any model")

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    # Resize and normalize the image
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_with_model(preprocessed_image):
    """Make prediction with the loaded model based on model type"""
    global model, model_type
    
    if model_type in ["keras", "h5"]:
        # Standard Keras prediction
        prediction = model.predict(preprocessed_image, verbose=0)
        return float(prediction[0][0])
    
    elif model_type == "tf_saved_model":
        try:
            # Try the serving signature first
            serving_fn = model.signatures['serving_default']
            input_name = list(serving_fn.structured_input_signature[1].keys())[0]
            
            # Convert to tensor
            tensor_input = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)
            
            # Make prediction
            pred_tensor = serving_fn(**{input_name: tensor_input})
            
            # Extract prediction value
            output_name = list(pred_tensor.keys())[0]
            prediction = float(pred_tensor[output_name].numpy()[0][0])
            return prediction
        except Exception as e:
            logger.error(f"Error using serving signature: {e}")
            
            # Fallback to direct call
            try:
                result = model(tf.constant(preprocessed_image, dtype=tf.float32))
                if isinstance(result, dict):
                    # If result is a dictionary, try to get the first value
                    prediction = float(list(result.values())[0].numpy()[0][0])
                else:
                    # Otherwise assume it's a tensor
                    prediction = float(result.numpy()[0][0])
                return prediction
            except Exception as e2:
                logger.error(f"Error using direct call: {e2}")
                raise ValueError(f"Failed to get prediction: {e2}")
    
    elif model_type == "tflite":
        # TFLite prediction
        input_details = model.input_details
        output_details = model.output_details
        
        # Ensure input has the right shape
        input_shape = input_details[0]['shape']
        if preprocessed_image.shape != tuple(input_shape):
            preprocessed_image = tf.image.resize(preprocessed_image, (input_shape[1], input_shape[2]))
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        
        # Set input tensor
        model.set_tensor(input_details[0]['index'], preprocessed_image)
        
        # Run inference
        model.invoke()
        
        # Get output tensor
        output_data = model.get_tensor(output_details[0]['index'])
        return float(output_data[0][0])
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict cancer from a chest CT scan"""
    global model, threshold
    
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
        
        # Make prediction using appropriate method based on model type
        prediction = predict_with_model(processed_image)
        
        # Use threshold for classification
        diagnosis = "Cancer" if prediction > threshold else "Normal"
        confidence = prediction if diagnosis == "Cancer" else 1 - prediction
        
        return {
            "filename": file.filename,
            "diagnosis": diagnosis,
            "cancer_probability": float(prediction),
            "confidence": float(confidence),
            "threshold_used": threshold,
            "model_type": model_type
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during prediction: {str(e)}"}
        )

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "No model loaded"}
    
    return {
        "model_loaded": model is not None,
        "model_type": model_type,
        "threshold": threshold
    }

@app.get("/")
async def redirect_to_docs():
    """Redirect root to documentation"""
    return {"message": "API Running", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None, "model_type": model_type}

# Add at the bottom of your app.py file
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)