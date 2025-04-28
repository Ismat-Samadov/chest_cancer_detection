# app.py
import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf
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
threshold = 0.7416  # Default optimal threshold

@app.on_event("startup")
async def startup_event():
    global model
    
    # Load only the TensorFlow SavedModel format
    models_dir = "models"
    tf_model_dir = os.path.join(models_dir, "chest_ct_binary_classifier_densenet_tf_20250427_182239")
    
    try:
        logger.info(f"Loading TF SavedModel from: {tf_model_dir}")
        model = tf.saved_model.load(tf_model_dir)
        logger.info("Successfully loaded TF SavedModel")
        
        # Verify serving function exists
        serving_fn = model.signatures['serving_default']
        input_name = list(serving_fn.structured_input_signature[1].keys())[0]
        logger.info(f"Model loaded with input signature: {input_name}")
    except Exception as e:
        logger.error(f"Failed to load TF SavedModel: {e}")
        model = None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

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
        
        # Get the serving signature and make prediction
        serving_fn = model.signatures['serving_default']
        input_name = list(serving_fn.structured_input_signature[1].keys())[0]
        
        # Convert to tensor
        tensor_input = tf.convert_to_tensor(processed_image, dtype=tf.float32)
        
        # Make prediction
        pred_tensor = serving_fn(**{input_name: tensor_input})
        
        # Extract prediction value
        output_name = list(pred_tensor.keys())[0]
        prediction = float(pred_tensor[output_name].numpy()[0][0])
        
        # Use threshold for classification
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
    return {
        "model_loaded": model is not None,
        "model_type": "tf_saved_model",
        "threshold": threshold
    }

@app.get("/")
async def redirect_to_docs():
    """Redirect root to documentation"""
    return {"message": "API Running", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)