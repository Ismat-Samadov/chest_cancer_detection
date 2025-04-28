# app.py - Lightweight TFLite implementation for Render deployment
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
interpreter = None
input_details = None
output_details = None
threshold = 0.7416  # Default optimal threshold

@app.on_event("startup")
async def startup_event():
    global interpreter, input_details, output_details
    
    # Load only the TFLite model which requires much less memory
    models_dir = "models"
    tflite_model_path = os.path.join(models_dir, "chest_ct_binary_classifier_densenet_20250427_182239.tflite")
    
    try:
        logger.info(f"Loading TFLite model from: {tflite_model_path}")
        
        # Set num_threads to 1 to reduce memory usage
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=1)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Successfully loaded TFLite model with input shape: {input_details[0]['shape']}")
        logger.info(f"Output details: {output_details[0]['shape']}")
    except Exception as e:
        logger.error(f"Failed to load TFLite model: {e}")
        interpreter = None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Get the expected input shape from model
    if input_details:
        input_shape = input_details[0]['shape']
        # Ensure we have the right shape
        if len(input_shape) == 4:  # [batch, height, width, channels]
            target_size = (input_shape[1], input_shape[2])
            if image_array.shape[:2] != target_size:
                image = image.resize(target_size)
                image_array = np.array(image, dtype=np.float32) / 255.0
    
    return np.expand_dims(image_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict cancer from a chest CT scan"""
    global interpreter, input_details, output_details, threshold
    
    if interpreter is None or input_details is None or output_details is None:
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
        
        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
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
        "model_loaded": interpreter is not None,
        "model_type": "tflite",
        "threshold": threshold,
        "status": "active" if interpreter is not None else "not loaded"
    }

@app.get("/")
async def redirect_to_docs():
    """Redirect root to documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chest Cancer Detection API</title>
            <meta http-equiv="refresh" content="0;url=/static/index.html">
        </head>
        <body>
            <p>Redirecting to application...</p>
        </body>
    </html>
    """
    return JSONResponse(
        content={"message": "API Running", "docs": "/docs", "app": "/static/index.html"},
        headers={"Content-Type": "application/json"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": interpreter is not None}

# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)