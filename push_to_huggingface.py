#!/usr/bin/env python3
import os
import subprocess
import argparse
import shutil
import time

# Parse arguments
parser = argparse.ArgumentParser(description='Push chest cancer detection model to Hugging Face Space')
parser.add_argument('--username', type=str, required=True, help='Your Hugging Face username')
parser.add_argument('--space_name', type=str, default='chest-cancer-detection', help='Name for your Hugging Face space')
parser.add_argument('--model_path', type=str, required=True, help='Path to your SavedModel directory')
parser.add_argument('--token', type=str, help='Hugging Face API token (or set HF_TOKEN environment variable)')
args = parser.parse_args()

# Get token from args or environment
hf_token = args.token or os.environ.get('HF_TOKEN')
if not hf_token:
    print("Please provide a Hugging Face token via --token or set the HF_TOKEN environment variable")
    exit(1)

# Set up directory names
space_repo = f"{args.username}/{args.space_name}"
local_dir = f"hf-{args.space_name}"

# Create app.py
app_code = '''
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2
import io
import os
matplotlib.use('Agg')  # Use non-interactive backend

# Load the model using SavedModel format
MODEL_PATH = "model"
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]  # Get the inference function

# Get input and output tensor names
input_tensor_name = list(infer.structured_input_signature[1].keys())[0]
output_tensor_name = list(infer.structured_outputs.keys())[0]

# Image size - matching what the model was trained on
IMG_SIZE = 256

# Function for preprocessing
def preprocess_image(image):
    img = Image.fromarray(image).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)  # Cast to float32 for TF

# Make prediction with the SavedModel
def predict_with_saved_model(image_tensor):
    # Create the input tensor with the right name
    input_dict = {input_tensor_name: image_tensor}
    # Run inference
    output = infer(**input_dict)
    # Get the prediction value
    prediction = output[output_tensor_name].numpy()[0][0]
    return prediction

# Generate attention map
def generate_attention_map(img_array, prediction):
    # Convert to grayscale
    gray = cv2.cvtColor(img_array[0].astype(np.float32), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use edge detection to find interesting regions
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-1
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    # Weight by prediction confidence
    weight = 0.5 + (prediction - 0.5) * 0.5  # Scale between 0.5-1 based on prediction
    magnitude = magnitude * weight
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * magnitude), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap, magnitude

# Prediction function with visualization
def predict_and_explain(image):
    if image is None:
        return None, "Please upload an image.", 0.0
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Make prediction
    prediction = predict_with_saved_model(preprocessed)
    
    # Generate attention map
    heatmap, attention = generate_attention_map(preprocessed, prediction)
    
    # Create overlay
    original_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    superimposed = (0.6 * original_resized) + (0.4 * heatmap)
    superimposed = superimposed.astype(np.uint8)
    
    # Create visualization with matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_resized)
    axes[0].set_title("Original CT Scan")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("Feature Map")
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    axes[2].set_title(f"Overlay")
    axes[2].axis('off')
    
    # Add prediction information
    result_text = f"{'Cancer' if prediction > 0.5 else 'Normal'} (Confidence: {abs(prediction if prediction > 0.5 else 1-prediction):.2%})"
    fig.suptitle(result_text, fontsize=16)
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    result_image = np.array(Image.open(buf))
    
    # Return prediction information
    prediction_class = "Cancer" if prediction > 0.5 else "Normal"
    confidence = float(prediction if prediction > 0.5 else 1-prediction)
    
    return result_image, prediction_class, confidence

# Create Gradio interface
with gr.Blocks(title="Chest CT Scan Cancer Detection") as demo:
    gr.Markdown("# Chest CT Scan Cancer Detection")
    gr.Markdown("Upload a chest CT scan image to detect the presence of cancer.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload CT Scan Image", type="numpy")
            submit_btn = gr.Button("Analyze Image")
        
        with gr.Column():
            output_image = gr.Image(label="Analysis Results")
            prediction_label = gr.Label(label="Prediction")
            confidence_score = gr.Number(label="Confidence Score")
    
    gr.Markdown("### How it works")
    gr.Markdown("""
    This application uses a deep learning model based on DenseNet121 architecture to classify chest CT scans as either 'Normal' or 'Cancer'.
    
    The visualization shows:
    - Left: Original CT scan
    - Middle: Feature map highlighting areas with distinctive patterns
    - Right: Overlay of the feature map on the original image
    
    The model was trained on a dataset of chest CT scans containing normal images and various types of lung cancer (adenocarcinoma, squamous cell carcinoma, and large cell carcinoma).
    """)
    
    submit_btn.click(
        predict_and_explain,
        inputs=input_image,
        outputs=[output_image, prediction_label, confidence_score]
    )

demo.launch()
'''

# Create README.md
readme_content = '''# Chest CT Scan Cancer Detection

This application uses a deep learning model to detect cancer in chest CT scan images. The model is based on a DenseNet121 architecture and trained on the Chest CT-Scan Images dataset from Kaggle.

## How to Use

1. Upload a chest CT scan image using the interface
2. Click "Analyze Image" to get results
3. View the prediction (Normal or Cancer) and visualization

## About the Model

- **Architecture**: Modified DenseNet121
- **Task**: Binary Classification (Normal vs. Cancer)
- **Input**: Chest CT scan images (resized to 256x256)
- **Performance**: ~90% accuracy on test set

## Limitations

- The model works best with chest CT scans similar to those in the training data
- This is a research tool and should not be used for clinical diagnosis without professional medical oversight

## Citation

If you use this model in your research, please cite:
@misc{samadov2025chestcancer,
author = {Ismat Samadov},
title = {Chest Cancer Detection Using Deep Learning},
year = {2025},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/Ismat-Samadov/chest_cancer_detection}}
}
'''

# Create requirements.txt
requirements_content = '''tensorflow==2.19.0
numpy==2.1.3
Pillow==11.2.1
matplotlib==3.10.1
gradio==4.9.0
opencv-python-headless==4.9.0.80
'''

# Create .gitattributes for Git LFS
gitattributes_content = '''*.pb filter=lfs diff=lfs merge=lfs -text
model/variables/variables.data-00000-of-00001 filter=lfs diff=lfs merge=lfs -text
'''

print(f"Creating local repository in {local_dir}")

# Create local directory
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)
os.makedirs(local_dir)

# Initialize git
os.chdir(local_dir)
subprocess.run(['git', 'init'], check=True)

# Create files
with open('app.py', 'w') as f:
    f.write(app_code)

with open('README.md', 'w') as f:
    f.write(readme_content)

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

with open('.gitattributes', 'w') as f:
    f.write(gitattributes_content)

# Create model directory and copy files
os.makedirs('model/variables', exist_ok=True)
shutil.copy(os.path.join(args.model_path, 'saved_model.pb'), 'model/')
shutil.copy(os.path.join(args.model_path, 'fingerprint.pb'), 'model/')
shutil.copy(os.path.join(args.model_path, 'variables/variables.index'), 'model/variables/')
shutil.copy(os.path.join(args.model_path, 'variables/variables.data-00000-of-00001'), 'model/variables/')

# Initialize Git LFS
subprocess.run(['git', 'lfs', 'install'], check=True)
subprocess.run(['git', 'lfs', 'track', 'model/variables/variables.data-00000-of-00001'], check=True)
subprocess.run(['git', 'lfs', 'track', '*.pb'], check=True)

# Prepare git
subprocess.run(['git', 'add', '.'], check=True)
subprocess.run(['git', 'commit', '-m', 'Initial commit with chest cancer detection model'], check=True)

print(f"Creating new Hugging Face Space: {space_repo}")

# Login to Hugging Face
subprocess.run(['huggingface-cli', 'login', '--token', hf_token], check=True)

# Create the Hugging Face space
subprocess.run(['huggingface-cli', 'repo', 'create', space_repo, '--type', 'space', '--space-sdk', 'gradio'], check=True)

print("Pushing model to Hugging Face Space (this may take some time for large models)...")
# Push to Hugging Face
subprocess.run(['git', 'remote', 'add', 'origin', f'https://huggingface.co/spaces/{space_repo}'], check=True)
subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)

print(f"\nSuccess! Your model has been deployed to: https://huggingface.co/spaces/{space_repo}")
print("It may take a few minutes for the space to build and become available.")