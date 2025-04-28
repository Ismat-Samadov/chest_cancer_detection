import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2

# Set page configuration
st.set_page_config(
    page_title="Chest Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('models/chest_ct_binary_classifier.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize(target_size)
    # Convert to RGB
    image = image.convert('RGB')
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    # Expand dims to create batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_heatmap(model, img_array, original_img):
    """Generate Grad-CAM heatmap for the image"""
    try:
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
        
        if last_conv_layer is None:
            # Try to find a specific layer for DenseNet
            for layer in model.layers:
                if 'conv5_block' in layer.name:
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is None:
            return None
            
        # Create a model that maps the input image to the activations
        # of the last conv layer and the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [last_conv_layer.output, model.output]
        )
        
        # Compute gradient of the top predicted class with respect
        # to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, 0]
        
        # Gradient of the output neuron with respect to the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of mean intensity of gradient over feature map
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by how important it is
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original image to BGR (for OpenCV)
        original_img_cv = np.array(original_img)
        original_img_cv = cv2.cvtColor(original_img_cv, cv2.COLOR_RGB2BGR)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap * 0.4 + original_img_cv
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        return heatmap, superimposed_img
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")
        return None, None

def main():
    # Header
    st.title("Chest Cancer Detection System")
    st.markdown("""
    This application uses deep learning to detect cancer in chest CT scan images.
    Upload a chest CT scan image, and the model will classify it as either normal or cancerous.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses a DenseNet121-based deep learning model to classify chest CT scan images.
    
    The model is trained to detect:
    - Normal CT scans
    - CT scans with cancer (adenocarcinoma, squamous cell carcinoma, or large cell carcinoma)
    
    The model has been trained on the Chest CT-Scan Images dataset from Kaggle.
    """)
    
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for cancer classification"
    )
    
    show_heatmap = st.sidebar.checkbox("Show Attention Heatmap", value=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if model files exist.")
        return
    
    # File uploader
    st.subheader("Upload a Chest CT Scan Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Process uploaded image
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
        
        # Preprocess and predict
        img_array = preprocess_image(image)
        
        with st.spinner("Running prediction..."):
            prediction = model.predict(img_array)[0][0]
        
        # Determine class
        is_cancer = prediction > confidence_threshold
        class_name = "Cancer" if is_cancer else "Normal"
        confidence = float(prediction) if is_cancer else float(1 - prediction)
        
        # Display prediction
        with col2:
            st.subheader("Prediction Results")
            
            # Create a colored box based on the prediction
            color = "red" if is_cancer else "green"
            st.markdown(f"""
            <div style="padding: 20px; background-color: {color}; color: white; border-radius: 10px; text-align: center;">
                <h2>{class_name}</h2>
                <h3>Confidence: {confidence:.2%}</h3>
                <p>Cancer Probability: {prediction:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if show_heatmap and is_cancer:
                st.subheader("Attention Heatmap")
                heatmap, superimposed_img = generate_heatmap(model, img_array, image)
                
                if superimposed_img is not None:
                    st.image(superimposed_img, caption="Regions of Interest", use_column_width=True)
                    st.info("The heatmap highlights areas the model is focusing on to make its prediction.")
        
        # Technical details section
        with st.expander("Technical Details"):
            st.write(f"""
            - Raw prediction value: {prediction:.6f}
            - Confidence threshold: {confidence_threshold}
            - Model architecture: DenseNet121
            - Input image size: 256x256 pixels
            """)
            
            # Display class probabilities as a bar chart
            fig, ax = plt.subplots(figsize=(10, 2))
            classes = ['Normal', 'Cancer']
            probs = [1-prediction, prediction]
            ax.barh(classes, probs, color=['green', 'red'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Class Probabilities')
            
            for i, prob in enumerate(probs):
                ax.text(prob + 0.01, i, f"{prob:.2%}", va='center')
                
            st.pyplot(fig)
    
    # Display information when no image is uploaded
    else:
        st.info("Please upload a chest CT scan image to get predictions.")
        
        # Display sample images
        st.subheader("Sample Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Normal_CT_scan_of_the_chest.jpg/320px-Normal_CT_scan_of_the_chest.jpg", 
                    caption="Example: Normal Chest CT", width=300)
        
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Secondary_cancer_in_the_lungs_%28CT_scan%29.jpg/320px-Secondary_cancer_in_the_lungs_%28CT_scan%29.jpg", 
                    caption="Example: Chest CT with Cancer", width=300)

if __name__ == "__main__":
    main()