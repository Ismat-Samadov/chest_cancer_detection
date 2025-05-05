# Chest Cancer Detection

A deep learning project for detecting cancer in chest CT scan images using a binary classification approach based on DenseNet121 architecture with attention mechanisms.

## Overview

This project implements a convolutional neural network to classify chest CT scan images as either "Normal" or "Cancer". It uses transfer learning with DenseNet121 as the base model and includes custom preprocessing techniques specific to medical imaging, along with attention mechanisms to improve focus on relevant image regions.

## System Architecture

```mermaid
flowchart TD
    A[Input CT Scan Image] --> B[Preprocessing]
    B --> |CT-Specific Techniques| C[Augmentation]
    C --> D[Deep Learning Model]
    D --> |Feature Extraction| E[Base Model: DenseNet/ResNet/EfficientNet]
    E --> F[Attention Mechanism]
    F --> G[Classification Head]
    G --> H[Binary Output: Normal/Cancer]
    
    subgraph Preprocessing
    I[CLAHE] --> J[Contrast Adjustment]
    J --> K[Sharpness Enhancement]
    K --> L[CT Windowing]
    end
    
    subgraph Attention
    M[Channel Attention] --> N[Spatial Attention]
    end
```

## Data Processing Pipeline

```mermaid
flowchart LR
    A[Raw CT Scan Images] --> B[Split: Train/Valid/Test]
    B --> C[Class Balancing]
    C --> D[Custom Data Generator]
    D --> E[Image Preprocessing]
    E --> F[Data Augmentation]
    F --> G[MixUp Augmentation]
    G --> H[Model Training]
```

## Training Workflow

```mermaid
flowchart TD
    A[Initialize Model] --> B[Phase 1: Frozen Base Model]
    B --> C[Train Top Layers]
    C --> D[Phase 2: Fine-tuning]
    D --> E[Unfreeze Deeper Layers]
    E --> F[Continue Training]
    F --> G[Evaluate on Test Set]
    G --> H[Save Model in Multiple Formats]
    
    subgraph Evaluation
    I[Accuracy] --> J[AUC]
    J --> K[Precision/Recall]
    K --> L[Confusion Matrix]
    L --> M[ROC Curve]
    end
```

## Model Architecture

```mermaid
flowchart TD
    A[Input: 256x256x3] --> B[Base Model]
    B --> C[Global Average Pooling]
    C --> D[Attention Modules]
    D --> E[Dense Layer: 512 units]
    E --> F[Batch Normalization]
    F --> G[Dropout: 0.5]
    G --> H[Dense Layer: 256 units]
    H --> I[Batch Normalization]
    I --> J[Dropout: 0.5]
    J --> K[Output: Sigmoid]
    
    subgraph Attention Mechanism
    L[Channel Attention] --> M[Spatial Attention]
    end
```

## Attention Mechanism Details

```mermaid
flowchart LR
    subgraph "Channel Attention"
    A[Feature Map] --> B[Global Average Pooling]
    B --> C[Dense: Channel/Ratio]
    C --> D[ReLU]
    D --> E[Dense: Channel]
    E --> F[Sigmoid]
    F --> G[Reshape]
    G --> H[Multiply with Original]
    end
    
    subgraph "Spatial Attention"
    I[Feature Map] --> J[Max Pool Across Channels]
    I --> K[Avg Pool Across Channels]
    J --> L[Concatenate]
    K --> L
    L --> M[Conv2D: 7x7]
    M --> N[Sigmoid]
    N --> O[Multiply with Original]
    end
```

## Dataset

The project uses the Chest CT-Scan Images dataset available on Kaggle: [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data)

The dataset contains chest CT scan images organized in the following structure:
- Training set: Contains 613 images (148 normal scans and 465 cancer scans)
- Validation set: Contains 72 images (13 normal scans and 59 cancer scans)
- Test set: Contains 315 images (54 normal scans and 261 cancer scans)

Cancer scans include different types of lung cancer such as squamous cell carcinoma, adenocarcinoma, and large cell carcinoma.

```mermaid
pie
    title Dataset Distribution
    "Training Normal": 148
    "Training Cancer": 465
    "Validation Normal": 13
    "Validation Cancer": 59
    "Test Normal": 54
    "Test Cancer": 261
```

## CT-Specific Preprocessing

```mermaid
flowchart LR
    A[Original CT Image] --> B[CLAHE]
    B --> C[Random Contrast]
    C --> D[Random Sharpness]
    D --> E[CT Windowing]
    E --> F[Processed Image]
    
    subgraph "CT Windowing"
    G[Scale to Hounsfield Units] --> H[Select Window Width/Center]
    H --> I[Apply Window]
    I --> J[Rescale to 0-1]
    end
```

## Two-Phase Training Strategy

```mermaid
flowchart TD
    A[Base Model: Pre-trained on ImageNet] --> B[Phase 1: Frozen Base]
    B --> C[Train Only New Layers]
    C --> D[Early Stopping on AUC]
    D --> E[Phase 2: Fine-tuning]
    E --> F[Unfreeze Some Base Layers]
    F --> G[Lower Learning Rate]
    G --> H[Continue Training]
    H --> I[Save Best Model]

    subgraph "Progressive Unfreezing"
    J[DenseNet: Unfreeze Last Half] 
    K[ResNet: Unfreeze Last 50 Layers]
    L[EfficientNet: Unfreeze Last 30 Layers]
    end
```

## Kaggle Notebook

A complete implementation of this project is available as a Kaggle notebook:
- **Notebook**: [CT DenseNet Chest Cancer Detector](https://www.kaggle.com/code/ismetsemedov/ct-densenet-chest-cancer-detector)
- **Features**: The notebook includes all code for data exploration, model building, training, and evaluation with interactive visualizations
- **Environment**: Runs in Kaggle's GPU-accelerated environment for faster training
- **Reproducibility**: Contains all necessary code to reproduce the results

You can:
- View the notebook on Kaggle
- Fork it to run your own experiments
- Download it to run locally

## Demo Application Architecture

```mermaid
flowchart LR
    A[User Interface] --> B[Upload CT Scan]
    B --> C[FastAPI Backend]
    C --> D[Image Preprocessing]
    D --> E[Model Prediction]
    E --> F[Results Displayed to User]
    
    subgraph "Model Loading"
    G[Try Keras Model] --> |Fallback| H[Try H5 Model]
    H --> |Fallback| I[Try TF SavedModel]
    I --> |Fallback| J[Try TFLite Model]
    end
```

## Performance Metrics

The model achieves the following performance on the test set:

- **AUC Score**: 0.847 (95% CI: 0.785-0.903)
- **Sensitivity (Recall)**: 0.980
- **Specificity**: 0.185
- **Precision (PPV)**: 0.848
- **F1 Score**: 0.909
- **Optimal Threshold**: 0.471
- **Diagnostic Odds Ratio**: 11.136

```mermaid
graph LR
    subgraph "ROC Curve (AUC = 0.847)"
        A((0,0)) --> B((0.2,0.65))
        B --> C((0.4,0.85))
        C --> D((0.6,0.92))
        D --> E((0.8,0.97))
        E --> F((1,1))
        
        %% Diagonal reference line
        A -.-> |"Random Classifier"| F
        
        %% Optimal threshold point
        C --- G[("Optimal Threshold (0.471)")]
    end
    
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style B fill:#d1e7dd,stroke:#333,stroke-width:2px
    style C fill:#a3cfbb,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style D fill:#75b798,stroke:#333,stroke-width:2px
    style E fill:#479f76,stroke:#333,stroke-width:2px
    style F fill:#198754,stroke:#333,stroke-width:2px
    style G fill:#dc3545,stroke:#333,stroke-width:2px
```

## Features

- **Binary Classification**: Detects the presence of cancer (regardless of type) in chest CT scans
- **Attention Mechanisms**: Implements channel and spatial attention to help the model focus on relevant features
- **Medical-specific Image Preprocessing**: Implements techniques like CLAHE, contrast adjustment, sharpness enhancement, and CT windowing optimized for CT scans
- **Balanced Training**: Uses a custom data generator that ensures balanced class representation during training
- **MixUp Augmentation**: Applies MixUp technique to improve model generalization
- **Two-phase Training**: Initial training with frozen base model followed by fine-tuning of deeper layers
- **Comprehensive Evaluation**: Includes accuracy, AUC, precision, recall, confusion matrix, and ROC curve analysis
- **Multiple Export Formats**: Saves the model in various formats (Keras, TensorFlow SavedModel, TFLite, H5)

## Project Structure

```
.
├── app.py                 # FastAPI application for serving the model
├── LICENSE                # MIT License
├── models                 # Directory containing trained models
│   ├── binary_model_densenet_checkpoint.keras
│   ├── chest_ct_binary_classifier_densenet_20250427_182239.h5
│   ├── chest_ct_binary_classifier_densenet_20250427_182239.keras
│   ├── chest_ct_binary_classifier_densenet_20250427_182239.tflite
│   └── chest_ct_binary_classifier_densenet_tf_20250427_182239
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── static                 # Static files for web interface
│   ├── index.html
│   ├── script.js
│   └── styles.css
└── trainer.py             # Main training script
```

## Technical Implementation

- **Architecture**: Modified DenseNet121 with custom dense layers and regularization
- **Attention Modules**: Channel attention and spatial attention to focus on relevant regions
- **Data Augmentation**: Extensive augmentation including rotation, shifts, zooming, flips, and MixUp
- **Regularization**: Implements dropout, batch normalization, and L2 regularization to prevent overfitting
- **Learning Rate Management**: Uses ReduceLROnPlateau to adaptively adjust learning rate
- **Early Stopping**: Implements early stopping to prevent overfitting during training
- **Custom Loss Function**: Implementation of Focal Loss for handling class imbalance
- **Optimal Threshold**: Automatically determines optimal classification threshold from ROC curve

## Requirements

- Python 3.8+
- TensorFlow 2.19.0
- NumPy 2.1.3
- Pandas 2.2.3
- Matplotlib 3.10.1
- scikit-learn 1.6.1
- Pillow 11.2.1
- FastAPI 0.115.12

See `requirements.txt` for a complete list of dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ismat-Samadov/chest_cancer_detection.git
cd chest_cancer_detection
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the appropriate directory.

## Usage

### Training the Model

To train the model using the provided script:

```bash
python trainer.py
```

The script will automatically:
1. Load and preprocess the dataset
2. Create and train the model in two phases (frozen base model followed by fine-tuning)
3. Evaluate the model on the test set
4. Generate performance plots and metrics
5. Save the trained model in multiple formats

### Running the Web Application

To run the FastAPI application locally:

```bash
uvicorn app:app --reload
```

Then navigate to `http://localhost:8000` in your browser.

### Making Predictions

The model includes a utility function for making predictions on single images:

```python
from app import preprocess_image
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('models/chest_ct_binary_classifier_densenet_20250427_182239.keras')

# Preprocess an image
img, img_array = preprocess_image('path/to/ct_scan_image.png')

# Make a prediction
prediction = float(model.predict(img_array)[0][0])
diagnosis = "Cancer" if prediction > 0.471 else "Normal"  # Using optimal threshold
confidence = prediction if diagnosis == "Cancer" else 1 - prediction

# Display results
print(f"Prediction: {diagnosis}")
print(f"Confidence: {confidence:.2%}")
print(f"Cancer Probability: {prediction:.2%}")
```

## Future Improvements

Potential enhancements to the project:
1. Gradual unfreezing of more DenseNet layers
2. Ensemble of multiple models
3. Exploring other architectures (EfficientNet, Vision Transformer)
4. Integrating with a web or mobile application
5. Deployment to an edge device for on-device inference
6. Multi-class classification to identify specific cancer types
7. Implementing Grad-CAM for better visualization of model decisions
8. Incorporating 3D CNN for volumetric analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Chest CT-Scan Images dataset by Mohamed Hany on Kaggle
- TensorFlow and Keras teams for developing the deep learning framework
- DenseNet121 architecture developers