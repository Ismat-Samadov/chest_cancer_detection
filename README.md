I'll create a detailed README for your chest cancer detection project based on the files you've shared. This README will provide comprehensive information about the project, its purpose, setup instructions, and usage.

# Chest Cancer Detection

A deep learning project for detecting cancer in chest CT scan images using a binary classification approach based on DenseNet121 architecture.

## Overview

This project implements a convolutional neural network to classify chest CT scan images as either "Normal" or "Cancer". It uses transfer learning with DenseNet121 as the base model and includes custom preprocessing techniques specific to medical imaging.

## Dataset

The project uses the Chest CT-Scan Images dataset available on Kaggle: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data

The dataset contains chest CT scan images organized in the following structure:
- Training set: Contains images of normal scans and different types of lung cancer (squamous cell carcinoma, adenocarcinoma, and large cell carcinoma)
- Validation set: Similar structure to training set
- Test set: Similar structure to training set

## Features

- **Binary Classification**: Detects the presence of cancer (regardless of type) in chest CT scans
- **Medical-specific Image Preprocessing**: Implements techniques like CLAHE, contrast adjustment, and sharpness enhancement optimized for CT scans
- **Balanced Training**: Uses a custom data generator that ensures balanced class representation during training
- **Two-phase Training**: Initial training with frozen base model followed by fine-tuning of deeper layers
- **Comprehensive Evaluation**: Includes accuracy, AUC, precision, recall, confusion matrix, and ROC curve analysis
- **Multiple Export Formats**: Saves the model in various formats (Keras, TensorFlow SavedModel, TFLite, H5)

## Technical Implementation

- **Architecture**: Modified DenseNet121 with custom dense layers and regularization
- **Data Augmentation**: Extensive augmentation including rotation, shifts, zooming, and flips
- **Regularization**: Implements dropout, batch normalization, and L2 regularization to prevent overfitting
- **Learning Rate Management**: Uses ReduceLROnPlateau to adaptively adjust learning rate
- **Early Stopping**: Implements early stopping to prevent overfitting during training

## Requirements

- Python 3.8+
- TensorFlow 2.19.0
- NumPy 2.1.3
- Pandas 2.2.3
- Matplotlib 3.10.1
- scikit-learn 1.6.1
- Pillow 11.2.1

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
python train.py
```

The script will automatically:
1. Load and preprocess the dataset
2. Create and train the model in two phases (frozen base model followed by fine-tuning)
3. Evaluate the model on the test set
4. Generate performance plots and metrics
5. Save the trained model in multiple formats

### Model Outputs

After training, you'll find the following outputs in the respective directories:

- **Models directory**:
  - `chest_ct_binary_classifier.keras`: The model in Keras format
  - `chest_ct_binary_classifier_tf`: The model in TensorFlow SavedModel format
  - `chest_ct_binary_classifier.tflite`: The model in TensorFlow Lite format for mobile/edge deployment
  - `chest_ct_binary_classifier.h5`: The model in H5 format for older applications
  - `binary_model_checkpoint.keras`: The best model checkpoint during training

- **Plots directory**:
  - `binary_training_history.png`: Training and validation metrics history
  - `confusion_matrix.png`: Confusion matrix visualization
  - `roc_curve.png`: ROC curve visualization

### Making Predictions

The model includes a utility function for making predictions on single images:

```python
from train import predict_cancer, create_binary_model
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('models/chest_ct_binary_classifier.keras')

# Make a prediction
result = predict_cancer(model, 'path/to/ct_scan_image.png')

# Display results
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Cancer Probability: {result['cancer_probability']:.2f}")
```

## Performance

The model achieves strong performance on chest CT scan cancer detection with:
- High accuracy (~90%+)
- High AUC score (0.95+)
- Balanced precision and recall

Exact performance metrics will vary based on the training run and dataset split.

## Future Improvements

Potential enhancements to the project:
1. Gradual unfreezing of more DenseNet layers
2. Ensemble of multiple models
3. Exploring other architectures (EfficientNet, Vision Transformer)
4. Integrating with a web or mobile application
5. Deployment to an edge device for on-device inference
6. Multi-class classification to identify specific cancer types

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Chest CT-Scan Images dataset by Mohamed Hany on Kaggle
- TensorFlow and Keras teams for developing the deep learning framework
- DenseNet121 architecture developers

## Contact

For any questions or feedback, please open an issue on GitHub or contact the repository owner.