# Chest CT Cancer Binary Classification
# -------------------------------------------------------
# A clean and well-organized implementation for binary cancer detection from chest CT scans

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224  # Standard input size for many CNN architectures
BATCH_SIZE = 16  # Smaller batch size for better generalization
EPOCHS = 20  # Total training epochs (including fine-tuning phase)
BASE_DIR = "/kaggle/input/chest-ctscan-images/Data"  # Your dataset location
MODELS_DIR = "models"  # Directory to save models
PLOTS_DIR = "plots"  # Directory to save plots

# Create directories for outputs
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------------------------------
# Cell 1: Understand the dataset structure
# -------------------------------------------------------

# List all directories
print("Examining dataset structure...")
train_dir = os.path.join(BASE_DIR, 'train')
valid_dir = os.path.join(BASE_DIR, 'valid')
test_dir = os.path.join(BASE_DIR, 'test')

# Check classes in each split
train_classes = os.listdir(train_dir)
valid_classes = os.listdir(valid_dir)
test_classes = os.listdir(test_dir)

print(f"Training classes: {train_classes}")
print(f"Validation classes: {valid_classes}")
print(f"Test classes: {test_classes}")

# Count images in each class
train_counts = {}
for cls in train_classes:
    class_dir = os.path.join(train_dir, cls)
    if os.path.isdir(class_dir):
        n_imgs = len([f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        train_counts[cls] = n_imgs

print(f"Training image counts per class: {train_counts}")
print(f"Total training images: {sum(train_counts.values())}")

# -------------------------------------------------------
# Cell 2: Define Custom Preprocessing Functions
# -------------------------------------------------------

# These preprocessing steps enhance features in medical images
def apply_clahe(img, chance=0.5):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Why: CLAHE improves local contrast and helps highlight subtle 
    features in medical images by equalizing histogram in small regions
    """
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        img = ImageOps.equalize(img)
        return np.array(img) / 255.0
    return img

def apply_random_contrast(img, chance=0.5, factor_range=(0.5, 1.5)):
    """
    Apply random contrast adjustment
    
    Why: Helps model become robust to variations in image contrast,
    which is common in CT scans from different machines/protocols
    """
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def apply_random_sharpness(img, chance=0.5, factor_range=(0.5, 2.0)):
    """
    Apply random sharpness adjustment
    
    Why: Enhances tissue boundaries and fine structures in CT scans
    which can help identify abnormal growth patterns
    """
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def custom_preprocessing(img):
    """Combine all preprocessing steps"""
    img = apply_clahe(img, chance=0.3)
    img = apply_random_contrast(img, chance=0.3)
    img = apply_random_sharpness(img, chance=0.3)
    return img

# -------------------------------------------------------
# Cell 3: Create Data Generators with Augmentation
# -------------------------------------------------------

# Extensive data augmentation helps with generalization and handling small datasets
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rescale=1./255,                # Normalize pixel values
    rotation_range=40,             # Randomly rotate images
    width_shift_range=0.3,         # Randomly shift image horizontally
    height_shift_range=0.3,        # Randomly shift image vertically
    shear_range=0.2,               # Shear transformations
    zoom_range=0.3,                # Zoom in/out
    horizontal_flip=True,          # Flip images horizontally
    vertical_flip=True,            # Flip images vertically (valid for CT scans)
    brightness_range=[0.7, 1.3],   # Adjust brightness
    fill_mode='reflect',           # Fill strategy for created pixels
    validation_split=0.1           # Optional: further split training data
)

# For validation and testing, only rescale (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# -------------------------------------------------------
# Cell 4: Create Balanced Dataset for Binary Classification
# -------------------------------------------------------

# Function to collect file paths and assign binary labels
def create_binary_dataset(base_dir, mode):
    """
    Create a dataframe of image paths and binary labels (0=normal, 1=cancer)
    
    Args:
        base_dir: Base directory containing class folders
        mode: 'train', 'valid', or 'test'
    
    Returns:
        DataFrame with 'filename' and 'class' columns
    """
    directory = os.path.join(BASE_DIR, mode)
    cancer_classes = [c for c in os.listdir(directory) if c != 'normal']
    all_classes = ['normal'] + cancer_classes
    
    paths = []
    labels = []
    
    for i, class_name in enumerate(all_classes):
        class_dir = os.path.join(directory, class_name)
        if os.path.exists(class_dir):
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
            paths.extend(files)
            # 0 for normal, 1 for any cancer type
            label = 0 if class_name == 'normal' else 1
            labels.extend([label] * len(files))
    
    return pd.DataFrame({'filename': paths, 'class': labels})

# Create dataframes for each split
train_df = create_binary_dataset(BASE_DIR, 'train')
valid_df = create_binary_dataset(BASE_DIR, 'valid')
test_df = create_binary_dataset(BASE_DIR, 'test')

# Display class distribution
print("Class distribution in datasets:")
print(f"Training: {train_df['class'].value_counts()}")
print(f"Validation: {valid_df['class'].value_counts()}")
print(f"Testing: {test_df['class'].value_counts()}")

# -------------------------------------------------------
# Cell 5: Create a Custom Balanced Data Generator
# -------------------------------------------------------

# This generator ensures equal representation of both classes in each batch
class BalancedBinaryDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=16, target_size=(224, 224), 
                 shuffle=True, datagen=None, balanced=True):
        """
        Custom generator that creates balanced batches for binary classification
        
        Why: Handles class imbalance by sampling equal numbers of each class,
        which is crucial for medical datasets where pathology is often rarer
        than normal cases
        """
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.datagen = datagen if datagen is not None else ImageDataGenerator(rescale=1./255)
        self.balanced = balanced
        
        # Split indices by class
        self.normal_indices = dataframe[dataframe['class'] == 0].index.tolist()
        self.cancer_indices = dataframe[dataframe['class'] == 1].index.tolist()
        
        # Check if we have both classes
        self.has_both_classes = len(self.normal_indices) > 0 and len(self.cancer_indices) > 0
        
        if self.balanced and not self.has_both_classes:
            print(f"WARNING: Can't create balanced batches. Normal samples: {len(self.normal_indices)}, "
                  f"Cancer samples: {len(self.cancer_indices)}. Switching to unbalanced sampling.")
            self.balanced = False
        
        # Number of batches
        self.batches_per_epoch = 50 if self.balanced else max(1, len(dataframe) // batch_size)
        
        # Reset state for each epoch
        self.on_epoch_end()
    
    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        batch_indices = []
        
        if self.balanced and self.has_both_classes:
            # For each batch, select half normal and half cancer
            half_batch = self.batch_size // 2
            
            # Get indices for this batch
            batch_normal_indices = np.random.choice(self.normal_indices, size=half_batch, replace=True)
            batch_cancer_indices = np.random.choice(self.cancer_indices, size=half_batch, replace=True)
            batch_indices = np.concatenate([batch_normal_indices, batch_cancer_indices])
            
            # Shuffle indices
            if self.shuffle:
                np.random.shuffle(batch_indices)
        else:
            # For unbalanced batches, just select randomly from all indices
            idx_offset = (idx * self.batch_size) % len(self.dataframe)
            batch_indices = np.arange(idx_offset, min(idx_offset + self.batch_size, len(self.dataframe)))
            if self.shuffle:
                np.random.shuffle(batch_indices)
        
        # Load images and labels
        batch_x = []
        batch_y = []
        
        for i in batch_indices:
            filename = self.dataframe.iloc[i]['filename']
            class_label = self.dataframe.iloc[i]['class']
            
            # Load image
            img = Image.open(filename)
            img = img.resize(self.target_size)
            img = img.convert('RGB')  # Ensure 3 channels
            img_array = np.array(img) / 255.0  # Normalize
            
            # Apply data augmentation if available
            if self.datagen is not None:
                # Convert to batch of 1 for augmentation
                img_array = np.expand_dims(img_array, axis=0)
                # Get augmented image
                augmented = next(self.datagen.flow(img_array, batch_size=1, shuffle=False))
                img_array = augmented[0]
            
            batch_x.append(img_array)
            batch_y.append(class_label)
        
        return np.array(batch_x), np.array(batch_y)
    
    def on_epoch_end(self):
        """Called at the end of each epoch during training"""
        if self.shuffle:
            np.random.shuffle(self.normal_indices)
            np.random.shuffle(self.cancer_indices)

# Create generator instances
train_generator = BalancedBinaryDataGenerator(
    train_df, 
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    datagen=train_datagen,
    balanced=True
)

valid_generator = BalancedBinaryDataGenerator(
    valid_df,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    datagen=valid_datagen,
    shuffle=False,
    balanced=True
)

test_generator = BalancedBinaryDataGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    datagen=test_datagen,
    shuffle=False,
    balanced=False  # Don't force balancing for test set
)

# -------------------------------------------------------
# Cell 6: Create the Model Architecture
# -------------------------------------------------------

def create_binary_model():
    """
    Create a binary classification model based on DenseNet121
    
    Why DenseNet121: 
    - Excellent at capturing fine-grained patterns in medical images
    - Dense connections help with gradient flow during training
    - Achieves high accuracy with fewer parameters than other architectures
    - Transfer learning leverages knowledge from natural images
    """
    # Load pre-trained DenseNet121 model (no top layers)
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create new model with regularization to prevent overfitting
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Gaussian noise improves robustness
    x = tf.keras.layers.GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # First dense layer with regularization
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)  # Normalize activations for stable training
    x = Dropout(0.7)(x)  # High dropout to prevent overfitting
    
    # Second dense layer
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    
    # Output layer with sigmoid for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model with binary crossentropy
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower initial learning rate for stability
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

# Create the model
model, base_model = create_binary_model()
model.summary()

# -------------------------------------------------------
# Cell 7: Define Training Callbacks
# -------------------------------------------------------

# Callbacks to improve training
callbacks = [
    # Stop training when validation accuracy doesn't improve
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,  # Number of epochs with no improvement
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when metrics plateau
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,  # Multiply lr by this factor
        patience=3,  # Number of epochs with no improvement
        min_lr=1e-6,
        verbose=1
    ),
    # Save the best model
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'binary_model_checkpoint.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# -------------------------------------------------------
# Cell 8: Two-Phase Training - Initial Phase
# -------------------------------------------------------

# Phase 1: Train only the top layers while keeping the base model frozen
print("Phase 1: Training top layers with frozen base model...")
history_phase1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.batches_per_epoch,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=10,  # Initial shorter training phase
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------
# Cell 9: Two-Phase Training - Fine-tuning Phase
# -------------------------------------------------------

# Phase 2: Fine-tune the model by unfreezing some layers of the base model
print("Phase 2: Fine-tuning the model by unfreezing deeper layers...")

# Unfreeze the base model but keep early layers frozen
base_model.trainable = True
# Freeze first 100 layers, unfreeze the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Continue training
history_phase2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.batches_per_epoch,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    initial_epoch=len(history_phase1.history['loss']),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------
# Cell 10: Combine Training Histories and Visualize
# -------------------------------------------------------

# Combine histories from both phases
history_combined = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'auc': history_phase1.history['auc'] + history_phase2.history['auc'],
    'val_auc': history_phase1.history['val_auc'] + history_phase2.history['val_auc'],
    'precision': history_phase1.history['precision'] + history_phase2.history['precision'],
    'val_precision': history_phase1.history['val_precision'] + history_phase2.history['val_precision'],
    'recall': history_phase1.history['recall'] + history_phase2.history['recall'],
    'val_recall': history_phase1.history['val_recall'] + history_phase2.history['val_recall']
}

# Plot training metrics
plt.figure(figsize=(16, 12))

# Plot accuracy
plt.subplot(2, 2, 1)
plt.plot(history_combined['accuracy'], label='Training Accuracy')
plt.plot(history_combined['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.axvline(x=len(history_phase1.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning start')

# Plot loss
plt.subplot(2, 2, 2)
plt.plot(history_combined['loss'], label='Training Loss')
plt.plot(history_combined['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.axvline(x=len(history_phase1.history['loss'])-1, color='r', linestyle='--')

# Plot AUC
plt.subplot(2, 2, 3)
plt.plot(history_combined['auc'], label='Training AUC')
plt.plot(history_combined['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.axvline(x=len(history_phase1.history['auc'])-1, color='r', linestyle='--')

# Plot Precision-Recall
plt.subplot(2, 2, 4)
plt.plot(history_combined['precision'], label='Training Precision')
plt.plot(history_combined['val_precision'], label='Validation Precision')
plt.plot(history_combined['recall'], label='Training Recall')
plt.plot(history_combined['val_recall'], label='Validation Recall')
plt.title('Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.axvline(x=len(history_phase1.history['precision'])-1, color='r', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'binary_training_history.png'))
plt.show()

# -------------------------------------------------------
# Cell 11: Evaluate on Test Set
# -------------------------------------------------------

# Evaluate on test data
test_results = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")
print(f"Test Precision: {test_results[3]:.4f}")
print(f"Test Recall: {test_results[4]:.4f}")

# Get predictions for confusion matrix and ROC curve
test_steps = len(test_generator)
test_y_true = []
test_y_pred = []

for i in range(test_steps):
    x_batch, y_batch = test_generator[i]
    test_y_true.extend(y_batch)
    preds = model.predict(x_batch)
    test_y_pred.extend(preds.flatten())

# Convert to numpy arrays
test_y_true = np.array(test_y_true)
test_y_pred = np.array(test_y_pred)

# Convert probabilities to binary predictions using threshold of 0.5
test_y_pred_binary = (test_y_pred > 0.5).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(test_y_true, test_y_pred_binary)

# Display confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Cancer'])
plt.yticks(tick_marks, ['Normal', 'Cancer'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations to the confusion matrix
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.show()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(test_y_true, test_y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(test_y_true, test_y_pred_binary, target_names=['Normal', 'Cancer']))

# -------------------------------------------------------
# Cell 12: Save the Final Model in Multiple Formats
# -------------------------------------------------------

# Save the model in Keras format
model_keras_path = os.path.join(MODELS_DIR, 'chest_ct_binary_classifier.keras')
model.save(model_keras_path)
print(f"Model saved in Keras format: {model_keras_path}")

# Save the model in TensorFlow SavedModel format
model_tf_path = os.path.join(MODELS_DIR, 'chest_ct_binary_classifier_tf')
tf.saved_model.save(model, model_tf_path)
print(f"Model saved in TensorFlow SavedModel format: {model_tf_path}")

# Save the model in TFLite format for mobile/edge deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(MODELS_DIR, 'chest_ct_binary_classifier.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"Model saved in TFLite format: {tflite_path}")

# Also save as h5 for older applications
model_h5_path = os.path.join(MODELS_DIR, 'chest_ct_binary_classifier.h5')
model.save(model_h5_path)
print(f"Model saved in h5 format: {model_h5_path}")

# -------------------------------------------------------
# Cell 13: Define Functions for Single Image Prediction
# -------------------------------------------------------

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_cancer(model, image_path, threshold=0.5):
    """
    Predict whether a CT scan shows cancer or normal tissue
    
    Args:
        model: Loaded TensorFlow/Keras model
        image_path: Path to the image file
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        dict: Prediction results with class name and confidence
    """
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    prediction = float(model.predict(img_array)[0][0])
    predicted_class = int(prediction > threshold)
    
    # Class names
    class_names = ['Normal', 'Cancer']
    
    # Create result dictionary
    result = {
        "prediction": class_names[predicted_class],
        "confidence": prediction if predicted_class == 1 else 1 - prediction,
        "cancer_probability": prediction,
        "classification_threshold": threshold
    }
    
    return result

# Example usage (uncomment to test)
# test_image_path = "/path/to/test/image.png"
# result = predict_cancer(model, test_image_path)
# print(f"Prediction: {result['prediction']}")
# print(f"Confidence: {result['confidence']:.2f}")
# print(f"Cancer Probability: {result['cancer_probability']:.2f}")

# -------------------------------------------------------
# Cell 14: Summary of Results and Next Steps
# -------------------------------------------------------

print("\n=== Binary Cancer Detection Model Summary ===")
print(f"Model Architecture: DenseNet121 (transfer learning)")
print(f"Input Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Training Images: {len(train_df)}")
print(f"Validation Images: {len(valid_df)}")
print(f"Test Images: {len(test_df)}")
print(f"Training Class Balance: {dict(train_df['class'].value_counts())}")
print(f"Final Test Accuracy: {test_results[1]:.4f}")
print(f"Final Test AUC: {test_results[2]:.4f}")
print("\nModel saved in multiple formats for deployment.")
print("\nNext steps could include:")
print("1. Gradual unfreezing of more DenseNet layers")
print("2. Ensemble of multiple models")
print("3. Exploring other architectures (EfficientNet, Vision Transformer)")
print("4. Integrating with a web or mobile application")
print("5. Deployment to an edge device for on-device inference")