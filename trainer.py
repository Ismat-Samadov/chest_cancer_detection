# Chest CT Cancer Binary Classification with Built-in Components
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.applications import DenseNet121, ResNet50V2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
BASE_DIR = "/kaggle/input/chest-ctscan-images/Data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
USE_ENSEMBLE = True
MODEL_ARCHITECTURE = "ensemble"  # "densenet", "resnet", "efficientnet", or "ensemble"
APPLY_MIXUP = True

# Create directories for outputs
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Custom focal loss implementation
def sigmoid_focal_crossentropy(alpha=0.25, gamma=2.0):
    """Return sigmoid focal loss function"""
    def loss_fn(y_true, y_pred):
        # Get binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Convert y_true to float32
        y_true = tf.cast(y_true, dtype=tf.float32)
        
        # Calculate the focal loss factors
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        
        # Apply the factors and return
        return alpha_factor * modulating_factor * bce
    return loss_fn

# Enhanced preprocessing functions
def apply_clahe(img, chance=0.5):
    """Apply CLAHE for better contrast in CT scans"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        img = ImageOps.equalize(img)
        return np.array(img) / 255.0
    return img

def apply_random_contrast(img, chance=0.5, factor_range=(0.5, 1.8)):
    """Apply contrast adjustment for CT scans"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def apply_random_sharpness(img, chance=0.5, factor_range=(0.5, 2.2)):
    """Apply sharpness for better tissue boundary detection"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def apply_windowing(img, chance=0.4, window_width_range=(1500, 2500), window_center_range=(-600, 500)):
    """Apply CT windowing to enhance specific tissues"""
    if np.random.random() < chance:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Simulate Hounsfield unit range
        hu_img = (img * 2000) - 1000
        
        # Randomly select window width and center
        window_width = np.random.uniform(*window_width_range)
        window_center = np.random.uniform(*window_center_range)
        
        # Apply windowing
        min_value = window_center - window_width/2
        max_value = window_center + window_width/2
        windowed = np.clip(hu_img, min_value, max_value)
        
        # Rescale to 0-1
        windowed = (windowed - min_value) / (max_value - min_value)
        windowed = np.clip(windowed, 0, 1)
        
        return windowed
    return img

def custom_preprocessing(img):
    """Enhanced preprocessing pipeline for CT scans"""
    img = apply_clahe(img, chance=0.4)
    img = apply_random_contrast(img, chance=0.4)
    img = apply_random_sharpness(img, chance=0.4)
    img = apply_windowing(img, chance=0.3)
    return img

# Simple data generator
class BalancedGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=16, target_size=(256, 256), 
                 shuffle=True, augment=True, apply_mixup=False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.apply_mixup = apply_mixup
        
        # Split indices by class
        self.normal_indices = dataframe[dataframe['class'] == 0].index.tolist()
        self.cancer_indices = dataframe[dataframe['class'] == 1].index.tolist()
        
        # Check if we have both classes
        self.has_both_classes = len(self.normal_indices) > 0 and len(self.cancer_indices) > 0
        
        # Number of batches
        self.batches_per_epoch = 75 if self.has_both_classes else max(1, len(dataframe) // batch_size)
        
        # Create augmentation generator
        if self.augment:
            self.datagen = ImageDataGenerator(
                preprocessing_function=custom_preprocessing,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='reflect'
            )
        else:
            self.datagen = ImageDataGenerator(rescale=1./255)
        
        # Reset state for each epoch
        self.on_epoch_end()
    
    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        batch_indices = []
        
        if self.has_both_classes:
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
            # For unbalanced batches
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
            
            # Load image with error handling
            try:
                img = Image.open(filename)
                img = img.resize(self.target_size)
                img = img.convert('RGB')  # Ensure 3 channels
                img_array = np.array(img) / 255.0  # Normalize
                
                # Apply augmentation
                if self.augment:
                    img_array = np.expand_dims(img_array, axis=0)
                    augmented = self.datagen.flow(img_array, batch_size=1, shuffle=False)
                    img_array = next(augmented)[0]
                
                batch_x.append(img_array)
                batch_y.append(class_label)
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                continue
        
        # Convert to numpy arrays
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y).reshape(-1, 1)  # Reshape to match model output
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of each epoch during training"""
        if self.shuffle:
            np.random.shuffle(self.normal_indices)
            np.random.shuffle(self.cancer_indices)

# Attention modules using Keras layers
def add_attention_module(x, ratio=8):
    """Add attention module using Keras layers (Squeeze-and-Excitation)"""
    channel = x.shape[-1]
    
    # Squeeze operation (global information embedding)
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Excitation operation (adaptive recalibration)
    se = tf.keras.layers.Dense(channel // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(channel, activation='sigmoid')(se)
    
    # Reshape for broadcasting using Keras reshape layer
    se = tf.keras.layers.Reshape((1, 1, channel))(se)
    
    # Scale the input using Keras multiply layer
    x = tf.keras.layers.Multiply()([x, se])
    
    return x

def add_spatial_attention(x, kernel_size=7):
    """Add spatial attention module using Keras layers"""
    # Max pooling along channel axis using Lambda layer
    max_pool = tf.keras.layers.Lambda(
        lambda x: tf.reduce_max(x, axis=3, keepdims=True)
    )(x)
    
    # Average pooling along channel axis using Lambda layer
    avg_pool = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=3, keepdims=True)
    )(x)
    
    # Concatenate features using Keras concatenate layer
    concat = tf.keras.layers.Concatenate(axis=3)([max_pool, avg_pool])
    
    # Apply convolution using Keras Conv2D
    attention = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(concat)
    
    # Apply attention using Keras multiply layer
    x = tf.keras.layers.Multiply()([x, attention])
    
    return x

# Create DenseNet model with built-in attention
def create_densenet_model(use_attention=True):
    """Create a DenseNet121-based model using built-in components"""
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add Gaussian noise for robustness
    x = tf.keras.layers.GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    
    # Apply attention if enabled
    if use_attention:
        x = add_attention_module(x, ratio=8)
        x = add_spatial_attention(x, kernel_size=7)
    
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0015))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0015))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

# Create ResNet model with built-in attention
def create_resnet_model(use_attention=True):
    """Create a ResNet50V2-based model using built-in components"""
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add Gaussian noise for robustness
    x = tf.keras.layers.GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    
    # Apply attention if enabled
    if use_attention:
        x = add_attention_module(x, ratio=8)
        x = add_spatial_attention(x, kernel_size=7)
    
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

# Create EfficientNet model with built-in attention
def create_efficientnet_model(use_attention=True):
    """Create an EfficientNetB0-based model using built-in components"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add Gaussian noise for robustness
    x = tf.keras.layers.GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    
    # Apply attention if enabled
    if use_attention:
        x = add_attention_module(x, ratio=8)
        x = add_spatial_attention(x, kernel_size=7)
    
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

# Create ensemble model
def create_ensemble_model():
    """Create an ensemble of multiple model architectures"""
    # Create base models
    densenet_base = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    resnet_base = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    efficientnet_base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Common input
    input_layer = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # DenseNet branch
    d = tf.keras.layers.GaussianNoise(0.1)(input_layer)
    d = densenet_base(d)
    d = add_attention_module(d)
    d = add_spatial_attention(d)
    d = tf.keras.layers.GlobalAveragePooling2D()(d)
    d = tf.keras.layers.Dense(256, activation='relu')(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.Dropout(0.5)(d)
    d_output = tf.keras.layers.Dense(1, activation='sigmoid', name='densenet_output')(d)
    
    # ResNet branch
    r = tf.keras.layers.GaussianNoise(0.1)(input_layer)
    r = resnet_base(r)
    r = add_attention_module(r)
    r = add_spatial_attention(r)
    r = tf.keras.layers.GlobalAveragePooling2D()(r)
    r = tf.keras.layers.Dense(256, activation='relu')(r)
    r = tf.keras.layers.BatchNormalization()(r)
    r = tf.keras.layers.Dropout(0.5)(r)
    r_output = tf.keras.layers.Dense(1, activation='sigmoid', name='resnet_output')(r)
    
    # EfficientNet branch
    e = tf.keras.layers.GaussianNoise(0.1)(input_layer)
    e = efficientnet_base(e)
    e = add_attention_module(e)
    e = add_spatial_attention(e)
    e = tf.keras.layers.GlobalAveragePooling2D()(e)
    e = tf.keras.layers.Dense(256, activation='relu')(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Dropout(0.5)(e)
    e_output = tf.keras.layers.Dense(1, activation='sigmoid', name='efficientnet_output')(e)
    
    # Average the outputs
    ensemble_output = tf.keras.layers.Average()([d_output, r_output, e_output])
    
    # Create ensemble model
    ensemble_model = tf.keras.models.Model(inputs=input_layer, outputs=ensemble_output)
    
    # Compile model
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return ensemble_model

# Test-time augmentation
def test_time_augmentation(model, image, num_augmentations=5):
    """Apply test-time augmentation for more robust predictions"""
    # Create augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Expand dimensions for batch
    image_batch = np.expand_dims(image, axis=0)
    
    # Initialize predictions
    predictions = []
    
    # Original prediction
    predictions.append(model.predict(image_batch)[0])
    
    # Augmented predictions
    augmented = datagen.flow(image_batch, batch_size=1)
    for i in range(num_augmentations):
        aug_image = next(augmented)[0]
        pred = model.predict(np.expand_dims(aug_image, axis=0))[0]
        predictions.append(pred)
    
    # Return average prediction
    return np.mean(predictions, axis=0)

# Create dataset
def create_binary_dataset(base_dir, mode):
    """Create a dataframe of image paths and binary labels (0=normal, 1=cancer)"""
    directory = os.path.join(base_dir, mode)
    
    # Get all classes
    cancer_classes = [c for c in os.listdir(directory) if c != 'normal']
    all_classes = ['normal'] + cancer_classes
    
    paths = []
    labels = []
    
    for class_name in all_classes:
        class_dir = os.path.join(directory, class_name)
        if os.path.exists(class_dir):
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
            paths.extend(files)
            
            # 0 for normal, 1 for any cancer type
            label = 0 if class_name == 'normal' else 1
            labels.extend([label] * len(files))
    
    return pd.DataFrame({'filename': paths, 'class': labels})

# Create datasets
train_df = create_binary_dataset(BASE_DIR, 'train')
valid_df = create_binary_dataset(BASE_DIR, 'valid')
test_df = create_binary_dataset(BASE_DIR, 'test')

# Display class distribution
print("Class distribution in datasets:")
print(f"Training: {train_df['class'].value_counts()}")
print(f"Validation: {valid_df['class'].value_counts()}")
print(f"Testing: {test_df['class'].value_counts()}")

# Create data generators
train_generator = BalancedGenerator(
    train_df, 
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    augment=True,
    apply_mixup=APPLY_MIXUP
)

valid_generator = BalancedGenerator(
    valid_df,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    augment=False,
    shuffle=False
)

test_generator = BalancedGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    augment=False,
    shuffle=False
)

# Create callbacks
callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, f'binary_model_{MODEL_ARCHITECTURE}_checkpoint.keras'),
        monitor='val_auc',
        save_best_only=True,
        verbose=1
    )
]

# Train model based on selected architecture
if MODEL_ARCHITECTURE == "ensemble":
    # Create ensemble model
    model = create_ensemble_model()
    
    # Train ensemble model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.batches_per_epoch,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    history_combined = history.history
    
else:
    # Create single architecture model
    if MODEL_ARCHITECTURE == "densenet":
        model, base_model = create_densenet_model(use_attention=True)
    elif MODEL_ARCHITECTURE == "resnet":
        model, base_model = create_resnet_model(use_attention=True)
    elif MODEL_ARCHITECTURE == "efficientnet":
        model, base_model = create_efficientnet_model(use_attention=True)
    
    # Phase 1: Train only the top layers
    print("Phase 1: Training top layers with frozen base model...")
    history_phase1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.batches_per_epoch,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune the model
    print("Phase 2: Fine-tuning the model...")
    
    # Unfreeze the base model but keep early layers frozen
    base_model.trainable = True
    
    # Progressive unfreezing
    if MODEL_ARCHITECTURE == "densenet":
        for layer in base_model.layers[:len(base_model.layers)//2]:
            layer.trainable = False
    elif MODEL_ARCHITECTURE == "resnet":
        for layer in base_model.layers[:-50]:
            layer.trainable = False
    elif MODEL_ARCHITECTURE == "efficientnet":
        for layer in base_model.layers[:-30]:
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss=sigmoid_focal_crossentropy(),
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
    
    # Combine histories
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

# Evaluate model
print("\n=== Final Evaluation on Test Set ===")
test_results = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")
print(f"Test Precision: {test_results[3]:.4f}")
print(f"Test Recall: {test_results[4]:.4f}")

# Get predictions with test-time augmentation
test_steps = len(test_generator)
test_y_true = []
test_y_pred = []

for i in range(test_steps):
    x_batch, y_batch = test_generator[i]
    test_y_true.extend(y_batch)
    
    # Apply test-time augmentation for each image in batch
    batch_preds = []
    for j in range(len(x_batch)):
        # Get predictions with test-time augmentation
        pred = test_time_augmentation(model, x_batch[j], num_augmentations=3)
        batch_preds.append(pred)
    
    test_y_pred.extend(batch_preds)

# Convert to numpy arrays
test_y_true = np.array(test_y_true).flatten()
test_y_pred = np.array(test_y_pred).flatten()

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(test_y_true, test_y_pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold from ROC analysis: {optimal_threshold:.4f}")

# Apply optimal threshold
test_y_pred_binary = (test_y_pred > optimal_threshold).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(test_y_true, test_y_pred_binary)

# Print classification report
print("\nClassification Report (Using Optimal Threshold):")
print(classification_report(test_y_true, test_y_pred_binary, target_names=['Normal', 'Cancer']))

# Calculate clinical metrics
def calculate_clinical_metrics(y_true, y_pred, threshold=0.5):
   """Calculate clinically relevant metrics for model evaluation"""
   # Apply threshold
   y_pred_binary = (y_pred > threshold).astype(int)
   
   # Calculate confusion matrix elements
   tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
   
   # Calculate metrics
   sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
   specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
   ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
   npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
   f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
   dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else 0
   nns = 1 / (sensitivity * (sum(y_true) / len(y_true))) if (sensitivity * (sum(y_true) / len(y_true))) > 0 else 0
   
   return {
       "Sensitivity (Recall)": sensitivity,
       "Specificity": specificity,
       "Precision (PPV)": ppv,
       "Negative Predictive Value": npv,
       "F1 Score": f1,
       "Diagnostic Odds Ratio": dor,
       "Number Needed to Screen": nns
   }

# Calculate metrics using optimal threshold
clinical_metrics = calculate_clinical_metrics(test_y_true, test_y_pred, threshold=optimal_threshold)

# Print clinical metrics
print("\nClinical Performance Metrics:")
for metric, value in clinical_metrics.items():
   print(f"{metric}: {value:.4f}")

# Enhanced prediction function with test-time augmentation
def predict_cancer(model, image_path, threshold=None):
   """
   Predict cancer with test-time augmentation
   
   Args:
       model: Loaded model
       image_path: Path to image file
       threshold: Classification threshold (default: optimal threshold)
       
   Returns:
       Prediction results
   """
   if threshold is None:
       threshold = optimal_threshold
   
   # Load and preprocess image
   img = Image.open(image_path)
   img = img.resize((IMG_SIZE, IMG_SIZE))
   img = img.convert('RGB')
   img_array = np.array(img) / 255.0
   
   # Apply test-time augmentation
   prediction = test_time_augmentation(model, img_array, num_augmentations=5)
   
   # Create result dictionary
   result = {
       "prediction": "Cancer" if prediction > threshold else "Normal",
       "confidence": float(prediction if prediction > threshold else 1 - prediction),
       "cancer_probability": float(prediction),
       "threshold_used": float(threshold)
   }
   
   return result

# Save model in multiple formats
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save in Keras format
model_keras_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_{timestamp}.keras')
model.save(model_keras_path)
print(f"Model saved in Keras format: {model_keras_path}")

# Save in TensorFlow SavedModel format
model_tf_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_tf_{timestamp}')
tf.saved_model.save(model, model_tf_path)
print(f"Model saved in TensorFlow SavedModel format: {model_tf_path}")

# Save in TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_{timestamp}.tflite')
with open(tflite_path, 'wb') as f:
   f.write(tflite_model)
print(f"Model saved in TFLite format: {tflite_path}")

# Save in H5 format for compatibility
model_h5_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_{timestamp}.h5')
model.save(model_h5_path)
print(f"Model saved in H5 format: {model_h5_path}")

# Summary
print("\n=== Chest Cancer Detection Model Summary ===")
print(f"Architecture: {MODEL_ARCHITECTURE}")
print(f"Input Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Training Images: {len(train_df)}")
print(f"Validation Images: {len(valid_df)}")
print(f"Test Images: {len(test_df)}")
print(f"Training Class Balance: {dict(train_df['class'].value_counts())}")
print(f"Final Test Accuracy: {test_results[1]:.4f}")
print(f"Final Test AUC: {test_results[2]:.4f}")
print(f"Optimal Classification Threshold: {optimal_threshold:.4f}")

# Save metrics
current_metrics = {
   'architecture': MODEL_ARCHITECTURE,
   'accuracy': float(test_results[1]),
   'auc': float(test_results[2]),
   'precision': float(test_results[3]),
   'recall': float(test_results[4]),
   'optimal_threshold': float(optimal_threshold),
   'timestamp': timestamp
}

# Save metrics as JSON
import json
with open(os.path.join(PLOTS_DIR, f'metrics_{MODEL_ARCHITECTURE}_{timestamp}.json'), 'w') as f:
   json.dump(current_metrics, f)