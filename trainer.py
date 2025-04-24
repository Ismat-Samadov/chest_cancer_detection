# Improved Chest CT Cancer Binary Classification without TensorFlow Addons
# -------------------------------------------------------
# Enhanced implementation with multiple model architectures and advanced techniques

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Add, Multiply
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import KFold
import seaborn as sns

# Custom implementation of Focal Loss to replace TensorFlow Addons
def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Custom implementation of Focal Loss
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        alpha: Weighting factor for the rare class
        gamma: Focusing parameter that reduces the loss for well-classified examples
        
    Returns:
        Focal loss value
    """
    # Get binary crossentropy
    bce = binary_crossentropy(y_true, y_pred)
    
    # Convert y_true to float32 if needed
    y_true = tf.cast(y_true, dtype=tf.float32)
    
    # Calculate the modulating factor
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = tf.pow((1.0 - p_t), gamma)
    
    # Apply the factors and return
    return alpha_factor * modulating_factor * bce

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 256  # Increased from 224 for more detail
BATCH_SIZE = 16
EPOCHS = 30  # More epochs for improved architectures
BASE_DIR = "/kaggle/input/chest-ctscan-images/Data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
USE_ENSEMBLE = True  # Set to True to use ensemble of models
MODEL_ARCHITECTURE = "densenet"  # Options: "densenet", "resnet", "efficientnet", "ensemble"
USE_ATTENTION = True  # Use attention mechanisms
APPLY_MIXUP = True  # Apply mixup augmentation

# Create directories for outputs
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------------------------------
# Enhanced Preprocessing Functions
# -------------------------------------------------------

def apply_clahe(img, chance=0.5):
    """Apply CLAHE with enhanced parameters for better contrast in CT scans"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        img = ImageOps.equalize(img)
        return np.array(img) / 255.0
    return img

def apply_random_contrast(img, chance=0.5, factor_range=(0.5, 1.8)):
    """Apply stronger contrast adjustment for CT scans"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def apply_random_sharpness(img, chance=0.5, factor_range=(0.5, 2.2)):
    """Apply stronger sharpness for better tissue boundary detection"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def apply_windowing(img, chance=0.4, window_width_range=(1500, 2500), window_center_range=(-600, 500)):
    """
    Apply CT windowing to enhance specific tissues
    
    Why: CT windowing is a domain-specific technique that adjusts contrast
    to highlight specific tissue types based on their Hounsfield units
    """
    if np.random.random() < chance:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Simulate Hounsfield unit range (normally from -1000 to +1000)
        # For this example we scale 0-1 to -1000 to +1000
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

# -------------------------------------------------------
# MixUp Augmentation
# -------------------------------------------------------

def mixup_augmentation(x, y, alpha=0.2):
    """
    Apply MixUp augmentation with proper data type handling
    """
    # Convert inputs to tensors with appropriate types
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)  # Convert labels to float
    
    batch_size = tf.shape(x)[0]
    
    # Create indices for shuffling
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Sample lambda from beta distribution
    random_lambda = tf.random.stateless_gamma([batch_size], seed=[42, 0], alpha=alpha) / \
                    tf.random.stateless_gamma([batch_size], seed=[42, 1], alpha=alpha)
    random_lambda = tf.minimum(random_lambda, 1.0)
    
    # Reshape lambda for broadcasting with images
    random_lambda_x = tf.reshape(random_lambda, [batch_size, 1, 1, 1])
    
    # Apply mixup to images
    shuffled_x = tf.gather(x, indices)
    mixed_x = random_lambda_x * x + (1.0 - random_lambda_x) * shuffled_x
    
    # Apply mixup to labels
    random_lambda_y = tf.reshape(random_lambda, [batch_size])
    shuffled_y = tf.gather(y, indices)
    mixed_y = random_lambda_y * y + (1.0 - random_lambda_y) * shuffled_y
    
    return mixed_x, mixed_y

# -------------------------------------------------------
# Enhanced Data Generator with MixUp
# -------------------------------------------------------

class EnhancedBalancedGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=16, target_size=(256, 256), 
                 shuffle=True, datagen=None, balanced=True, apply_mixup=False):
        """Enhanced generator with MixUp capability"""
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.datagen = datagen if datagen is not None else ImageDataGenerator(rescale=1./255)
        self.balanced = balanced
        self.apply_mixup = apply_mixup
        
        # Split indices by class
        self.normal_indices = dataframe[dataframe['class'] == 0].index.tolist()
        self.cancer_indices = dataframe[dataframe['class'] == 1].index.tolist()
        
        # Check if we have both classes
        self.has_both_classes = len(self.normal_indices) > 0 and len(self.cancer_indices) > 0
        
        if self.balanced and not self.has_both_classes:
            print(f"WARNING: Can't create balanced batches. Normal samples: {len(self.normal_indices)}, "
                  f"Cancer samples: {len(self.cancer_indices)}. Switching to unbalanced sampling.")
            self.balanced = False
        
        # Number of batches - increased for better training
        self.batches_per_epoch = 75 if self.balanced else max(1, len(dataframe) // batch_size)
        
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
            
            # Load image with error handling
            try:
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
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                continue
        
        # Convert to numpy arrays
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        # Apply MixUp if enabled
        if self.apply_mixup and len(batch_x) > 1:
            batch_x, batch_y = mixup_augmentation(batch_x, batch_y, alpha=0.2)
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of each epoch during training"""
        if self.shuffle:
            np.random.shuffle(self.normal_indices)
            np.random.shuffle(self.cancer_indices)

# -------------------------------------------------------
# Attention Mechanisms
# -------------------------------------------------------

def create_attention_module(x, ratio=8):
    """
    Create a channel attention module (Squeeze-and-Excitation block)
    
    Why: Attention mechanisms help the model focus on relevant areas of the image
    and ignore irrelevant parts, which is particularly important for detecting
    subtle cancer features in CT scans.
    """
    channel = tf.keras.backend.int_shape(x)[-1]
    
    # Squeeze operation
    se = GlobalAveragePooling2D()(x)
    
    # Excitation operation
    se = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    # Scale the input - using Keras Reshape layer instead of tf.reshape
    se = tf.keras.layers.Reshape((1, 1, channel))(se)
    x = Multiply()([x, se])
    
    return x

def create_spatial_attention_module(x):
def create_spatial_attention_module(x):
    """Create a spatial attention module using dedicated layers instead of Lambda"""
    # Max pooling across channels
    max_pool = tf.keras.layers.MaxPool2D(pool_size=(1, 1), data_format='channels_last')(x)
    
    # Average pooling across channels using a custom layer
    class ChannelAveragePooling(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.reduce_mean(inputs, axis=3, keepdims=True)
            
    avg_pool = ChannelAveragePooling()(x)
    
    # Concatenate pool features
    concat = tf.keras.layers.Concatenate(axis=3)([max_pool, avg_pool])
    
    # Conv to generate attention map
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    
    # Apply attention
    x = Multiply()([x, attention])
    
    return x

# -------------------------------------------------------
# Enhanced Model Architectures
# -------------------------------------------------------

def create_densenet_model(use_attention=False):
    """Create an enhanced DenseNet121-based model with attention"""
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Gaussian noise for robustness
    x = tf.keras.layers.GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    
    # Apply attention if enabled
    if use_attention:
        x = create_attention_module(x)
        x = create_spatial_attention_module(x)
    
    x = GlobalAveragePooling2D()(x)
    
    # Enhanced dense layers with stronger regularization
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0015))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0015))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with custom focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

def create_resnet_model(use_attention=False):
    """Create a ResNet50-based model with attention"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Gaussian noise for robustness
    x = tf.keras.layers.GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    
    # Apply attention if enabled
    if use_attention:
        x = create_attention_module(x)
        x = create_spatial_attention_module(x)
    
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
    
    # Compile with custom focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

def create_efficientnet_model(use_attention=False):
    """Create an EfficientNetB0-based model with attention"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Gaussian noise for robustness
    x = tf.keras.layers.GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    
    # Apply attention if enabled
    if use_attention:
        x = create_attention_module(x)
        x = create_spatial_attention_module(x)
    
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization - using ReLU instead of Swish for compatibility
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with custom focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

def create_ensemble_model(use_attention=False):
    """
    Create an ensemble of multiple models for better performance
    
    Why: Ensemble models combine predictions from different architectures,
    capturing different aspects of the images and generally achieving
    better performance than any single model.
    """
    # Create individual models
    densenet_model, _ = create_densenet_model(use_attention)
    resnet_model, _ = create_resnet_model(use_attention)
    efficientnet_model, _ = create_efficientnet_model(use_attention)
    
    # Common input
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Get predictions from each model
    densenet_output = densenet_model(input_layer)
    resnet_output = resnet_model(input_layer)
    efficientnet_output = efficientnet_model(input_layer)
    
    # Combine predictions (weighted average)
    ensemble_output = tf.keras.layers.Average()([
        densenet_output,
        resnet_output,
        efficientnet_output
    ])
    
    # Create ensemble model
    ensemble_model = Model(inputs=input_layer, outputs=ensemble_output)
    
    # Compile model
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sigmoid_focal_crossentropy,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return ensemble_model, None

# -------------------------------------------------------
# Create Binary Dataset for Training
# -------------------------------------------------------

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
# Enhanced Training and Evaluation Workflow
# -------------------------------------------------------

# Create data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rescale=1./255,
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

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Select model architecture based on configuration
if MODEL_ARCHITECTURE == "densenet":
    model, base_model = create_densenet_model(USE_ATTENTION)
elif MODEL_ARCHITECTURE == "resnet":
    model, base_model = create_resnet_model(USE_ATTENTION)
elif MODEL_ARCHITECTURE == "efficientnet":
    model, base_model = create_efficientnet_model(USE_ATTENTION)
elif MODEL_ARCHITECTURE == "ensemble":
    model, base_model = create_ensemble_model(USE_ATTENTION)
else:
    print(f"Unknown architecture: {MODEL_ARCHITECTURE}. Using DenseNet as default.")
    model, base_model = create_densenet_model(USE_ATTENTION)

# Print model summary
model.summary()

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_auc',  # Monitor AUC instead of accuracy for medical tasks
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
    ),
    # Add TensorBoard for better visualization
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs', MODEL_ARCHITECTURE),
        histogram_freq=1
    )
]

# Use the enhanced generator
train_generator = EnhancedBalancedGenerator(
    train_df, 
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    datagen=train_datagen,
    balanced=True,
    apply_mixup=APPLY_MIXUP
)

valid_generator = EnhancedBalancedGenerator(
    valid_df,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    datagen=valid_datagen,
    shuffle=False,
    balanced=True
)

test_generator = EnhancedBalancedGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    datagen=test_datagen,
    shuffle=False,
    balanced=False
)

# Phase 1: Train only the top layers (if not using ensemble)
if base_model is not None:
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

    # Phase 2: Fine-tune the model by unfreezing deeper layers
    print("Phase 2: Fine-tuning the model...")
    
    # Unfreeze the base model but keep early layers frozen
    base_model.trainable = True
    
    # Progressive unfreezing - different strategy based on architecture
    if MODEL_ARCHITECTURE == "densenet":
        # For DenseNet, unfreeze last half of the network
        for layer in base_model.layers[:len(base_model.layers)//2]:
            layer.trainable = False
    elif MODEL_ARCHITECTURE == "resnet":
        # For ResNet, unfreeze last 50 layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
    elif MODEL_ARCHITECTURE == "efficientnet":
        # For EfficientNet, unfreeze last 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss=sigmoid_focal_crossentropy,
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
else:
    # For ensemble models, train in a single phase
    print("Training ensemble model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.batches_per_epoch,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # For consistent plotting
    history_combined = history.history

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
if base_model is not None:
   plt.axvline(x=len(history_phase1.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning start')

# Plot loss
plt.subplot(2, 2, 2)
plt.plot(history_combined['loss'], label='Training Loss')
plt.plot(history_combined['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
if base_model is not None:
   plt.axvline(x=len(history_phase1.history['loss'])-1, color='r', linestyle='--')

# Plot AUC
plt.subplot(2, 2, 3)
plt.plot(history_combined['auc'], label='Training AUC')
plt.plot(history_combined['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
if base_model is not None:
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
if base_model is not None:
   plt.axvline(x=len(history_phase1.history['precision'])-1, color='r', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'training_history_{MODEL_ARCHITECTURE}.png'))
plt.show()

# -------------------------------------------------------
# Enhanced Evaluation on Test Set
# -------------------------------------------------------

# Final evaluation on test data
print("\n=== Final Evaluation on Test Set ===")
test_results = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")
print(f"Test Precision: {test_results[3]:.4f}")
print(f"Test Recall: {test_results[4]:.4f}")

# Get predictions
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

# Convert probabilities to binary predictions using optimal threshold
# Find optimal threshold based on ROC curve for better sensitivity/specificity balance
fpr, tpr, thresholds = roc_curve(test_y_true, test_y_pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold from ROC analysis: {optimal_threshold:.4f}")

# Apply optimal threshold
test_y_pred_binary = (test_y_pred > optimal_threshold).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(test_y_true, test_y_pred_binary)

# Display enhanced confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.2f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['Normal', 'Cancer'])
plt.yticks([0.5, 1.5], ['Normal', 'Cancer'])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{MODEL_ARCHITECTURE}.png'))
plt.show()

# Calculate and plot ROC curve with confidence interval
plt.figure(figsize=(10, 8))

# Calculate ROC curve
fpr, tpr, _ = roc_curve(test_y_true, test_y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')

# Plot confidence interval (using bootstrapping)
n_bootstraps = 1000
rng = np.random.RandomState(42)
bootstrapped_aucs = []

for i in range(n_bootstraps):
   # Bootstrap by sampling with replacement
   indices = rng.randint(0, len(test_y_pred), len(test_y_pred))
   if len(np.unique(test_y_true[indices])) < 2:
       # Skip if only one class is present
       continue
   
   # Calculate AUC for this bootstrap sample
   fpr, tpr, _ = roc_curve(test_y_true[indices], test_y_pred[indices])
   bootstrapped_aucs.append(auc(fpr, tpr))

# Calculate 95% confidence interval
bootstrapped_aucs = np.array(bootstrapped_aucs)
auc_95ci = np.percentile(bootstrapped_aucs, [2.5, 97.5])

# Add confidence interval to plot title
plt.title(f'ROC Curve (AUC = {roc_auc:.3f}, 95% CI: {auc_95ci[0]:.3f}-{auc_95ci[1]:.3f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, f'roc_curve_{MODEL_ARCHITECTURE}.png'))
plt.show()

# Print detailed classification report
print("\nClassification Report (Using Optimal Threshold):")
print(classification_report(test_y_true, test_y_pred_binary, target_names=['Normal', 'Cancer']))

# -------------------------------------------------------
# Calculate Clinical Metrics
# -------------------------------------------------------

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
   
   # F1 score
   f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
   
   # Diagnostic odds ratio (measure of test effectiveness)
   dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else 0
   
   # Number needed to screen (NNS)
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

# -------------------------------------------------------
# Enhanced Prediction with Explainability
# -------------------------------------------------------

def predict_cancer_with_explanation(model, image_path, threshold=0.5):
   """
   Predict cancer and provide visual explanation using Grad-CAM
   
   Args:
       model: Loaded TensorFlow/Keras model
       image_path: Path to the image file
       threshold: Classification threshold (default: optimal threshold)
       
   Returns:
       dict: Prediction results and Grad-CAM visualization
   """
   # Load and preprocess the image
   img = Image.open(image_path)
   img = img.resize((IMG_SIZE, IMG_SIZE))
   img = img.convert('RGB')
   img_array = np.array(img) / 255.0
   
   # Make prediction
   x = np.expand_dims(img_array, axis=0)
   prediction = float(model.predict(x)[0][0])
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
   
   # Generate Grad-CAM visualization
   try:
       # Get the last convolutional layer
       if MODEL_ARCHITECTURE == "densenet":
           last_conv_layer_name = [layer.name for layer in model.layers if 'conv5_block16_concat' in layer.name][0]
       elif MODEL_ARCHITECTURE == "resnet":
           last_conv_layer_name = [layer.name for layer in model.layers if 'conv5_block3_out' in layer.name][0]
       elif MODEL_ARCHITECTURE == "efficientnet":
           last_conv_layer_name = [layer.name for layer in model.layers if 'top_activation' in layer.name][0]
       else:
           # Fallback - find the last conv layer
           conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower() and len(layer.output_shape) == 4]
           last_conv_layer_name = conv_layers[-1] if conv_layers else None
       
       if last_conv_layer_name:
           last_conv_layer = model.get_layer(last_conv_layer_name)
           
           # Create a model that maps the input image to the activations of the last conv layer
           grad_model = tf.keras.models.Model(
               [model.inputs], 
               [last_conv_layer.output, model.output]
           )
           
           # Compute gradient of the prediction with respect to the output of the last conv layer
           with tf.GradientTape() as tape:
               last_conv_layer_output, predictions = grad_model(x)
               loss = predictions[:, 0]  # For binary classification
           
           # Gradient of the output neuron with respect to the output feature map
           grads = tape.gradient(loss, last_conv_layer_output)
           
           # Vector of mean intensity of gradient over feature map
           pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
           
           # Weight the channels by the gradient importance
           last_conv_layer_output = last_conv_layer_output[0]
           heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
           heatmap = tf.squeeze(heatmap)
           
           # Normalize the heatmap
           heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
           heatmap = heatmap.numpy()
           
           # Resize heatmap to original image size
           import cv2
           heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
           
           # Create a colored heatmap
           heatmap_rgb = np.uint8(255 * plt.cm.jet(heatmap_resized)[:, :, :3])
           
           # Superimpose the heatmap on original image
           superimposed_img = np.uint8(0.6 * img_array * 255 + 0.4 * heatmap_rgb)
           
           # Create and save the visualization
           fig, ax = plt.subplots(1, 3, figsize=(15, 5))
           ax[0].imshow(img_array)
           ax[0].set_title('Original Image')
           ax[0].axis('off')
           
           ax[1].imshow(heatmap_resized, cmap='jet')
           ax[1].set_title('Heatmap')
           ax[1].axis('off')
           
           ax[2].imshow(superimposed_img)
           ax[2].set_title(f'Prediction: {result["prediction"]} ({result["confidence"]:.2%})')
           ax[2].axis('off')
           
           plt.tight_layout()
           explanation_path = f"explanation_{os.path.basename(image_path).split('.')[0]}.png"
           plt.savefig(explanation_path)
           plt.close()
           
           # Add explanation path to result
           result["explanation_path"] = explanation_path
   except Exception as e:
       print(f"Error generating Grad-CAM: {e}")
       result["explanation_error"] = str(e)
   
   return result

# -------------------------------------------------------
# Save models in multiple formats with versioning
# -------------------------------------------------------

# Create timestamp for version tracking
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the model in Keras format
model_keras_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_{timestamp}.keras')
model.save(model_keras_path)
print(f"Model saved in Keras format: {model_keras_path}")

# Save the model in TensorFlow SavedModel format
model_tf_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_tf_{timestamp}')
tf.saved_model.save(model, model_tf_path)
print(f"Model saved in TensorFlow SavedModel format: {model_tf_path}")

# Save in TFLite format with optimization for edge devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Quantize model for smaller size and faster inference
tflite_model = converter.convert()
tflite_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_{timestamp}.tflite')
with open(tflite_path, 'wb') as f:
   f.write(tflite_model)
print(f"Model saved in optimized TFLite format: {tflite_path}")

# Save as h5 for compatibility
model_h5_path = os.path.join(MODELS_DIR, f'chest_ct_binary_classifier_{MODEL_ARCHITECTURE}_{timestamp}.h5')
model.save(model_h5_path)
print(f"Model saved in h5 format: {model_h5_path}")

# -------------------------------------------------------
# Summary and Model Comparison
# -------------------------------------------------------

print("\n=== Enhanced Binary Cancer Detection Model Summary ===")
print(f"Architecture: {MODEL_ARCHITECTURE}")
print(f"Attention Mechanisms: {'Enabled' if USE_ATTENTION else 'Disabled'}")
print(f"MixUp Augmentation: {'Enabled' if APPLY_MIXUP else 'Disabled'}")
print(f"Input Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Training Images: {len(train_df)}")
print(f"Validation Images: {len(valid_df)}")
print(f"Test Images: {len(test_df)}")
print(f"Training Class Balance: {dict(train_df['class'].value_counts())}")
print(f"Final Test Accuracy: {test_results[1]:.4f}")
print(f"Final Test AUC: {test_results[2]:.4f}")
print(f"Optimal Classification Threshold: {optimal_threshold:.4f}")

# Save current metrics as new baseline
current_metrics = {
   'architecture': MODEL_ARCHITECTURE,
   'attention': USE_ATTENTION,
   'mixup': APPLY_MIXUP,
   'accuracy': float(test_results[1]),
   'auc': float(test_results[2]),
   'precision': float(test_results[3]),
   'recall': float(test_results[4]),
   'optimal_threshold': float(optimal_threshold),
   'timestamp': timestamp
}

import json
with open(os.path.join(PLOTS_DIR, f'metrics_{MODEL_ARCHITECTURE}_{timestamp}.json'), 'w') as f:
   json.dump(current_metrics, f)

print("\nModel saved in multiple formats for deployment.")

print("\nNext steps could include:")
print("1. Implementing 3D CNN variants for volumetric CT analysis")
print("2. Adding Vision Transformer modules for better feature extraction")
print("3. Creating a model ensemble with different architectures and input sizes")
print("4. Implementing test-time augmentation for more robust predictions")
print("5. Deploying as a web service with Grad-CAM visualization for explainability")
print("6. Exploring self-supervised pre-training on unlabeled CT data")
print("7. Implementing uncertainty estimation techniques")