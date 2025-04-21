import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, GaussianNoise, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageEnhance, ImageOps

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20  # Reduced from 40 to prevent overtraining
DATA_DIR = "/kaggle/input/chest-ctscan-images/Data"
MODEL_PATH = "chest_ct_binary_classifier.keras"

# Create directory to save models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Define custom preprocessing functions for more advanced augmentation
def random_clahe(img, chance=0.5):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with random probability"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        img = ImageOps.equalize(img)
        return np.array(img) / 255.0
    return img

def random_contrast(img, chance=0.5, factor_range=(0.5, 1.5)):
    """Apply random contrast with given probability"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

def random_sharpness(img, chance=0.5, factor_range=(0.5, 2.0)):
    """Apply random sharpness with given probability"""
    if np.random.random() < chance:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype(np.uint8))
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
        return np.array(img) / 255.0
    return img

# Create a custom preprocessing function that combines all augmentations
def custom_preprocessing(img):
    """Apply several preprocessing techniques to a single image"""
    img = random_clahe(img, chance=0.3)
    img = random_contrast(img, chance=0.3)
    img = random_sharpness(img, chance=0.3)
    return img

# Create a more aggressive data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='reflect',
    validation_split=0.1
)

# Only rescale validation and test sets
test_datagen = ImageDataGenerator(rescale=1./255)

# Define classes for binary classification
cancer_classes = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]
all_classes = ['normal'] + cancer_classes

print("Loading training data...")
# First, get all the image paths
train_paths = []
train_labels = []
for i, class_name in enumerate(all_classes):
    class_dir = os.path.join(DATA_DIR, 'train', class_name)
    if os.path.exists(class_dir):
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        train_paths.extend(files)
        # 0 for normal, 1 for any cancer type
        label = 0 if class_name == 'normal' else 1
        train_labels.extend([label] * len(files))

# Create binary classification training data
train_df = pd.DataFrame({
    'filename': train_paths,
    'class': train_labels
})

# Verify class distribution
print(f"Training set class distribution: {train_df['class'].value_counts()}")

# Do the same for validation and test sets
valid_paths = []
valid_labels = []
for i, class_name in enumerate(all_classes):
    class_dir = os.path.join(DATA_DIR, 'valid', class_name)
    if os.path.exists(class_dir):
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        valid_paths.extend(files)
        label = 0 if class_name == 'normal' else 1
        valid_labels.extend([label] * len(files))

valid_df = pd.DataFrame({
    'filename': valid_paths,
    'class': valid_labels
})
print(f"Validation set class distribution: {valid_df['class'].value_counts()}")

test_paths = []
test_labels = []
for i, class_name in enumerate(all_classes):
    class_dir = os.path.join(DATA_DIR, 'test', class_name)
    if os.path.exists(class_dir):
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        test_paths.extend(files)
        label = 0 if class_name == 'normal' else 1
        test_labels.extend([label] * len(files))

test_df = pd.DataFrame({
    'filename': test_paths,
    'class': test_labels
})
print(f"Test set class distribution: {test_df['class'].value_counts()}")

# Attempt to fix test dataset issue by combining validation data into test when needed
# If test set is missing one class, add some samples from validation set
if 0 not in test_df['class'].values or 1 not in test_df['class'].values:
    print("WARNING: Test set is missing one or more classes. Adding validation samples to test set.")
    
    # Determine which class is missing
    missing_class = 1 if 1 not in test_df['class'].values else 0
    
    # Get samples of missing class from validation set
    missing_samples = valid_df[valid_df['class'] == missing_class]
    
    if len(missing_samples) > 0:
        # Add some validation samples to test set (use 25% of validation samples)
        num_samples_to_add = max(int(len(missing_samples) * 0.25), 1)
        samples_to_add = missing_samples.sample(n=num_samples_to_add)
        
        # Append to test dataframe
        test_df = pd.concat([test_df, samples_to_add], ignore_index=True)
        
        print(f"Added {num_samples_to_add} samples of class {missing_class} to test set.")
        print(f"Updated test set class distribution: {test_df['class'].value_counts()}")
    else:
        print(f"ERROR: Cannot find samples of class {missing_class} in validation set either.")

# Custom data generator for balanced binary classification
class BalancedBinaryDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=16, target_size=(224, 224), 
                 shuffle=True, seed=None, datagen=None, balanced=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.seed = seed
        self.datagen = datagen if datagen is not None else ImageDataGenerator(rescale=1./255)
        self.balanced = balanced
        
        # Split by class
        self.normal_indices = dataframe[dataframe['class'] == 0].index.tolist()
        self.cancer_indices = dataframe[dataframe['class'] == 1].index.tolist()
        
        # Check if we have both classes - if not, can't do balanced batches
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
                # Get augmented image - use next() instead of .next()
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

# Create custom generators
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
    datagen=test_datagen,
    shuffle=False,
    balanced=True
)

# For test generator, don't force balancing since we might not have both classes
test_generator = BalancedBinaryDataGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    datagen=test_datagen,
    shuffle=False,
    balanced=False  # Don't force balancing for test set
)

# Define model using DenseNet121 for binary classification
def create_binary_model():
    # Load pre-trained DenseNet121 model
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create new model on top with regularization
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Add noise for better generalization
    x = GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Add more regularization
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)  # Higher dropout
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)  # Higher dropout
    
    # Output layer with sigmoid for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model with binary crossentropy
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower initial learning rate for stability
        loss='binary_crossentropy',  # Binary classification loss
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

# Create binary classification model
print("Creating binary classification model with DenseNet121 architecture...")
model, base_model = create_binary_model()
model.summary()

# Define callbacks for training with increased patience
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,  # Larger reduction factor
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='models/binary_model_checkpoint.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# First phase: Train the top layers with frozen base model
print("Phase 1: Training top layers...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.batches_per_epoch,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=10,  # Shorter initial training phase
    callbacks=callbacks,
    verbose=1
)

# Second phase: Fine-tune the model by unfreezing some layers of the base model
print("Phase 2: Fine-tuning the model...")
# Unfreeze the top layers of the base model
base_model.trainable = True
# Freeze first 100 layers, unfreeze the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Even lower learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Continue training
history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=train_generator.batches_per_epoch,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Helper function to safely combine histories
def combine_histories(hist1, hist2, metric_name):
    """Safely combine histories even if metric names changed"""
    # Check for renamed metrics (e.g., 'auc' might become 'auc_1' in fine-tuning)
    metrics1 = list(hist1.history.keys())
    metrics2 = list(hist2.history.keys())
    
    m1 = metric_name
    m2 = metric_name
    
    # If metric_name not in second history, try with suffix '_1'
    if metric_name not in metrics2:
        alt_name = f"{metric_name}_1"
        if alt_name in metrics2:
            m2 = alt_name
        else:
            # If neither is found, return empty list for second history
            return hist1.history.get(m1, []) + []
            
    return hist1.history.get(m1, []) + hist2.history.get(m2, [])

# Combine the histories safely
combined_history = {
    'accuracy': combine_histories(history, history_fine_tuning, 'accuracy'),
    'val_accuracy': combine_histories(history, history_fine_tuning, 'val_accuracy'),
    'loss': combine_histories(history, history_fine_tuning, 'loss'),
    'val_loss': combine_histories(history, history_fine_tuning, 'val_loss'),
    'auc': combine_histories(history, history_fine_tuning, 'auc'),
    'val_auc': combine_histories(history, history_fine_tuning, 'val_auc'),
    'precision': combine_histories(history, history_fine_tuning, 'precision'),
    'val_precision': combine_histories(history, history_fine_tuning, 'val_precision'),
    'recall': combine_histories(history, history_fine_tuning, 'recall'),
    'val_recall': combine_histories(history, history_fine_tuning, 'val_recall')
}

# Plot training history
plt.figure(figsize=(16, 12))

# Plot accuracy
plt.subplot(2, 2, 1)
plt.plot(combined_history['accuracy'], label='Training Accuracy')
plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning start')

# Plot loss
plt.subplot(2, 2, 2)
plt.plot(combined_history['loss'], label='Training Loss')
plt.plot(combined_history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--')

# Plot AUC
plt.subplot(2, 2, 3)
plt.plot(combined_history['auc'], label='Training AUC')
plt.plot(combined_history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.axvline(x=len(history.history['auc'])-1, color='r', linestyle='--')

# Plot Precision-Recall
plt.subplot(2, 2, 4)
plt.plot(combined_history['precision'], label='Training Precision')
plt.plot(combined_history['val_precision'], label='Validation Precision')
plt.plot(combined_history['recall'], label='Training Recall')
plt.plot(combined_history['val_recall'], label='Validation Recall')
plt.title('Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.axvline(x=len(history.history['precision'])-1, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('plots/binary_training_history.png')
plt.show()

# Handle the test evaluation differently if we're missing a class
print("\nSaving and preparing model for deployment...")

# Save the model
print(f"Saving binary model to {MODEL_PATH}...")
model.save(f'models/{MODEL_PATH}')

# Save a smaller TFLite model version for deployment
print("Converting to TFLite format for deployment...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('models/chest_ct_binary_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Create a prediction function for binary classification deployment
def create_binary_prediction_model():
    """Create a function for binary classification deployment"""
    # Binary class names
    binary_class_names = ['Normal', 'Cancer']
    
    def predict_image(image_path=None, image_array=None, threshold=0.5):
        """
        Predict whether a CT scan shows cancer or normal tissue
        
        Args:
            image_path: Path to the image file
            image_array: Preprocessed image array (alternative to image_path)
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            dict: Prediction results with class name and confidence
        """
        if image_path is not None:
            # Load and preprocess the image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        elif image_array is not None:
            img_array = image_array
        else:
            raise ValueError("Either image_path or image_array must be provided")
        
        # Make prediction
        prediction = float(model.predict(img_array)[0][0])
        predicted_class = int(prediction > threshold)
        
        # Create result dictionary
        result = {
            "prediction": binary_class_names[predicted_class],
            "confidence": prediction if predicted_class == 1 else 1 - prediction,
            "cancer_probability": prediction,
            "classification_threshold": threshold
        }
        
        return result
    
    return predict_image

# Create and save binary prediction model for API deployment
binary_predict_function = create_binary_prediction_model()

# Example of API usage for binary classification
"""
# Example API Usage for Binary Classification
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the binary prediction function
binary_predict_function = create_binary_prediction_model()

@app.route('/predict', methods=['POST'])
def predict():
    threshold = request.args.get('threshold', default=0.5, type=float)
    
    if 'file' not in request.files:
        # Check if image is sent as base64
        if 'image' in request.json:
            base64_img = request.json['image']
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            result = binary_predict_function(image_array=img_array, threshold=threshold)
            return jsonify(result)
        return jsonify({'error': 'No file part or base64 image in request'}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    result = binary_predict_function(image_array=img_array, threshold=threshold)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
"""

print("Binary model training and preparation complete.")
print("The model has been saved in three formats:")
print(f"1. Full Keras model: models/{MODEL_PATH}")
print("2. TFLite model: models/chest_ct_binary_model.tflite")
print("3. Binary prediction function prepared for deployment")