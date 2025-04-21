import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, GaussianNoise
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.regularizers import l2
# Fix for the import error - use scikit-learn instead
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image, ImageEnhance, ImageOps


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
DATA_DIR = "/kaggle/input/chest-ctscan-images/Data"  # Updated path based on your shared location
MODEL_PATH = "chest_ct_binary_classifier.keras"

# Create directory to save models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Define custom preprocessing for advanced augmentation
def custom_preprocessing(img):
    """Apply advanced preprocessing techniques to a single image"""
    # Convert to PIL Image if it's a numpy array
    if not isinstance(img, Image.Image):
        img = Image.fromarray((img * 255).astype(np.uint8))
    
    # Randomly apply CLAHE
    if np.random.random() < 0.3:
        img = ImageOps.equalize(img)
    
    # Randomly adjust contrast
    if np.random.random() < 0.3:
        factor = np.random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
    
    # Randomly adjust sharpness
    if np.random.random() < 0.3:
        factor = np.random.uniform(0.5, 2.0)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
    
    return np.array(img) / 255.0

# Create data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='reflect',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Define cancer and normal class paths
cancer_classes = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

# Prepare dataframes for the data generators
train_files = []
train_labels = []
valid_files = []
valid_labels = []
test_files = []
test_labels = []

# Add normal class
for img_file in os.listdir(os.path.join(DATA_DIR, 'train', 'normal')):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(DATA_DIR, 'train', 'normal', img_file)
        train_files.append(img_path)
        train_labels.append(0)  # 0 for normal

for img_file in os.listdir(os.path.join(DATA_DIR, 'valid', 'normal')):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(DATA_DIR, 'valid', 'normal', img_file)
        valid_files.append(img_path)
        valid_labels.append(0)  # 0 for normal

for img_file in os.listdir(os.path.join(DATA_DIR, 'test', 'normal')):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(DATA_DIR, 'test', 'normal', img_file)
        test_files.append(img_path)
        test_labels.append(0)  # 0 for normal

# Add cancer classes (all labeled as 1)
for cancer_class in cancer_classes:
    # Training data
    class_dir = os.path.join(DATA_DIR, 'train', cancer_class)
    if os.path.exists(class_dir):
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                train_files.append(img_path)
                train_labels.append(1)  # 1 for cancer
    
    # Validation data
    class_dir = os.path.join(DATA_DIR, 'valid', cancer_class)
    if os.path.exists(class_dir):
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                valid_files.append(img_path)
                valid_labels.append(1)  # 1 for cancer

# For test data - the structure is different with just the cancer type name
test_cancer_classes = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma']
for cancer_class in test_cancer_classes:
    test_class_dir = os.path.join(DATA_DIR, 'test', cancer_class)
    if os.path.exists(test_class_dir):
        for img_file in os.listdir(test_class_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_class_dir, img_file)
                test_files.append(img_path)
                test_labels.append(1)  # 1 for cancer

# Create dataframes
train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
valid_df = pd.DataFrame({'filename': valid_files, 'class': valid_labels})
test_df = pd.DataFrame({'filename': test_files, 'class': test_labels})

print(f"Training set: {len(train_df)} images")
print(f"Validation set: {len(valid_df)} images")
print(f"Test set: {len(test_df)} images")
print(f"Training class distribution: {train_df['class'].value_counts()}")
print(f"Validation class distribution: {valid_df['class'].value_counts()}")
print(f"Test class distribution: {test_df['class'].value_counts()}")

# Create balanced generators
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['class']), y=train_df['class'])
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weight_dict}")

# Convert class labels to strings for compatibility with binary class_mode
train_df['class'] = train_df['class'].astype(str)
valid_df['class'] = valid_df['class'].astype(str)
test_df['class'] = test_df['class'].astype(str)

# Flow from dataframe
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col="filename",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filename",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Create binary classification model using DenseNet121
def create_binary_model():
    # Load pre-trained DenseNet121
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create model architecture
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = GaussianNoise(0.1)(inputs)  # Add noise for better generalization
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Add regularization layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Binary output
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model, base_model

# Create and compile the model
print("Creating model...")
model, base_model = create_binary_model()
model.summary()

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
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

# First phase: Train with frozen base model
print("Phase 1: Training top layers...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=8,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Second phase: Fine-tune by unfreezing some layers
print("Phase 2: Fine-tuning the model...")
# Unfreeze the top layers of the base model
base_model.trainable = True
# Keep first 100 layers frozen, unfreeze the rest for fine-tuning
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Continue training
fine_tuning_history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1,
    initial_epoch=len(history.history['loss'])
)

# Combine histories for plotting
history_dict = {}
for metric in ['accuracy', 'loss', 'auc', 'precision', 'recall', 
               'val_accuracy', 'val_loss', 'val_auc', 'val_precision', 'val_recall']:
    if metric in history.history and metric in fine_tuning_history.history:
        # Handle possible name changes
        alt_metric = metric
        if metric not in fine_tuning_history.history and f"{metric}_1" in fine_tuning_history.history:
            alt_metric = f"{metric}_1"
        
        history_dict[metric] = history.history[metric] + fine_tuning_history.history[alt_metric]

# Plot training history
plt.figure(figsize=(16, 10))

# Plot accuracy
plt.subplot(2, 2, 1)
plt.plot(history_dict['accuracy'], label='Train')
plt.plot(history_dict['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()
plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning start')

# Plot loss
plt.subplot(2, 2, 2)
plt.plot(history_dict['loss'], label='Train')
plt.plot(history_dict['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--')

# Plot AUC
plt.subplot(2, 2, 3)
plt.plot(history_dict['auc'], label='Train')
plt.plot(history_dict['val_auc'], label='Validation')
plt.title('AUC')
plt.legend()
plt.axvline(x=len(history.history['auc'])-1, color='r', linestyle='--')

# Plot Precision-Recall
plt.subplot(2, 2, 4)
plt.plot(history_dict['precision'], label='Train Precision')
plt.plot(history_dict['val_precision'], label='Validation Precision')
plt.plot(history_dict['recall'], label='Train Recall')
plt.plot(history_dict['val_recall'], label='Validation Recall')
plt.title('Precision and Recall')
plt.legend()
plt.axvline(x=len(history.history['precision'])-1, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('plots/training_history.png')

# Evaluate the model on test set
print("Evaluating model on test set...")
test_results = model.evaluate(test_generator, verbose=1)
print(f"Test results: {dict(zip(model.metrics_names, test_results))}")

# Save the model in different formats
print(f"Saving model...")

# Save model in Keras format
model.save(f'models/{MODEL_PATH}')
print(f"Model saved as models/{MODEL_PATH}")

# Save model in TFLite format for mobile/edge deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('models/chest_ct_binary_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as models/chest_ct_binary_model.tflite")

# Create a simple function to test the model with a single image
def predict_cancer(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    class_name = "Cancer" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return {
        "prediction": class_name,
        "confidence": float(confidence),
        "raw_score": float(prediction)
    }

# Test the model with a sample image
if len(test_files) > 0:
    sample_img = test_files[0]
    result = predict_cancer(sample_img)
    print(f"Sample prediction for {sample_img}:")
    print(result)

print("Model training and saving complete.")