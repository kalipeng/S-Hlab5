# Azure ML Model Training Script
# TECHIN515 Lab 5 - Edge-Cloud Offloading

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from azureml.core import Workspace, Dataset
import joblib

# Connect to Azure ML workspace
ws = Workspace.from_config()
print(f"Connected to workspace: {ws.name}")

# Load dataset from Azure Blob Storage
# Update this path to match your dataset location
dataset = Dataset.File.from_files(path=(ws.get_default_datastore(), 'UI/2025-04-05_231528_UTC/**'))
mount_point = dataset.mount()
mount_point.start()

print(f"Dataset mounted at: {mount_point.mount_point}")

# Data loading and preprocessing function
def load_gesture_data(data_path):
    """
    Load gesture data from CSV files
    Expected structure: gesture_class/samples.csv
    """
    features = []
    labels = []
    
    for gesture_class in os.listdir(data_path):
        class_path = os.path.join(data_path, gesture_class)
        if os.path.isdir(class_path):
            print(f"Loading {gesture_class} samples...")
            
            for file in os.listdir(class_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(class_path, file)
                    try:
                        # Read CSV file
                        df = pd.read_csv(file_path)
                        
                        # Extract accelerometer data (assuming columns: ax, ay, az)
                        if 'ax' in df.columns and 'ay' in df.columns and 'az' in df.columns:
                            # Flatten the accelerometer data
                            sample_features = df[['ax', 'ay', 'az']].values.flatten()
                            features.append(sample_features)
                            labels.append(gesture_class)
                        else:
                            print(f"Warning: {file_path} missing required columns")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    return np.array(features), np.array(labels)

# Load the training data
data_path = mount_point.mount_point
X, y = load_gesture_data(data_path)

print(f"Loaded {len(X)} samples with {len(set(y))} classes")
print(f"Feature shape: {X[0].shape}")
print(f"Classes: {set(y)}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"Label mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")

# Normalize features
X_normalized = X / np.max(np.abs(X))

# Reshape for CNN input (samples, timesteps, features)
timesteps = X.shape[1] // 3  # Number of time steps
X_reshaped = X_normalized.reshape(X.shape[0], timesteps, 3)

print(f"Reshaped data: {X_reshaped.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Build the model
def create_gesture_model(input_shape, num_classes):
    """Create a CNN model for gesture recognition"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional layers
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        layers.Conv1D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile model
input_shape = (X_train.shape[1], X_train.shape[2])
model = create_gesture_model(input_shape, num_classes)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Training callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]

# Train the model
print("Starting training...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Save the model
model.save('wand_model.h5')
print("Model saved as wand_model.h5")

# Save label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved as label_encoder.pkl")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Print final results
print("\n=== Training Complete ===")
print(f"Final test accuracy: {test_accuracy:.4f}")
print(f"Number of classes: {num_classes}")
print(f"Classes: {list(label_encoder.classes_)}")
print(f"Model input shape: {model.input_shape}")
print(f"Model saved successfully!")

# Unmount dataset
mount_point.stop()