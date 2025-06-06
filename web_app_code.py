#!/usr/bin/env python3
"""
Flask Web Application for Gesture Recognition Model Serving
TECHIN515 Lab 5 - Edge-Cloud Offloading
"""

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
class_names = ['O', 'V', 'Z']  # Update these based on your gesture classes

def load_model():
    """Load the trained gesture recognition model"""
    global model
    try:
        model = keras.models.load_model('wand_model.h5')
        logger.info("Model loaded successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def preprocess_features(features):
    """
    Preprocess the raw features for model inference
    Args:
        features: List of raw accelerometer readings [x1, y1, z1, x2, y2, z2, ...]
    Returns:
        Preprocessed numpy array ready for model inference
    """
    try:
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)
        
        # Reshape to match model input shape
        # Assuming model expects shape (batch_size, timesteps, features)
        # where timesteps = len(features) / 3 and features = 3 (x, y, z)
        timesteps = len(features) // 3
        features_reshaped = features_array.reshape(1, timesteps, 3)
        
        logger.info(f"Features shape after preprocessing: {features_reshaped.shape}")
        return features_reshaped
    except Exception as e:
        logger.error(f"Error preprocessing features: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Gesture Recognition Server',
        'model_loaded': model is not None,
        'supported_gestures': class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict gesture from raw accelerometer data
    Expected input format:
    {
        "features": [x1, y1, z1, x2, y2, z2, ...]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing features in request',
                'expected_format': {'features': 'list of float values'}
            }), 400
        
        features = data['features']
        
        # Validate features
        if not isinstance(features, list) or len(features) == 0:
            return jsonify({
                'error': 'Features must be a non-empty list'
            }), 400
        
        if len(features) % 3 != 0:
            return jsonify({
                'error': f'Features length ({len(features)}) must be divisible by 3 (x, y, z)'
            }), 400
        
        logger.info(f"Received {len(features)} features from client")
        
        # Preprocess features
        processed_features = preprocess_features(features)
        
        # Make prediction
        predictions = model.predict(processed_features, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        predicted_gesture = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else 'unknown'
        
        # Log prediction details
        logger.info(f"Prediction: {predicted_gesture} ({confidence:.2f}%)")
        logger.info(f"Raw predictions: {predictions[0]}")
        
        # Return prediction result
        response = {
            'gesture': predicted_gesture,
            'confidence': round(confidence, 2),
            'raw_predictions': {
                class_names[i]: round(float(predictions[0][i]) * 100, 2) 
                for i in range(len(class_names))
            },
            'features_received': len(features),
            'timestamp': str(np.datetime64('now'))
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        return jsonify({
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'num_classes': len(class_names),
            'class_names': class_names,
            'model_summary': str(model.summary())
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Load the model before starting the server
        load_model()
        
        # Start the Flask server
        logger.info("Starting Flask server...")
        app.run(
            host='0.0.0.0',  # Allow external connections
            port=5000,
            debug=True,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        exit(1)