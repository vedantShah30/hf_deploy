import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

def classify_image(model_zoo, img_path):
    """
    Classifies the image into one of: ['FCC', 'IR', 'Optical', 'SAR']
    """
    # 1. Get the pre-loaded classifier model
    classifier = model_zoo.classifier
    if classifier is None:
        raise ValueError("Classifier model not loaded in ModelZoo.")

    # 2. Define class names (Must match your training order!)
    class_names = ['FCC', 'IR', 'Optical', 'SAR'] 

    # 3. Preprocess Image
    # Note: Keras models usually expect a specific target size (e.g., 224x224)
    # Ensure this matches what you used in training!
    target_size = (224, 224) 
    
    try:
        # Load and resize
        img = keras_image.load_img(img_path, target_size=target_size)
        
        # Convert to array and cast
        img_array = keras_image.img_to_array(img).astype("float32")
        
        # Add batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 4. Predict
        preds = classifier.predict(img_array, verbose=0) # verbose=0 suppresses the progress bar
        
        # 5. Get Result
        predicted_index = np.argmax(preds[0])
        pred_class = class_names[predicted_index]
        confidence = float(np.max(preds[0]))
        
        print(f"🖼️ Image Classified: {pred_class} (Conf: {confidence:.2f})")
        return pred_class.lower() # Return lowercase for easier string matching later ('fcc', 'sar', etc.)

    except Exception as e:
        print(f"❌ Error during classification: {e}")
        # Fallback to 'optical' if classification fails so the pipeline doesn't crash
        return 'optical'