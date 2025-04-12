import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/Users/hritikgupta/Documents/brain_mri_detection_model.h5')


# Predict new image function
def predict_image(image_path, model):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Resize to the input size of the model
    image = cv2.resize(image, (128, 128))
    
    # Reshape and normalize
    image = image.reshape(1, 128, 128, 1) / 255.0
    
    # Make prediction
    prediction = model.predict(image)
    
    # Display the prediction result
    result = 'Tumor' if np.argmax(prediction) == 1 else 'No Tumor'
    return result, image  # Return the result and the image for plotting

# Example usage
if __name__ == "__main__":
    # Update this path to point to an actual MRI image you want to test
    image_path = '/Users/hritikgupta/Documents/test_image.jpg'  # Change this to your MRI image path
    try:
        result, image = predict_image(image_path, model)
        print(result)  # Print the result
        
        # Plot the image and prediction result
        plt.imshow(image.reshape(128, 128), cmap='gray')
        plt.title(f'Prediction: {result}')
        plt.axis('off')
        plt.show()  # Show the image with the prediction title

    except ValueError as e:
        print(e)
