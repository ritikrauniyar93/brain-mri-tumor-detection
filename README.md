# 🧠 Brain MRI Tumor Detection

This is a deep learning project built using Convolutional Neural Networks (CNN) to detect brain tumors from MRI images. The model classifies MRI scans as **Tumor** or **Non-Tumor** based on visual patterns learned from the dataset.

## 🔍 Features
- **Binary Classification**: Classifies MRI images as either Tumor or No Tumor.
- **Technology Stack**: Built using Python, TensorFlow, and Keras.
- **Dataset**: Trained on a real MRI image dataset with labeled data.
- **Easy-to-use**: Includes a simple script to make predictions on new images.

## 📁 Dataset Structure
The dataset is divided into two folders:
- **tumor/**: Contains images labeled as Tumor.
- **no_tumor/**: Contains images labeled as No Tumor.

Example of dataset structure:
Dataset/ ├── tumor/ │ ├── image1.jpg │ ├── image2.jpg │ └── ... └── no_tumor/ ├── image1.jpg ├── image2.jpg └── ...

bash
Copy
Edit

## 🚀 Getting Started

To get the project up and running locally, follow these steps:

### 1. Clone the repository:
```bash
git clone https://github.com/ritikrauniyar93/brain-mri-tumor-detection.git
2. Install dependencies:
bash
Copy
Edit
pip install tensorflow matplotlib numpy
3. Run the detection script:
bash
Copy
Edit
python brain_mri_detection.py
🧑‍💻 Code Explanation
brain_mri_detection.py: Main script for training the model and making predictions.

model.h5: Pre-trained model saved in H5 format.

utils.py: Utility functions for preprocessing images and visualizing results.

