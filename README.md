# Pneumonia Detection Web Application

## Overview
This repository contains a Streamlit-based web application for pneumonia detection using a Convolutional Neural Network (CNN). The application allows users to upload chest X-ray images and predict whether they indicate pneumonia, leveraging a pre-trained CNN model with a training accuracy of 94% and a test accuracy of 90%.

## Repository Structure
- **main.py**: The Streamlit web interface for the pneumonia detection application. It handles user inputs, image uploads, and displays prediction results.
- **model.py**: Contains the CNN model architecture and code for training and evaluation.
- **model.h5** (or **model.keras**): The pre-trained CNN model file (either `.h5` or `.keras` format) with 94% training accuracy and 90% test accuracy.
- **Database**: Uses SQLite for storing relevant data (e.g., prediction history or user inputs).

## Features
- **Web Interface**: Built with Streamlit, allowing users to upload X-ray images and receive pneumonia predictions.
- **CNN Model**: A convolutional neural network implemented in `model.py` for pneumonia detection from chest X-ray images.
- **Database Integration**: SQLite database for storing prediction results or other metadata.
- **Model Performance**: 
  - Training Accuracy: 94%
  - Test Accuracy: 90%

## Requirements
To run the application, ensure you have the following dependencies installed:
- Python 3.8+
- Streamlit
- TensorFlow (or Keras)
- SQLite
- NumPy
- Pillow (for image processing)
- Other dependencies listed in `requirements.txt` (if applicable)

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/bhargav1213/pneumonia.git
   cd pneumonia
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Pre-trained Model**:
   Ensure the `model.h5` or `model.keras` file is present in the repository. If not, train the model using `model.py` or download it from a specified source (if provided).

4. **Run the Streamlit Application**:
   ```bash
   streamlit run main.py
   ```

5. **Access the Web Interface**:
   Open your browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).

## Usage
1. Launch the Streamlit app using the command above.
2. Upload a chest X-ray image through the web interface.
3. The CNN model will process the image and display the prediction (e.g., "Pneumonia" or "Normal").
4. Prediction results may be stored in the SQLite database for future reference.

## Model Training
To retrain the CNN model:
1. Ensure you have a dataset of chest X-ray images (e.g., from Kaggle's Pneumonia Detection dataset).
2. Run `model.py` to train the model:
   ```bash
   python model.py
   ```
3. The script will generate a new `model.h5` or `model.keras` file with the trained weights.

## Model Performance
- **Training Accuracy**: 94%
- **Test Accuracy**: 90%

The model uses a CNN architecture optimized for binary classification (pneumonia vs. normal) on chest X-ray images.
