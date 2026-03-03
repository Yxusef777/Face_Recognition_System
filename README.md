# AT&T Face Recognition System

This project implements face recognition using the AT&T Face Database (also known as Olivetti Faces) with multiple algorithms:

1. Support Vector Machine (SVM) - Supervised
2. Neural Network (MLP) - Supervised
3. Principal Component Analysis (PCA) with SVM - Dimensionality reduction + supervised
4. K-means Clustering - Unsupervised

The system includes comprehensive visualization for model performance, including accuracy metrics, loss curves, confusion matrices, and classification reports. It also provides both web-based (Streamlit) and desktop (Tkinter) interfaces for interactive face recognition.

## Dataset

The AT&T Face Database contains 400 grayscale images of 40 individuals (10 images per person), with variations in:
- Facial expressions (open/closed eyes, smiling/not smiling)
- Facial details (glasses/no glasses)
- Lighting conditions
- Head pose (tilting, looking left/right/up/down)

The database is automatically downloaded via scikit-learn's `fetch_olivetti_faces()` function.

## Features

- **Multiple Algorithms**: Compare the performance of different face recognition approaches
- **Performance Visualization**: 
  - Train and test accuracy
  - Loss curves for CNN
  - Confusion matrices
  - Classification reports
- **Interactive Face Recognition**: Upload a new image to predict which person it belongs to
- **Eigenfaces Visualization**: View the principal components from PCA
- **Dataset Exploration**: Browse through the dataset images by person

## Applications

### Main Script (MLPP.py)

The core functionality is implemented in `MLPP.py`, which includes:
- Dataset loading and preprocessing
- Model training (SVM, CNN, PCA+SVM, K-means)
- Performance evaluation
- Visualization functions
- Prediction for new images

### Streamlit Web App (streamlit_app.py)

A web-based interface built with Streamlit that provides:
- Interactive model performance visualization
- Face recognition from uploaded images
- Dataset exploration

### Tkinter Desktop App (tkinter_app.py)

A desktop application with a graphical interface that offers:
- Face recognition from uploaded images or webcam
- Performance metrics visualization
- Dataset browsing

## Requirements

See `requirements.txt` for the full list of dependencies.

## How to Use

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the main script to train the models:
   ```
   python MLPP.py
   ```

3. Run the Streamlit web app:
   ```
   streamlit run streamlit_app.py
   ```

4. Run the Tkinter desktop app:
   ```
   python tkinter_app.py
   ```

## Working with the Code

### Training Custom Models

You can customize the model parameters in `MLPP.py`:

```python
frs = FaceRecognitionSystem()
frs.train_svm(C=1.0, kernel='rbf', gamma='scale')
frs.train_cnn(epochs=50, batch_size=32)
frs.train_pca_svm(n_components=50)
frs.train_kmeans(n_clusters=40)
