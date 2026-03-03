import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import io
from MLPP import FaceRecognitionSystem

# Set page configuration
st.set_page_config(
    page_title="AT&T Face Recognition App",
    page_icon="👤",
    layout="wide"
)

# Initialize the face recognition system
@st.cache_resource
def load_face_recognition_system():
    frs = FaceRecognitionSystem()
    # Check if models folder exists
    if os.path.exists('models'):
        frs.load_models()
    else:
        with st.spinner("Training models for the first time... This may take a few minutes"):
            frs.train_svm()
            frs.train_cnn(max_iter=20)
            frs.train_pca_svm(n_components=50)
            frs.train_kmeans(n_clusters=40)
            frs.save_models()
    return frs

# Main function
def main():
    st.title("AT&T Face Recognition System")
    st.write("This application demonstrates face recognition using the AT&T Face Database with multiple algorithms.")
    
    # Load the face recognition system
    frs = load_face_recognition_system()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Home", "Model Performance", "Face Recognition", "Dataset Exploration"]
    selection = st.sidebar.radio("Go to", pages)
    
    # Page content
    if selection == "Home":
        show_home_page(frs)
    elif selection == "Model Performance":
        show_performance_page(frs)
    elif selection == "Face Recognition":
        show_recognition_page(frs)
    elif selection == "Dataset Exploration":
        show_dataset_page(frs)

def show_home_page(frs):
    st.header("Welcome to the AT&T Face Recognition System")
    
    st.write("""
    This application demonstrates face recognition using the AT&T Face Database (formerly Olivetti Faces).
    
    ### Implemented Algorithms:
    - **Support Vector Machine (SVM)**: A supervised learning algorithm for classification
    - **Convolutional Neural Network (CNN)**: A deep learning approach for image classification
    - **Principal Component Analysis (PCA) + SVM**: Dimensionality reduction combined with SVM
    - **K-means Clustering**: An unsupervised approach for grouping similar faces
    
    ### Dataset Information:
    - 400 grayscale face images (10 images per person)
    - 40 different subjects
    - Various poses, expressions, and lighting conditions
    
    ### Use the sidebar to navigate through different sections of the application.
    """)
    
    # Show sample images from dataset
    st.subheader("Sample Images from Dataset")
    
    # Display 10 random images
    indices = np.random.choice(len(frs.X), 10, replace=False)
    cols = st.columns(5)
    
    for i, idx in enumerate(indices):
        col_idx = i % 5
        with cols[col_idx]:
            st.image(frs.X[idx], caption=f"Person {frs.y[idx]}", width=100)
            
    # Display model accuracy comparison if available
    if all(result['train_acc'] is not None for result in [frs.results['svm'], frs.results['cnn'], frs.results['pca_svm']]):
        st.subheader("Model Accuracy Comparison")
        
        if os.path.exists('accuracy_comparison.png'):
            st.image('accuracy_comparison.png')
        else:
            frs.plot_accuracy_comparison()
            st.image('accuracy_comparison.png')

def show_performance_page(frs):
    st.header("Model Performance")
    
    # Select model to display
    model_selection = st.selectbox(
        "Select Model",
        ["SVM", "CNN", "PCA+SVM", "K-means"]
    )
    
    if model_selection == "SVM":
        show_svm_performance(frs)
    elif model_selection == "CNN":
        show_cnn_performance(frs)
    elif model_selection == "PCA+SVM":
        show_pca_svm_performance(frs)
    elif model_selection == "K-means":
        show_kmeans_performance(frs)

def show_svm_performance(frs):
    st.subheader("SVM Model Performance")
    
    if frs.results['svm']['train_acc'] is None:
        st.warning("SVM model has not been trained yet")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Accuracy", f"{frs.results['svm']['train_acc']:.4f}")
    
    with col2:
        st.metric("Test Accuracy", f"{frs.results['svm']['test_acc']:.4f}")
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    confusion_matrix_file = 'confusion_matrix_svm.png'
    
    if not os.path.exists(confusion_matrix_file):
        frs.plot_confusion_matrix('svm')
    
    st.image(confusion_matrix_file)
    
    # Display classification report
    st.subheader("Classification Report")
    report = frs.print_classification_report('svm')
    st.text(report)

def show_cnn_performance(frs):
    st.subheader("Neural Network Model Performance")
    
    if frs.results['cnn']['train_acc'] is None:
        st.warning("Neural Network model has not been trained yet")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Accuracy", f"{frs.results['cnn']['train_acc']:.4f}")
    
    with col2:
        st.metric("Test Accuracy", f"{frs.results['cnn']['test_acc']:.4f}")
    
    # Display training history
    st.subheader("Training History")
    history_file = 'cnn_history.png'
    
    if not os.path.exists(history_file):
        frs.plot_cnn_history()
    
    st.image(history_file)
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    confusion_matrix_file = 'confusion_matrix_cnn.png'
    
    if not os.path.exists(confusion_matrix_file):
        frs.plot_confusion_matrix('cnn')
    
    st.image(confusion_matrix_file)
    
    # Display classification report
    st.subheader("Classification Report")
    report = frs.print_classification_report('cnn')
    st.text(report)

def show_pca_svm_performance(frs):
    st.subheader("PCA+SVM Model Performance")
    
    if frs.results['pca_svm']['train_acc'] is None:
        st.warning("PCA+SVM model has not been trained yet")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Accuracy", f"{frs.results['pca_svm']['train_acc']:.4f}")
    
    with col2:
        st.metric("Test Accuracy", f"{frs.results['pca_svm']['test_acc']:.4f}")
    
    # Display eigenfaces
    st.subheader("Eigenfaces (PCA Components)")
    eigenfaces_file = 'eigenfaces.png'
    
    if os.path.exists(eigenfaces_file):
        st.image(eigenfaces_file)
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    confusion_matrix_file = 'confusion_matrix_pca_svm.png'
    
    if not os.path.exists(confusion_matrix_file):
        frs.plot_confusion_matrix('pca_svm')
    
    st.image(confusion_matrix_file)
    
    # Display classification report
    st.subheader("Classification Report")
    report = frs.print_classification_report('pca_svm')
    st.text(report)

def show_kmeans_performance(frs):
    st.subheader("K-means Clustering Performance")
    
    if frs.results['kmeans']['accuracy'] is None:
        st.warning("K-means model has not been trained yet")
        return
    
    st.metric("Clustering Accuracy", f"{frs.results['kmeans']['accuracy']:.4f}")
    
    st.write("""
    Note: K-means is an unsupervised learning algorithm, so the accuracy 
    measure is based on matching cluster assignments to the most common 
    true label within each cluster.
    """)
    
    # Display cluster-to-label mapping
    st.subheader("Cluster-to-Label Mapping")
    
    if 'clusters_to_labels' in frs.results['kmeans']:
        mapping = frs.results['kmeans']['clusters_to_labels']
        
        # Create a more visual representation
        cols = st.columns(4)
        for i, (cluster, label) in enumerate(mapping.items()):
            col_idx = i % 4
            cols[col_idx].metric(f"Cluster {cluster}", f"Person {label}")

def show_recognition_page(frs):
    st.header("Face Recognition")
    
    st.write("""
    Upload a face image to recognize the person using the trained models.
    
    For best results:
    - The image should be a frontal face
    - Good lighting conditions
    - Clear background
    """)
    
    # Select model to use
    model_selection = st.selectbox(
        "Select Model for Recognition",
        ["SVM", "CNN", "PCA+SVM"]
    )
    
    model_name_map = {
        "SVM": "svm",
        "CNN": "cnn",
        "PCA+SVM": "pca_svm"
    }
    
    # Upload image
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Convert to format suitable for prediction
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Save the image temporarily
        temp_path = "temp_uploaded_image.jpg"
        image.convert('L').save(temp_path)
        
        # Make prediction
        with st.spinner("Recognizing face..."):
            result = frs.predict_new_image(temp_path, model_name=model_name_map[model_selection])
        
        if result is not None:
            st.success(f"Prediction complete!")
            
            # Display result
            st.subheader("Recognition Result")
            
            st.metric("Predicted Person ID", result['person_id'])
            st.metric("Confidence", f"{result['confidence']:.4f}")
            
            # Display a sample of the predicted person from the dataset
            st.subheader("Samples of the Predicted Person")
            
            # Find all images of the predicted person
            person_images = [img for img, label in zip(frs.X, frs.y) if label == result['person_id']]
            
            if person_images:
                # Display up to 5 sample images
                cols = st.columns(min(5, len(person_images)))
                for i, img in enumerate(person_images[:5]):
                    cols[i].image(img, width=100)
            else:
                st.warning("No sample images found for the predicted person")
        else:
            st.error("Error in prediction. Please try a different image or model.")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def show_dataset_page(frs):
    st.header("Dataset Exploration")
    
    st.write("""
    The AT&T Face Database contains 400 grayscale images of 40 individuals.
    Each individual has 10 different images, taken with various facial expressions,
    lighting conditions, and facial details.
    """)
    
    # Select a person to display
    person_id = st.selectbox(
        "Select Person ID",
        list(range(40)),
        format_func=lambda x: f"Person {x}"
    )
    
    # Find all images of the selected person
    person_images = [img for img, label in zip(frs.X, frs.y) if label == person_id]
    
    if person_images:
        st.subheader(f"Images of Person {person_id}")
        
        # Display all images of the person
        cols = st.columns(5)
        for i, img in enumerate(person_images):
            col_idx = i % 5
            cols[col_idx].image(img, width=100)
    else:
        st.warning(f"No images found for Person {person_id}")

if __name__ == "__main__":
    main()
