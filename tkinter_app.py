import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import threading
from PIL import Image, ImageTk
from MLPP import FaceRecognitionSystem

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AT&T Face Recognition System")
        self.root.geometry("1000x650")
        self.root.minsize(1000, 650)
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize face recognition system
        self.init_system()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize variables
        self.selected_model = tk.StringVar(value="svm")
        self.selected_person = tk.IntVar(value=0)
        self.uploaded_image_path = None
        self.webcam_active = False
        self.webcam_thread = None
        self.cap = None
        
        # Create tabs
        self.create_tabs()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def init_system(self):
        """Initialize the face recognition system"""
        # Create status_var if it doesn't exist yet
        if not hasattr(self, 'status_var'):
            self.status_var = tk.StringVar()
        self.status_var.set("Loading models...")
        self.frs = FaceRecognitionSystem()
        
        # Check if models folder exists
        if os.path.exists('models'):
            self.frs.load_models()
        else:
            # Show a loading window while training
            loading_window = tk.Toplevel(self.root)
            loading_window.title("Training Models")
            loading_window.geometry("300x100")
            loading_window.transient(self.root)
            loading_window.grab_set()
            
            ttk.Label(loading_window, text="Training models for the first time...", padding=10).pack()
            progress = ttk.Progressbar(loading_window, mode='indeterminate')
            progress.pack(padx=10, pady=10, fill=tk.X)
            progress.start()
            
            # Train models in a separate thread
            def train_models():
                self.frs.train_svm()
                self.frs.train_cnn(max_iter=20)
                self.frs.train_pca_svm(n_components=100)
                self.frs.train_kmeans(n_clusters=40)
                self.frs.save_models()
                loading_window.destroy()
                self.status_var.set("Models trained and loaded.")
            
            thread = threading.Thread(target=train_models)
            thread.daemon = True
            thread.start()
            
            # Wait for the thread to finish
            self.root.wait_window(loading_window)
    
    def create_tabs(self):
        """Create tabs for different functionalities"""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create individual tabs
        self.home_tab = ttk.Frame(self.notebook)
        self.recognition_tab = ttk.Frame(self.notebook)
        self.performance_tab = ttk.Frame(self.notebook)
        self.dataset_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.home_tab, text="Home")
        self.notebook.add(self.recognition_tab, text="Recognition")
        self.notebook.add(self.performance_tab, text="Performance")
        self.notebook.add(self.dataset_tab, text="Dataset")
        
        # Setup each tab
        self.setup_home_tab()
        self.setup_recognition_tab()
        self.setup_performance_tab()
        self.setup_dataset_tab()
    
    def setup_home_tab(self):
        """Set up the home tab with general information"""
        # Title
        title_label = ttk.Label(self.home_tab, text="AT&T Face Recognition System", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Description
        desc_text = """
        This application demonstrates face recognition using the AT&T Face Database.
        
        Implemented Algorithms:
        - Support Vector Machine (SVM): A supervised learning algorithm for classification
        - Convolutional Neural Network (CNN): A deep learning approach for image classification
        - Principal Component Analysis (PCA) + SVM: Dimensionality reduction combined with SVM
        - K-means Clustering: An unsupervised approach for grouping similar faces
        
        Dataset Information:
        - 400 grayscale face images (10 images per person)
        - 40 different subjects
        - Various poses, expressions, and lighting conditions
        
        Use the tabs above to navigate through different functionalities.
        """
        
        desc_label = ttk.Label(self.home_tab, text=desc_text, wraplength=800, justify=tk.LEFT)
        desc_label.pack(pady=10, padx=20, anchor=tk.W)
        
        # Sample images frame
        ttk.Label(self.home_tab, text="Sample Images from Dataset:", font=("Arial", 12, "bold")).pack(pady=(20, 10), anchor=tk.W, padx=20)
        
        samples_frame = ttk.Frame(self.home_tab)
        samples_frame.pack(fill=tk.X, padx=20)
        
        # Display 10 random images
        indices = np.random.choice(len(self.frs.X), 10, replace=False)
        
        for i, idx in enumerate(indices):
            img_frame = ttk.Frame(samples_frame)
            img_frame.grid(row=i//5, column=i%5, padx=5, pady=5)
            
            # Convert image to PhotoImage
            img = self.frs.X[idx]
            img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Keep a reference to prevent garbage collection
            img_frame.img_tk = img_tk
            
            img_label = ttk.Label(img_frame, image=img_tk)
            img_label.pack()
            
            ttk.Label(img_frame, text=f"Person {self.frs.y[idx]}").pack()
    
    def setup_recognition_tab(self):
        """Set up the recognition tab for image upload and face recognition"""
        # Create left and right frames
        left_frame = ttk.Frame(self.recognition_tab, width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = ttk.Frame(self.recognition_tab, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - Upload and capture
        ttk.Label(left_frame, text="Face Recognition", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Model selection
        ttk.Label(left_frame, text="Select Model:").pack(anchor=tk.W, pady=(10, 5))
        
        models_frame = ttk.Frame(left_frame)
        models_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(models_frame, text="SVM", variable=self.selected_model, value="svm").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(models_frame, text="CNN", variable=self.selected_model, value="cnn").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(models_frame, text="PCA+SVM", variable=self.selected_model, value="pca_svm").pack(side=tk.LEFT)
        
        # Image source options
        ttk.Label(left_frame, text="Source:").pack(anchor=tk.W, pady=(10, 5))
        
        source_frame = ttk.Frame(left_frame)
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        upload_btn = ttk.Button(source_frame, text="Upload Image", command=self.upload_image)
        upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        webcam_btn = ttk.Button(source_frame, text="Webcam", command=self.toggle_webcam)
        webcam_btn.pack(side=tk.LEFT)
        
        # Image display
        self.image_frame = ttk.LabelFrame(left_frame, text="Image")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recognize button
        self.recognize_btn = ttk.Button(left_frame, text="Recognize Face", command=self.recognize_face, state=tk.DISABLED)
        self.recognize_btn.pack(pady=10, fill=tk.X)
        
        # Right frame - Results
        ttk.Label(right_frame, text="Recognition Results", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        self.results_frame = ttk.LabelFrame(right_frame, text="Prediction")
        self.results_frame.pack(fill=tk.X, pady=10)
        
        # Initially empty results
        ttk.Label(self.results_frame, text="No prediction yet. Upload an image and click Recognize.").pack(pady=20)
        
        # Sample images of predicted person
        self.samples_label = ttk.Label(right_frame, text="Samples of Predicted Person:", font=("Arial", 12))
        self.samples_label.pack(anchor=tk.W, pady=(10, 5))
        self.samples_label.pack_forget()  # Initially hidden
        
        self.samples_frame = ttk.Frame(right_frame)
        self.samples_frame.pack(fill=tk.BOTH, expand=True)
    
    def setup_performance_tab(self):
        """Set up the performance tab to display model metrics"""
        # Create control frame
        control_frame = ttk.Frame(self.performance_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Select Model:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Model selection combobox
        self.perf_model_var = tk.StringVar(value="SVM")
        model_combo = ttk.Combobox(control_frame, textvariable=self.perf_model_var, 
                                   values=["SVM", "CNN", "PCA+SVM", "K-means"])
        model_combo.pack(side=tk.LEFT)
        model_combo.bind("<<ComboboxSelected>>", self.update_performance_display)
        
        # Create content frame
        self.perf_content_frame = ttk.Frame(self.performance_tab)
        self.perf_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial update
        self.update_performance_display()
    
    def setup_dataset_tab(self):
        """Set up the dataset tab to explore the AT&T Face Database"""
        # Control frame
        control_frame = ttk.Frame(self.dataset_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Select Person ID:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Person selection combobox
        person_combo = ttk.Combobox(control_frame, textvariable=self.selected_person, 
                                    values=[f"{i}" for i in range(40)])
        person_combo.pack(side=tk.LEFT)
        person_combo.bind("<<ComboboxSelected>>", self.update_dataset_display)
        
        # Create display frame
        self.dataset_content_frame = ttk.Frame(self.dataset_tab)
        self.dataset_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial update
        self.update_dataset_display()
    
    def upload_image(self):
        """Open file dialog to upload an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.uploaded_image_path = file_path
            self.display_image(file_path)
            self.recognize_btn.config(state=tk.NORMAL)
            
            # Stop webcam if active
            if self.webcam_active:
                self.toggle_webcam()
    
    def toggle_webcam(self):
        """Toggle webcam capture on/off"""
        if not self.webcam_active:
            # Start webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            self.webcam_active = True
            self.webcam_thread = threading.Thread(target=self.update_webcam, daemon=True)
            self.webcam_thread.start()
            
            # Update button text
            self.notebook.nametowidget(f"{self.recognition_tab}.!frame.!button2").config(text="Stop Webcam")
            
            # Enable recognize button
            self.recognize_btn.config(state=tk.NORMAL)
        else:
            # Stop webcam
            self.webcam_active = False
            if self.cap:
                self.cap.release()
            
            # Update button text
            self.notebook.nametowidget(f"{self.recognition_tab}.!frame.!button2").config(text="Webcam")
            
            # Disable recognize button if no image is uploaded
            if not self.uploaded_image_path:
                self.recognize_btn.config(state=tk.DISABLED)
    
    def update_webcam(self):
        """Update webcam feed"""
        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Save current frame for recognition
            cv2.imwrite("temp_webcam.jpg", frame)
            self.uploaded_image_path = "temp_webcam.jpg"
            
            # Convert to grayscale for display consistency
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert to PIL format and then to Tkinter format
            img_pil = Image.fromarray(gray)
            img_pil = img_pil.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Update image
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # Keep a reference
            
            # Short sleep to reduce CPU usage
            self.root.after(10)
    
    def display_image(self, image_path):
        """Display the uploaded image"""
        img = Image.open(image_path)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((300, 300), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference
    
    def recognize_face(self):
        """Recognize the face in the uploaded image"""
        if not self.uploaded_image_path:
            messagebox.showerror("Error", "No image to recognize")
            return
        
        model_name = self.selected_model.get()
        
        # Update status
        self.status_var.set(f"Recognizing face using {model_name.upper()}...")
        
        # Make prediction
        result = self.frs.predict_new_image(self.uploaded_image_path, model_name=model_name)
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        for widget in self.samples_frame.winfo_children():
            widget.destroy()
        
        if result is not None:
            # Display results
            person_id = result['person_id']
            confidence = result['confidence']
            
            ttk.Label(self.results_frame, text=f"Predicted Person ID: {person_id}", font=("Arial", 12, "bold")).pack(pady=(10, 5))
            ttk.Label(self.results_frame, text=f"Confidence: {confidence:.4f}").pack(pady=(0, 10))
            
            # Show sample images of the predicted person
            self.samples_label.pack(anchor=tk.W, pady=(10, 5))
            
            # Find all images of the predicted person
            person_images = [img for img, label in zip(self.frs.X, self.frs.y) if label == person_id]
            
            if person_images:
                # Display up to 5 sample images
                for i, img in enumerate(person_images[:5]):
                    img_frame = ttk.Frame(self.samples_frame)
                    img_frame.grid(row=0, column=i, padx=5, pady=5)
                    
                    # Convert image to PhotoImage
                    img_display = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_display)
                    img_tk = ImageTk.PhotoImage(img_pil)
                    
                    # Keep a reference to prevent garbage collection
                    img_frame.img_tk = img_tk
                    
                    img_label = ttk.Label(img_frame, image=img_tk)
                    img_label.pack()
            else:
                ttk.Label(self.samples_frame, text="No sample images found for the predicted person").pack(pady=10)
        else:
            ttk.Label(self.results_frame, text="Error in prediction. Please try a different image or model.").pack(pady=20)
        
        # Update status
        self.status_var.set("Ready")
    
    def update_performance_display(self, event=None):
        """Update the performance display based on selected model"""
        model = self.perf_model_var.get()
        
        # Clear previous content
        for widget in self.perf_content_frame.winfo_children():
            widget.destroy()
        
        # Get model data
        model_map = {
            "SVM": "svm",
            "CNN": "cnn",
            "PCA+SVM": "pca_svm",
            "K-means": "kmeans"
        }
        
        model_key = model_map[model]
        model_data = self.frs.results[model_key]
        
        # Create metrics frame
        metrics_frame = ttk.Frame(self.perf_content_frame)
        metrics_frame.pack(fill=tk.X, pady=10)
        
        if model_key == 'kmeans':
            ttk.Label(metrics_frame, text=f"Clustering Accuracy: {model_data['accuracy']:.4f}", 
                     font=("Arial", 12, "bold")).pack(pady=5)
            
            ttk.Label(metrics_frame, text="""
            Note: K-means is an unsupervised learning algorithm, so the accuracy 
            measure is based on matching cluster assignments to the most common 
            true label within each cluster.
            """, wraplength=800).pack(pady=5)
            
            # Display cluster mapping if available
            if 'clusters_to_labels' in model_data:
                mapping_frame = ttk.LabelFrame(self.perf_content_frame, text="Cluster-to-Label Mapping")
                mapping_frame.pack(fill=tk.X, pady=10, padx=5)
                
                # Create a grid of labels
                mapping = model_data['clusters_to_labels']
                cols = 4
                
                for i, (cluster, label) in enumerate(mapping.items()):
                    row = i // cols
                    col = i % cols
                    
                    ttk.Label(mapping_frame, text=f"Cluster {cluster} → Person {label}").grid(
                        row=row, column=col, padx=10, pady=5, sticky=tk.W)
        else:
            # Regular supervised models
            if model_data['train_acc'] is not None:
                ttk.Label(metrics_frame, text=f"Training Accuracy: {model_data['train_acc']:.4f}", 
                         font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
                
                ttk.Label(metrics_frame, text=f"Test Accuracy: {model_data['test_acc']:.4f}", 
                         font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
            
            # For CNN, display training history
            if model_key == 'cnn' and 'train_loss' in model_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Plot accuracy
                ax1.plot(model_data['train_acc_history'], label='Training')
                ax1.plot(model_data['val_acc_history'], label='Validation')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot loss
                ax2.plot(model_data['train_loss'], label='Training')
                ax2.plot(model_data['val_loss'], label='Validation')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                fig.tight_layout()
                
                # Embed the figure in Tkinter
                canvas_frame = ttk.Frame(self.perf_content_frame)
                canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
                
                canvas = FigureCanvasTkAgg(fig, canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # For PCA+SVM, display eigenfaces if available
            if model_key == 'pca_svm' and os.path.exists('eigenfaces.png'):
                eigenfaces_img = Image.open('eigenfaces.png')
                eigenfaces_img = eigenfaces_img.resize((800, 350), Image.LANCZOS)
                eigenfaces_tk = ImageTk.PhotoImage(eigenfaces_img)
                
                # Keep reference to prevent garbage collection
                self.perf_content_frame.eigenfaces_tk = eigenfaces_tk
                
                eigenfaces_label = ttk.Label(self.perf_content_frame, image=eigenfaces_tk)
                eigenfaces_label.pack(pady=10)
            
            # Display confusion matrix if available
            confusion_matrix_file = f'confusion_matrix_{model_key}.png'
            if os.path.exists(confusion_matrix_file):
                cm_img = Image.open(confusion_matrix_file)
                cm_img = cm_img.resize((500, 400), Image.LANCZOS)
                cm_tk = ImageTk.PhotoImage(cm_img)
                
                # Keep reference to prevent garbage collection
                self.perf_content_frame.cm_tk = cm_tk
                
                ttk.Label(self.perf_content_frame, text="Confusion Matrix:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 5))
                cm_label = ttk.Label(self.perf_content_frame, image=cm_tk)
                cm_label.pack(pady=5)
    
    def update_dataset_display(self, event=None):
        """Update the dataset display based on selected person"""
        person_id = self.selected_person.get()
        
        # Clear previous content
        for widget in self.dataset_content_frame.winfo_children():
            widget.destroy()
        
        # Find all images of the selected person
        person_images = [img for img, label in zip(self.frs.X, self.frs.y) if label == person_id]
        
        if person_images:
            ttk.Label(self.dataset_content_frame, text=f"Images of Person {person_id}", 
                     font=("Arial", 12, "bold")).pack(pady=(0, 10))
            
            # Create a frame for the images
            images_frame = ttk.Frame(self.dataset_content_frame)
            images_frame.pack(fill=tk.BOTH, expand=True)
            
            # Display all images of the person
            for i, img in enumerate(person_images):
                img_frame = ttk.Frame(images_frame)
                img_frame.grid(row=i//5, column=i%5, padx=5, pady=5)
                
                # Convert image to PhotoImage
                img_display = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_display)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Keep a reference to prevent garbage collection
                img_frame.img_tk = img_tk
                
                img_label = ttk.Label(img_frame, image=img_tk)
                img_label.pack()
                
                ttk.Label(img_frame, text=f"Image {i+1}").pack()
        else:
            ttk.Label(self.dataset_content_frame, text=f"No images found for Person {person_id}").pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()