import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
import cv2
import os
import pickle

class FaceRecognitionSystem:
    def __init__(self, test_size=0.2, random_state=42):
        # Load the AT&T Face Database (Olivetti Faces)
        self.dataset = fetch_olivetti_faces(shuffle=True, random_state=random_state)
        self.X = self.dataset.images
        self.y = self.dataset.target
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Initialize models
        self.svm_model = None
        self.cnn_model = None
        self.pca_model = None
        self.kmeans_model = None
        
        # Store results
        self.results = {
            'svm': {'train_acc': None, 'test_acc': None},
            'cnn': {'train_acc': None, 'test_acc': None, 'train_loss': [], 'test_loss': [], 'train_acc_history': [], 'val_acc_history': []},
            'pca_svm': {'train_acc': None, 'test_acc': None},
            'kmeans': {'accuracy': None}
        }
        
        print(f"Dataset loaded: {self.X.shape[0]} images of {self.X.shape[1]}x{self.X.shape[2]} pixels")
        print(f"Number of classes (individuals): {len(np.unique(self.y))}")
        print(f"Training set: {self.X_train.shape[0]} images")
        print(f"Test set: {self.X_test.shape[0]} images")
    
    def train_svm(self, C=1.0, kernel='rbf', gamma='scale'):
        # Reshape images for SVM
        X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
        X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Train SVM model
        print("Training SVM model...")
        self.svm_model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        self.svm_model.fit(X_train_scaled, self.y_train)
        
        # Calculate accuracy
        y_train_pred = self.svm_model.predict(X_train_scaled)
        y_test_pred = self.svm_model.predict(X_test_scaled)
        
        self.results['svm']['train_acc'] = accuracy_score(self.y_train, y_train_pred)
        self.results['svm']['test_acc'] = accuracy_score(self.y_test, y_test_pred)
        self.results['svm']['y_pred'] = y_test_pred
        self.results['svm']['scaler'] = scaler
        
        print(f"SVM - Training Accuracy: {self.results['svm']['train_acc']:.4f}")
        print(f"SVM - Test Accuracy: {self.results['svm']['test_acc']:.4f}")
        
        return self.results['svm']
    
    def train_cnn(self, max_iter=50, batch_size=32, validation_fraction=0.1):
        # Reshape images for neural network (flatten)
        X_train_nn = self.X_train.reshape(self.X_train.shape[0], -1)
        X_test_nn = self.X_test.reshape(self.X_test.shape[0], -1)
        
        # Create Neural Network model (MLPClassifier)
        print("Training Neural Network model...")
        
        # Define hidden layer sizes for a deep network
        # Using a structure that mimics a CNN's feature extraction
        hidden_layers = (512, 256, 128)
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size=batch_size,
            learning_rate='adaptive',
            max_iter=max_iter,
            random_state=42,
            verbose=True,
            validation_fraction=validation_fraction
        )
        
        # Train the model
        model.fit(X_train_nn, self.y_train)
        
        # Save the model
        self.cnn_model = model
        
        # Evaluate the model
        train_acc = model.score(X_train_nn, self.y_train)
        test_acc = model.score(X_test_nn, self.y_test)
        
        self.results['cnn']['train_acc'] = train_acc
        self.results['cnn']['test_acc'] = test_acc
        
        # Store loss curve
        self.results['cnn']['train_loss'] = model.loss_curve_
        
        # Since MLPClassifier doesn't provide validation metrics during training
        # we'll create placeholder data to maintain compatibility
        self.results['cnn']['val_loss'] = []
        self.results['cnn']['train_acc_history'] = []
        self.results['cnn']['val_acc_history'] = []
        
        # Get predictions
        y_pred = model.predict(X_test_nn)
        y_pred_probs = model.predict_proba(X_test_nn)
        self.results['cnn']['y_pred'] = y_pred
        
        print(f"Neural Network - Training Accuracy: {train_acc:.4f}")
        print(f"Neural Network - Test Accuracy: {test_acc:.4f}")
        
        return self.results['cnn']
    
    def train_pca_svm(self, n_components=50, C=1.0, kernel='rbf', gamma='scale'):
        # Reshape images for PCA
        X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
        X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        # Apply PCA
        print("Training PCA+SVM model...")
        self.pca_model = PCA(n_components=n_components)
        X_train_pca = self.pca_model.fit_transform(X_train_flat)
        X_test_pca = self.pca_model.transform(X_test_flat)
        
        # Train SVM on PCA features
        pca_svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        pca_svm.fit(X_train_pca, self.y_train)
        
        # Calculate accuracy
        y_train_pred = pca_svm.predict(X_train_pca)
        y_test_pred = pca_svm.predict(X_test_pca)
        
        self.results['pca_svm']['train_acc'] = accuracy_score(self.y_train, y_train_pred)
        self.results['pca_svm']['test_acc'] = accuracy_score(self.y_test, y_test_pred)
        self.results['pca_svm']['y_pred'] = y_test_pred
        self.results['pca_svm']['model'] = pca_svm
        
        print(f"PCA+SVM - Training Accuracy: {self.results['pca_svm']['train_acc']:.4f}")
        print(f"PCA+SVM - Test Accuracy: {self.results['pca_svm']['test_acc']:.4f}")
        
        # Visualize eigenfaces (first few components)
        if n_components >= 10:
            plt.figure(figsize=(12, 5))
            for i in range(10):
                plt.subplot(2, 5, i+1)
                eigenface = self.pca_model.components_[i].reshape(64, 64)
                plt.imshow(eigenface, cmap='gray')
                plt.title(f'Eigenface {i+1}')
                plt.axis('off')
            plt.tight_layout()
            plt.suptitle('First 10 Eigenfaces (PCA Components)')
            plt.subplots_adjust(top=0.85)
            plt.savefig('eigenfaces.png')
            plt.close()
        
        return self.results['pca_svm']
    
    def train_kmeans(self, n_clusters=40, random_state=42):
        # Reshape images for K-means
        X_flat = self.X.reshape(self.X.shape[0], -1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply K-means clustering
        print("Training K-means clustering model...")
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = self.kmeans_model.fit_predict(X_pca)
        
        # Evaluate clustering (unsupervised, so we use a different approach)
        # We assign each cluster to the most common true label within it
        clusters_to_labels = {}
        for i in range(n_clusters):
            mask = (clusters == i)
            if np.sum(mask) > 0:  # Ensure the cluster has samples
                most_common_label = np.bincount(self.y[mask]).argmax()
                clusters_to_labels[i] = most_common_label
        
        # Map cluster assignments to predicted labels
        y_pred = np.array([clusters_to_labels.get(c, -1) for c in clusters])
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == self.y)
        self.results['kmeans']['accuracy'] = accuracy
        self.results['kmeans']['clusters'] = clusters
        self.results['kmeans']['clusters_to_labels'] = clusters_to_labels
        
        print(f"K-means - Clustering Accuracy: {accuracy:.4f}")
        
        return self.results['kmeans']
    
    def plot_accuracy_comparison(self):
        models = []
        train_accuracies = []
        test_accuracies = []
        
        if self.results['svm']['train_acc'] is not None:
            models.append('SVM')
            train_accuracies.append(self.results['svm']['train_acc'])
            test_accuracies.append(self.results['svm']['test_acc'])
        
        if self.results['cnn']['train_acc'] is not None:
            models.append('CNN')
            train_accuracies.append(self.results['cnn']['train_acc'])
            test_accuracies.append(self.results['cnn']['test_acc'])
        
        if self.results['pca_svm']['train_acc'] is not None:
            models.append('PCA+SVM')
            train_accuracies.append(self.results['pca_svm']['train_acc'])
            test_accuracies.append(self.results['pca_svm']['test_acc'])
        
        if self.results['kmeans']['accuracy'] is not None:
            models.append('K-means')
            # For K-means, we use the same value for both train and test
            train_accuracies.append(self.results['kmeans']['accuracy'])
            test_accuracies.append(self.results['kmeans']['accuracy'])
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_accuracies, width, label='Training Accuracy')
        plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for i, v in enumerate(train_accuracies):
            plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
        
        for i, v in enumerate(test_accuracies):
            plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png')
        plt.close()
    
    def plot_cnn_history(self):
        if self.results['cnn']['train_acc'] is None:
            print("Neural Network model has not been trained yet.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot loss curve
        if 'train_loss' in self.results['cnn'] and len(self.results['cnn']['train_loss']) > 0:
            ax.plot(self.results['cnn']['train_loss'], label='Training Loss')
            ax.set_title('Neural Network Loss Curve')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss history available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig('cnn_history.png')
        plt.close()
    
    def plot_confusion_matrix(self, model_name):
        if model_name not in self.results or 'y_pred' not in self.results[model_name]:
            print(f"{model_name} model predictions not available.")
            return
        
        y_pred = self.results[model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.close()
    
    def print_classification_report(self, model_name):
        if model_name not in self.results or 'y_pred' not in self.results[model_name]:
            print(f"{model_name} model predictions not available.")
            return
        
        y_pred = self.results[model_name]['y_pred']
        report = classification_report(self.y_test, y_pred, zero_division=0)
        
        print(f"Classification Report - {model_name.upper()}:")
        print(report)
        
        return report
    
    def predict_new_image(self, image_path, model_name='svm'):
        """
        Predict the class of a new face image using the specified model.
        
        Args:
            image_path: Path to the image file
            model_name: Model to use for prediction ('svm', 'cnn', or 'pca_svm')
        
        Returns:
            Predicted class (person ID)
        """
        # Read and preprocess the image
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Resize to 64x64
            img = cv2.resize(img, (64, 64))
            
            # Normalize pixel values to [0, 1]
            img = img / 255.0
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        
        # Make prediction based on the specified model
        if model_name == 'svm':
            if self.svm_model is None:
                print("SVM model has not been trained yet.")
                return None
            
            # Flatten and scale the image
            img_flat = img.reshape(1, -1)
            img_scaled = self.results['svm']['scaler'].transform(img_flat)
            
            # Predict
            pred_class = self.svm_model.predict(img_scaled)[0]
            pred_proba = self.svm_model.predict_proba(img_scaled)[0]
            confidence = pred_proba[pred_class]
            
        elif model_name == 'cnn':
            if self.cnn_model is None:
                print("CNN model has not been trained yet.")
                return None
            
            # Reshape for CNN
            img_cnn = img.reshape(1, 64, 64, 1)
            
            # Predict
            pred_proba = self.cnn_model.predict(img_cnn)[0]
            pred_class = np.argmax(pred_proba)
            confidence = pred_proba[pred_class]
            
        elif model_name == 'pca_svm':
            if self.pca_model is None or 'model' not in self.results['pca_svm']:
                print("PCA+SVM model has not been trained yet.")
                return None
            
            # Flatten the image
            img_flat = img.reshape(1, -1)
            
            # Apply PCA transformation
            img_pca = self.pca_model.transform(img_flat)
            
            # Predict
            pca_svm = self.results['pca_svm']['model']
            pred_class = pca_svm.predict(img_pca)[0]
            pred_proba = pca_svm.predict_proba(img_pca)[0]
            confidence = pred_proba[pred_class]
            
        else:
            print(f"Unsupported model: {model_name}")
            return None
        
        return {
            'class': int(pred_class),
            'confidence': float(confidence),
            'person_id': int(pred_class)
        }
    
    def save_models(self, folder='models'):
        """Save all trained models to disk"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Save SVM model
        if self.svm_model is not None:
            with open(f'{folder}/svm_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.svm_model,
                    'scaler': self.results['svm']['scaler']
                }, f)
        
        # Save Neural Network model
        if self.cnn_model is not None:
            with open(f'{folder}/nn_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.cnn_model,
                    'results': self.results['cnn']
                }, f)
        
        # Save PCA model and PCA+SVM model
        if self.pca_model is not None and 'model' in self.results['pca_svm']:
            with open(f'{folder}/pca_svm_model.pkl', 'wb') as f:
                pickle.dump({
                    'pca': self.pca_model,
                    'svm': self.results['pca_svm']['model']
                }, f)
        
        # Save K-means model
        if self.kmeans_model is not None:
            with open(f'{folder}/kmeans_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.kmeans_model,
                    'clusters_to_labels': self.results['kmeans']['clusters_to_labels']
                }, f)
        
        print(f"Models saved to {folder} folder")
    
    def load_models(self, folder='models'):
        """Load all models from disk"""
        # Load SVM model
        svm_path = f'{folder}/svm_model.pkl'
        if os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                svm_dict = pickle.load(f)
                self.svm_model = svm_dict['model']
                self.results['svm']['scaler'] = svm_dict['scaler']
            print("SVM model loaded")
        
        # Load Neural Network model
        nn_path = f'{folder}/nn_model.pkl'
        if os.path.exists(nn_path):
            with open(nn_path, 'rb') as f:
                nn_dict = pickle.load(f)
                self.cnn_model = nn_dict['model']
                self.results['cnn'] = nn_dict['results']
            print("Neural Network model loaded")
        
        # Load PCA+SVM model
        pca_svm_path = f'{folder}/pca_svm_model.pkl'
        if os.path.exists(pca_svm_path):
            with open(pca_svm_path, 'rb') as f:
                pca_svm_dict = pickle.load(f)
                self.pca_model = pca_svm_dict['pca']
                self.results['pca_svm']['model'] = pca_svm_dict['svm']
            print("PCA+SVM model loaded")
        
        # Load K-means model
        kmeans_path = f'{folder}/kmeans_model.pkl'
        if os.path.exists(kmeans_path):
            with open(kmeans_path, 'rb') as f:
                kmeans_dict = pickle.load(f)
                self.kmeans_model = kmeans_dict['model']
                self.results['kmeans']['clusters_to_labels'] = kmeans_dict['clusters_to_labels']
            print("K-means model loaded")


def main():
    # Create the face recognition system
    frs = FaceRecognitionSystem(test_size=0.2, random_state=42)
    
    # Train models
    frs.train_svm()
    frs.train_cnn(max_iter=20)  # Use max_iter for MLPClassifier
    frs.train_pca_svm(n_components=100)
    frs.train_kmeans(n_clusters=40)
    
    # Plot results
    frs.plot_accuracy_comparison()
    frs.plot_cnn_history()
    
    # Plot confusion matrices
    frs.plot_confusion_matrix('svm')
    frs.plot_confusion_matrix('cnn')
    frs.plot_confusion_matrix('pca_svm')
    
    # Print classification reports
    frs.print_classification_report('svm')
    frs.print_classification_report('cnn')
    frs.print_classification_report('pca_svm')
    
    # Save models for future use
    frs.save_models()


if __name__ == "__main__":
    main()