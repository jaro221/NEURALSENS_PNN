"""
Polynomial Neural Network for Temperature Classification
Using real sensor data from in-sensor computing experiments

Classification approach: Temperature ranges
- Class 0: T < 32°C (Low)
- Class 1: 32°C ≤ T < 35°C (Medium)
- Class 2: 35°C ≤ T < 38°C (High)
- Class 3: T ≥ 38°C (Very High)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from keras import layers
from keras.models import Model
import seaborn as sns

# ===========================================================================================
# PART 1: Load and preprocess real sensor data
# ===========================================================================================

print("Loading sensor data...")
data = pd.read_csv('DATA_[1,2V,1,4V;1,6;1.8]_TempSweep_2026-02-18_19-44-46.csv')

# Extract features (currents from 4 sensors) and target (temperature)
X_raw = data[['I_D1_1.2V', 'I_D2_1.4V', 'I_D3_1.6V', 'I_D4_1.8V']].values
y_temp = data['act_temp'].values

# Create classification labels based on temperature ranges
def create_temperature_classes(temperatures):
    """Create 4 temperature classes"""
    classes = np.zeros(len(temperatures), dtype=int)
    classes[temperatures < 32] = 0      # Low
    classes[(temperatures >= 32) & (temperatures < 35)] = 1   # Medium
    classes[(temperatures >= 35) & (temperatures < 38)] = 2   # High
    classes[temperatures >= 38] = 3     # Very High
    return classes

y_class = create_temperature_classes(y_temp)

print(f"Data shape: {X_raw.shape}")
print(f"Temperature range: {y_temp.min():.1f}°C - {y_temp.max():.1f}°C")
print(f"Class distribution: {np.bincount(y_class)}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# ===========================================================================================
# PART 2: Polynomial characteristics functions
# ===========================================================================================

def poly_features(X, degree=2):
    """
    Generate polynomial features from input
    For degree=2: includes original features and all 2nd order interactions
    """
    X_poly = X.copy()
    
    # Add polynomial features (squared terms)
    for i in range(X.shape[1]):
        X_poly = np.column_stack([X_poly, X[:, i]**2])
    
    # Add some interaction terms
    if degree >= 2 and X.shape[1] >= 2:
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                X_poly = np.column_stack([X_poly, X[:, i] * X[:, j]])
    
    return X_poly

def apply_poly_transform(X, degree=2):
    """Apply polynomial transformation to create new feature space"""
    X_transformed = poly_features(X, degree)
    # Standardize the new features
    scaler_poly = StandardScaler()
    return scaler_poly.fit_transform(X_transformed)

# Generate polynomial features
print("\nGenerating polynomial features...")
X_train_poly = apply_poly_transform(X_train, degree=2)
X_test_poly = apply_poly_transform(X_test, degree=2)

print(f"Polynomial feature shape: {X_train_poly.shape}")

# ===========================================================================================
# PART 3: Custom Polynomial Neural Network (PNN) - Classification
# ===========================================================================================

class PolynomialNNClassifier:
    def __init__(self, input_size, hidden_size, output_size=4, poly_degree=2):
        self.poly_degree = poly_degree
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Polynomial coefficient matrices for transformation
        self.poly_coeff = np.random.randn(input_size, hidden_size, poly_degree+1) * 0.1
        
        # Regular hidden to output weights
        self.W1 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def poly_transform(self, X):
        """Apply polynomial transformation to features"""
        n_samples = X.shape[0]
        output = np.zeros((n_samples, self.hidden_size))
        
        for h in range(self.hidden_size):
            for s in range(n_samples):
                # For each hidden unit, apply polynomial to each input feature
                result = 0
                for i in range(X.shape[1]):
                    poly = np.poly1d(self.poly_coeff[i, h, :])
                    result += poly(X[s, i])
                output[s, h] = result
        
        return output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Polynomial feature transformation
        self.z1_poly = self.poly_transform(X)
        
        # Hidden layer
        self.z1 = np.dot(self.z1_poly, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer with softmax
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

print("\n" + "="*80)
print("CUSTOM POLYNOMIAL NEURAL NETWORK CLASSIFIER")
print("="*80)

pnn = PolynomialNNClassifier(X_train_poly.shape[1], hidden_size=8, output_size=4)
pnn_pred = pnn.predict(X_test_poly)

print(f"\nPNN Classification Accuracy: {accuracy_score(y_test, pnn_pred):.4f}")
print("\nPNN Classification Report:")
print(classification_report(y_test, pnn_pred, target_names=['Low (<32°C)', 'Medium (32-35°C)', 'High (35-38°C)', 'Very High (≥38°C)']))

# ===========================================================================================
# PART 4: Standard Keras Neural Network - Classification
# ===========================================================================================

print("\n" + "="*80)
print("STANDARD KERAS NEURAL NETWORK CLASSIFIER")
print("="*80)

model_standard = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model_standard.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

print("\nTraining standard NN...")
history_standard = model_standard.fit(
    X_train, y_train,
    batch_size=16,
    epochs=50,
    verbose=0,
    validation_split=0.2
)

y_pred_standard = model_standard.predict(X_test, verbose=0)
y_pred_standard_class = np.argmax(y_pred_standard, axis=1)

print(f"\nStandard NN Accuracy: {accuracy_score(y_test, y_pred_standard_class):.4f}")
print("\nStandard NN Classification Report:")
print(classification_report(y_test, y_pred_standard_class, target_names=['Low (<32°C)', 'Medium (32-35°C)', 'High (35-38°C)', 'Very High (≥38°C)']))

# ===========================================================================================
# PART 5: Polynomial Feature-based Keras NN
# ===========================================================================================

print("\n" + "="*80)
print("POLYNOMIAL FEATURE + KERAS NEURAL NETWORK")
print("="*80)

model_poly = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train_poly.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model_poly.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

print("\nTraining polynomial feature NN...")
history_poly = model_poly.fit(
    X_train_poly, y_train,
    batch_size=16,
    epochs=50,
    verbose=0,
    validation_split=0.2
)

y_pred_poly = model_poly.predict(X_test_poly, verbose=0)
y_pred_poly_class = np.argmax(y_pred_poly, axis=1)

print(f"\nPolynomial Feature NN Accuracy: {accuracy_score(y_test, y_pred_poly_class):.4f}")
print("\nPolynomial Feature NN Classification Report:")
print(classification_report(y_test, y_pred_poly_class, target_names=['Low (<32°C)', 'Medium (32-35°C)', 'High (35-38°C)', 'Very High (≥38°C)']))

# ===========================================================================================
# PART 6: Visualizations
# ===========================================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create visualisation folder if needed
import os
os.makedirs('visualisation', exist_ok=True)

# Plot 1: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

class_names = ['Low\n(<32°C)', 'Medium\n(32-35°C)', 'High\n(35-38°C)', 'Very High\n(≥38°C)']

# PNN confusion matrix
cm_pnn = confusion_matrix(y_test, pnn_pred)
sns.heatmap(cm_pnn, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=class_names, yticklabels=class_names)
axes[0].set_title('PNN Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Standard NN confusion matrix
cm_standard = confusion_matrix(y_test, y_pred_standard_class)
sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
axes[1].set_title('Standard NN Confusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# Polynomial Feature NN confusion matrix
cm_poly = confusion_matrix(y_test, y_pred_poly_class)
sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Oranges', ax=axes[2], xticklabels=class_names, yticklabels=class_names)
axes[2].set_title('Polynomial Feature NN Confusion Matrix', fontsize=12, fontweight='bold')
axes[2].set_ylabel('True Label')
axes[2].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('visualisation/07_confusion_matrices_classification.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_standard.history['loss'], label='Standard NN', linewidth=2)
axes[0].plot(history_poly.history['loss'], label='Polynomial Feature NN', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_standard.history['accuracy'], label='Standard NN', linewidth=2)
axes[1].plot(history_poly.history['accuracy'], label='Polynomial Feature NN', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Training Accuracy Comparison', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisation/08_training_history_classification.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Accuracy Comparison
accuracies = [
    accuracy_score(y_test, pnn_pred),
    accuracy_score(y_test, y_pred_standard_class),
    accuracy_score(y_test, y_pred_poly_class)
]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['Custom PNN', 'Standard NN', 'Polynomial\nFeature NN'], accuracies, 
              color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.6, edgecolor='black', linewidth=2)

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Classification Model Comparison - Accuracy', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualisation/09_accuracy_comparison_classification.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Predictions vs True Labels
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].scatter(range(len(y_test)), y_test, alpha=0.6, label='True', s=50, color='blue')
axes[0].scatter(range(len(y_test)), pnn_pred, alpha=0.6, label='Predicted', s=30, marker='x', color='red')
axes[0].set_ylabel('Class', fontsize=11)
axes[0].set_xlabel('Sample Index', fontsize=11)
axes[0].set_title('Custom PNN Predictions', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].set_ylim([-0.5, 3.5])
axes[0].grid(True, alpha=0.3)

axes[1].scatter(range(len(y_test)), y_test, alpha=0.6, label='True', s=50, color='blue')
axes[1].scatter(range(len(y_test)), y_pred_standard_class, alpha=0.6, label='Predicted', s=30, marker='x', color='red')
axes[1].set_ylabel('Class', fontsize=11)
axes[1].set_xlabel('Sample Index', fontsize=11)
axes[1].set_title('Standard NN Predictions', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].set_ylim([-0.5, 3.5])
axes[1].grid(True, alpha=0.3)

axes[2].scatter(range(len(y_test)), y_test, alpha=0.6, label='True', s=50, color='blue')
axes[2].scatter(range(len(y_test)), y_pred_poly_class, alpha=0.6, label='Predicted', s=30, marker='x', color='red')
axes[2].set_ylabel('Class', fontsize=11)
axes[2].set_xlabel('Sample Index', fontsize=11)
axes[2].set_title('Polynomial Feature NN Predictions', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].set_ylim([-0.5, 3.5])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisation/10_predictions_comparison_classification.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Visualizations saved to 'visualisation' folder:")
print("  - 07_confusion_matrices_classification.png")
print("  - 08_training_history_classification.png")
print("  - 09_accuracy_comparison_classification.png")
print("  - 10_predictions_comparison_classification.png")

print("\n" + "="*80)
print("CLASSIFICATION NEURAL NETWORK ANALYSIS COMPLETE")
print("="*80)
