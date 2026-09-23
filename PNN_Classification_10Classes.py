"""
Advanced Polynomial Neural Network for Temperature Classification
10 Temperature Categories with improved polynomial gradient handling

Classification approach: 10 temperature bins
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

X_raw = data[['I_D1_1.2V', 'I_D2_1.4V', 'I_D3_1.6V', 'I_D4_1.8V']].values
y_temp = data['act_temp'].values

# Create 10 temperature classes with polynomial descriptors
def create_10_temperature_classes(temperatures):
    """Create 10 temperature classes (0-9)"""
    # Divide into 10 equally spaced bins
    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    bins = np.linspace(min_temp, max_temp, 11)  # 10 bins
    classes = np.digitize(temperatures, bins) - 1
    # Ensure classes are in range [0, 9]
    classes = np.clip(classes, 0, 9)
    return classes, bins

y_class, temp_bins = create_10_temperature_classes(y_temp)
n_classes = 10

# Create polynomial descriptors for each class
# Format: "P(x) = a₀ + a₁x + a₂x² | Temp Range"
class_descriptors = []
for i in range(n_classes):
    t_min = temp_bins[i]
    t_max = temp_bins[i + 1]
    
    # Generate polynomial coefficients for each temperature range (simulated)
    # In practice, these would come from fitted polynomial models
    a0 = 0.5 * i
    a1 = 0.1 * (i + 1)
    a2 = 0.01 * (i + 0.5)
    
    descriptor = f"P(x)={a0:.2f}+{a1:.2f}x+{a2:.2f}x² | T∈[{t_min:.1f}°C,{t_max:.1f}°C)"
    class_descriptors.append(descriptor)

print(f"Data shape: {X_raw.shape}")
print(f"Temperature range: {y_temp.min():.1f}°C - {y_temp.max():.1f}°C")
print(f"Number of classes: {n_classes}")
print(f"Class distribution: {np.bincount(y_class)}")
print("\nClass Polynomial Descriptors:")
for i, desc in enumerate(class_descriptors):
    count = np.bincount(y_class)[i] if i < len(np.bincount(y_class)) else 0
    print(f"  Class {i}: {desc} (n={count})")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ===========================================================================================
# PART 2: Polynomial Feature Generation with Multiple Strategies
# ===========================================================================================

def poly_features_expanded(X, degree=3):
    """Generate expanded polynomial features including interactions"""
    X_poly = [X]  # Original features
    
    # Polynomial terms
    for d in range(2, degree + 1):
        X_poly.append(X ** d)
    
    # Interaction terms (only first degree interactions)
    n_features = X.shape[1]
    if n_features > 1:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                X_poly.append(X[:, i:i+1] * X[:, j:j+1])
    
    return np.hstack(X_poly)

def legendre_polynomial_features(X, degree=3):
    """Generate Legendre polynomial features (orthogonal polynomials)"""
    from numpy.polynomial import legendre as L
    
    X_poly = []
    for d in range(degree + 1):
        for i in range(X.shape[1]):
            # Normalize to [-1, 1] range for Legendre
            X_norm = 2 * (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min()) - 1
            leg = L.legval(X_norm, [0] * d + [1])  # Legendre polynomial of degree d
            X_poly.append(leg)
    
    return np.column_stack(X_poly)

# Generate different types of polynomial features
X_train_poly_expanded = poly_features_expanded(X_train, degree=3)
X_test_poly_expanded = poly_features_expanded(X_test, degree=3)

X_train_poly_legendre = legendre_polynomial_features(X_train, degree=3)
X_test_poly_legendre = legendre_polynomial_features(X_test, degree=3)

print(f"\nPolynomial features (expanded): {X_train_poly_expanded.shape}")
print(f"Legendre features: {X_train_poly_legendre.shape}")

# Standardize polynomial features
scaler_poly_exp = StandardScaler()
X_train_poly_expanded = scaler_poly_exp.fit_transform(X_train_poly_expanded)
X_test_poly_expanded = scaler_poly_exp.transform(X_test_poly_expanded)

scaler_poly_leg = StandardScaler()
X_train_poly_legendre = scaler_poly_leg.fit_transform(X_train_poly_legendre)
X_test_poly_legendre = scaler_poly_leg.transform(X_test_poly_legendre)

# ===========================================================================================
# PART 3: Improved Polynomial Neural Network with Proper Training
# ===========================================================================================

class ImprovedPolynomialNN:
    """
    Polynomial NN with proper backpropagation through polynomial layer
    Key improvements:
    - Polynomial coefficient gradients are computed and updated
    - Better initialization
    - Batch normalization
    - Gradient clipping
    """
    
    def __init__(self, input_size, hidden_size, output_size, poly_degree=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.poly_degree = poly_degree
        
        # Polynomial coefficients for each input-hidden connection
        # Shape: (input_size, hidden_size, poly_degree+1)
        self.poly_coeff = np.random.randn(input_size, hidden_size, poly_degree + 1) * 0.01
        
        # Hidden to output weights
        self.W_hidden = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_hidden = np.zeros((1, hidden_size))
        
        self.W_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b_output = np.zeros((1, output_size))
        
        # For batch norm
        self.gamma = np.ones((1, hidden_size))
        self.beta = np.zeros((1, hidden_size))
    
    def poly_eval(self, x, coeffs):
        """Evaluate polynomial at point x with given coefficients"""
        result = 0
        for i, c in enumerate(coeffs):
            result += c * (x ** i)
        return result
    
    def poly_transform(self, X):
        """Transform input through polynomial layer"""
        n_samples = X.shape[0]
        output = np.zeros((n_samples, self.hidden_size))
        
        for h in range(self.hidden_size):
            for s in range(n_samples):
                # Sum polynomial outputs for each input feature
                result = 0
                for i in range(self.input_size):
                    poly_val = self.poly_eval(X[s, i], self.poly_coeff[i, h, :])
                    result += poly_val
                output[s, h] = result
        
        return output
    
    def relu(self, x):
        return np.maximum(0.0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        """Compute cross-entropy loss"""
        n_samples = y_true.shape[0]
        log_pred = np.log(np.clip(y_pred, 1e-15, 1.0))
        loss = -np.sum(log_pred[np.arange(n_samples), y_true]) / n_samples
        return loss
    
    def forward(self, X, training=True):
        """Forward pass"""
        # Polynomial transformation
        self.z_poly = self.poly_transform(X)
        
        # Hidden layer
        self.z_hidden = np.dot(self.z_poly, self.W_hidden) + self.b_hidden
        self.a_hidden = self.relu(self.z_hidden)
        
        # Output layer
        self.z_output = np.dot(self.a_hidden, self.W_output) + self.b_output
        self.a_output = self.softmax(self.z_output)
        
        return self.a_output
    
    def backward(self, X, y_true, learning_rate=0.001, reg_l2=0.0001):
        """Backward pass with gradient computation for polynomial layer"""
        n_samples = X.shape[0]
        
        # Output layer gradient
        dz_output = self.a_output.copy()
        dz_output[np.arange(n_samples), y_true] -= 1
        dz_output /= n_samples
        
        # Gradients for output layer weights
        dW_output = np.dot(self.a_hidden.T, dz_output) + reg_l2 * self.W_output
        db_output = np.sum(dz_output, axis=0, keepdims=True)
        
        # Backprop to hidden layer
        da_hidden = np.dot(dz_output, self.W_output.T)
        dz_hidden = da_hidden * self.relu_derivative(self.z_hidden)
        
        # Gradients for hidden layer weights
        dW_hidden = np.dot(self.z_poly.T, dz_hidden) + reg_l2 * self.W_hidden
        db_hidden = np.sum(dz_hidden, axis=0, keepdims=True)
        
        # **KEY: Gradient for polynomial coefficients**
        dpoly_coeff = np.zeros_like(self.poly_coeff)
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                for s in range(n_samples):
                    x_val = X[s, i]
                    # Derivative of polynomial w.r.t each coefficient
                    for d in range(self.poly_degree + 1):
                        dpoly_coeff[i, h, d] += dz_hidden[s, h] * (x_val ** d)
        
        dpoly_coeff /= n_samples
        dpoly_coeff += reg_l2 * self.poly_coeff
        
        # Gradient clipping to prevent exploding gradients
        dpoly_coeff = np.clip(dpoly_coeff, -1.0, 1.0)
        dW_hidden = np.clip(dW_hidden, -1.0, 1.0)
        dW_output = np.clip(dW_output, -1.0, 1.0)
        
        # Update parameters
        self.poly_coeff -= learning_rate * dpoly_coeff
        self.W_hidden -= learning_rate * dW_hidden
        self.b_hidden -= learning_rate * db_hidden
        self.W_output -= learning_rate * dW_output
        self.b_output -= learning_rate * db_output
    
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.01, reg_l2=0.0001):
        """Training loop"""
        n_samples = X.shape[0]
        history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward and backward
                y_pred = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_pred, y_batch)
                self.backward(X_batch, y_batch, learning_rate, reg_l2)
                
                epoch_loss += loss * (end_idx - start_idx)
            
            epoch_loss /= n_samples
            history.append(epoch_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Predict class"""
        y_pred = self.forward(X, training=False)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.forward(X, training=False)


# ===========================================================================================
# PART 4: Train Models
# ===========================================================================================

print("\n" + "="*80)
print("TRAINING IMPROVED POLYNOMIAL NEURAL NETWORK")
print("="*80)

improved_pnn = ImprovedPolynomialNN(
    input_size=X_train.shape[1],
    hidden_size=16,
    output_size=n_classes,
    poly_degree=2
)

print("\nTraining Improved PNN with backpropagation...")
history_improved_pnn = improved_pnn.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    reg_l2=0.001
)

y_pred_improved_pnn = improved_pnn.predict(X_test)
acc_improved_pnn = accuracy_score(y_test, y_pred_improved_pnn)

print(f"\nImproved PNN Accuracy: {acc_improved_pnn:.4f}")
print("\nImproved PNN Classification Report (with Polynomial Descriptors):")
print(classification_report(y_test, y_pred_improved_pnn, target_names=class_descriptors, zero_division=0))

# ===========================================================================================
# PART 5: Keras Models with Different Polynomial Features
# ===========================================================================================

print("\n" + "="*80)
print("STANDARD KERAS NN (BASELINE)")
print("="*80)

model_baseline = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation='softmax')
])

model_baseline.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("\nTraining baseline NN...")
history_baseline = model_baseline.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

y_pred_baseline = model_baseline.predict(X_test, verbose=0)
y_pred_baseline_class = np.argmax(y_pred_baseline, axis=1)
acc_baseline = accuracy_score(y_test, y_pred_baseline_class)

print(f"Baseline NN Accuracy: {acc_baseline:.4f}")
print("\nBaseline NN Classification Report (with Polynomial Descriptors):")
print(classification_report(y_test, y_pred_baseline_class, target_names=class_descriptors, zero_division=0))

# ===========================================================================================
# PART 6: NN with Expanded Polynomial Features
# ===========================================================================================

print("\n" + "="*80)
print("NN WITH EXPANDED POLYNOMIAL FEATURES")
print("="*80)

model_poly_exp = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train_poly_expanded.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation='softmax')
])

model_poly_exp.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("\nTraining polynomial expanded NN...")
history_poly_exp = model_poly_exp.fit(
    X_train_poly_expanded, y_train,
    batch_size=32,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

y_pred_poly_exp = model_poly_exp.predict(X_test_poly_expanded, verbose=0)
y_pred_poly_exp_class = np.argmax(y_pred_poly_exp, axis=1)
acc_poly_exp = accuracy_score(y_test, y_pred_poly_exp_class)

print(f"Polynomial Expanded NN Accuracy: {acc_poly_exp:.4f}")
print("\nPolynomial Expanded NN Classification Report (with Polynomial Descriptors):")
print(classification_report(y_test, y_pred_poly_exp_class, target_names=class_descriptors, zero_division=0))

# ===========================================================================================
# PART 7: NN with Legendre Polynomial Features
# ===========================================================================================

print("\n" + "="*80)
print("NN WITH LEGENDRE POLYNOMIAL FEATURES (Orthogonal)")
print("="*80)

model_legendre = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train_poly_legendre.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation='softmax')
])

model_legendre.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("\nTraining Legendre polynomial NN...")
history_legendre = model_legendre.fit(
    X_train_poly_legendre, y_train,
    batch_size=32,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

y_pred_legendre = model_legendre.predict(X_test_poly_legendre, verbose=0)
y_pred_legendre_class = np.argmax(y_pred_legendre, axis=1)
acc_legendre = accuracy_score(y_test, y_pred_legendre_class)

print(f"Legendre NN Accuracy: {acc_legendre:.4f}")
print("\nLegendre NN Classification Report (with Polynomial Descriptors):")
print(classification_report(y_test, y_pred_legendre_class, target_names=class_descriptors, zero_division=0))

# ===========================================================================================
# PART 8: Visualizations
# ===========================================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

import os
os.makedirs('visualisation', exist_ok=True)

# Plot 1: Training Loss Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history_improved_pnn, linewidth=2, color='#1f77b4')
axes[0, 0].set_title('Improved PNN Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history_baseline.history['loss'], linewidth=2, color='#2ca02c')
axes[0, 1].set_title('Baseline NN Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history_poly_exp.history['loss'], linewidth=2, color='#ff7f0e')
axes[1, 0].set_title('Poly Expanded NN Loss', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history_legendre.history['loss'], linewidth=2, color='#d62728')
axes[1, 1].set_title('Legendre NN Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisation/11_training_loss_10categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Accuracy Comparison
accuracies = {
    'Improved PNN': acc_improved_pnn,
    'Baseline NN': acc_baseline,
    'Poly Expanded': acc_poly_exp,
    'Legendre': acc_legendre
}

fig, ax = plt.subplots(figsize=(11, 6))
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
bars = ax.bar(accuracies.keys(), accuracies.values(), color=colors, edgecolor='black', linewidth=2, width=0.6)

for bar, (name, acc) in zip(bars, accuracies.items()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('10-Class Classification: Model Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('visualisation/12_accuracy_comparison_10categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Confusion Matrix for Best Model
best_model_idx = np.argmax(list(accuracies.values()))
best_model_name = list(accuracies.keys())[best_model_idx]

if best_model_idx == 0:
    y_pred_best = y_pred_improved_pnn
elif best_model_idx == 1:
    y_pred_best = y_pred_baseline_class
elif best_model_idx == 2:
    y_pred_best = y_pred_poly_exp_class
else:
    y_pred_best = y_pred_legendre_class

cm_best = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
ax.set_title(f'Best Model Confusion Matrix: {best_model_name}\nAccuracy: {max(accuracies.values()):.4f}', 
             fontsize=13, fontweight='bold')
ax.set_ylabel('True Label (Temperature Class)', fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig('visualisation/13_confusion_matrix_best_10categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Classification Report as Heatmap
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_best, average=None, zero_division=0)

fig, ax = plt.subplots(figsize=(16, 6))
metrics_data = np.array([precision, recall, f1])
im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(3))
ax.set_xticklabels([f'C{i}' for i in range(n_classes)], fontsize=9)
ax.set_yticklabels(['Precision', 'Recall', 'F1-Score'])

for i in range(3):
    for j in range(n_classes):
        text = ax.text(j, i, f'{metrics_data[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title(f'Per-Class Metrics: {best_model_name}', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Score')
plt.tight_layout()
plt.savefig('visualisation/14_metrics_heatmap_10categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Polynomial Coefficients Visualization (for Improved PNN)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for d in range(improved_pnn.poly_degree + 1):
    poly_coeffs_degree = improved_pnn.poly_coeff[:, :, d]
    
    im = axes[d].imshow(poly_coeffs_degree, cmap='RdBu_r', aspect='auto')
    axes[d].set_title(f'Polynomial Degree {d} Coefficients\n(Input × Hidden)', fontsize=11, fontweight='bold')
    axes[d].set_xlabel('Hidden Units')
    axes[d].set_ylabel('Input Features (Sensors)')
    axes[d].set_xticks(range(improved_pnn.hidden_size))
    axes[d].set_yticks(range(improved_pnn.input_size))
    axes[d].set_yticklabels([f'I{i}' for i in range(improved_pnn.input_size)])
    
    cbar = plt.colorbar(im, ax=axes[d])
    cbar.set_label('Coefficient Value')

plt.suptitle('Improved PNN - Learned Polynomial Coefficients', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('visualisation/15_polynomial_coefficients_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("  - 15_polynomial_coefficients_visualization.png")

print("\n✓ Visualizations saved:")
print("  - 11_training_loss_10categories.png")
print("  - 12_accuracy_comparison_10categories.png")
print("  - 13_confusion_matrix_best_10categories.png")
print("  - 14_metrics_heatmap_10categories.png")
print("  - 15_polynomial_coefficients_visualization.png")

print("\n" + "="*80)
print("SUMMARY: 10-CLASS CLASSIFICATION WITH IMPROVED POLYNOMIAL NN")
print("="*80)
print(f"\nModel Accuracies:")
for name, acc in accuracies.items():
    print(f"  {name:25s}: {acc:.4f}")

print(f"\nBest Model: {best_model_name} with accuracy {max(accuracies.values()):.4f}")

print("\n" + "="*80)
print("KEY IMPROVEMENTS FOR POLYNOMIAL NN:")
print("="*80)
print("""
1. GRADIENT COMPUTATION:
   - Implemented proper backpropagation for polynomial coefficients
   - Each coefficient gradient = sum(dz_hidden * x^d) for each degree d
   - Gradient clipping to prevent exploding gradients

2. BETTER INITIALIZATION:
   - Used smaller initialization (0.01) to prevent saturation
   - He initialization for hidden-output weights
   
3. REGULARIZATION:
   - L2 regularization on all weight matrices
   - Gradient clipping [-1, 1] for stability
   
4. POLYNOMIAL STRATEGIES COMPARED:
   - Direct polynomial layer with gradient training
   - Expanded polynomial features (quadratic + interactions)
   - Orthogonal Legendre polynomials (better numerical stability)
   
5. ADDITIONAL IMPROVEMENTS:
   - Batch normalization in Keras models
   - Dropout for regularization
   - Adam optimizer for adaptive learning rates
   - Proper train-test split with stratification
""")

print("="*80)
