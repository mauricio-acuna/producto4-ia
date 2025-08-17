# 🔬 Módulo A - Fundamentos de AI Engineering

**Duración:** 3-4 semanas  
**Objetivo:** Establecer base técnica sólida en AI/ML con hands-on practice

## 🎯 Objetivos de Aprendizaje

Al finalizar este módulo serás capaz de:

1. **Dominar** fundamentos matemáticos aplicados a AI/ML
2. **Implementar** algoritmos de ML desde cero y con libraries
3. **Configurar** entorno de desarrollo Python profesional
4. **Aplicar** técnicas de data preprocessing y feature engineering
5. **Desplegar** modelos básicos con MLOps fundamentals

## 📋 Contenido del Módulo

### Semana 1: Mathematical Foundations
- **1.1** Linear algebra para ML (vectores, matrices, eigenvalues)
- **1.2** Statistics y probability theory aplicada
- **1.3** Calculus y optimization basics (gradients, chain rule)
- **1.4** Information theory fundamentals (entropy, KL divergence)

### Semana 2: Algorithm Implementation
- **2.1** Linear regression desde cero y con scikit-learn
- **2.2** Logistic regression y classification metrics
- **2.3** Neural networks básicos con NumPy y PyTorch
- **2.4** Decision trees y ensemble methods

### Semana 3: Python AI/ML Ecosystem
- **3.1** Environment setup (conda, venv, Docker)
- **3.2** Core libraries: NumPy, Pandas, Matplotlib, Seaborn
- **3.3** ML libraries: scikit-learn, PyTorch, TensorFlow
- **3.4** Data tools: Jupyter, VS Code, Git workflows

### Semana 4: Data Engineering & MLOps Basics
- **4.1** Data preprocessing y cleaning pipelines
- **4.2** Feature engineering y selection techniques
- **4.3** Model training, validation y testing
- **4.4** Basic deployment con FastAPI y Docker

## 🧮 Mathematical Foundations Deep Dive

### Linear Algebra Essentials

#### Vector Operations & Spaces
```python
import numpy as np
import matplotlib.pyplot as plt

# Vector fundamentals for ML
def vector_fundamentals():
    """Core vector operations used in ML algorithms"""
    
    # Vectors as feature representations
    x1 = np.array([2.5, 1.8, 3.2, 0.9])  # Sample 1 features (e.g., height, weight, age, income)
    x2 = np.array([1.9, 2.1, 2.8, 1.2])  # Sample 2 features
    
    # Dot product - similarity measure
    # High dot product = similar feature vectors = similar samples
    similarity = np.dot(x1, x2)
    print(f"Dot product (similarity): {similarity:.2f}")
    
    # L2 norm - magnitude/distance  
    # Represents the "length" of the feature vector in high-dimensional space
    magnitude_x1 = np.linalg.norm(x1)
    distance = np.linalg.norm(x1 - x2)  # Euclidean distance between samples
    print(f"L2 norm of x1: {magnitude_x1:.2f}")
    print(f"Euclidean distance: {distance:.2f}")
    
    # Cosine similarity - normalized similarity (range: -1 to 1)
    # Measures angle between vectors, independent of magnitude
    # Used in recommendation systems, text similarity, etc.
    cosine_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    print(f"Cosine similarity: {cosine_sim:.3f}")
    
    # Practical interpretation:
    # - cosine_sim ≈ 1: Very similar samples
    # - cosine_sim ≈ 0: Orthogonal/unrelated samples  
    # - cosine_sim ≈ -1: Opposite samples
    
    return x1, x2, similarity, distance

# Matrix operations for neural networks
def matrix_operations():
    """Matrix operations fundamental to neural networks"""
    
    # Weight matrix (3 neurons, 4 inputs)
    # Each row represents one neuron's weights for all inputs
    W = np.random.randn(3, 4) * 0.1
    # Bias vector (one bias per neuron)
    b = np.random.randn(3) * 0.1
    # Input batch (2 samples, 4 features each)
    X = np.random.randn(2, 4)
    
    # Forward pass: Y = XW^T + b
    # This is the core computation in neural networks
    # X: (batch_size, input_dim) -> Y: (batch_size, output_dim)
    Y = np.dot(X, W.T) + b
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")  
    print(f"Output shape: {Y.shape}")
    
    # Eigendecomposition for PCA (Principal Component Analysis)
    # Finds the directions of maximum variance in the data
    cov_matrix = np.cov(X.T)  # Covariance matrix of features
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print(f"Eigenvalues (variance along each principal component): {eigenvalues}")
    
    # Practical applications:
    # - Eigenvalues tell us how much variance each component captures
    # - Eigenvectors are the directions of principal components
    # - Used for dimensionality reduction, data visualization
    
    return W, b, X, Y, eigenvalues, eigenvectors

# Real-world example: Document similarity using vectors
def document_similarity_example():
    """Practical example: Finding similar documents using vector operations"""
    
    # Simulate TF-IDF vectors for 3 documents
    # Each element represents the importance of a word in the document
    doc1 = np.array([0.2, 0.8, 0.1, 0.0, 0.3])  # "machine learning python"
    doc2 = np.array([0.1, 0.7, 0.2, 0.0, 0.4])  # "python machine learning"  
    doc3 = np.array([0.0, 0.1, 0.0, 0.9, 0.0])  # "cat dog animal"
    
    # Calculate cosine similarities
    sim_1_2 = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
    sim_1_3 = np.dot(doc1, doc3) / (np.linalg.norm(doc1) * np.linalg.norm(doc3))
    sim_2_3 = np.dot(doc2, doc3) / (np.linalg.norm(doc2) * np.linalg.norm(doc3))
    
    print(f"Similarity doc1-doc2 (both about ML): {sim_1_2:.3f}")
    print(f"Similarity doc1-doc3 (different topics): {sim_1_3:.3f}")
    print(f"Similarity doc2-doc3 (different topics): {sim_2_3:.3f}")
    
    # Expected result: doc1-doc2 high similarity, others low
    # This is the foundation of search engines, recommendation systems
    
    return sim_1_2, sim_1_3, sim_2_3

# Example usage with detailed explanations
if __name__ == "__main__":
    print("=== Basic Vector Operations ===")
    vector_fundamentals()
    
    print("\n=== Neural Network Matrix Operations ===")
    matrix_operations()
    
    print("\n=== Real-world Application: Document Similarity ===")
    document_similarity_example()
```

#### Probability & Statistics for ML
```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import softmax

def probability_fundamentals():
    """Core probability concepts for ML"""
    
    # Gaussian distribution - central to many ML algorithms
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 100)
    gaussian = stats.norm.pdf(x, mu, sigma)
    
    # Central Limit Theorem demonstration
    sample_means = []
    for _ in range(1000):
        sample = np.random.exponential(2, 30)  # Non-normal distribution
        sample_means.append(np.mean(sample))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(x, gaussian)
    plt.title('Standard Normal Distribution')
    
    plt.subplot(132)
    plt.hist(sample_means, bins=30, density=True, alpha=0.7)
    plt.title('Sample Means (CLT)')
    
    # Bayes' Theorem example
    # P(Disease|Test+) = P(Test+|Disease) * P(Disease) / P(Test+)
    p_disease = 0.01  # Prior: 1% have disease
    p_test_given_disease = 0.95  # Test sensitivity
    p_test_given_no_disease = 0.05  # False positive rate
    
    p_test_positive = (p_test_given_disease * p_disease + 
                      p_test_given_no_disease * (1 - p_disease))
    p_disease_given_test = (p_test_given_disease * p_disease) / p_test_positive
    
    print(f"P(Disease|Test+) = {p_disease_given_test:.3f}")
    
    return gaussian, sample_means, p_disease_given_test

def information_theory():
    """Information theory concepts for ML"""
    
    # Entropy calculation
    def entropy(probabilities):
        """Calculate Shannon entropy"""
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # Example: Binary classification entropy
    p_class_1 = np.array([0.5, 0.9, 0.1, 0.7])
    p_class_0 = 1 - p_class_1
    
    entropies = []
    for p1, p0 in zip(p_class_1, p_class_0):
        h = entropy(np.array([p1, p0]))
        entropies.append(h)
        print(f"P(class=1)={p1:.1f}, Entropy={h:.3f}")
    
    # KL divergence
    def kl_divergence(p, q):
        """Calculate KL divergence between distributions"""
        return np.sum(p * np.log(p / q))
    
    # True distribution vs predicted distribution
    p_true = np.array([0.7, 0.2, 0.1])
    p_pred = np.array([0.6, 0.3, 0.1])
    kl_div = kl_divergence(p_true, p_pred)
    print(f"KL divergence: {kl_div:.4f}")
    
    return entropies, kl_div

# Example usage
probability_fundamentals()
information_theory()
```

### Calculus & Optimization

#### Gradients & Backpropagation
```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_fundamentals():
    """Understanding gradients for optimization with detailed explanations"""
    
    # Simple quadratic function: f(x) = x^2 + 2x + 1 = (x+1)^2
    # This function has a global minimum at x = -1, f(-1) = 0
    def f(x):
        return x**2 + 2*x + 1
    
    def df_dx(x):
        return 2*x + 2  # Derivative: slope of the function at point x
    
    # Gradient descent algorithm
    x = 5.0  # Starting point (far from minimum)
    learning_rate = 0.1  # Step size - too high causes oscillation, too low is slow
    history = [x]
    
    print("Gradient Descent Optimization Process:")
    print(f"Target: Find minimum of f(x) = x² + 2x + 1")
    print(f"Starting at x = {x:.4f}, f(x) = {f(x):.4f}")
    print(f"True minimum at x = -1.0, f(-1) = 0.0\n")
    
    for i in range(20):
        gradient = df_dx(x)  # Calculate slope at current point
        x_new = x - learning_rate * gradient  # Move opposite to gradient direction
        
        print(f"Step {i+1:2d}: x = {x:8.4f} → {x_new:8.4f}, " +
              f"f(x) = {f(x):8.4f}, gradient = {gradient:6.3f}")
        
        x = x_new
        history.append(x)
        
        # Stop if we're very close to the minimum
        if abs(gradient) < 1e-6:
            print(f"Converged! Gradient is nearly zero: {gradient:.2e}")
            break
    
    # Visualization
    x_plot = np.linspace(-4, 6, 100)
    y_plot = f(x_plot)
    
    plt.figure(figsize=(12, 8))
    
    # Main plot: function and optimization path
    plt.subplot(2, 2, 1)
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x² + 2x + 1')
    plt.plot(history, [f(x) for x in history], 'ro-', linewidth=2, 
             markersize=6, label='Gradient Descent Path')
    plt.plot(-1, 0, 'g*', markersize=15, label='True Minimum')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('Gradient Descent Optimization')
    plt.grid(True, alpha=0.3)
    
    # Plot convergence over iterations
    plt.subplot(2, 2, 2)
    plt.plot([f(x) for x in history], 'b-o', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value f(x)')
    plt.title('Convergence: Function Value Over Time')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better see convergence
    
    # Plot gradient magnitude over iterations
    plt.subplot(2, 2, 3)
    gradients = [abs(df_dx(x)) for x in history[:-1]]  # Gradients at each step
    plt.plot(gradients, 'r-o', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('|Gradient|')
    plt.title('Gradient Magnitude (Should → 0)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Learning rate effect visualization
    plt.subplot(2, 2, 4)
    learning_rates = [0.01, 0.1, 0.5, 1.1]
    colors = ['blue', 'green', 'orange', 'red']
    
    for lr, color in zip(learning_rates, colors):
        x_temp = 5.0
        history_temp = []
        for _ in range(15):
            history_temp.append(x_temp)
            gradient = df_dx(x_temp)
            x_temp = x_temp - lr * gradient
            if abs(x_temp) > 10:  # Prevent divergence
                break
                
        plt.plot(range(len(history_temp)), history_temp, 
                color=color, marker='o', label=f'LR = {lr}')
    
    plt.axhline(y=-1, color='black', linestyle='--', alpha=0.5, label='Target x = -1')
    plt.xlabel('Iteration')
    plt.ylabel('x position')
    plt.title('Learning Rate Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal result: x = {history[-1]:.6f}, f(x) = {f(history[-1]):.6f}")
    print(f"Error from true minimum: {abs(history[-1] + 1):.6f}")
    
    return history

def multivariate_gradients():
    """Gradients for multivariate functions with detailed visualization"""
    
    # Function: f(x,y) = x² + y² - 2x - 4y + 5
    # This is a paraboloid with minimum at (1, 2), f(1,2) = 0
    def f(x, y):
        return x**2 + y**2 - 2*x - 4*y + 5
    
    def gradient(x, y):
        # Partial derivatives
        df_dx = 2*x - 2  # ∂f/∂x
        df_dy = 2*y - 4  # ∂f/∂y
        return np.array([df_dx, df_dy])
    
    # Gradient descent in 2D
    point = np.array([5.0, 5.0])  # Starting point
    learning_rate = 0.1
    path = [point.copy()]
    
    print("2D Gradient Descent:")
    print(f"Function: f(x,y) = x² + y² - 2x - 4y + 5")
    print(f"Starting at ({point[0]:.1f}, {point[1]:.1f})")
    print(f"True minimum at (1.0, 2.0), f(1,2) = 0\n")
    
    for i in range(30):
        grad = gradient(point[0], point[1])
        grad_magnitude = np.linalg.norm(grad)
        
        if i < 5 or i % 5 == 0:  # Print first few and every 5th iteration
            print(f"Iter {i:2d}: ({point[0]:6.3f}, {point[1]:6.3f}), " +
                  f"f = {f(point[0], point[1]):8.4f}, |grad| = {grad_magnitude:.4f}")
        
        point = point - learning_rate * grad
        path.append(point.copy())
        
        if grad_magnitude < 1e-6:
            print(f"Converged at iteration {i}")
            break
    
    path = np.array(path)
    
    # Create comprehensive visualization
    x = np.linspace(-1, 6, 100)
    y = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Contour plot with optimization path
    ax1 = axes[0, 0]
    contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(path[:, 0], path[:, 1], 'ro-', linewidth=3, markersize=6, 
             label='Optimization Path')
    ax1.plot(1, 2, 'g*', markersize=20, label='Global Minimum (1,2)')
    ax1.plot(path[0, 0], path[0, 1], 'bs', markersize=10, label='Start (5,5)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('2D Gradient Descent Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D surface plot
    ax2 = axes[0, 1] = plt.subplot(2, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
    ax2.plot(path[:, 0], path[:, 1], [f(p[0], p[1]) for p in path], 
             'ro-', linewidth=3, markersize=4, label='Path')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_title('3D Optimization Path')
    
    # Convergence analysis
    ax3 = axes[1, 0]
    function_values = [f(p[0], p[1]) for p in path]
    ax3.plot(function_values, 'b-o', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Function Value')
    ax3.set_title('Function Value Convergence')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Distance from optimum
    ax4 = axes[1, 1]
    distances = [np.linalg.norm(p - np.array([1, 2])) for p in path]
    ax4.plot(distances, 'r-o', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Distance from Optimum')
    ax4.set_title('Distance to True Minimum')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal point: ({path[-1][0]:.6f}, {path[-1][1]:.6f})")
    print(f"Final function value: {f(path[-1][0], path[-1][1]):.6f}")
    print(f"Distance from true minimum: {np.linalg.norm(path[-1] - np.array([1, 2])):.6f}")
    
    return path

# Example usage with comprehensive explanations
print("=== Single Variable Gradient Descent ===")
gradient_fundamentals()
print("\n" + "="*60 + "\n")
print("=== Multivariate Gradient Descent ===")
multivariate_gradients()
```

# Neural network gradient example
def simple_neural_network_gradients():
    """Backpropagation in a simple neural network"""
    
    # Simple 2-layer network
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # Network parameters
    np.random.seed(42)
    W1 = np.random.randn(2, 3) * 0.5  # Input to hidden
    b1 = np.random.randn(3) * 0.5
    W2 = np.random.randn(3, 1) * 0.5  # Hidden to output
    b2 = np.random.randn(1) * 0.5
    
    # Training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR function
    
    learning_rate = 1.0
    losses = []
    
    for epoch in range(1000):
        # Forward pass
        z1 = np.dot(X, W1.T) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2.T) + b2
        a2 = sigmoid(z2)
        
        # Loss (MSE)
        loss = np.mean((a2 - y)**2)
        losses.append(loss)
        
        # Backward pass
        # Output layer gradients
        dL_da2 = 2 * (a2 - y) / len(X)
        da2_dz2 = sigmoid_derivative(z2)
        dL_dz2 = dL_da2 * da2_dz2
        
        dL_dW2 = np.dot(dL_dz2.T, a1)
        dL_db2 = np.sum(dL_dz2, axis=0)
        
        # Hidden layer gradients
        dL_da1 = np.dot(dL_dz2, W2)
        da1_dz1 = sigmoid_derivative(z1)
        dL_dz1 = dL_da1 * da1_dz1
        
        dL_dW1 = np.dot(dL_dz1.T, X)
        dL_db1 = np.sum(dL_dz1, axis=0)
        
        # Update parameters
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    # Test the trained network
    print("\nFinal predictions:")
    z1 = np.dot(X, W1.T) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2.T) + b2
    predictions = sigmoid(z2)
    
    for i, (input_val, target, pred) in enumerate(zip(X, y, predictions)):
        print(f"Input: {input_val}, Target: {target[0]}, Prediction: {pred[0]:.4f}")
    
    return losses, W1, W2, b1, b2

# Example usage
gradient_fundamentals()
multivariate_gradients()
simple_neural_network_gradients()
```

## 🐍 Python AI/ML Ecosystem Mastery

### Environment Setup & Best Practices

#### Professional Development Environment
```bash
# Create conda environment for AI/ML projects
conda create -n ai-engineering python=3.11
conda activate ai-engineering

# Core scientific computing
conda install numpy pandas matplotlib seaborn jupyter

# Machine learning libraries
conda install scikit-learn pytorch torchvision torchaudio -c pytorch
pip install transformers datasets accelerate

# Data visualization and analysis
pip install plotly dash streamlit

# MLOps and experimentation
pip install mlflow wandb optuna

# Development tools
pip install black isort flake8 pytest pre-commit

# API development
pip install fastapi uvicorn pydantic

# Containerization
pip install docker-compose
```

#### Project Structure Template
```
ai-project/
├── .env                          # Environment variables
├── .gitignore                    # Git ignore patterns
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── Dockerfile                    # Container configuration
├── docker-compose.yml            # Multi-service setup
├── README.md                     # Project documentation
├── src/                          # Source code
│   ├── __init__.py
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── neural_networks.py
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── api/                      # API endpoints
│       ├── __init__.py
│       └── main.py
├── notebooks/                    # Jupyter notebooks
│   ├── exploration.ipynb
│   └── experimentation.ipynb
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── configs/                      # Configuration files
│   └── model_config.yaml
└── scripts/                      # Utility scripts
    ├── download_data.py
    └── setup.py
```

### Core Libraries Deep Dive

#### NumPy for Numerical Computing
```python
import numpy as np
import time

def numpy_fundamentals():
    """Essential NumPy operations for ML"""
    
    # Array creation and manipulation
    print("=== Array Creation ===")
    
    # Various ways to create arrays
    zeros = np.zeros((3, 4))
    ones = np.ones((2, 3))
    identity = np.eye(4)
    random_uniform = np.random.uniform(0, 1, (3, 3))
    random_normal = np.random.normal(0, 1, (3, 3))
    
    print(f"Zeros shape: {zeros.shape}")
    print(f"Random uniform:\n{random_uniform}")
    
    # Broadcasting - crucial for vectorized operations
    print("\n=== Broadcasting ===")
    matrix = np.random.randn(4, 3)
    row_means = np.mean(matrix, axis=1, keepdims=True)
    centered_matrix = matrix - row_means
    
    print(f"Original shape: {matrix.shape}")
    print(f"Row means shape: {row_means.shape}")
    print(f"Centered shape: {centered_matrix.shape}")
    
    # Vectorization performance comparison
    print("\n=== Vectorization Performance ===")
    
    # Slow loop-based approach
    def slow_dot_product(a, b):
        result = 0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result
    
    # Fast vectorized approach
    def fast_dot_product(a, b):
        return np.dot(a, b)
    
    # Performance test
    large_vector = np.random.randn(100000)
    
    start_time = time.time()
    slow_result = slow_dot_product(large_vector, large_vector)
    slow_time = time.time() - start_time
    
    start_time = time.time()
    fast_result = fast_dot_product(large_vector, large_vector)
    fast_time = time.time() - start_time
    
    print(f"Slow approach: {slow_time:.4f}s")
    print(f"Fast approach: {fast_time:.4f}s")
    print(f"Speedup: {slow_time/fast_time:.1f}x")
    
    return matrix, centered_matrix

def advanced_numpy_operations():
    """Advanced NumPy for ML algorithms"""
    
    # Boolean indexing and masking
    data = np.random.randn(1000, 5)
    
    # Find outliers (> 2 standard deviations)
    outlier_mask = np.abs(data) > 2
    outlier_indices = np.where(outlier_mask)
    
    print(f"Total outliers: {np.sum(outlier_mask)}")
    print(f"Outlier percentage: {np.mean(outlier_mask)*100:.2f}%")
    
    # Advanced indexing
    # Select specific rows and columns
    selected_data = data[::10, [0, 2, 4]]  # Every 10th row, columns 0,2,4
    
    # Sorting and argsort
    scores = np.random.randn(100)
    sorted_indices = np.argsort(scores)[::-1]  # Descending order
    top_10_indices = sorted_indices[:10]
    
    print(f"Top 10 scores: {scores[top_10_indices]}")
    
    # Array splitting and concatenation
    arrays = np.array_split(data, 5)  # Split into 5 parts
    reconstructed = np.concatenate(arrays, axis=0)
    
    assert np.array_equal(data, reconstructed), "Reconstruction failed"
    
    return data, selected_data, top_10_indices

# Example usage
numpy_fundamentals()
advanced_numpy_operations()
```

#### Pandas for Data Manipulation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pandas_fundamentals():
    """Essential Pandas operations for data preprocessing"""
    
    # Create sample dataset
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'user_id': np.random.randint(1, 101, 1000),
        'feature_1': np.random.normal(50, 15, 1000),
        'feature_2': np.random.exponential(2, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
        'target': np.random.binomial(1, 0.3, 1000)
    })
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, 50, replace=False)
    df.loc[missing_indices, 'feature_1'] = np.nan
    
    print("=== Data Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Data exploration
    print("\n=== Statistical Summary ===")
    print(df.describe())
    
    # Categorical analysis
    print("\n=== Category Distribution ===")
    print(df['category'].value_counts(normalize=True))
    
    return df

def advanced_pandas_operations(df):
    """Advanced Pandas for feature engineering"""
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Rolling statistics
    df = df.sort_values('date')
    df['feature_1_rolling_mean'] = df['feature_1'].rolling(window=7).mean()
    df['feature_1_rolling_std'] = df['feature_1'].rolling(window=7).std()
    
    # Lag features
    df['feature_1_lag_1'] = df['feature_1'].shift(1)
    df['feature_1_lag_7'] = df['feature_1'].shift(7)
    
    # Aggregation by groups
    user_stats = df.groupby('user_id').agg({
        'feature_1': ['mean', 'std', 'count'],
        'target': 'mean'
    }).round(3)
    
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
    user_stats = user_stats.reset_index()
    
    # Merge back to original data
    df = df.merge(user_stats, on='user_id', how='left')
    
    # Handle missing values
    # Forward fill for time series
    df['feature_1'] = df['feature_1'].fillna(method='ffill')
    
    # Fill remaining with median
    df['feature_1'] = df['feature_1'].fillna(df['feature_1'].median())
    
    # Categorical encoding
    df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')
    
    print("=== Feature Engineering Results ===")
    print(f"New shape: {df_encoded.shape}")
    print(f"New columns: {list(df_encoded.columns[-10:])}")
    
    return df_encoded, user_stats

def data_quality_checks(df):
    """Data quality assessment functions"""
    
    def check_data_quality(dataframe):
        """Comprehensive data quality report"""
        
        quality_report = {
            'total_rows': len(dataframe),
            'total_columns': len(dataframe.columns),
            'missing_values': dataframe.isnull().sum().sum(),
            'duplicate_rows': dataframe.duplicated().sum(),
            'memory_usage': dataframe.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Check for constant columns
        constant_columns = []
        for col in dataframe.columns:
            if dataframe[col].nunique() <= 1:
                constant_columns.append(col)
        quality_report['constant_columns'] = constant_columns
        
        # Check for high cardinality columns
        high_cardinality = []
        for col in dataframe.select_dtypes(include=['object']).columns:
            if dataframe[col].nunique() > 0.8 * len(dataframe):
                high_cardinality.append(col)
        quality_report['high_cardinality_columns'] = high_cardinality
        
        return quality_report
    
    # Run quality checks
    quality_report = check_data_quality(df)
    
    print("=== Data Quality Report ===")
    for key, value in quality_report.items():
        print(f"{key}: {value}")
    
    # Correlation analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_value
                ))
    
    print("\n=== High Correlation Pairs (>0.8) ===")
    for col1, col2, corr in high_corr_pairs:
        print(f"{col1} - {col2}: {corr:.3f}")
    
    return quality_report, correlation_matrix

# Example usage
df = pandas_fundamentals()
df_encoded, user_stats = advanced_pandas_operations(df)
quality_report, correlation_matrix = data_quality_checks(df_encoded)
```

## 🤖 Algorithm Implementation from Scratch

### Linear Regression Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegressionFromScratch:
    """Linear Regression implemented from scratch"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Train the linear regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_predicted = self.predict(X)
            
            # Cost function (MSE)
            cost = np.mean((y_predicted - y) ** 2)
            self.cost_history.append(cost)
            
            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
    
    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Test implementation
def test_linear_regression():
    """Test linear regression implementation"""
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train custom model
    model = LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training R² score: {train_score:.4f}")
    print(f"Testing R² score: {test_score:.4f}")
    
    # Plot cost history
    plt.figure(figsize=(10, 6))
    plt.plot(model.cost_history)
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    plt.show()
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test

# Run test
model, X_train, X_test, y_train, y_test = test_linear_regression()
```

### Neural Network from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetworkFromScratch:
    """Simple neural network implemented from scratch"""
    
    def __init__(self, layer_sizes, learning_rate=0.01, max_iterations=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def forward_propagation(self, X):
        """Forward propagation through the network"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:  # Output layer
                activation = self.sigmoid(z)
            else:  # Hidden layers
                activation = self.relu(z)
            
            activations.append(activation)
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values):
        """Backward propagation to compute gradients"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer error
        dz = activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            dw = (1/m) * np.dot(activations[i].T, dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            if i > 0:  # Not the first layer
                da_prev = np.dot(dz, self.weights[i].T)
                dz = da_prev * self.relu_derivative(z_values[i-1])
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-8) + 
                              (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss
    
    def fit(self, X, y):
        """Train the neural network"""
        for iteration in range(self.max_iterations):
            # Forward propagation
            activations, z_values = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(y, activations[-1])
            self.loss_history.append(loss)
            
            # Backward propagation
            gradients_w, gradients_b = self.backward_propagation(X, y, activations, z_values)
            
            # Update parameters
            self.update_parameters(gradients_w, gradients_b)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def predict_classes(self, X, threshold=0.5):
        """Predict binary classes"""
        probabilities = self.predict(X)
        return (probabilities > threshold).astype(int)

# Test neural network implementation
def test_neural_network():
    """Test neural network implementation"""
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_redundant=0, 
                             n_informative=10, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape y for the network
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Create and train network
    # Architecture: 10 input -> 16 hidden -> 8 hidden -> 1 output
    nn = NeuralNetworkFromScratch(layer_sizes=[10, 16, 8, 1], 
                                 learning_rate=0.01, max_iterations=1000)
    
    nn.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_predictions = nn.predict_classes(X_train_scaled)
    test_predictions = nn.predict_classes(X_test_scaled)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(nn.loss_history)
    plt.title('Loss Function Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True)
    plt.show()
    
    return nn, X_train_scaled, X_test_scaled, y_train, y_test

# Run test
nn, X_train, X_test, y_train, y_test = test_neural_network()
```

## 🏗️ Actividades Prácticas

### Actividad 1: Mathematical Foundation Project
**Tiempo:** 8 horas durante 1 semana  
**Entregable:** Jupyter notebook con implementaciones matemáticas

**Objetivos específicos:**
1. **Linear Algebra Toolkit:** Implement matrix operations from scratch
   - Vector operations (dot product, norms, distances)
   - Matrix multiplication, inversion, eigendecomposition
   - Compare performance: pure Python vs NumPy vs optimized BLAS
   - Visualize eigenvectors and eigenvalues for 2D/3D data

2. **Statistics Calculator:** Build probability distribution functions
   - Implement common distributions (Normal, Binomial, Poisson)
   - Central Limit Theorem demonstration with animations
   - Bayesian inference examples (medical testing, spam filtering)
   - Monte Carlo simulations for complex probability problems

3. **Optimization Visualizer:** Create gradient descent animations
   - Single variable: parabolas, cubic functions, non-convex functions
   - Multivariate: 2D/3D contour plots with optimization paths
   - Learning rate effects: oscillation, convergence, divergence
   - Different optimizers: SGD, Momentum, Adam (basic implementations)

4. **Information Theory:** Calculate entropy and KL divergence for datasets
   - Text analysis: character/word entropy in different languages
   - Image analysis: entropy in different image types (natural vs synthetic)
   - Model comparison: KL divergence between predicted and true distributions
   - Mutual information for feature selection

5. **Performance Comparison:** Benchmark NumPy vs pure Python implementations
   - Matrix operations timing for different sizes (10x10 to 1000x1000)
   - Memory usage profiling and optimization
   - Vectorization benefits demonstration
   - SIMD and parallel processing impact

**Evaluation Criteria:**
- Code quality and documentation (25%)
- Mathematical correctness (30%)
- Visualization clarity and insights (25%)
- Performance analysis depth (20%)

### Actividad 2: Algorithm Implementation Challenge
**Tiempo:** 12 horas durante 1.5 semanas  
**Entregable:** Complete ML algorithm library

**Detailed Implementation Requirements:**

1. **Linear Models:** Linear/Logistic regression with regularization
   - **Linear Regression:**
     - Normal equation method vs gradient descent
     - Ridge regression (L2) and Lasso regression (L1)
     - Feature scaling and normalization impact
     - Cross-validation for hyperparameter tuning
   - **Logistic Regression:**
     - Sigmoid activation and log-likelihood
     - Multi-class extension (one-vs-rest, softmax)
     - Regularization to prevent overfitting
     - Feature importance interpretation

2. **Tree-based Methods:** Decision tree from scratch with pruning
   - **Decision Tree Implementation:**
     - Information gain vs Gini impurity splitting criteria
     - Handling continuous and categorical features
     - Pre-pruning (max depth, min samples) and post-pruning
     - Feature importance calculation
   - **Ensemble Extensions:**
     - Bagging implementation (Bootstrap Aggregating)
     - Random Forest with feature subsampling
     - Out-of-bag error estimation

3. **Neural Networks:** Multi-layer perceptron with backpropagation
   - **Architecture Design:**
     - Flexible layer sizes and activation functions
     - Weight initialization strategies (Xavier, He, etc.)
     - Bias handling and regularization (dropout, L2)
   - **Training Implementation:**
     - Forward propagation with matrix operations
     - Backpropagation derivation and implementation
     - Learning rate scheduling and momentum
     - Batch processing for efficiency

4. **Clustering:** K-means algorithm with initialization strategies
   - **Core Algorithm:**
     - Lloyd's algorithm implementation
     - Distance metrics (Euclidean, Manhattan, Cosine)
     - Convergence criteria and iteration limits
   - **Initialization Methods:**
     - Random initialization
     - K-means++ for better starting points
     - Multiple restarts for stability
   - **Evaluation:**
     - Within-cluster sum of squares (WCSS)
     - Silhouette score calculation
     - Elbow method for optimal K

5. **Evaluation Framework:** Cross-validation and metrics calculation
   - **Cross-Validation:**
     - K-fold, stratified K-fold, time series splits
     - Train/validation/test split strategies
     - Nested cross-validation for model selection
   - **Metrics Implementation:**
     - Classification: accuracy, precision, recall, F1, ROC-AUC
     - Regression: MSE, MAE, R², adjusted R²
     - Clustering: silhouette, adjusted rand index
   - **Statistical Testing:**
     - Paired t-tests for model comparison
     - McNemar's test for classifier comparison
     - Confidence intervals for performance metrics

**Comparison Framework:**
- Implement each algorithm and compare against scikit-learn
- Performance benchmarking (speed and memory usage)
- Accuracy comparison on standard datasets
- Detailed analysis of differences and optimizations

**Deliverable Structure:**
```
ml_algorithms/
├── linear_models/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   └── regularization.py
├── trees/
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── pruning.py
├── neural_networks/
│   ├── mlp.py
│   ├── activations.py
│   └── optimizers.py
├── clustering/
│   ├── kmeans.py
│   └── initialization.py
├── evaluation/
│   ├── cross_validation.py
│   ├── metrics.py
│   └── statistical_tests.py
├── benchmarks/
│   └── comparison_study.ipynb
└── tests/
    └── test_algorithms.py
```

**Success Criteria:**
- All algorithms achieve >95% accuracy match with scikit-learn
- Comprehensive unit tests with >90% coverage
- Performance analysis showing optimization opportunities
- Clear documentation explaining mathematical foundations

### Actividad 3: Data Engineering Pipeline
**Tiempo:** 10 horas durante 1 semana  
**Entregable:** Production-ready data processing pipeline

1. **Data Ingestion:** Multiple format support (CSV, JSON, Parquet)
2. **Quality Assessment:** Automated data quality reports
3. **Feature Engineering:** Automated feature generation framework
4. **Pipeline Orchestration:** Configurable processing workflows
5. **Monitoring:** Data drift detection and alerting

### Actividad 4: MLOps Foundation Setup
**Tiempo:** 8 horas durante 1 semana  
**Entregable:** Complete MLOps development environment

1. **Environment Setup:** Docker containers for reproducible environments
2. **Version Control:** Git workflows for ML projects
3. **Experiment Tracking:** MLflow or Weights & Biases integration
4. **Model Deployment:** FastAPI service with Docker
5. **CI/CD Pipeline:** GitHub Actions for automated testing

### Actividad 5: Capstone Foundation Project
**Tiempo:** 15 horas durante 2 semanas  
**Entregable:** End-to-end ML project prototype

1. **Problem Definition:** Real-world AI/ML problem selection
2. **Data Collection:** Gather and prepare relevant datasets
3. **Exploratory Analysis:** Comprehensive data exploration
4. **Baseline Model:** Simple model implementation and evaluation
5. **Documentation:** Professional project documentation

## 📊 Entregables por Semana

### ✅ Semana 1: Mathematical Foundations
- [ ] **Linear Algebra Notebook** con implementaciones desde cero
- [ ] **Statistics & Probability Dashboard** con visualizaciones interactivas
- [ ] **Optimization Visualizer** con gradient descent animations
- [ ] **Information Theory Calculator** para análisis de datasets
- [ ] **Performance Benchmarks** NumPy vs implementaciones propias

### ✅ Semana 2: Algorithm Mastery
- [ ] **ML Algorithm Library** con linear models, trees, neural nets
- [ ] **Evaluation Framework** con cross-validation y métricas
- [ ] **Algorithm Comparison Study** en multiple datasets
- [ ] **Implementation Documentation** con mathematical derivations
- [ ] **Performance Analysis** tiempo y memory usage

### ✅ Semana 3: Python Ecosystem Proficiency
- [ ] **Professional Development Environment** setup completo
- [ ] **Data Processing Pipeline** con automated quality checks
- [ ] **Feature Engineering Framework** reusable y configurable
- [ ] **Visualization Dashboard** con multiple plot types
- [ ] **Testing Suite** para data processing functions

### ✅ Semana 4: MLOps & Deployment
- [ ] **Containerized ML Service** con FastAPI y Docker
- [ ] **Experiment Tracking System** configurado y functional
- [ ] **CI/CD Pipeline** con automated testing y deployment
- [ ] **Model Monitoring Dashboard** con performance metrics
- [ ] **Capstone Project Proposal** con technical specifications

## 🏆 Success Metrics

### Technical Proficiency
- [ ] **Algorithm Implementation:** 5+ ML algorithms desde cero con >90% accuracy vs sklearn
- [ ] **Mathematical Understanding:** Derivar gradients y explain optimization processes
- [ ] **Python Mastery:** Efficient NumPy/Pandas code con vectorization
- [ ] **Environment Setup:** Professional development environment funcional

### Practical Skills
- [ ] **Data Processing:** Handle datasets >1GB con efficient memory usage
- [ ] **Pipeline Development:** End-to-end ML pipeline con monitoring
- [ ] **Documentation:** Professional-grade documentation y code comments
- [ ] **Testing:** Comprehensive test suite con >80% code coverage

### Project Outcomes
- [ ] **Capstone Foundation:** Well-defined project con clear objectives
- [ ] **Technical Stack:** Configured environment para advanced modules
- [ ] **Portfolio Start:** Professional GitHub profile con quality projects
- [ ] **Learning Path:** Clear progression plan para specialized AI domains

## 🚀 Preparación para Módulos Avanzados

Al completar el Módulo A, estarás preparado para:

### Módulo B - Desarrollo del Proyecto
- **Project Management:** Planificación y execution de proyectos complejos
- **Architecture Design:** System design para AI applications
- **Code Quality:** Advanced testing y CI/CD practices

### Módulo C - Benchmarks
- **Performance Evaluation:** Scientific approach a model evaluation
- **Statistical Analysis:** A/B testing y significance testing
- **Cost Optimization:** Resource efficiency en production systems

### Módulo D - Documentación
- **Technical Writing:** Clear communication de complex concepts
- **API Documentation:** Professional documentation standards
- **Stakeholder Communication:** Business-focused presentations

### Módulos E & F - Interview & Career Preparation
- **Technical Confidence:** Deep understanding para technical interviews
- **Portfolio Quality:** Professional projects para job applications
- **Market Readiness:** Skills alignment con industry demands

## 📞 Soporte y Recursos

**Office Hours:** Lunes y Miércoles 6-7 PM GMT-5  
**Lab Sessions:** Sábados 10 AM-12 PM GMT-5  
**Study Groups:** Slack channels por topic área

### Recursos Adicionales
- **Mathematical References:** Khan Academy, 3Blue1Brown videos
- **Python Practice:** LeetCode, HackerRank challenges
- **ML Theory:** Andrew Ng's Course, Fast.ai
- **Documentation:** NumPy, Pandas, scikit-learn official docs

¡Construye tu foundation sólida en AI Engineering! 🔬🚀
