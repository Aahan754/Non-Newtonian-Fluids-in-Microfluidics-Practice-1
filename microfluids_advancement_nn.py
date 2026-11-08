import numpy as np

# =====================================================================
# 1. PHYSICS-ALIGNED ACTIVATION FUNCTIONS
# =====================================================================

def tanh(Z):
    """
    Hyperbolic Tangent Activation.
    Preferred for physics networks over ReLU because fluid dynamics curves 
    (like velocity parabolas) are smooth and continuously differentiable.
    """
    return np.tanh(Z)

def tanh_derivative(A):
    """
    Derivative of Tanh.
    Note: Input 'A' is already tanh(Z) from the forward pass.
    """
    return 1.0 - A**2


# =====================================================================
# 2. ACCURACY UTILITY FUNCTIONS FOR CONTINUOUS REGRESSION FIELDS
# =====================================================================

def calculate_r2_accuracy(y_true, y_pred):
    """
    Computes the Coefficient of Determination (R^2 Score) as an Accuracy Percentage.
    Formula: Accuracy = 1 - (Residual Sum of Squares / Total Sum of Squares)
    Returns a clean scalar percentage string representing variance coverage.
    """
    # Sum of squared discrepancies between true targets and model predictions
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    
    # Sum of squared discrepancies between true targets and the statistical global mean
    mean_true = np.mean(y_true)
    total_sum_squares = np.sum((y_true - mean_true) ** 2)
    
    # Safety condition to prevent division by zero errors on empty channels
    if total_sum_squares == 0:
        return 0.0
        
    r2_score = 1.0 - (residual_sum_squares / total_sum_squares)
    
    # Convert score bounds into a clean percentage scale (clamped to a 0-100 threshold)
    accuracy_percentage = max(0.0, r2_score * 100.0)
    return accuracy_percentage


# =====================================================================
# 3. FLUID DYNAMICS SURROGATE NETWORK (3 HIDDEN LAYERS)
# =====================================================================

class FluidFieldPredictor:
    def __init__(self, input_size=7, hidden_size=256, output_size=400, learning_rate=0.001):
        """
        Structural Layout:
        - Input Layer: 7 neurons (Width, Height, Flow Rate, Density, n, K, Lambda)
        - Hidden Layer 1: 256 Neurons (Maps channel geometries and solid boundaries)
        - Hidden Layer 2: 256 Neurons (Extracts non-Newtonian viscosity profiles)
        - Hidden Layer 3: 256 Neurons (Fuses geometries and velocity continuity)
        - Output Layer: 400 Neurons (Continuous spatial grid velocity vectors)
        """
        self.lr = learning_rate
        self.beta = 0.9  # Momentum factor
        
        # --- Layer 1 (Input -> Hidden 1) ---
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        # --- Layer 2 (Hidden 1 -> Hidden 2) ---
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        
        # --- Layer 3 (Hidden 2 -> Hidden 3) ---
        self.W3 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.b3 = np.zeros((1, hidden_size))
        
        # --- Layer 4 (Hidden 3 -> Output Field Grid) ---
        self.W4 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b4 = np.zeros((1, output_size))
        
        # --- Momentum Cache Tracking Vectors ---
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.vW3, self.vb3 = np.zeros_like(self.W3), np.zeros_like(self.b3)
        self.vW4, self.vb4 = np.zeros_like(self.W4), np.zeros_like(self.b4)

    def forward(self, X):
        """Feeds the fluid design configurations forward into a continuous spatial grid."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = tanh(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = tanh(self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = tanh(self.Z3)
        
        # Output layer remains completely LINEAR (No activation function applied)
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.Z4  
        return self.A4

    def backward(self, X, y_true, y_hat):
        """Calculates regression adjustments based on standard calculus of Mean Squared Error."""
        m = X.shape[0]  # Isolate row integer explicitly
        
        # --- Output Layer Error (Derivative of MSE Loss function) ---
        dZ4 = 2.0 * (y_hat - y_true) / m
        dW4 = np.dot(self.A3.T, dZ4)
        db4 = np.sum(dZ4, axis=0, keepdims=True)
        
        # --- Hidden Layer 3 Error ---
        dA3 = np.dot(dZ4, self.W4.T)
        dZ3 = dA3 * tanh_derivative(self.A3)
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        # --- Hidden Layer 2 Error ---
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * tanh_derivative(self.A2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # --- Hidden Layer 1 Error ---
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * tanh_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Physics Update: Blend Smooth Momentum Caches ---
        self.vW4 = self.beta * self.vW4 + (1 - self.beta) * dW4
        self.vb4 = self.beta * self.vb4 + (1 - self.beta) * db4
        self.vW3 = self.beta * self.vW3 + (1 - self.beta) * dW3
        self.vb3 = self.beta * self.vb3 + (1 - self.beta) * db3
        self.vW2 = self.beta * self.vW2 + (1 - self.beta) * dW2
        self.vb2 = self.beta * self.vb2 + (1 - self.beta) * db2
        self.vW1 = self.beta * self.vW1 + (1 - self.beta) * dW1
        self.vb1 = self.beta * self.vb1 + (1 - self.beta) * db1
        
        # Adjust weight values
        self.W4 -= self.lr * self.vW4
        self.b4 -= self.lr * self.vb4
        self.W3 -= self.lr * self.vW3
        self.b3 -= self.lr * self.vb3
        self.W2 -= self.lr * self.vW2
        self.b2 -= self.lr * self.vb2
        self.W1 -= self.lr * self.vW1
        self.b1 -= self.lr * self.vb1

    def train(self, X_train, y_train, batch_size=32, epochs=100):
        """Processes parameter sweeps to lock down fluid profiles."""
        num_samples = X_train.shape[0]  # FIXED: Grabs row integer directly to prevent IndexError
        
        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                y_hat = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_hat)
                
            # Metric evaluation step happening every 10 cycles
            if (epoch + 1) % 10 == 0:
                full_preds = self.forward(X_train)
                mse_loss = np.mean((full_preds - y_train) ** 2)
                
                # Call precision tracker to report variance metric percentage
                accuracy = calculate_r2_accuracy(y_train, full_preds)
                print(f"Epoch {epoch + 1:3d}/{epochs} | MSE Loss: {mse_loss:.8f} | Physics R² Accuracy: {accuracy:.2f}%")

# =====================================================================
# 4. PIPELINE SANDBOX EXECUTION ENVIRONMENT
# =====================================================================

if __name__ == "__main__":
    print("Generating simulated non-Newtonian training matrices (1000 configurations)...")
    num_samples = 1000
    
    # Create 1,000 random test samples containing our 7 exact physical inputs
    X_fake_inputs = np.random.rand(num_samples, 7)
    # Create 1,000 random output target grids containing 400 continuous velocity components
    y_fake_fields = np.random.rand(num_samples, 400)

    # Initialize advanced 3-hidden-layer physics model
    nn = FluidFieldPredictor(input_size=7, hidden_size=256, output_size=400, learning_rate=0.001)
    
    print("\nBeginning Training Cycles across Data Sweeps...")
    nn.train(X_fake_inputs, y_fake_fields, batch_size=32, epochs=100)
    
    print("\n[Preserving Model Knowledge] Saving trained weight matrices to files...")
    np.save('W1.npy', nn.W1)
    np.save('b1.npy', nn.b1)
    np.save('W2.npy', nn.W2)
    np.save('b2.npy', nn.b2)
    np.save('W3.npy', nn.W3)
    np.save('b3.npy', nn.b3)
    np.save('W4.npy', nn.W4)
    np.save('b4.npy', nn.b4)
    print("Done! Weight arrays saved successfully.")
