import numpy as np

# =====================================================================
# 1. PHYSICS-ALIGNED ACTIVATION FUNCTIONS
# =====================================================================

def tanh(Z):
    """
    Hyperbolic Tangent Activation.
    Clamps values smoothly between -1.0 and +1.0. Perfect for modeling continuous
    fluid dynamics features (like parabolic velocity curves) without jagged edges.
    """
    return np.tanh(Z)

def tanh_derivative(A):
    """
    Derivative of Tanh.
    Uses the mathematical pre-calculated optimization where 'A' is already tanh(Z).
    Acts as a scaling filter for physics gradients during backpropagation.
    """
    return 1.0 - A**2


# =====================================================================
# 2. 3-HIDDEN-LAYER FLUID DYNAMICS SURROGATE NETWORK
# =====================================================================

class FluidFieldPredictor:
    def __init__(self, input_size=7, hidden_size=256, output_size=400, learning_rate=0.01):
        """
        Structural Sweet Spot Architecture:
        - Input Layer: 7 fluid/channel parameters (w, h, Vin, rho, n, K, lambda)
        - Hidden Layer 1: 256 Neurons (Maps channel boundary wall geometry limits)
        - Hidden Layer 2: 256 Neurons (Extracts non-Newtonian shear & viscosity profiles)
        - Hidden Layer 3: 256 Neurons (Blends boundary limits and fluid physics trends)
        - Output Layer: 400 Neurons (Continuous spatial grid velocity vectors, e.g., 20x20)
        """
        self.lr = learning_rate
        self.beta = 0.9  # Momentum cache tracking retention factor (90%)
        
        # --- Weight and Bias Matrix Initializations (Xavier/Glorot) ---
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        
        self.W3 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.b3 = np.zeros((1, hidden_size))
        
        self.W4 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b4 = np.zeros((1, output_size))
        
        # --- Momentum Cache Velocity Vector Maps ---
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.vW3, self.vb3 = np.zeros_like(self.W3), np.zeros_like(self.b3)
        self.vW4, self.vb4 = np.zeros_like(self.W4), np.zeros_like(self.b4)

    def forward(self, X):
        """Feeds design parameters forward through the layers into continuous outputs."""
        # Layer 1: Geometry Constraints Mapping
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = tanh(self.Z1)
        
        # Layer 2: Non-Newtonian Shear Mapping
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = tanh(self.Z2)
        
        # Layer 3: Field Fusion & Fluid Continuity
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = tanh(self.Z3)
        
        # Layer 4: Linear Output Layer (NO activation function)
        # Allows output neurons to scale directly to absolute fluid velocities (e.g., 0.035 m/s)
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.Z4  
        return self.A4

    def backward(self, X, y_true, y_hat):
        """Calculates continuous gradients using Mean Squared Error (MSE) physics calculus."""
        m = X.shape[0]  # Number of samples in active batch
        
        # --- Output Layer Error (Derivative of MSE Loss) ---
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
        
        # --- Physics Updates: Momentum Cache Velocity Vector Accumulation ---
        self.vW4 = self.beta * self.vW4 + (1 - self.beta) * dW4
        self.vb4 = self.beta * self.vb4 + (1 - self.beta) * db4
        self.vW3 = self.beta * self.vW3 + (1 - self.beta) * dW3
        self.vb3 = self.beta * self.vb3 + (1 - self.beta) * db3
        self.vW2 = self.beta * self.vW2 + (1 - self.beta) * dW2
        self.vb2 = self.beta * self.vb2 + (1 - self.beta) * db2
        self.vW1 = self.beta * self.vW1 + (1 - self.beta) * dW1
        self.vb1 = self.beta * self.vb1 + (1 - self.beta) * db1
        
        # Apply momentum updates back to structural connection values
        self.W4 -= self.lr * self.vW4
        self.b4 -= self.lr * self.vb4
        self.W3 -= self.lr * self.vW3
        self.b3 -= self.lr * self.vb3
        self.W2 -= self.lr * self.vW2
        self.b2 -= self.lr * self.vb2
        self.W1 -= self.lr * self.vW1
        self.b1 -= self.lr * self.vb1

    def train(self, X_train, y_train, batch_size=32, epochs=100):
        """Runs optimization iterations across training data packages."""
        num_samples = X_train.shape[0]  # Isolated row index limit to avoid index arrays errors
        
        for epoch in range(epochs):
            # Shuffle data sweeps at the beginning of each learning path pass
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                y_hat = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_hat)
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                full_preds = self.forward(X_train)
                mse_loss = np.mean((full_preds - y_train) ** 2)
                print(f"Epoch {epoch + 1:3d}/{epochs} | Global Evaluation MSE Loss: {mse_loss:.8f}")

# =====================================================================
# 3. COMPILATION UTILITIES FOR INTEGRATION SAVES
# =====================================================================

if __name__ == "__main__":
    print("Initialising Fluid Network Framework Sandbox Simulation...")
    
    # Generate dummy training data replicating OpenFOAM parameter outputs
    # 500 spatial iterations paired with 7 design options and 400 vector metrics
    num_mock_simulations = 500
    X_mock_parameters = np.random.rand(num_mock_simulations, 7)
    y_mock_openfoam_grids = np.random.rand(num_mock_simulations, 400) * 0.05  # Scale mock velocities 
    
    # Initialize sweet-spot predictor network
    predictor = FluidFieldPredictor(input_size=7, hidden_size=256, output_size=400, learning_rate=0.01)
    
    print("Executing structural network baseline compilation loops...")
    predictor.train(X_mock_parameters, y_mock_openfoam_grids, batch_size=32, epochs=20)
    
    # Save network structures
    print("\n[Saving Data Arrays] Locking optimized configuration files to disk...")
    np.save('W1.npy', predictor.W1)
    np.save('b1.npy', predictor.b1)
    np.save('W2.npy', predictor.W2)
    np.save('b2.npy', predictor.b2)
    np.save('W3.npy', predictor.W3)
    np.save('b3.npy', predictor.b3)
    np.save('W4.npy', predictor.W4)
    np.save('b4.npy', predictor.b4)
    print("Pipeline compilation complete! Ready for raw GitHub integration commits.")
