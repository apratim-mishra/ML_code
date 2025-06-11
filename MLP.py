"""
Simple Multi-Layer Perceptron (MLP) Implementation with PyTorch
A basic neural network for both classification and regression tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron neural network.
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    hidden_sizes : list
        List of hidden layer sizes
    output_size : int
        Number of output neurons
    task_type : str
        Either 'classification' or 'regression'
    activation : str
        Activation function ('relu', 'tanh', 'sigmoid')
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, task_type='classification', activation='relu'):
        super(MLP, self).__init__()
        
        self.task_type = task_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Add activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # For classification, add softmax if multi-class
        if task_type == 'classification' and output_size > 1:
            layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
        self.to(self.device)
        
        print(f"MLP created on device: {self.device}")
        print(f"Network architecture: {self.network}")
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class MLPTrainer:
    """
    Trainer class for MLP neural network.
    
    Parameters:
    -----------
    model : MLP
        The MLP model to train
    optimizer_name : str
        Optimizer to use ('adam', 'sgd')
    learning_rate : float
        Learning rate for optimization
    loss_function : str
        Loss function ('mse', 'cross_entropy', 'bce')
    """
    
    def __init__(self, model, optimizer_name='adam', learning_rate=0.001, loss_function='auto'):
        self.model = model
        self.device = model.device
        
        # Set up optimizer
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer must be 'adam' or 'sgd'")
        
        # Set up loss function
        if loss_function == 'auto':
            if model.task_type == 'classification':
                self.criterion = nn.CrossEntropyLoss()
            else:  # regression
                self.criterion = nn.MSELoss()
        elif loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == 'bce':
            self.criterion = nn.BCELoss()
        else:
            raise ValueError("Loss function must be 'mse', 'cross_entropy', 'bce', or 'auto'")
        
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, X, y):
        """Convert data to PyTorch tensors and move to device."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if self.model.task_type == 'classification':
            y_tensor = torch.LongTensor(y).to(self.device)
        else:  # regression
            y_tensor = torch.FloatTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """Split data into train/validation/test sets."""
        n_samples = len(X)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train_epoch(self, X_train, y_train):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        # Prepare data
        X_tensor, y_tensor = self.prepare_data(X_train, y_train)
        
        # Forward pass
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, X_val, y_val):
        """Validate the model."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor, y_tensor = self.prepare_data(X_val, y_val)
            outputs = self.model(X_tensor)
            val_loss = self.criterion(outputs, y_tensor)
        
        return val_loss.item()
    
    def fit(self, X, y, epochs=100, verbose=True):
        """
        Train the model.
        
        Parameters:
        -----------
        X : list of lists
            Input features
        y : list
            Target values/labels
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print training progress
        """
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X, y)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train)
            self.train_losses.append(train_loss)
            
            # Validate
            if X_val:
                val_loss = self.validate(X_val, y_val)
                self.val_losses.append(val_loss)
            else:
                val_loss = 0.0
            
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Test evaluation
        if X_test:
            test_loss = self.validate(X_test, y_test)
            test_accuracy = self.evaluate(X_test, y_test)
            print(f"\nFinal Test Loss: {test_loss:.4f}")
            if self.model.task_type == 'classification':
                print(f"Test Accuracy: {test_accuracy:.4f}")
            else:
                print(f"Test MAE: {test_accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions on new data."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            
            if self.model.task_type == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                return predicted.cpu().numpy()
            else:  # regression
                return outputs.cpu().numpy().flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        
        if self.model.task_type == 'classification':
            # Calculate accuracy
            correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
            return correct / len(y_test)
        else:  # regression
            # Calculate MAE
            mae = sum(abs(pred - true) for pred, true in zip(predictions, y_test)) / len(y_test)
            return mae


def generate_classification_data(sample_size, features_size, num_classes=2):
    """Generate random classification data."""
    X = []
    y = []
    
    for _ in range(sample_size):
        # Generate random features between 0 and 1
        features = [random.random() for _ in range(features_size)]
        X.append(features)
        
        # Generate label based on simple rule (sum of features)
        if sum(features) > features_size / 2:
            label = 1 if num_classes == 2 else random.randint(1, num_classes-1)
        else:
            label = 0
        y.append(label)
    
    return X, y


def generate_regression_data(sample_size, features_size):
    """Generate random regression data."""
    X = []
    y = []
    
    for _ in range(sample_size):
        # Generate random features between -2 and 2
        features = [random.uniform(-2, 2) for _ in range(features_size)]
        X.append(features)
        
        # Generate target as weighted sum with some non-linearity
        target = sum(f * (i+1) for i, f in enumerate(features))  # weighted sum
        target += sum(f**2 * 0.1 for f in features)  # add some non-linearity
        target += random.normal(0, 0.1)  # add noise
        y.append(target)
    
    return X, y


# Example usage
if __name__ == "__main__":
    print("Simple MLP Neural Network Example")
    print("=" * 40)
    
    # Configuration parameters
    SAMPLE_SIZE_CLS = 200
    FEATURES_SIZE_CLS = 4
    NUM_CLASSES = 3
    
    SAMPLE_SIZE_REG = 300
    FEATURES_SIZE_REG = 3
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Classification Example
    print("\n1. CLASSIFICATION EXAMPLE:")
    print("-" * 30)
    print(f"Sample size: {SAMPLE_SIZE_CLS}, Features: {FEATURES_SIZE_CLS}, Classes: {NUM_CLASSES}")
    
    # Generate random classification data
    X_cls, y_cls = generate_classification_data(SAMPLE_SIZE_CLS, FEATURES_SIZE_CLS, NUM_CLASSES)
    
    print(f"Sample data points:")
    for i in range(3):
        print(f"  {X_cls[i]} -> Class {y_cls[i]}")
    
    # Create and train classification model
    hidden_sizes = [max(8, FEATURES_SIZE_CLS * 2), max(4, FEATURES_SIZE_CLS)]
    model_cls = MLP(input_size=FEATURES_SIZE_CLS, hidden_sizes=hidden_sizes, 
                    output_size=NUM_CLASSES, task_type='classification', activation='relu')
    
    trainer_cls = MLPTrainer(model_cls, optimizer_name='adam', learning_rate=0.01)
    trainer_cls.fit(X_cls, y_cls, epochs=100, verbose=True)
    
    # Test predictions on new random data
    test_X, test_y = generate_classification_data(5, FEATURES_SIZE_CLS, NUM_CLASSES)
    predictions = trainer_cls.predict(test_X)
    print(f"\nClassification predictions on new data:")
    for point, pred, actual in zip(test_X, predictions, test_y):
        print(f"  {[round(x, 2) for x in point]} -> Predicted: {pred}, Actual: {actual}")
    
    # Regression Example
    print("\n2. REGRESSION EXAMPLE:")
    print("-" * 25)
    print(f"Sample size: {SAMPLE_SIZE_REG}, Features: {FEATURES_SIZE_REG}")
    
    # Generate random regression data
    X_reg, y_reg = generate_regression_data(SAMPLE_SIZE_REG, FEATURES_SIZE_REG)
    
    print(f"Sample data points:")
    for i in range(3):
        print(f"  {[round(x, 2) for x in X_reg[i]]} -> {y_reg[i]:.3f}")
    
    # Create and train regression model
    hidden_sizes = [max(10, FEATURES_SIZE_REG * 3), max(5, FEATURES_SIZE_REG)]
    model_reg = MLP(input_size=FEATURES_SIZE_REG, hidden_sizes=hidden_sizes, 
                    output_size=1, task_type='regression', activation='relu')
    
    trainer_reg = MLPTrainer(model_reg, optimizer_name='adam', learning_rate=0.001)
    trainer_reg.fit(X_reg, y_reg, epochs=100, verbose=True)
    
    # Test predictions on new random data
    test_X, test_y = generate_regression_data(5, FEATURES_SIZE_REG)
    predictions = trainer_reg.predict(test_X)
    print(f"\nRegression predictions on new data:")
    for point, pred, actual in zip(test_X, predictions, test_y):
        print(f"  {[round(x, 2) for x in point]} -> Predicted: {pred:.3f}, Actual: {actual:.3f}")
    
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") 