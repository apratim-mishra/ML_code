"""
Simple K-Nearest Neighbors (KNN) Implementation
A basic KNN algorithm for both classification and regression using only Python.
"""

import math


class KNN:
    """
    Simple K-Nearest Neighbors algorithm.
    
    Parameters:
    -----------
    k : int
        Number of nearest neighbors to consider
    task_type : str
        Either 'classification' or 'regression'
    """
    
    def __init__(self, k=3, task_type='classification'):
        self.k = k
        self.task_type = task_type
        self.X_train = []
        self.y_train = []
    
    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def fit(self, X, y):
        """
        Store training data.
        
        Parameters:
        -----------
        X : list of lists
            Training features
        y : list
            Training labels/targets
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """
        Make predictions for test data.
        
        Parameters:
        -----------
        X : list of lists
            Test features
            
        Returns:
        --------
        predictions : list
            Predicted labels/values
        """
        predictions = []
        
        for test_point in X:
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self.distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Make prediction based on task type
            if self.task_type == 'classification':
                # Majority vote for classification
                neighbor_labels = [label for _, label in k_nearest]
                prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            else:  # regression
                # Average for regression
                neighbor_values = [value for _, value in k_nearest]
                prediction = sum(neighbor_values) / len(neighbor_values)
            
            predictions.append(prediction)
        
        return predictions
    
    def predict_single(self, point):
        """Predict for a single point."""
        return self.predict([point])[0]


def generate_classification_data(sample_size, features_size, num_classes=2):
    """Generate random classification data."""
    import random
    X = []
    y = []
    
    for _ in range(sample_size):
        # Generate random features between 0 and 10
        features = [random.uniform(0, 10) for _ in range(features_size)]
        X.append(features)
        
        # Generate label based on simple rule (sum of features)
        if sum(features) > features_size * 5:  # threshold at middle
            label = 1 if num_classes == 2 else random.randint(1, num_classes-1)
        else:
            label = 0
        y.append(label)
    
    return X, y


def generate_regression_data(sample_size, features_size):
    """Generate random regression data."""
    import random
    X = []
    y = []
    
    for _ in range(sample_size):
        # Generate random features between 0 and 10
        features = [random.uniform(0, 10) for _ in range(features_size)]
        X.append(features)
        
        # Generate target as weighted sum with noise
        target = sum(f * (i+1) * 0.5 for i, f in enumerate(features))  # weighted sum
        target += random.uniform(-2, 2)  # add noise
        y.append(target)
    
    return X, y


# Example usage
if __name__ == "__main__":
    import random
    
    print("Simple KNN Example")
    print("=" * 30)
    
    # Configuration parameters
    SAMPLE_SIZE_CLS = 50
    FEATURES_SIZE_CLS = 3
    NUM_CLASSES = 2
    K_NEIGHBORS_CLS = 3
    
    SAMPLE_SIZE_REG = 60
    FEATURES_SIZE_REG = 2
    K_NEIGHBORS_REG = 5
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Classification Example
    print("\n1. CLASSIFICATION EXAMPLE:")
    print("-" * 25)
    print(f"Sample size: {SAMPLE_SIZE_CLS}, Features: {FEATURES_SIZE_CLS}, Classes: {NUM_CLASSES}")
    
    # Generate random classification data
    X_train_cls, y_train_cls = generate_classification_data(SAMPLE_SIZE_CLS, FEATURES_SIZE_CLS, NUM_CLASSES)
    
    print(f"\nSample training data:")
    for i in range(5):
        features_rounded = [round(x, 2) for x in X_train_cls[i]]
        print(f"  {features_rounded} -> Class {y_train_cls[i]}")
    
    # Generate test data
    X_test_cls, y_test_cls = generate_classification_data(10, FEATURES_SIZE_CLS, NUM_CLASSES)
    
    # Create and fit classifier
    knn_cls = KNN(k=K_NEIGHBORS_CLS, task_type='classification')
    knn_cls.fit(X_train_cls, y_train_cls)
    predictions_cls = knn_cls.predict(X_test_cls)
    
    print(f"\nTest predictions:")
    correct = 0
    for i, (features, pred, actual) in enumerate(zip(X_test_cls, predictions_cls, y_test_cls)):
        features_rounded = [round(x, 2) for x in features]
        print(f"  {features_rounded} -> Predicted: {pred}, Actual: {actual}")
        if pred == actual:
            correct += 1
    
    accuracy = correct / len(predictions_cls)
    print(f"Classification Accuracy: {accuracy:.2f}")
    
    # Regression Example
    print("\n2. REGRESSION EXAMPLE:")
    print("-" * 20)
    print(f"Sample size: {SAMPLE_SIZE_REG}, Features: {FEATURES_SIZE_REG}")
    
    # Generate random regression data
    X_train_reg, y_train_reg = generate_regression_data(SAMPLE_SIZE_REG, FEATURES_SIZE_REG)
    
    print(f"\nSample training data:")
    for i in range(5):
        features_rounded = [round(x, 2) for x in X_train_reg[i]]
        print(f"  {features_rounded} -> {y_train_reg[i]:.2f}")
    
    # Generate test data
    X_test_reg, y_test_reg = generate_regression_data(8, FEATURES_SIZE_REG)
    
    # Create and fit regressor
    knn_reg = KNN(k=K_NEIGHBORS_REG, task_type='regression')
    knn_reg.fit(X_train_reg, y_train_reg)
    predictions_reg = knn_reg.predict(X_test_reg)
    
    print(f"\nTest predictions:")
    total_error = 0
    for features, pred, actual in zip(X_test_reg, predictions_reg, y_test_reg):
        features_rounded = [round(x, 2) for x in features]
        error = abs(pred - actual)
        total_error += error
        print(f"  {features_rounded} -> Predicted: {pred:.2f}, Actual: {actual:.2f}, Error: {error:.2f}")
    
    mae = total_error / len(predictions_reg)
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Show nearest neighbors for first test point
    print(f"\n3. NEAREST NEIGHBORS DETAILS:")
    print("-" * 30)
    test_point = X_test_reg[0]
    test_point_rounded = [round(x, 2) for x in test_point]
    print(f"For test point {test_point_rounded}:")
    
    # Calculate distances to all training points
    distances = []
    for i, train_point in enumerate(X_train_reg):
        dist = knn_reg.distance(test_point, train_point)
        distances.append((dist, train_point, y_train_reg[i]))
    
    # Sort and show k nearest
    distances.sort(key=lambda x: x[0])
    print(f"{K_NEIGHBORS_REG} nearest neighbors:")
    for i in range(K_NEIGHBORS_REG):
        dist, point, target = distances[i]
        point_rounded = [round(x, 2) for x in point]
        print(f"  {point_rounded} (distance: {dist:.2f}) -> {target:.2f}")
    
    avg_target = sum(distances[i][2] for i in range(K_NEIGHBORS_REG)) / K_NEIGHBORS_REG
    print(f"Average prediction: {avg_target:.2f}") 