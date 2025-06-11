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


# Example usage
if __name__ == "__main__":
    print("Simple KNN Example")
    print("=" * 30)
    
    # Classification Example
    print("\n1. CLASSIFICATION EXAMPLE:")
    print("-" * 25)
    
    # Training data: [height, weight] -> gender (0=female, 1=male)
    X_train_cls = [
        [160, 50], [165, 55], [170, 60], [175, 70], [180, 80],
        [155, 45], [162, 52], [172, 65], [178, 75], [185, 85]
    ]
    y_train_cls = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]  # 0=female, 1=male
    
    # Test data
    X_test_cls = [[168, 58], [182, 78]]
    
    # Create and fit classifier
    knn_cls = KNN(k=3, task_type='classification')
    knn_cls.fit(X_train_cls, y_train_cls)
    predictions_cls = knn_cls.predict(X_test_cls)
    
    print("Training data (height, weight -> gender):")
    for i, (features, label) in enumerate(zip(X_train_cls, y_train_cls)):
        gender = "female" if label == 0 else "male"
        print(f"  {features} -> {gender}")
    
    print("\nTest predictions:")
    for features, pred in zip(X_test_cls, predictions_cls):
        gender = "female" if pred == 0 else "male"
        print(f"  {features} -> {gender}")
    
    # Regression Example
    print("\n2. REGRESSION EXAMPLE:")
    print("-" * 20)
    
    # Training data: [size, rooms] -> price
    X_train_reg = [
        [100, 2], [120, 3], [80, 1], [150, 4], [90, 2],
        [110, 3], [130, 3], [95, 2], [140, 4], [85, 1]
    ]
    y_train_reg = [200, 280, 150, 350, 180, 250, 300, 190, 330, 160]  # prices in thousands
    
    # Test data
    X_test_reg = [[105, 2], [125, 3]]
    
    # Create and fit regressor
    knn_reg = KNN(k=3, task_type='regression')
    knn_reg.fit(X_train_reg, y_train_reg)
    predictions_reg = knn_reg.predict(X_test_reg)
    
    print("Training data (size, rooms -> price):")
    for features, price in zip(X_train_reg, y_train_reg):
        print(f"  {features} -> ${price}k")
    
    print("\nTest predictions:")
    for features, pred in zip(X_test_reg, predictions_reg):
        print(f"  {features} -> ${pred:.1f}k")
    
    # Show nearest neighbors for first test point
    print(f"\n3. NEAREST NEIGHBORS DETAILS:")
    print("-" * 30)
    test_point = X_test_reg[0]  # [105, 2]
    print(f"For test point {test_point}:")
    
    # Calculate distances to all training points
    distances = []
    for i, train_point in enumerate(X_train_reg):
        dist = knn_reg.distance(test_point, train_point)
        distances.append((dist, train_point, y_train_reg[i]))
    
    # Sort and show k nearest
    distances.sort(key=lambda x: x[0])
    print(f"3 nearest neighbors:")
    for i in range(3):
        dist, point, price = distances[i]
        print(f"  {point} (distance: {dist:.1f}) -> ${price}k")
    
    avg_price = sum(distances[i][2] for i in range(3)) / 3
    print(f"Average price: ${avg_price:.1f}k") 