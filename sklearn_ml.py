"""
Simple Scikit-Learn ML Implementation
Basic machine learning algorithms using scikit-learn for classification and regression.
"""

import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import xgboost as xgb


class SimpleMLPipeline:
    """
    Simple ML pipeline for classification and regression.
    
    Parameters:
    -----------
    task_type : str
        Either 'classification' or 'regression'
    algorithm : str
        Algorithm to use: 'knn', 'naive_bayes', 'random_forest', 'xgboost'
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, task_type='classification', algorithm='random_forest', 
                 test_size=0.2, random_state=42):
        self.task_type = task_type
        self.algorithm = algorithm
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on task type and algorithm."""
        if self.task_type == 'classification':
            if self.algorithm == 'knn':
                self.model = KNeighborsClassifier(n_neighbors=5)
            elif self.algorithm == 'naive_bayes':
                self.model = GaussianNB()
            elif self.algorithm == 'random_forest':
                self.model = RandomForestClassifier(n_estimators=50, max_depth=10, 
                                                  random_state=self.random_state)
            elif self.algorithm == 'xgboost':
                self.model = xgb.XGBClassifier(n_estimators=50, max_depth=6, 
                                             learning_rate=0.1, random_state=self.random_state)
            else:
                raise ValueError("Algorithm must be 'knn', 'naive_bayes', 'random_forest', or 'xgboost'")
        
        else:  # regression
            if self.algorithm == 'knn':
                self.model = KNeighborsRegressor(n_neighbors=5)
            elif self.algorithm == 'naive_bayes':
                raise ValueError("Naive Bayes is not suitable for regression. Use other algorithms.")
            elif self.algorithm == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=50, max_depth=10, 
                                                 random_state=self.random_state)
            elif self.algorithm == 'xgboost':
                self.model = xgb.XGBRegressor(n_estimators=50, max_depth=6, 
                                            learning_rate=0.1, random_state=self.random_state)
            else:
                raise ValueError("Algorithm must be 'knn', 'random_forest', or 'xgboost' for regression")
    
    def fit(self, X, y):
        """
        Fit the model with train-test split.
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values/labels
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Fit model
        print(f"Training {self.algorithm} for {self.task_type}...")
        self.model.fit(self.X_train, self.y_train)
        
        return self
    
    def predict(self, X=None):
        """Make predictions on test set or provided data."""
        if X is None:
            X = self.X_test
        else:
            X = np.array(X)
        
        return self.model.predict(X)
    
    def evaluate(self):
        """Evaluate model performance on test set."""
        if self.X_test is None:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict()
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(self.y_test, predictions)
            print(f"\nClassification Results:")
            print(f"Test Accuracy: {accuracy:.4f}")
            
            # Show some predictions
            print(f"\nSample predictions (first 10):")
            for i in range(min(10, len(predictions))):
                print(f"  Predicted: {predictions[i]}, Actual: {self.y_test[i]}")
            
            return accuracy
        
        else:  # regression
            mae = mean_absolute_error(self.y_test, predictions)
            print(f"\nRegression Results:")
            print(f"Mean Absolute Error: {mae:.4f}")
            
            # Show some predictions
            print(f"\nSample predictions (first 10):")
            for i in range(min(10, len(predictions))):
                print(f"  Predicted: {predictions[i]:.3f}, Actual: {self.y_test[i]:.3f}")
            
            return mae


def generate_classification_data(sample_size, features_size, num_classes=3):
    """Generate random classification data."""
    X = []
    y = []
    
    for _ in range(sample_size):
        # Generate random features
        features = [random.uniform(-2, 2) for _ in range(features_size)]
        X.append(features)
        
        # Generate label based on feature combination
        feature_sum = sum(features)
        if feature_sum < -features_size/2:
            label = 0
        elif feature_sum > features_size/2:
            label = min(2, num_classes-1)
        else:
            label = 1
        
        y.append(label)
    
    return X, y


def generate_regression_data(sample_size, features_size):
    """Generate random regression data."""
    X = []
    y = []
    
    for _ in range(sample_size):
        # Generate random features
        features = [random.uniform(-3, 3) for _ in range(features_size)]
        X.append(features)
        
        # Generate target with some pattern and noise
        target = sum(f * (i+1) * 0.7 for i, f in enumerate(features))  # weighted sum
        target += sum(f**2 * 0.1 for f in features)  # add non-linearity
        target += np.random.normal(0, 0.5)  # add noise
        
        y.append(target)
    
    return X, y


def run_algorithm_comparison(sample_size, features_size, task_type='classification'):
    """Run comparison of different algorithms."""
    print(f"\n{'='*60}")
    print(f"ALGORITHM COMPARISON - {task_type.upper()}")
    print(f"Sample size: {sample_size}, Features: {features_size}")
    print(f"{'='*60}")
    
    # Generate data
    if task_type == 'classification':
        X, y = generate_classification_data(sample_size, features_size)
        algorithms = ['knn', 'naive_bayes', 'random_forest', 'xgboost']
    else:
        X, y = generate_regression_data(sample_size, features_size)
        algorithms = ['knn', 'random_forest', 'xgboost']  # No naive bayes for regression
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n{'-'*40}")
        print(f"Testing {algorithm.upper()}")
        print(f"{'-'*40}")
        
        try:
            # Create and train model
            pipeline = SimpleMLPipeline(task_type=task_type, algorithm=algorithm)
            pipeline.fit(X, y)
            
            # Evaluate
            score = pipeline.evaluate()
            results[algorithm] = score
            
        except Exception as e:
            print(f"Error with {algorithm}: {e}")
            results[algorithm] = None
    
    # Summary
    print(f"\n{'='*40}")
    print(f"SUMMARY - {task_type.upper()}")
    print(f"{'='*40}")
    metric_name = "Accuracy" if task_type == 'classification' else "MAE"
    
    for algorithm, score in results.items():
        if score is not None:
            print(f"{algorithm.upper():15}: {metric_name} = {score:.4f}")
        else:
            print(f"{algorithm.upper():15}: Failed")


# Example usage
if __name__ == "__main__":
    print("Simple Scikit-Learn ML Pipeline")
    print("=" * 50)
    
    # Configuration parameters
    SAMPLE_SIZE = 200
    FEATURES_SIZE = 4
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test individual algorithm
    print("\n1. INDIVIDUAL ALGORITHM TEST:")
    print("-" * 35)
    
    # Classification example
    print(f"\nClassification with Random Forest:")
    X_cls, y_cls = generate_classification_data(SAMPLE_SIZE, FEATURES_SIZE)
    
    pipeline_cls = SimpleMLPipeline(task_type='classification', algorithm='random_forest')
    pipeline_cls.fit(X_cls, y_cls)
    pipeline_cls.evaluate()
    
    # Regression example
    print(f"\nRegression with XGBoost:")
    X_reg, y_reg = generate_regression_data(SAMPLE_SIZE, FEATURES_SIZE)
    
    pipeline_reg = SimpleMLPipeline(task_type='regression', algorithm='xgboost')
    pipeline_reg.fit(X_reg, y_reg)
    pipeline_reg.evaluate()
    
    # Algorithm comparison
    print("\n2. ALGORITHM COMPARISONS:")
    print("-" * 30)
    
    # Compare classification algorithms
    run_algorithm_comparison(SAMPLE_SIZE, FEATURES_SIZE, 'classification')
    
    # Compare regression algorithms
    run_algorithm_comparison(SAMPLE_SIZE, FEATURES_SIZE, 'regression')
    
    print(f"\n{'='*50}")
    print("All tests completed successfully!")
    print(f"{'='*50}") 