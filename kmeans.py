"""
Simple K-Means Clustering Implementation
A basic, easy-to-understand K-means algorithm using only Python.
"""

import random
import math


class KMeans:
    """
    Simple K-Means clustering algorithm.
    
    Parameters:
    -----------
    k : int
        Number of clusters
    max_iters : int, default=10
        Number of iterations to run
    """
    
    def __init__(self, k, max_iters=10):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.labels = []
    
    def distance(self, point1, point2):
        """Calculate distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def fit(self, data):
        """
        Fit K-means to data.
        
        Parameters:
        -----------
        data : list of lists
            Input data where each inner list is a data point
        """
        # Step 1: Initialize centroids randomly
        self.centroids = random.sample(data, self.k)
        
        # Step 2: Run K-means for max_iters iterations
        for _ in range(self.max_iters):
            # Assign each point to nearest centroid
            self.labels = []
            for point in data:
                distances = [self.distance(point, centroid) for centroid in self.centroids]
                closest = distances.index(min(distances))
                self.labels.append(closest)
            
            # Update centroids
            new_centroids = []
            for i in range(self.k):
                # Find all points assigned to cluster i
                cluster_points = [data[j] for j in range(len(data)) if self.labels[j] == i]
                
                if cluster_points:
                    # Calculate mean of cluster points
                    n_features = len(cluster_points[0])
                    centroid = []
                    for feature in range(n_features):
                        mean_val = sum(point[feature] for point in cluster_points) / len(cluster_points)
                        centroid.append(mean_val)
                    new_centroids.append(centroid)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids.append(self.centroids[i])
            
            self.centroids = new_centroids
        
        return self
    
    def predict(self, data):
        """Predict cluster for new data points."""
        labels = []
        for point in data:
            distances = [self.distance(point, centroid) for centroid in self.centroids]
            closest = distances.index(min(distances))
            labels.append(closest)
        return labels


# Example usage
if __name__ == "__main__":
    # Sample data: 2D points
    data = [
        [1, 2], [1, 4], [1, 0],
        [10, 2], [10, 4], [10, 0],
        [5, 5], [6, 5], [4, 5]
    ]
    
    print("Simple K-Means Example")
    print("=" * 30)
    
    # Create and fit model
    kmeans = KMeans(k=3, max_iters=5)
    kmeans.fit(data)
    
    print("Original data:")
    for i, point in enumerate(data):
        print(f"  Point {i+1}: {point} -> Cluster {kmeans.labels[i]}")
    
    print("\nFinal centroids:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"  Cluster {i}: [{centroid[0]:.1f}, {centroid[1]:.1f}]")
    
    # Test prediction on new data
    new_data = [[2, 3], [9, 1]]
    predictions = kmeans.predict(new_data)
    print("\nPredictions for new data:")
    for point, pred in zip(new_data, predictions):
        print(f"  {point} -> Cluster {pred}") 