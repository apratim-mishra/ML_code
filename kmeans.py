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


def generate_clustering_data(sample_size, features_size, num_clusters=3):
    """Generate random data with natural clusters."""
    data = []
    
    # Create cluster centers
    cluster_centers = []
    for _ in range(num_clusters):
        center = [random.uniform(-5, 5) for _ in range(features_size)]
        cluster_centers.append(center)
    
    # Generate points around each cluster center
    points_per_cluster = sample_size // num_clusters
    remaining_points = sample_size % num_clusters
    
    for i, center in enumerate(cluster_centers):
        # Add extra points to first clusters if sample_size not evenly divisible
        cluster_size = points_per_cluster + (1 if i < remaining_points else 0)
        
        for _ in range(cluster_size):
            # Generate point near cluster center with some noise
            point = []
            for j in range(features_size):
                noise = random.uniform(-1.5, 1.5)  # Add noise around center
                point.append(center[j] + noise)
            data.append(point)
    
    # Shuffle the data so clusters are mixed
    random.shuffle(data)
    return data


# Example usage
if __name__ == "__main__":
    print("Simple K-Means Example")
    print("=" * 30)
    
    # Configuration parameters
    SAMPLE_SIZE = 100
    FEATURES_SIZE = 2
    K_CLUSTERS = 3
    MAX_ITERATIONS = 10
    
    print(f"Sample size: {SAMPLE_SIZE}, Features: {FEATURES_SIZE}, Clusters: {K_CLUSTERS}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate random clustering data
    data = generate_clustering_data(SAMPLE_SIZE, FEATURES_SIZE, K_CLUSTERS)
    
    print(f"\nSample data points:")
    for i in range(5):
        print(f"  Point {i+1}: {[round(x, 2) for x in data[i]]}")
    
    # Create and fit model
    kmeans = KMeans(k=K_CLUSTERS, max_iters=MAX_ITERATIONS)
    kmeans.fit(data)
    
    print(f"\nCluster assignments (first 10 points):")
    for i in range(min(10, len(data))):
        print(f"  Point {i+1}: {[round(x, 2) for x in data[i]]} -> Cluster {kmeans.labels[i]}")
    
    print(f"\nFinal centroids:")
    for i, centroid in enumerate(kmeans.centroids):
        centroid_rounded = [round(x, 2) for x in centroid]
        print(f"  Cluster {i}: {centroid_rounded}")
    
    # Test prediction on new random data
    test_data = generate_clustering_data(5, FEATURES_SIZE, K_CLUSTERS)
    predictions = kmeans.predict(test_data)
    print(f"\nPredictions for new data:")
    for point, pred in zip(test_data, predictions):
        print(f"  {[round(x, 2) for x in point]} -> Cluster {pred}")
    
    # Count points in each cluster
    cluster_counts = [0] * K_CLUSTERS
    for label in kmeans.labels:
        cluster_counts[label] += 1
    
    print(f"\nCluster sizes:")
    for i, count in enumerate(cluster_counts):
        print(f"  Cluster {i}: {count} points") 