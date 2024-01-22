import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def k_means(X, k, max_iters=100, tol=1e-4):
    """
    K-means clustering algorithm.

    Parameters:
        X (numpy.ndarray): Input data with shape (m, n), where m is the number of samples and n is the number of features.
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.

    Returns:
        centroids (numpy.ndarray): Final cluster centroids with shape (k, n).
        labels (numpy.ndarray): Labels assigned to each data point.
    """
    m, n = X.shape

    # Initialize centroids randomly
    centroids = X[np.random.choice(m, k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X - centroids[:, np.newaxis], axis=2), axis=0)

        # Update centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels

# Example usage:
# Generate some random data for testing
np.random.seed(42)
data = np.random.rand(100, 2)

# Set the number of clusters (k)
num_clusters = 3

# Run k-means algorithm
final_centroids, cluster_labels = k_means(data, num_clusters)

# Visualize the clustered data
plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
