import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    # Randomly select k data points as initial centroids
    return X[np.random.choice(len(X), k, replace=False)]

def assign_to_clusters(X, centroids):
    # Assign each data point to the cluster with the closest centroid
    return np.argmin(np.linalg.norm(X - centroids[:, np.newaxis], axis=2), axis=0)

def update_centroids(X, labels, k):
    # Move each centroid to the mean of its partition
    new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
    return new_centroids

def k_means(X, k, max_iters=100, tol=1e-4):
    # Step 1: Initialize centroids
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        # Step 2: Assign data points to clusters
        labels = assign_to_clusters(X, centroids)

        # Step 3: Update centroids
        new_centroids = update_centroids(X, labels, k)

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