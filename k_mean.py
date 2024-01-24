import numpy as np
import matplotlib.pyplot as plt

def k_means(data, k, max_iters=100):
    # Randomly initialize centroids
    centroids = data[np.random.choice(len(data), k, replace=False)]
    
    for _ in range(max_iters):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(data - centroids[:, np.newaxis], axis=2), axis=0)
        
        # Update centroids based on the mean of the assigned points
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Example usage:
def plot_clusters(data, labels, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5, edgecolors='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()


data = np.array([[1, 2], [3, 4], [4, 3], [2, 1]])
k = 2
labels, centroids = k_means(data, k)
plot_clusters(data, labels, centroids)