import torch
# import cupy as cp
# import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------------------

def measure_time(func, *args):
    # Ensure tensors are on the correct device
    torch.cuda.synchronize() if torch.cuda.is_available() else None  

    start_time = time.time()  # Start the timer
    result = func(*args)  # Run the function
    torch.cuda.synchronize() if torch.cuda.is_available() else None  

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    return result, elapsed_time

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):
    """
    Compute the pairwise cosine distance between rows of X and Y.
    
    Args:
        X (torch.Tensor): Tensor of shape (m, d)
        Y (torch.Tensor): Tensor of shape (n, d)
    
    Returns:
        torch.Tensor: Cosine distance matrix of shape (m, n)
    """
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)  # Normalize along features (d)
    Y_norm = torch.nn.functional.normalize(Y, p=2, dim=1)
    
    cosine_similarity = torch.mm(X_norm, Y_norm.T)  
    
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance


def distance_l2(X, Y):
    """
    Compute the pairwise Euclidean distance (L2 distance) between rows of X and Y.
    
    Args:
        X (torch.Tensor): Tensor of shape (m, d)
        Y (torch.Tensor): Tensor of shape (n, d)
    
    Returns:
        torch.Tensor: Euclidean distance matrix of shape (m, n)
    """
    # Compute squared norms of each row in X and Y
    X_norm_sq = torch.sum(X ** 2, dim=1, keepdim=True)  # Shape: (m, 1)
    Y_norm_sq = torch.sum(Y ** 2, dim=1, keepdim=True)  # Shape: (n, 1)

    # Compute pairwise squared distances using broadcasting
    squared_dist = X_norm_sq + Y_norm_sq.T - 2 * torch.mm(X, Y.T)

    # Ensure numerical stability (avoid sqrt of negative values due to precision errors)
    squared_dist = torch.clamp(squared_dist, min=0.0)

    # Compute the square root to get final distances
    return torch.sqrt(squared_dist)


def distance_dot(X, Y):
    pass

def distance_manhattan(X, Y):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K): 
    """
    Find the top-K nearest vectors to X in the set A using Euclidean distance.

    Args:
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (torch.Tensor): Tensor of shape (N, D), collection of vectors
        X (torch.Tensor): Tensor of shape (D,), query vector
        K (int): Number of nearest neighbors to find
    
    Returns:
        torch.Tensor: Indices of the top K nearest neighbors
    """
    # Determine device first
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to PyTorch tensors directly on the device
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)  # Creates tensor on GPU if available
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)

    # Ensure X has correct shape (1, D) for broadcasting in distance_l2
    # X = X.unsqueeze(0)  # Shape: (1, D)

    # Compute pairwise Euclidean distances using the distance_l2 function
    # distances = distance_l2(A, X).squeeze(1)  # Shape: (N,)
    distances = torch.norm(A - X, dim=1, p=2)  # Shape: (N,)
    # -> much faster...

    # Implement custom top-K selection using argsort
    indices = torch.argsort(distances)  # Sort in ascending order

    # Take the first K indices (smallest distances)
    top_k_indices = indices[:K]

    return top_k_indices.cpu()  # Move back to CPU if needed

'''
Possible optimisations:
- heap (O(NlogK)) or quickselect-based (O(N)) selection, rather than argsort (O(NlogN))
- parallelise parts of the selection process using CUDA kernels
'''

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K, max_iters=100, tol = 1e-4):
    """
    Perform K-Means clustering on a dataset.

    Args:
        N (int): Number of data points
        D (int): Dimension of each data point
        A (torch.Tensor): Tensor of shape (N, D), dataset
        K (int): Number of clusters
        max_iters (int): Maximum number of iterations
        tol (float): Convergence tolerance (threshold for centroid movement)
    
    Returns:
        torch.Tensor: Tensor of shape (N,) with cluster assignments
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)

    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)

    # Step 1: Initialize K random points from A as initial centroids
    indices = torch.randperm(N)[:K]
    centroids = A[indices].clone()

    for _ in range(max_iters):
        old_centroids = centroids.clone()

        # Step 2a: Assignment Step - Compute distances and assign clusters
        distances = distance_l2(A, centroids)  # A: (N, D), centroids: (K, D)
        cluster_assignments = distances.argmin(dim=1)  # Shape (N,)

        # Step 2b: Update Step - Compute new centroids
        centroids = torch.zeros(K, D, device=device)  # Placeholder for new centroids
        counts = torch.zeros(K, device=device)  # Count points per cluster

        # Scatter reduce: sum points in each cluster
        centroids.scatter_add_(0, cluster_assignments.view(-1, 1).expand(-1, D), A)
        counts.scatter_add_(0, cluster_assignments, torch.ones(N, device=device))

        # Avoid division by zero
        mask = counts > 0
        centroids[mask] /= counts[mask].unsqueeze(1)

        # Step 3: Check for convergence
        if torch.norm(centroids - old_centroids, p=2) < tol:
            break

    return cluster_assignments


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K, K1=5, K2=10):
    """
    Approximate Nearest Neighbor (ANN) using K-Means and KNN.

    Args:
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (torch.Tensor): Tensor of shape (N, D), collection of vectors
        X (torch.Tensor): Tensor of shape (D,), query vector
        K (int): Number of nearest neighbors to find
        K1 (int): Number of nearest cluster centers to consider
        K2 (int): Number of nearest neighbors from K1 clusters

    Returns:
        torch.Tensor: Indices of the top K nearest neighbors
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to PyTorch tensors directly on the device
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Step 1: Run K-Means clustering to partition the dataset
    K_means = 10  # Number of clusters (tunable)
    centroids = A[torch.randperm(N)[:K_means]]  # Randomly select initial centroids
    prev_centroids = torch.zeros_like(centroids)
    cluster_assignments = torch.zeros(N, dtype=torch.long, device=device)
    
    # Run K-Means iterations
    for _ in range(10):  # Fixed iterations for simplicity
        distances = torch.cdist(A, centroids)  # Compute distances to centroids
        cluster_assignments = torch.argmin(distances, dim=1)  # Assign clusters
        
        # Update centroids
        for i in range(K_means):
            cluster_points = A[cluster_assignments == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(dim=0)
        
        # Check for convergence
        if torch.allclose(centroids, prev_centroids, atol=1e-4):
            break
        prev_centroids = centroids.clone()
    
    # Step 2: Find the nearest K1 cluster centers to X
    cluster_distances = torch.cdist(X.unsqueeze(0), centroids).squeeze(0)
    nearest_clusters = torch.argsort(cluster_distances)[:K1]  # Select top K1 clusters
    
    # Step 3: Gather all points in the selected clusters
    candidate_indices = torch.cat([torch.nonzero(cluster_assignments == c, as_tuple=True)[0] for c in nearest_clusters])
    candidate_vectors = A[candidate_indices]
    
    # Step 4: Find the top K2 nearest neighbors from these candidates
    knn_distances = torch.norm(candidate_vectors - X, dim=1, p=2)
    top_k2_indices = torch.argsort(knn_distances)[:K2]
    final_candidates = candidate_indices[top_k2_indices]
    
    # Step 5: Merge and return final K neighbors
    return final_candidates[:K].cpu()

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    # N, D, A, K = testdata_kmeans("test_file.json")
    N, D, A, K = testdata_kmeans("")
    
    kmeans_result, elapsed_time = measure_time(our_kmeans, N, D, A, K)

    # print("Kmeans clusters : ", kmeans_result.tolist())
    print(f"Kmeans time elapsed: {elapsed_time1:.6f} sec")


def test_knn():
    # N, D, A, X, K = testdata_knn("test_file.json")
    N, D, A, X, K = testdata_knn(1000, pow(2, 1), 10)

    knn_result, elapsed_time = measure_time(our_knn, N, D, A, X, K)

    print("K closest neighbors: ", knn_result.tolist())
    print(f"Time elapsed: {elapsed_time:.6f} sec")
    
def test_ann():
    # N, D, A, X, K = testdata_ann("test_file.json")
    N, D, A, X, K = testdata_ann("")

    knn_result, knn_elapsed_time = measure_time(our_knn, N, D, A, X, K)
    ann_result, ann_elapsed_time = measure_time(our_ann, N, D, A, X, K)

    print("KNN result: ", knn_result.tolist())
    print(f"KNN time elapsed: {knn_elapsed_time:.6f} sec")
    print("ANN result: ", ann_result.tolist())
    print(f"ANN time elapsed: {ann_elapsed_time:.6f} sec")

    recall = recall_rate(sorted(knn_result.tolist()), sorted(ann_result.tolist()))
    print(f"Recall rate: {recall:.4f}")
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_ann()
