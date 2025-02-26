import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):
    pass

def distance_l2(X, Y):
    return (X-Y).norm(2)

def distance_dot(X, Y):
    return X @ Y

def distance_manhattan(X, Y):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    def f(Y):
        return distance_l2(X,Y)
    distance = torch.vmap(f)(A)
    _, indices = distance.sort()
    return A[indices[:K]]

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

device = "cuda" if torch.cuda.is_available() else "cpu"

def our_kmeans(N, D, A, K):
    indices = torch.randperm(N, device=device)[:K]
    centroids = A[indices]
    def f(Y):
        return torch.vmap(lambda x: distance_l2(x, Y))(centroids)
    while True:
        distances = torch.vmap(f)(A)

        # Assign each point to the nearest centroid
        assignments = torch.argmin(distances, dim=1)

        # Update centroids using vectorized operations
        mask = torch.zeros(N, K, dtype=A.dtype, device=A.device)
        mask[torch.arange(N, device=A.device), assignments] = 1.0
        counts = mask.sum(dim=0)  # Shape (K,)
        sums = torch.mm(mask.T, A)  # Shape (K, D)
        new_centroids = sums / counts.unsqueeze(1)

        # Check for centroid convergence
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
        if centroid_shift < 1e-4:
            break
        centroids = new_centroids
    return assignments, centroids

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn_filter(A,X,K):        
    def f(Y):
        return distance_l2(X,Y)
    distance = torch.vmap(f)(A)
    _, indices = distance.sort()
    return indices < K

def our_ann(N, D, A, X, K):

    assignments, means = our_kmeans(N, D, X, 5)
    nearest_means = our_knn_filter(means, X, 2)
    mask = nearest_means[assignments]
    filtered = A[mask]
    return our_knn(N, D, filtered, X, K)

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
