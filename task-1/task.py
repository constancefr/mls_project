import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------------------
def warm_up_cpu():
    # Run a simple CPU operation to warm up caches and branch predictor
    warm_up_data = np.zeros(1, dtype=np.float32)  # Allocate a small array
    warm_up_data += 1  # Perform a simple operation

def warm_up_gpu():
    # Run a simple GPU operation to initialize CUDA and warm up the GPU
    warm_up_data = cp.zeros(1, dtype=cp.float32)  # Allocate a small array on the GPU
    warm_up_data += 1  # Perform a simple operation
    cp.cuda.Stream.null.synchronize()  # Synchronize to ensure the operation completes

def measure_time(func, device, *args):
    # Ensure tensors are on the correct device
    torch.cuda.synchronize() if device == "cuda" else None  

    if device == "cuda":
        warm_up_gpu()
    else:
        warm_up_cpu()

    start_time = time.time()  # Start the timer
    result = func(*args)  # Run the function
    torch.cuda.synchronize() if device == "cuda" else None  

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    return result, elapsed_time

def print_table(results):
    print("Benchmark - CPU vs GPU Performance")
    print(" N       | D        | Function              | CPU Time (s)  | GPU Time (s)  | Speedup ")
    print("---------------------------------------------------------------------------------------------------")
    for result in results:
        N, D, func, cpu_time, gpu_time = result
        speedup = cpu_time / gpu_time
        print(f"{N:<8} | {D:<8} | {func:<20} | {cpu_time:.6f}      | {gpu_time:.6f}      | {speedup:.2f}x")

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

def our_knn(N, D, A, X, K, device="cuda"): 
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

    # Convert to PyTorch tensors directly on the device
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)

    # Compute pairwise Euclidean distances using the distance_l2 function
    # distances = distance_l2(A, X).squeeze(1)  # Shape: (N,)
    distances = torch.norm(A - X, dim=1, p=2)  # faster than l2 implementation...

    indices = torch.argsort(distances)  # Sort in ascending order
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

def kmeans_cupy(N, D, A, K, max_iters=100, tol=1e-4):
    # Convert input data to CuPy arrays
    A = cp.asarray(A, dtype=cp.float32)
    
    # Step 1: Initialize K random centroids
    # print("A: ", A)
    indices = cp.random.choice(N, K, replace=False)
    centroids = A[indices].copy()
    # print("Init. centroid indices: ", indices)
    # print("Init. centroids: ", centroids)
    
    for _ in range(max_iters):
        # Step 2a: Compute distances and assign clusters
        # -> USE L2 / COSINE????
        distances = cp.linalg.norm(A[:, cp.newaxis] - centroids, axis=2)  # Shape: (N, K)
        cluster_assignments = cp.argmin(distances, axis=1)  # Shape: (N,)
        
        # Step 2b: Update centroids using advanced indexing and reduction
        new_centroids = cp.zeros((K, D), dtype=cp.float32)
        for k in range(K):
            mask = cluster_assignments == k
            if cp.any(mask):
                new_centroids[k] = cp.mean(A[mask], axis=0)
        
        # Step 3: Check for convergence
        if cp.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    
    return cluster_assignments.get()

# --------OPTIMISED CUPY KMEANS-------------
# Custom kernel for distance computation
distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void distance_kernel(const float* A, const float* centroids, float* distances, int N, int D, int K) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;  // Thread ID
    if (tid < N * K) {
        int n = tid / K;  // Data point index
        int k = tid % K;  // Centroid index
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = A[n * D + d] - centroids[k * D + d];
            dist += diff * diff;
        }
        distances[n * K + k] = sqrtf(dist);
    }
}
''', 'distance_kernel')

# Custom kernel for centroid updates
centroid_update_kernel = cp.RawKernel(r'''
extern "C" __global__
void centroid_update_kernel(const float* A, const int* cluster_assignments, float* new_centroids, int* counts, int N, int D, int K) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;  // Thread ID
    if (tid < N) {
        int k = cluster_assignments[tid];  // Cluster assignment for this point
        for (int d = 0; d < D; d++) {
            atomicAdd(&new_centroids[k * D + d], A[tid * D + d]);
        }
        atomicAdd(&counts[k], 1);
    }
}
''', 'centroid_update_kernel')

def kmeans_cupy_custom_kernels(N, D, A, K, max_iters=100, tol=1e-4):
    A = cp.asarray(A, dtype=cp.float32)
    
    # Step 1: Initialize K random centroids
    indices = cp.random.choice(N, K, replace=False)
    centroids = A[indices].copy()
    
    # Allocate memory for distances and cluster assignments
    distances = cp.zeros((N, K), dtype=cp.float32)
    cluster_assignments = cp.zeros(N, dtype=cp.int32)
    
    # Define thread and block sizes
    threads_per_block = 256
    blocks_per_grid_distance = (N * K + threads_per_block - 1) // threads_per_block
    blocks_per_grid_centroid = (N + threads_per_block - 1) // threads_per_block
    
    for _ in range(max_iters):
        old_centroids = centroids.copy()
        
        # Step 2a: Compute distances using the custom kernel
        distance_kernel((blocks_per_grid_distance,), (threads_per_block,), 
                        (A, centroids, distances, N, D, K))
        
        # Step 2b: Assign clusters
        cluster_assignments = cp.argmin(distances, axis=1)
        
        # Step 2c: Update centroids using the custom kernel
        new_centroids = cp.zeros((K, D), dtype=cp.float32)
        counts = cp.zeros(K, dtype=cp.int32)
        centroid_update_kernel((blocks_per_grid_centroid,), (threads_per_block,), 
                              (A, cluster_assignments, new_centroids, counts, N, D, K))
        
        # Normalize centroids
        mask = counts > 0
        new_centroids[mask] /= counts[mask, cp.newaxis]
        
        # Step 3: Check for convergence
        if cp.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    
    return cluster_assignments.get()

# ------------------------------------------


def our_kmeans(N, D, A, K, max_iters=100, tol = 1e-4, device="cuda"):
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

def our_ann(N, D, A, X, K, K1=5, K2=10, device="cuda"):
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

    # Convert to PyTorch tensors directly on the device
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Step 1: Run K-Means clustering to partition the dataset
    K_means = 10 # ???
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
def test_kmeans_torch():
    N, D, A, K = testdata_kmeans("")
    measure_time(our_kmeans, "cpu", N, D, A, 1) # warmup
    cpu_result, cpu_time = measure_time(our_kmeans, "cpu", N, D, A, K)
    gpu_torch_result, gpu_torch_time = measure_time(our_kmeans, "cuda", N, D, A, K)
    print(f"Kmeans CPU time: {cpu_time:.6f} sec")
    print(f"Kmeans Torch GPU time: {gpu_torch_time:.6f} sec")

    return N, D, "KMeans Torch", cpu_time, gpu_torch_time

def test_kmeans_cupy():
    N, D, A, K = testdata_kmeans("")
    cpu_result, cpu_time = measure_time(our_kmeans, "cpu", N, D, A, K)
    gpu_cupy_result, gpu_cupy_time = measure_time(kmeans_cupy, "cuda", N, D, A, K)
    print(f"Kmeans CPU time: {cpu_time:.6f} sec")
    print(f"Kmeans CuPy GPU time: {gpu_cupy_time:.6f} sec")

    return N, D, "KMeans CuPy", cpu_time, gpu_cupy_time

def test_kmeans_cupy_optimised():
    N, D, A, K = testdata_kmeans("")
    cpu_result, cpu_time = measure_time(our_kmeans, "cpu", N, D, A, K)
    gpu_cupy_opt_result, gpu_cupy_opt_time = measure_time(kmeans_cupy_custom_kernels, "cuda", N, D, A, K)
    print(f"Kmeans CPU time: {cpu_time:.6f} sec")
    print(f"Kmeans CuPy optimised GPU time: {gpu_cupy_opt_time:.6f} sec")

    return N, D, "KMeans CuPy Opt.", cpu_time, gpu_cupy_opt_time

def test_knn():
    N, D, A, X, K = testdata_knn("")
    cpu_result, cpu_time = measure_time(our_knn, "cpu", N, D, A, X, K)
    gpu_result, gpu_time = measure_time(our_knn, "cuda", N, D, A, X, K)
    print(f"KNN CPU time: {cpu_time:.6f} sec")
    print(f"KNN GPU time: {gpu_time:.6f} sec")

    return N, D, "KNN", cpu_time, gpu_time
    
def test_ann():
    N, D, A, X, K = testdata_ann("")

    ann_cpu_result, ann_cpu_time = measure_time(our_ann, "cpu", N, D, A, X, K)
    ann_gpu_result, ann_gpu_time = measure_time(our_ann, "cuda", N, D, A, X, K)
    print(f"ANN CPU time: {ann_cpu_time:.6f} sec")
    print(f"ANN GPU time: {ann_gpu_time:.6f} sec")

    knn_cpu_result, knn_cpu_time = measure_time(our_knn, "cpu", N, D, A, X, K)
    knn_gpu_result, knn_gpu_time = measure_time(our_knn, "cuda", N, D, A, X, K)
    print(f"KNN CPU time: {knn_cpu_time:.6f} sec")
    print(f"KNN GPU time: {knn_gpu_time:.6f} sec")

    recall = recall_rate(sorted(knn_gpu_result.tolist()), sorted(ann_gpu_result.tolist()))
    print(f"Recall rate of KNN vs ANN (GPU): {recall:.4f}")
    
    return N, D, "ANN", ann_cpu_time, ann_gpu_time

def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)


if __name__ == "__main__":
    np.random.seed(123)

    results = []
    results.append(test_kmeans_torch())
    results.append(test_kmeans_cupy())
    results.append(test_kmeans_cupy_optimised())
    results.append(test_knn())
    results.append(test_ann())
    print_table(results)
