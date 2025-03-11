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

    start_time = time.time()  # Start the timer
    result = func(*args)  # Run the function
    torch.cuda.synchronize() if device == "cuda" else None  

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    return result, elapsed_time

def print_table(results):
    print("Benchmark - CPU vs GPU Performance")
    print(" N       | D        | Function             | CPU Time (s)  | GPU Time (s)  | Speedup ")
    print("---------------------------------------------------------------------------------------------------")
    for result in results:
        N, D, func, cpu_time, gpu_time = result
        speedup = cpu_time / gpu_time
        print(f"{N:<8} | {D:<8} | {func:<20} | {cpu_time:.6f}      | {gpu_time:.6f}      | {speedup:.2f}x")

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def distance_l2_cupy(A, centroids):
    # print(f"A type: {type(A)}, centroids type: {type(centroids)}")
    
    diff = A[:, cp.newaxis, :] - centroids[cp.newaxis, :, :]  # Should be CuPy
    # print(f"diff type: {type(diff)}")  # 🔍 Check if it's CuPy or PyTorch

    distances = cp.sum(diff ** 2, axis=2)  
    return distances

def kmeans_cupy(N, D, A, K, max_iters=100, tol=1e-4):
    """
    Perform K-Means clustering on a dataset using CuPy.

    Args:
        N (int): Number of data points
        D (int): Dimension of each data point
        A (cp.ndarray): Array of shape (N, D), dataset
        K (int): Number of clusters
        max_iters (int): Maximum number of iterations
        tol (float): Convergence tolerance (threshold for centroid movement)
    
    Returns:
        cp.ndarray: Array of shape (N,) with cluster assignments
    """

    if isinstance(A, np.ndarray):
        A = cp.array(A, dtype=cp.float32)

    # Step 1: Initialize K random points from A as initial centroids
    indices = cp.random.permutation(N)[:K]
    centroids = A[indices].copy()

    if isinstance(A, torch.Tensor):
        A = cp.asarray(A.cpu().numpy(), dtype=cp.float32)
    if isinstance(centroids, torch.Tensor):
        centroids = cp.asarray(centroids.cpu().numpy(), dtype=cp.float32)

    for _ in range(max_iters):
        old_centroids = centroids.copy()

        # Step 2a: Assignment Step - Compute distances and assign clusters
        distances = distance_l2_cupy(A, centroids)  # A: (N, D), centroids: (K, D)
        cluster_assignments = distances.argmin(axis=1)  # Shape (N,)

        # Step 2b: Update Step - Compute new centroids
        centroids = cp.zeros((K, D), dtype=cp.float32)  # Placeholder for new centroids
        counts = cp.zeros(K, dtype=cp.float32)  # Count points per cluster

        # Scatter reduce: sum points in each cluster
        for k in range(K):
            mask = cluster_assignments == k
            if cp.any(mask):
                centroids[k] = cp.sum(A[mask], axis=0)
                counts[k] = cp.sum(mask)

        # Avoid division by zero
        mask = counts > 0
        centroids[mask] /= counts[mask][:, cp.newaxis]

        # Step 3: Check for convergence
        # if cp.linalg.norm(centroids - old_centroids, ord=2) < tol:
        #     break
        centroid_diff = centroids - old_centroids
        norm = cp.sqrt(cp.sum(centroid_diff ** 2))
        if norm < tol:
            break

    return cluster_assignments

# CUPY WITH KERNEL-----------------------------------------------------------

# Custom kernel for assignment and distance calculation
assign_and_distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void assign_and_distance(const float* A, const float* centroids, int* cluster_assignments, float* distances, int N, int D, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        float min_dist = 3.4e+38;
        int best_k = 0;

        for (int k = 0; k < K; k++) {
            float dist = 0.0;
            for (int d = 0; d < D; d++) {
                float diff = A[n * D + d] - centroids[k * D + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_k = k;
            }
        }

        cluster_assignments[n] = best_k;
        distances[n] = min_dist;
    }
}
''', 'assign_and_distance')

# Custom kernel for centroid updates
update_centroids_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_centroids(const float* A, const int* cluster_assignments, float* centroids, int* counts, int N, int D, int K) {
    extern __shared__ float shared_mem[];
    float* shared_centroids = shared_mem;
    int* shared_counts = (int*)&shared_centroids[K * D];

    int tid = threadIdx.x;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < K * D) {
        shared_centroids[tid] = 0.0;
    }
    if (tid < K) {
        shared_counts[tid] = 0;
    }
    __syncthreads();

    // Accumulate centroids and counts in shared memory
    if (n < N) {
        int k = cluster_assignments[n];
        for (int d = 0; d < D; d++) {
            atomicAdd(&shared_centroids[k * D + d], A[n * D + d]);
        }
        atomicAdd(&shared_counts[k], 1);
    }
    __syncthreads();

    // Write results to global memory
    if (tid < K * D) {
        atomicAdd(&centroids[tid], shared_centroids[tid]);
    }
    if (tid < K) {
        atomicAdd(&counts[tid], shared_counts[tid]);
    }
}
''', 'update_centroids')

def kmeans_cupy_kernel(N, D, A, K, max_iters=100, tol=1e-4):
    if isinstance(A, np.ndarray):
        A = cp.array(A, dtype=cp.float32)

    # Step 1: Initialize K random points from A as initial centroids
    indices = cp.random.permutation(N)[:K]
    centroids = A[indices].copy()

    for _ in range(max_iters):
        old_centroids = centroids.copy()

        # Step 2a: Assignment Step - Compute distances and assign clusters
        cluster_assignments = cp.zeros(N, dtype=cp.int32)
        distances = cp.zeros(N, dtype=cp.float32)

        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        assign_and_distance_kernel((grid_size,), (block_size,), (A, centroids, cluster_assignments, distances, N, D, K))

        # Step 2b: Update Step - Compute new centroids
        centroids = cp.zeros((K, D), dtype=cp.float32)
        counts = cp.zeros(K, dtype=cp.int32)

        shared_mem_size = (K * D) * cp.dtype(cp.float32).itemsize + K * cp.dtype(cp.int32).itemsize
        update_centroids_kernel((grid_size,), (block_size,), (A, cluster_assignments, centroids, counts, N, D, K), shared_mem=shared_mem_size)

        # Avoid division by zero
        mask = counts > 0
        centroids[mask] /= counts[mask][:, cp.newaxis]

        # Step 3: Check for convergence
        centroid_diff = centroids - old_centroids
        norm = cp.sqrt(cp.sum(centroid_diff ** 2))
        if norm < tol:
            break

    return cluster_assignments

# ---------------------------------------------------------------------------

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

def kmeans_torch(N, D, A, K, max_iters=100, tol = 1e-4, device="cuda"):
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
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_kmeans_torch(N, D, A, K):
    measure_time(kmeans_torch, "cpu", N, D, A, 1) # warmup
    cpu_result, cpu_time = measure_time(kmeans_torch, "cpu", N, D, A, K)
    gpu_torch_result, gpu_torch_time = measure_time(kmeans_torch, "cuda", N, D, A, K)

    return N, D, "KMeans Torch", cpu_time, gpu_torch_time

def test_kmeans_cupy(N, D, A, K):
    cpu_result, cpu_time = measure_time(kmeans_torch, "cpu", N, D, A, K)
    gpu_cupy_result, gpu_cupy_time = measure_time(kmeans_cupy, "cuda", N, D, A, K)

    return N, D, "KMeans CuPy", cpu_time, gpu_cupy_time

def test_kmeans_cupy_kernel(N, D, A, K):
    cpu_result, cpu_time = measure_time(kmeans_torch, "cpu", N, D, A, K)
    gpu_cupy_kernel_result, gpu_cupy_kernel_time = measure_time(kmeans_cupy_kernel, "cuda", N, D, A, K)

    return N, D, "KMeans CuPy Kernel", cpu_time, gpu_cupy_kernel_time

if __name__ == "__main__":
    np.random.seed(123)
    N, D, A, K = testdata_kmeans("")

    results = []
    results.append(test_kmeans_torch(N, D, A, K))
    results.append(test_kmeans_cupy(N, D, A, K))
    results.append(test_kmeans_cupy_kernel(N, D, A, K))

    print_table(results)

#  N       | D        | Function             | CPU Time (s)  | GPU Time (s)  | Speedup 
# ---------------------------------------------------------------------------------------------------
# 100000   | 100      | KMeans Torch         | 0.444396      | 0.406674      | 1.09x
# 100000   | 100      | KMeans CuPy          | 0.405463      | 5.590575      | 0.07x
# 100000   | 100      | KMeans CuPy Kernel   | 0.407877      | 2.777226      | 0.15x

#  N       | D        | Function             | CPU Time (s)  | GPU Time (s)  | Speedup 
# ---------------------------------------------------------------------------------------------------
# 10000    | 2        | KMeans Torch         | 0.122799      | 0.057809      | 2.12x
# 10000    | 2        | KMeans CuPy          | 0.047961      | 1.104833      | 0.04x
# 10000    | 2        | KMeans CuPy Kernel   | 0.084559      | 0.127401      | 0.66x

#  N       | D        | Function             | CPU Time (s)  | GPU Time (s)  | Speedup 
# ---------------------------------------------------------------------------------------------------
# 10000    | 1024     | KMeans Torch         | 0.156841      | 0.116008      | 1.35x
# 10000    | 1024     | KMeans CuPy          | 0.135100      | 1.599215      | 0.08x
# 10000    | 1024     | KMeans CuPy Kernel   | 0.163891      | 1.442933      | 0.11x