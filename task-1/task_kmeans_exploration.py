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
    print(" N       | D        | Function                       | CPU Time (s)  | GPU Time (s)  | Speedup ")
    print("---------------------------------------------------------------------------------------------------")
    for result in results:
        N, D, func, cpu_time, gpu_time = result
        speedup = cpu_time / gpu_time
        print(f"{N:<8} | {D:<8} | {func:<24}       | {cpu_time:.6f}      | {gpu_time:.6f}      | {speedup:.2f}x")

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# TORCH ------------------------------------------------------------------------------------------
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

# CUPY -------------------------------------------------------------------------------------------
l2_closest_kernel = """
extern "C" __global__
void closest_index(const float* x, float* y, int* output, const int N, const int M, const int D) {
    extern __shared__ char shared_mem[];
    float* shared_dist = (float*)shared_mem;
    int* shared_idx = (int*)(shared_dist + blockDim.x);

    int tid = threadIdx.x;
    int i = blockIdx.x;  // Each block handles one vector x[i]
    int num_threads = blockDim.x;

    float min_dist = 99999999.0;
    int min_index = -1;

    // Grid-stride loop over M y vectors
    // Each block handles one data point x[i], threads within the block compute l2 distances to all centroids y[i]
    for (int j = tid; j < M; j += num_threads) {
        float sum_sq = 0.0f;
        for (int k = 0; k < D; ++k) {
            float diff = x[i * D + k] - y[j * D + k];
            sum_sq += diff * diff;
        }
        if (sum_sq < min_dist) {
            min_dist = sum_sq;
            min_index = j;
        }
    }

    // Store local min into shared memory
    shared_dist[tid] = min_dist;
    shared_idx[tid] = min_index;
    __syncthreads();

    // Parallel reduction to find the global min (nearest centroid for each data pt)
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_dist[tid + s] < shared_dist[tid]) {
                shared_dist[tid] = shared_dist[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Write the result
    if (tid == 0) {
        output[i] = shared_idx[0];
    }
}
"""

closest_kernel = cp.RawKernel(l2_closest_kernel, "closest_index")

def cupy_closest_index(x, y, stream=None):
    '''
    Compute the nearest centroid for each data point using a custom CUDA kernel.
    
    Args:
        x (cp.ndarray): Data points A (N, D).
        y (cp.ndarray): Centroids (M, D).
        stream (cp.cuda.Stream): CUDA stream for asynchronous execution.
    
    Returns:
        output (cp.ndarray): Indices of nearest centroids for each data point.
    '''
    N, D = x.shape
    M, D_y = y.shape
    assert D == D_y, "Dimensions of x and y must match"
    
    threads_per_block = 256  # Use a power of two for optimal reduction
    blocks_per_grid = N
    output = cp.zeros(N, dtype=cp.int32)
    
    # shared memory size: floats + ints per thread
    shared_mem_size = (threads_per_block * cp.dtype(cp.float32).itemsize +
                       threads_per_block * cp.dtype(cp.int32).itemsize)
    
    
    # Launch the kernel in the stream
    closest_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (x, y, output, N, M, D),
        shared_mem=shared_mem_size,
        stream=stream
    )
    
    if stream is not None:
        stream.synchronize()
        
    return output

def kmeans_cupy(N,D,A,K, max_iters=100, tol=1e-4, random_state=None, batch_size=1000):
    """
    K-Means clustering using CuPy and a custom CUDA kernel for the assignment step.
    
    Args:
        N (int): Number of data points.
        D (int): Dimension of each data point.
        A (np.ndarray or cp.ndarray): Input data of shape (N, D).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence checking.
        random_state (int): Seed for random number generation.
        batch_size (int): Size of batches for processing data in chunks.
    
    Returns:
        centroids (cupy.ndarray): Final centroids of shape (n_clusters, D).
        assignments (cupy.ndarray): Indices of closest clusters for each data point.
    """

    # Convert to cupy
    if isinstance(A, np.ndarray):
        A = cp.asarray(A, dtype=cp.float32) # ASARRAY INSTEAD OF ARRAY TO AVOID COPY (unnecessary data transfer)

    # 1. Initialize centroids by randomly selecting K data points from A
    indices = cp.random.choice(N, size=K, replace=False)
    centroids = A[indices] # initial centroids, AVOID COPY
    assignments = cp.zeros(N, dtype=cp.int32) # array of zeros of size N for cluster assignments

    # ADDED: Create multiple streams for overlapping computation and data transfers
    num_streams = 2
    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    # 2. Iterate kmeans until convergence / max_iters 
    for _ in range(max_iters):
        # 2a. Assign each data point in A to nearest centroid
        # Process data in batches using multiple streams
        for i in range(0, N, batch_size):
            stream = streams[i // batch_size % num_streams]  # Cycle through streams
            batch = A[i:i + batch_size]
            assignments[i:i + batch_size] = cupy_closest_index(batch, centroids, stream=stream)

        # Synchronize all streams to ensure all batches are processed
        for stream in streams:
            stream.synchronize()

        # 2b. Update centroids
        # mask to indicate which point belongs to which cluster
        mask = cp.zeros((N, K), dtype=cp.float32)
        mask[cp.arange(N), assignments] = 1.0 # one-hot encoding: sets mask[i,k] = 1 if point i is assigned to cluster k

        counts = mask.sum(axis=0) # total pts in each cluster
        sums = cp.matmul(mask.T, A) # matrix mult. to sum pts per cluster
        new_centroids = sums/cp.expand_dims(counts + 1e-8, 1) # compute new centroids

        # 2c. Check for convergence
        centroid_shift = cp.linalg.norm(new_centroids - centroids, axis=1).max() # max shift in centroids between iterations
        if centroid_shift <= tol:
            break
        
        centroids = new_centroids

    return centroids, assignments

# NUMPY BASELINE ---------------------------------------------------------------------------------
def numpy_closest_index(x, y):
    """
    Compute the nearest centroid for each data point using NumPy.
    
    Args:
        x (np.ndarray): Data points of shape (N, D).
        y (np.ndarray): Centroids of shape (M, D).
    
    Returns:
        output (np.ndarray): Indices of nearest centroids for each data point.
    """
    N, D = x.shape
    M, D_y = y.shape
    assert D == D_y, "Dimensions of x and y must match"

    # Compute squared Euclidean distances between all data points and centroids
    distances = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=2)  # Shape (N, M)

    # Find the index of the nearest centroid for each data point
    output = np.argmin(distances, axis=1)  # Shape (N,)

    return output

def kmeans_numpy(N, D, A, K, max_iters=100, tol=1e-4, random_state=None):
    """
    K-Means clustering using NumPy (CPU implementation).
    
    Args:
        N (int): Number of data points.
        D (int): Dimension of each data point.
        A (np.ndarray): Input data of shape (N, D).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence checking.
        random_state (int): Seed for random number generation.
    
    Returns:
        centroids (np.ndarray): Final centroids of shape (K, D).
        assignments (np.ndarray): Indices of closest clusters for each data point.
    """
    # Ensure A is a NumPy array
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=np.float32)

    # Initialize centroids by randomly selecting K data points from A
    rng = np.random.RandomState(seed=random_state)
    indices = rng.choice(N, size=K, replace=False)
    centroids = A[indices].copy()  # Initial centroids
    assignments = np.zeros(N, dtype=np.int32)  # Array for cluster assignments

    # Iterate until convergence or max_iters is reached
    for _ in range(max_iters):
        # Step 1: Assignment Step - Compute distances and assign clusters
        assignments = numpy_closest_index(A, centroids)

        # Step 2: Update Step - Compute new centroids
        new_centroids = np.zeros((K, D), dtype=np.float32)
        counts = np.zeros(K, dtype=np.float32)

        for k in range(K):
            mask = assignments == k  # Points assigned to cluster k
            if np.any(mask):
                new_centroids[k] = np.mean(A[mask], axis=0)  # Mean of points in cluster k
                counts[k] = np.sum(mask)  # Number of points in cluster k

        # Step 3: Check for convergence
        centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        if centroid_shift <= tol:
            break

        # Update centroids for the next iteration
        centroids = new_centroids

    return centroids, assignments

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_kmeans_torch(N, D, A, K):
    measure_time(kmeans_torch, "cpu", N, D, A, 1) # warmup
    cpu_result, cpu_time = measure_time(kmeans_torch, "cpu", N, D, A, K)
    gpu_torch_result, gpu_torch_time = measure_time(kmeans_torch, "cuda", N, D, A, K)

    return N, D, "KMeans Torch CPU vs GPU", cpu_time, gpu_torch_time

def test_kmeans_cupy(N, D, A, K):
    # cpu_result, cpu_time = measure_time(kmeans_numpy, "cpu", N, D, A, K)
    cpu_time, cpu_time = measure_time(kmeans_numpy, "cpu", N, D, A, K)
    gpu_cupy_robin_result, gpu_cupy_robin_time = measure_time(kmeans_cupy, "cuda", N, D, A, K)

    return N, D, "KMeans NumPy vs CuPy", cpu_time, gpu_cupy_robin_time

if __name__ == "__main__":
    np.random.seed(123)
    N, D, A, K = testdata_kmeans("")

    results = []
    results.append(test_kmeans_torch(N, D, A, K))
    results.append(test_kmeans_cupy(N, D, A, K))
    print_table(results)


# Performance
#  N       | D        | Function                       | CPU Time (s)  | GPU Time (s)  | Speedup 
# ---------------------------------------------------------------------------------------------------
# 10000    | 1024     | KMeans Torch CPU vs GPU        | 0.138115      | 0.147309      | 0.94x
# 10000    | 1024     | KMeans NumPy vs CuPy           | 47.247719      | 0.628899      | 75.13x