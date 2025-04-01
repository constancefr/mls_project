import os
import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
from test import testdata_kmeans_dynam, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------------------
def warm_up_gpu():
    # Run a simple GPU operation to initialize CUDA and warm up the GPU
    warm_up_data = cp.zeros(1, dtype=cp.float32)  # Allocate a small array on the GPU
    warm_up_data += 1  # Perform a simple operation
    cp.cuda.Stream.null.synchronize()  # Synchronize to ensure the operation completes

def measure_time(func, device, *args, **kwargs):
    if device == "cuda":
        warm_up_gpu()
        if "torch" in func.__module__:
            torch.cuda.synchronize()
        elif "cupy" in func.__module__:
            cp.cuda.Stream.null.synchronize()

    start_time = time.time()

    result = func(*args, **kwargs) # run function

    if device == "cuda":
        if "torch" in func.__module__:
            torch.cuda.synchronize()
        elif "cupy" in func.__module__:
            cp.cuda.Stream.null.synchronize()

    end_time = time.time()
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
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine_torch(X, Y):
    """
    Computes the pairwise cosine distance between rows of X and Y.
    
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

def distance_cosine(X, Y):
    """
    CuPy implementation.
    Computes the pairwise cosine distance between rows of X and Y.
    
    Args:
        X (cp.ndarray): Array of shape (m, d)
        Y (cp.ndarray): Array of shape (n, d)
    
    Returns:
        cp.ndarray: Cosine distance matrix of shape (m, n)
    """
    # Normalize X and Y along the feature dimension (axis=1)
    X_norm = cp.linalg.norm(X, axis=1, keepdims=True)  # Shape: (m, 1)
    Y_norm = cp.linalg.norm(Y, axis=1, keepdims=True)  # Shape: (n, 1)
    
    # Avoid division by zero
    X_norm[X_norm == 0] = 1
    Y_norm[Y_norm == 0] = 1
    
    # Normalize the rows of X and Y
    X_normalized = X / X_norm  # Shape: (m, d)
    Y_normalized = Y / Y_norm  # Shape: (n, d)
    
    # Compute cosine similarity
    cosine_similarity = cp.dot(X_normalized, Y_normalized.T)  # Shape: (m, n)
    
    # Compute cosine distance
    cosine_distance = 1 - cosine_similarity  # Shape: (m, n)
    
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

# CUPY DESIGN 1 ----------------------------------------------------------------------------------
def kmeans_cupy_1(N, D, A, K, max_iters=100, tol=1e-4):
     # Convert input data to CuPy arrays
     A = cp.asarray(A, dtype=cp.float32)
     
     # Step 1: Initialize K random centroids
     indices = cp.random.choice(N, K, replace=False)
     centroids = A[indices].copy()
     
     for _ in range(max_iters):
         # Step 2a: Compute distances and assign clusters
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

# CUPY DESIGN 2 ----------------------------------------------------------------------------------
l2_closest_kernel = """
extern "C" __global__
void closest_index_l2(const float* x, float* y, int* output, const int N, const int M, const int D) {
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

cosine_closest_kernel = """
extern "C" __global__
void closest_index_cosine(const float* x, float* y, int* output, const int N, const int M, const int D) {
    extern __shared__ char shared_mem[];
    float* shared_dist = (float*)shared_mem;
    int* shared_idx = (int*)(shared_dist + blockDim.x);

    int tid = threadIdx.x;
    int i = blockIdx.x;  // Each block handles one vector x[i]
    int num_threads = blockDim.x;

    float max_sim = -99999999.0;  // Cosine similarity ranges from -1 to 1
    int max_index = -1;

    // Grid-stride loop over M y vectors
    for (int j = tid; j < M; j += num_threads) {
        float dot_product = 0.0f;
        float norm_x = 0.0f;
        float norm_y = 0.0f;

        // Compute dot product and norms
        for (int k = 0; k < D; ++k) {
            dot_product += x[i * D + k] * y[j * D + k];
            norm_x += x[i * D + k] * x[i * D + k];
            norm_y += y[j * D + k] * y[j * D + k];
        }

        // Normalize dot product to get cosine similarity
        float cosine_sim = dot_product / (sqrtf(norm_x) * sqrtf(norm_y));

        // Convert similarity to distance (1 - similarity)
        float cosine_dist = 1.0f - cosine_sim;

        // Find the minimum distance
        if (cosine_dist < max_sim) {
            max_sim = cosine_dist;
            max_index = j;
        }
    }

    // Store local min into shared memory
    shared_dist[tid] = max_sim;
    shared_idx[tid] = max_index;
    __syncthreads();

    // Parallel reduction to find the global min
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

l2_kernel = cp.RawKernel(l2_closest_kernel, "closest_index_l2")
cosine_kernel = cp.RawKernel(cosine_closest_kernel, "closest_index_cosine")

def cupy_closest_index(x, y, distance_metric="l2", stream=None):
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
    
    # choose kernel
    if distance_metric == "l2":
        kernel = l2_kernel
    elif distance_metric == "cosine":
        kernel = cosine_kernel
    else:
        raise ValueError("Unsupported distance metric. Use 'l2' or 'cosine'.")
    
    # Launch the kernel in the stream
    kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (x, y, output, N, M, D),
        shared_mem=shared_mem_size,
        stream=stream
    )
    
    if stream is not None:
        stream.synchronize()
        
    return output

def kmeans_cupy_2(N,D,A,K, max_iters=100, tol=1e-4, random_state=None, batch_size=1000, distance_metric="l2"):
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
        distance_metric (str): Distance metric to use ("l2" or "cosine").
    
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
        for i in range(0, N, batch_size): # process data in batches using multiple streams
            stream = streams[i // batch_size % num_streams]  # Cycle through streams
            batch = A[i:i + batch_size]
            assignments[i:i + batch_size] = cupy_closest_index(batch, centroids, distance_metric=distance_metric, stream=stream)

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

def test_kmeans_torch(N, D, A, K):
    # N, D, A, K = testdata_kmeans("")
    measure_time(kmeans_torch, "cpu", N, D, A, 1) # warmup
    cpu_result, cpu_time = measure_time(kmeans_torch, "cpu", N, D, A, K)
    gpu_torch_result, gpu_torch_time = measure_time(kmeans_torch, "cuda", N, D, A, K)

    return N, D, "KMeans Torch CPU vs GPU", cpu_time, gpu_torch_time

def test_kmeans_cupy_1(N, D, A, K):
    # N, D, A, K = testdata_kmeans("")
    cpu_result, cpu_time = measure_time(kmeans_numpy, "cpu", N, D, A, K)
    gpu_cupy_result, gpu_cupy_time = measure_time(kmeans_cupy_1, "cuda", N, D, A, K)

    return N, D, "KMeans CuPy 1", cpu_time, gpu_cupy_time

def test_kmeans_cupy_2(N, D, A, K):
    # N, D, A, K = testdata_kmeans("")
    cpu_result, cpu_time = measure_time(kmeans_numpy, "cpu", N, D, A, K)
    gpu_cupy_result, gpu_cupy_time = measure_time(kmeans_cupy_2, "cuda", N, D, A, K)

    return N, D, "KMeans CuPy 2", cpu_time, gpu_cupy_time

def test_kmeans_l2_cosine(N, D, A, K):
    # Compares performance of kmeans_cupy_1 with l2 vs. cosine distance kernels
    l2_cupy_result, l2_cupy_time = measure_time(kmeans_cupy_2, "cuda", N, D, A, K, distance_metric="l2")
    cosine_cupy_result, cosine_cupy_time = measure_time(kmeans_cupy_2, "cuda", N, D, A, K, distance_metric="cosine")

    return N, D, "KMeans l2 vs. cosine", l2_cupy_time, cosine_cupy_time

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

def create_or_load_memmap(filename, shape, dtype):
    """
    Create or load a memory-mapped file. If the file already exists, load it in read-only mode.
    If it doesn't exist, create it and fill it with random data.

    Args:
        filename (str): Path to the memory-mapped file.
        shape (tuple): Shape of the array (N, D).
        dtype (np.dtype): Data type of the array.

    Returns:
        np.memmap: The memory-mapped array.
    """
    if os.path.exists(filename):
        # Load the existing file in read-only mode
        print(f"Loading existing memory-mapped file: {filename}")
        return np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    else:
        # Create a new memory-mapped file and fill it with random data
        print(f"Creating new memory-mapped file: {filename}")
        large_A = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
        large_A[:] = np.random.rand(*shape).astype(dtype)
        large_A.flush()  # Ensure data is written to disk
        return large_A

if __name__ == "__main__":
    np.random.seed(123)

    # Test cases for smaller datasets
    test_cases = [
        [1000, 2, 10],
        [1000, 1024, 10],
        [4000000, 2, 10],
        [1000, 65536, 10]
    ]

    results = []
    for i in range(len(test_cases)):
        N = test_cases[i][0]
        D = test_cases[i][1]
        A = testdata_kmeans_dynam(N, D)
        K = test_cases[i][2]
        
        results.append(test_kmeans_torch(N, D, A, K))
        results.append(test_kmeans_cupy_1(N, D, A, K))
        results.append(test_kmeans_cupy_2(N, D, A, K))
        results.append(test_kmeans_l2_cosine(N, D, A, K))
        # results.append(test_knn())
        # results.append(test_ann())

    # Handle the large dataset using a memory-mapped file
    # filename = 'large_data.dat'
    # shape = (4000000, 65536)
    # dtype = np.float32

    # try:
    #     large_A = create_or_load_memmap(filename, shape, dtype)
    #     N_large, D_large = large_A.shape
    #     K = 10

    #     results.append(test_kmeans_torch(N_large, D_large, large_A, K))
    #     results.append(test_kmeans_cupy_1(N_large, D_large, large_A, K))
    #     results.append(test_kmeans_cupy_2(N_large, D_large, large_A, K))
    #     # results.append(test_knn())
    #     # results.append(test_ann())
    # except Exception as e:
    #     print(f"Error handling memory-mapped file: {e}")

    # Print the results
    print_table(results)


# KMeans performance
#  N       | D        | Function                       | CPU Time (s)  | GPU Time (s)  | Speedup 
# ---------------------------------------------------------------------------------------------------
# 10000    | 1024     | KMeans Torch CPU vs GPU        | 0.138115      | 0.147309      | 0.94x
# 10000    | 1024     | KMeans NumPy vs CuPy           | 47.247719      | 0.628899      | 75.13x

# TODO: 
#   run tests with bigger data
#   implement cupy cosine distance and test