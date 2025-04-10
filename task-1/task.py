import torch
import cupy as cp
import numpy as np
import time
import json
import sys
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------------------
def warm_up_gpu():
    # Run a simple GPU operation to initialise CUDA and warm up the GPU
    warm_up_data = cp.zeros(1, dtype=cp.float32)  # Allocate a small array on the GPU
    warm_up_data += 1  # Perform a simple operation
    cp.cuda.Stream.null.synchronize()  # Synchronise to ensure the operation completes

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

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def cosine_distance_cpu(X, Y):
    """
    Computes the pairwise cosine distance between rows of X and Y using NumPy on CPU.
    
    Args:
        X (np.ndarray): Array of shape (m, d)
        Y (np.ndarray): Array of shape (n, d)
    
    Returns:
        np.ndarray: Cosine distance matrix of shape (m, n)
    """
    # Start timer
    start_time = time.time()
    
    # Normalise X and Y
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Avoid division by zero
    X_norm[X_norm == 0] = 1
    Y_norm[Y_norm == 0] = 1
    
    # Compute cosine similarity
    cosine_sim = np.dot(X/X_norm, (Y/Y_norm).T)
    
    # Compute cosine distance
    cosine_dist = 1 - cosine_sim
    
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")
    
    return cosine_dist

def cosine_distance_cupy(X, Y):
    """
    CuPy implementation.
    Computes the pairwise cosine distance between rows of X and Y.
    
    Args:
        X (cp.ndarray): Array of shape (m, d)
        Y (cp.ndarray): Array of shape (n, d)
    
    Returns:
        cp.ndarray: Cosine distance matrix of shape (m, n)
    """
    # Start timer
    start_time = time.time()

    # Normalise X and Y along the feature dimension (axis=1)
    X_norm = cp.linalg.norm(X, axis=1, keepdims=True)  # Shape: (m, 1)
    Y_norm = cp.linalg.norm(Y, axis=1, keepdims=True)  # Shape: (n, 1)
    
    # Avoid division by zero
    X_norm[X_norm == 0] = 1
    Y_norm[Y_norm == 0] = 1
    
    # Normalise the rows of X and Y
    X_normalised = X / X_norm  # Shape: (m, d)
    Y_normalised = Y / Y_norm  # Shape: (n, d)
    
    # Compute cosine similarity
    cosine_similarity = cp.dot(X_normalised, Y_normalised.T)  # Shape: (m, n)
    
    # Compute cosine distance
    cosine_distance = 1 - cosine_similarity  # Shape: (m, n)

    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")
    
    return cosine_distance

cosine_kernel = r'''
extern "C" __global__
void cosine_distance_kernel(
    const float* X, const float* Y, float* output, 
    int m, int n, int d) {
    
    // Each block handles one row of X (i) and one row of Y (j)
    int i = blockIdx.x;
    int j = blockIdx.y;
    
    extern __shared__ float shared_mem[];
    float* dot_prod = shared_mem;
    float* x_norm = shared_mem + 1;
    float* y_norm = shared_mem + 2;
    
    if (threadIdx.x == 0) {
        *dot_prod = 0.0f;
        *x_norm = 0.0f;
        *y_norm = 0.0f;
    }
    __syncthreads();
    
    // Each thread computes partial sums
    float my_dot = 0.0f;
    float my_x_norm = 0.0f;
    float my_y_norm = 0.0f;
    
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        float x_val = X[i * d + k];
        float y_val = Y[j * d + k];
        my_dot += x_val * y_val;
        my_x_norm += x_val * x_val;
        my_y_norm += y_val * y_val;
    }
    
    // Parallel reduction within block
    atomicAdd(dot_prod, my_dot);
    atomicAdd(x_norm, my_x_norm);
    atomicAdd(y_norm, my_y_norm);
    __syncthreads();
    
    // Compute final cosine distance
    if (threadIdx.x == 0) {
        float norm_product = sqrtf(*x_norm * *y_norm);
        float cosine_sim = norm_product > 0 ? (*dot_prod / norm_product) : 0.0f;
        output[i * n + j] = 1.0f - cosine_sim;
    }
}
'''

cosine_kernel = cp.RawKernel(cosine_kernel, 'cosine_distance_kernel')

def cosine_distance_kernel(X, Y):
    """
    Computes pairwise cosine distance using a custom CUDA kernel.
    
    Args:
        X (cp.ndarray or np.ndarray): Shape (m, d)
        Y (cp.ndarray or np.ndarray): Shape (n, d)
    
    Returns:
        cp.ndarray: Cosine distance matrix (m, n)
    """
    # Start timer
    start_time = time.time()
    
    X_gpu = cp.asarray(X, dtype=cp.float32)
    Y_gpu = cp.asarray(Y, dtype=cp.float32)
    
    m, d = X_gpu.shape
    n = Y_gpu.shape[0]
    
    output = cp.zeros((m, n), dtype=cp.float32)
    
    # Launch configuration
    threads_per_block = 256
    blocks_per_grid = (m, n)
    
    shared_mem_size = 3 * 4  # 3 floats (dot_prod, x_norm, y_norm)
    
    cosine_kernel(
        blocks_per_grid,
        (threads_per_block,),
        (X_gpu, Y_gpu, output, m, n, d),
        shared_mem=shared_mem_size
    )
    
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")
    
    return output

# EXTRA ------------------------------------------------------------------------------------------
def distance_cosine_torch(X, Y):
    """
    Computes the pairwise cosine distance between rows of X and Y.
    
    Args:
        X (torch.Tensor): Tensor of shape (m, d)
        Y (torch.Tensor): Tensor of shape (n, d)
    
    Returns:
        torch.Tensor: Cosine distance matrix of shape (m, n)
    """
    X_norm = torch.nn.functional.normalise(X, p=2, dim=1)  # Normalize along features (d)
    Y_norm = torch.nn.functional.normalise(Y, p=2, dim=1)
    
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

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

# TORCH ------------------------------------------------------------------------------------------
# TODO:implement batching?
def kmeans_torch(N, D, A, K, max_iters=100, tol = 1e-4, device="cuda", distance_metric="l2"):
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

    # if isinstance(A, np.ndarray):
    #     A = torch.tensor(A, dtype=torch.float32, device=device)
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    elif not isinstance(A, torch.Tensor):
        raise TypeError(f"Input must be either numpy array or torch tensor, got {type(A)}.")

    # Step 1: Initialise K random points from A as initial centroids
    indices = torch.randperm(N)[:K]
    centroids = A[indices].clone()

    for _ in range(max_iters):
        old_centroids = centroids.clone()

        # Step 2a: Assignment Step - Compute distances and assign clusters
        if distance_metric == "l2":
            distances = distance_l2(A, centroids)  # A: (N, D), centroids: (K, D)
        elif distance_metric == "cosine":
            distances = distance_cosine_torch(A, centroids)
        else:
            raise ValueError("Invalid distance metric. Use 'l2' or 'cosine'.")
        
        cluster_assignments = distances.argmin(dim=1)

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

# NUMPY BASELINE ---------------------------------------------------------------------------------
def numpy_closest_index(x, y, distance_metric="l2"):
    """
    Compute the nearest centroid for each data point using NumPy.
    
    Args:
        x (np.ndarray): Data points of shape (N, D).
        y (np.ndarray): Centroids of shape (M, D).
        distance_metric (str): Distance metric to use ("l2" or "cosine").
    
    Returns:
        output (np.ndarray): Indices of nearest centroids for each data point.
    """
    N, D = x.shape
    M, D_y = y.shape
    assert D == D_y, "Dimensions of x and y must match"

    if distance_metric == "l2":
        # Compute squared L2 distances between all data points and centroids
        distances = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=2)  # Shape (N, M)
    elif distance_metric == "cosine":
        # Compute cosine distances (1 - cosine similarity)
        x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
        cosine_sim = np.dot(x_norm, y_norm.T)  # Shape (N, M)
        distances = 1 - cosine_sim  # Convert similarity to distance
    else:
        raise ValueError("Invalid distance metric. Use 'l2' or 'cosine'.")

    # Find the index of the nearest centroid for each data point
    output = np.argmin(distances, axis=1)  # Shape (N,)

    return output

def kmeans_numpy(N, D, A, K, max_iters=100, tol=1e-4, random_state=None, distance_metric="l2"):
    """
    K-Means clustering over CPU using NumPy with batched processing for large datasets.
    
    Args:
        N (int): Number of data points.
        D (int): Dimension of each data point.
        A (np.ndarray): Input data of shape (N, D).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence checking.
        random_state (int): Seed for random number generation.
        distance_metric (str): Distance metric to use ("l2" or "cosine").
    
    Returns:
        centroids (np.ndarray): Final centroids of shape (K, D).
        assignments (np.ndarray): Indices of closest clusters for each data point.

    Memory considerations:
        The function assumes that the dataset A is loaded entirely in memory, e.g using np.memap is A is too big.
    """

    # Start timer
    start_time = time.time()

    # Ensure A is a NumPy array
    # if not isinstance(A, np.ndarray):
    #     A = np.array(A, dtype=np.float32)

    memory_limit = 2**28 # assume 2GB of RAM
    batch_size = min(N, memory_limit // (D * A.itemsize))

    # Initialise centroids by randomly selecting K data points from A
    rng = np.random.RandomState(seed=random_state)
    indices = rng.choice(N, size=K, replace=False)
    centroids = np.array(A[indices])  # Initial centroids (copy??)
    assignments = np.zeros(N, dtype=np.int32)  # Array for cluster assignments

    # Iterate until convergence or max_iters is reached
    for _ in range(max_iters):

        # Assign clusters, processing A in batches
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            # Extract current batch (ensure it's an np array)
            batch_A = np.array(A[batch_start:batch_end], dtype=np.float32)

            # Find closest indices for the current batch given the chosen distance metric
            assignments[batch_start:batch_end] = numpy_closest_index(batch_A, centroids, distance_metric = distance_metric)

        # Update the centroids in batches
        new_centroids = np.zeros((K, D), dtype=np.float32)
        counts = np.zeros(K, dtype=np.float32)

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_A = np.array(A[batch_start:batch_end], dtype=np.float32)
            batch_assignments = assignments[batch_start:batch_end]

            for k in range(K):
                mask = batch_assignments == k  # points assigned to cluster k
                if np.any(mask):
                    new_centroids[k] += np.sum(batch_A[mask], axis=0)  # sum of points in cluster k (don't average yet)
                    counts[k] = np.sum(mask)  # store number of points in cluster k

        # Normalise by counts to get means
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                # Reinitialise empty clusters
                new_centroids[k] = A[rng.choice(N, 1)].squeeze(0)

        # Check for convergence
        centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        if centroid_shift <= tol:
            break

        # Update centroids for the next iteration
        centroids = new_centroids
    
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"Total Time: {execution_time:.6f} seconds")

    return centroids, assignments, execution_time

# CUPY DESIGN 1 ----------------------------------------------------------------------------------
def kmeans_cupy_1(N, D, A, K, max_iters=100, tol=1e-4, distance_metric="l2"):
    """
    K-Means clustering using CuPy with support for multiple distance metrics, with batched
    processing for large datasets.
    
    Args:
        N (int): Number of data points.
        D (int): Dimension of each data point.
        A (np.ndarray or cp.ndarray): Input data of shape (N, D).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence checking.
        distance_metric (str): Distance metric to use ("l2" or "cosine").
    
    Returns:
        cluster_assignments (np.ndarray): Indices of closest clusters for each data point.
    
    Memory considerations:
        The function assumes that the dataset A is loaded entirely in memory, e.g using np.memap is A is too big.
    """

    # Start timer
    start_time = time.time()

    memory_limit = 2**28 # assumer 2GB of RAM
    batch_size = min(N, memory_limit // (D * A.itemsize))

    # Convert input data to CuPy array
    # A = cp.asarray(A, dtype=cp.float32)
    
    # Initialise centroids
    indices = cp.random.choice(N, K, replace=False)
    centroids = cp.asarray(A[indices.get()]) # copy??
    
    for _ in range(max_iters):
        cluster_assignments = cp.zeros(N, dtype=cp.int32)

        # Assign clusters, processing A in batches
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            # Move batch to GPU
            batch_A = cp.asarray(A[batch_start:batch_end])

            # Compute distances based on selected metric
            if distance_metric == "l2":
                # Squared L2 distance (more efficient, same clustering result)
                distances = cp.sum((batch_A[:, cp.newaxis] - centroids)**2, axis=2)
            elif distance_metric == "cosine":
                # Cosine distance (1 - cosine similarity)
                norm_A = cp.linalg.norm(batch_A, axis=1, keepdims=True)
                norm_C = cp.linalg.norm(centroids, axis=1)
                distances = 1 - cp.dot(batch_A, centroids.T) / (norm_A * norm_C)
            else:
                raise ValueError("Invalid distance metric. Use 'l2' or 'cosine'.")
            
            # Assign clusters
            cluster_assignments[batch_start:batch_end] = cp.argmin(distances, axis=1)
        
        # Update centroids in batches
        new_centroids = cp.zeros((K, D), dtype=cp.float32)
        counts = cp.zeros(K, dtype=cp.float32)
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_A = cp.asarray(A[batch_start:batch_end])
            batch_assignments = cluster_assignments[batch_start:batch_end]
            
            for k in range(K):
                mask = batch_assignments == k
                if cp.any(mask):
                    new_centroids[k] += cp.sum(batch_A[mask], axis=0) # don't average yet
                    counts[k] += cp.sum(mask) # store counts
        
        # Normalise by counts and handle empty clusters
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                # Reinitialise empty cluster with random point
                new_centroids[k] = A[cp.random.randint(0, N, 1)].squeeze(0)
        
        # Check convergence
        centroid_shift = cp.linalg.norm(new_centroids - centroids, axis=1).max()
        if centroid_shift <= tol:
            break
            
        centroids = new_centroids
    
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"Total Time: {execution_time:.6f} seconds")
    
    return cluster_assignments.get(), execution_time

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

        // Normalise dot product to get cosine similarity
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
    K-Means clustering using CuPy with custom CUDA kernels for the assignment step,
    and batched processing for large datasets.
    
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

    Memory considerations:
        The function assumes that the dataset A is loaded entirely in memory, e.g using np.memap is A is too big.
    """

    # Start timer
    start_time = time.time()

    memory_limit = 2**28
    batch_size = min(N, memory_limit // (D * A.itemsize))

    # Convert to cupy
    # if isinstance(A, np.ndarray):
        # A = cp.asarray(A, dtype=cp.float32) # ASARRAY INSTEAD OF ARRAY TO AVOID COPY (unnecessary data transfer)

    # Initialise centroids by randomly selecting K data points from A
    cp.random.seed(random_state)
    indices = cp.random.choice(N, size=K, replace=False)
    centroids = cp.asarray(A[indices.get()]) # COPY??
    assignments = cp.zeros(N, dtype=cp.int32)

    # Create multiple streams for concurrent execution
    num_streams = 2
    streams = [cp.cuda.Stream() for _ in range(num_streams)]
    events = [cp.cuda.Event() for _ in range(num_streams)]

    # Iterate kmeans until convergence or max_iters is reached
    for _ in range(max_iters):
        # Assign points to the nearest centroid, processig A in batches
        for batch_idx, batch_start in enumerate(range(0, N, batch_size)):
            stream = streams[batch_idx % num_streams] # cycle through streams
            batch_end = min(batch_start + batch_size, N)

            # Move batch A to GPU
            batch_A = cp.asarray(A[batch_start:batch_end])
            
            with stream:
                assignments[batch_start:batch_end] = cupy_closest_index(
                    batch_A, centroids, distance_metric=distance_metric, stream=stream
                )
            events[batch_idx % num_streams].record(stream)

        # Wait for all batches to complete
        for event in events:
            event.synchronize()

        # Update centroids in batches
        sums = cp.zeros((K, D), dtype=cp.float32)
        counts = cp.zeros(K, dtype=cp.float32)
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            # Move batch A to GPU
            batch_A = cp.asarray(A[batch_start:batch_end])
            batch_assignments = assignments[batch_start:batch_end]
            
            # Accumulate sums and counts
            for k in range(K):
                mask = batch_assignments == k
                if cp.any(mask):
                    sums[k] += cp.sum(batch_A[mask], axis=0)
                    counts[k] += cp.sum(mask)

        # Compute new centroids
        new_centroids = cp.zeros((K, D), dtype=cp.float32)
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] = sums[k] / counts[k]
            else:
                # Reinitialise empty cluster
                rand_idx = np.random.randint(0, N, 1)  # CPU random index
                new_centroids[k] = cp.asarray(A[rand_idx], dtype=cp.float32).squeeze(0)
                # new_centroids[k] = A[cp.random.randint(0, N, 1)].squeeze(0)

                # indices = cp.random.choice(N, size=K, replace=False)
                # centroids = cp.asarray(A[indices.get()])


        # Check for convergence
        centroid_shift = cp.linalg.norm(new_centroids - centroids, axis=1).max() # max shift in centroids between iterations
        if centroid_shift <= tol:
            break
        
        centroids = new_centroids

    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"Total Time: {execution_time:.6f} seconds")
    
    return centroids, assignments, execution_time

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

def test_cosine(D):
    """
    Tests the Cosine distance calculation between two random vectors using different CPU, Cupy kernels and custom CUDA kernels.
    Args:
        D (list of int): A list of integers where each integer represents the dimensionality of the random vectors X and Y.
    Returns:
        None: This function prints the results of the distance calculations and the time spent, but does not return any values.
    """
    for d in D:
        print("D: ", d)
        for i in range(0,3):
            np.random.seed(42)
            X = np.random.randn(d).astype(np.float32)
            Y = np.random.randn(d).astype(np.float32)
            print(cosine_distance_cpu(X,Y))
            print(cosine_distance_cupy(X,Y))
            print(cosine_distance_kernel(X,Y))

def test_kmeans(random = False):

    """
    Tests three KMeans implementations using L2 and Cosine distance metrics, with batching to accommodate large data.
    The implementations are: CPU, CuPy with native functions, and CuPy with custom kernels.
    
    Args:
        random (bool, optional): If True, generates random test data; if False, loads test data from a JSON file. Defaults to False. When set to False,
                the function will look for a file called "test_data.json" in the "Data" folder. If the file is not found, it will raise an error. This JSON ç
                must have the following format:
                {
                    "n": Number of vector in the datastet,
                    "d": Dimension of each vector,
                    "a_file": path to the dataset file, which must be a .npy file,
                    "x_file": path to the query vector file, which must be a .npy file,
                    "k": number of elements to retrieve
                }
    Returns:
        None: This function prints the results of the KMeans and the time spent by each implementations, but does not return any values.
    """

    if random:
        N, D, A, K = testdata_kmeans("")
    else:
        N, D, A, K = testdata_kmeans("test_data.json")
    
    print("N: ", N)
    print("D: ", D)

    # print("\nL2")
    # for i in range(0, 1):
    #     print("Iteration: ", i)
    #     print("CPU (Numpy)")
    #     # kmeans_numpy(N, D, A, K, distance_metric="l2")
    #     print("GPU (Native CuPy Functions)")
    #     kmeans_cupy_1(N, D, A, K, distance_metric="l2")
    #     print("GPU (Custom Cuda Kernels)\n")
    #     kmeans_cupy_2(N, D, A, K, distance_metric="l2")

    # print("\nCOSINE")
    # for i in range(0, 1):
    #     print("Iteration: ", i)
    #     print("CPU (Numpy)")
    #     # kmeans_numpy(N, D, A, K, distance_metric="cosine")
    #     print("GPU (Native CuPy Functions)")
    #     kmeans_cupy_1(N, D, A, K, distance_metric="cosine")
    #     print("GPU (Custom Cuda Kernels)\n")
    #     kmeans_cupy_2(N, D, A, K, distance_metric="cosine")

    print_results(N, D, A, K, iterations=2)

def print_results(N, D, A, K, iterations=1):
    """
    Tests and compares different K-Means implementations (CPU and GPU versions) with both L2 and
    Cosine distance metrics, printing execution times and speedups in a table format.
    
    Args:
        random (bool): If True, generates random test data. If False, loads from test_data.json.
    """
    # Initialise results storage
    results = {
        "CPU (Numpy)": {"L2": [], "Cosine": []},
        "GPU (Native CuPy)": {"L2": [], "Cosine": []},
        "GPU (Custom Kernels)": {"L2": [], "Cosine": []}
    }
    
    # Test L2 distance
    print("\nRunning L2 distance tests ({iterations} iterations)...")
    for i in range(iterations):  # Single iteration for demonstration
        # GPU custom kernels test
        centroids_cupy_2, assignments_cupy_2, execution_time_cupy_2 = kmeans_cupy_2(N, D, A, K, distance_metric="l2")
        results["GPU (Custom Kernels)"]["L2"].append(execution_time_cupy_2)
        
        # GPU native test
        cluster_assignments_cupy_1, execution_time_cupy_1 = kmeans_cupy_1(N, D, A, K, distance_metric="l2")
        results["GPU (Native CuPy)"]["L2"].append(execution_time_cupy_1)

        # CPU test
        centroids_numpy, assignments_numpy, execution_time_numpy = kmeans_numpy(N, D, A, K, distance_metric="l2")
        results["CPU (Numpy)"]["L2"].append(execution_time_numpy)
    
    # Test Cosine distance
    print("\nRunning Cosine distance tests...")
    for i in range(iterations):  # Single iteration for demonstration
        # GPU custom kernels test
        centroids_cupy_2, assignments_cupy_2, execution_time_cupy_2 = kmeans_cupy_2(N, D, A, K, distance_metric="cosine")
        results["GPU (Custom Kernels)"]["Cosine"].append(execution_time_cupy_2)

        # GPU native test
        cluster_assignments_cupy_1, execution_time_cupy_1 = kmeans_cupy_1(N, D, A, K, distance_metric="cosine")
        results["GPU (Native CuPy)"]["Cosine"].append(execution_time_cupy_1)

        # CPU test
        centroids_numpy, assignments_numpy, execution_time_numpy = kmeans_numpy(N, D, A, K, distance_metric="cosine")
        results["CPU (Numpy)"]["Cosine"].append(execution_time_numpy)
    
    # Calculate averages and speedups
    cpu_l2_avg = sum(results["CPU (Numpy)"]["L2"])/iterations if iterations > 0 else 0
    cpu_cosine_avg = sum(results["CPU (Numpy)"]["Cosine"])/iterations if iterations > 0 else 0
    
    # Prepare formatted data for printing
    table_data = []
    for impl in results:
        l2_avg = sum(results[impl]["L2"])/iterations if iterations > 0 else 0
        cosine_avg = sum(results[impl]["Cosine"])/iterations if iterations > 0 else 0
        
        l2_speedup = cpu_l2_avg/l2_avg if impl != "CPU (Numpy)" and cpu_l2_avg > 0 else 0
        cosine_speedup = cpu_cosine_avg/cosine_avg if impl != "CPU (Numpy)" and cpu_cosine_avg > 0 else 0
        
        table_data.append({
            "Implementation": impl,
            "L2": f"{l2_avg:.4f}s",
            "L2 Speedup": f"{l2_speedup:.1f}x" if l2_speedup > 0 else "N/A",
            "Cosine": f"{cosine_avg:.4f}s",
            "Cosine Speedup": f"{cosine_speedup:.1f}x" if cosine_speedup > 0 else "N/A"
        })
    
    # Print results table
    print("\n" + "="*85)
    print(f"K-Means Performance Comparison (N={N}, D={D}, K={K}, iterations={iterations})")
    print("="*85)
    
    # Table header
    print("\n| {:<20} | {:^12} | {:^12} | {:^12} | {:^12} |".format(
        "Implementation", "L2", "L2 Speedup", "Cosine", "Cosine Speedup"))
    print("|" + "-"*22 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|")
    
    # Table rows
    for row in table_data:
        print("| {:<20} | {:>12} | {:>12} | {:>12} | {:>12} |".format(
            row["Implementation"],
            row["L2"],
            row["L2 Speedup"],
            row["Cosine"],
            row["Cosine Speedup"]
        ))
    
    print("="*85 + "\n")

if __name__ == "__main__":
    batch_start_time = time.time()

    test_kmeans(random=False)
    
    batch_end_time = time.time()
    execution_time = batch_end_time - batch_start_time

    print(f"Batch took {execution_time}s to execute.")
    # test_cosine(D = [2, 1024, 2**15,2**20])