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

distance_l2 = """
extern "C" __global__
void distance(const float* x, const float* y, float* result, int n) {
    __shared__ float shared_mem[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp = 0;
    while (i < n) {
        float diff = x[i] - y[i];
        temp += diff * diff;
        i += blockDim.x * gridDim.x;
    }
    
    shared_mem[tid] = temp;
    __syncthreads();
    
    // Parallel reduction in shared memory
    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
        // Only one thread in the entire grid should compute the sqrt
        __threadfence();  // Ensure all atomicAdds complete
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *result = sqrtf(*result);  // sqrtf for float
        }
    }
}
"""
l2= cp.RawKernel(distance_l2, "distance")

dot_product_code = """
extern "C" __global__
void dot_product(const float* x, const float* y, float* result, int n) {
    __shared__ float shared_mem[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp = 0;
    while (i < n) {
        temp += x[i] * y[i];
        i += blockDim.x * gridDim.x;
    }
    
    shared_mem[tid] = temp;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}
"""

dot_product= cp.RawKernel(dot_product_code, "dot_product")

def distance_cosine(X, Y):
    pass

def distance_l2(X, Y):
    n = X.shape[0]
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    result = cp.zeros(1, dtype=cp.float32)
    l2(
        (blocks_per_grid,), 
        (threads_per_block,), 
        (X, Y, result, n)
    )
    return result

def distance_dot(X, Y):
    n = X.shape[0]
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    result = cp.zeros(1, dtype=cp.float32)
    dot_product(
        (blocks_per_grid,), 
        (threads_per_block,), 
        (X, Y, result, n)
    )
    return result

def distance_manhattan(X, Y):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

d_l2_batch = """
extern "C" __global__
void batch_distance(const float* x, const float* y, float* output, const int N, const int D) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x;  // Each block handles one vector x[i]
    int offset = i * D;  // Starting index for x[i] in memory
    
    float temp = 0.0f;
    int j = tid;        // Thread-local index for vector components
    
    // Grid-stride loop across vector components
    while (j < D) {
        float diff = x[offset + j] - y[j];
        temp += diff * diff;
        j += blockDim.x;  // Jump by threads-per-block
    }
    
    shared_mem[tid] = temp;
    __syncthreads();
    
    // Parallel reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Write final result for this vector
    if (tid == 0) {
        output[i] = sqrtf(shared_mem[0]);
    }
}
"""
l2_b= cp.RawKernel(d_l2_batch, "batch_distance")

knn_code = r'''
    extern "C" __global__
    void knn_topk_kernel(const float* distances, int* indices, const int K, const int N) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (tid == 0) {
            float* temp_dist = new float[N];  // Copy of the distances
            for (int i = 0; i < N; i++) {
                temp_dist[i] = distances[i];
            }

            for (int k = 0; k < K; k++) {
                float min_dist = 1e10f;
                int min_idx = -1;

                for (int i = 0; i < N; i++) {
                    if (temp_dist[i] < min_dist) {
                        min_dist = temp_dist[i];
                        min_idx = i;
                    }
                }

                if (min_idx != -1) {
                    indices[k] = min_idx;  // Store the index of the nearest neighbor
                    temp_dist[min_idx] = 1e10f;  // Mark it as selected
                }
            }

            delete[] temp_dist;  // Free memory
        }
    }

    '''
knn = cp.RawKernel(knn_code, "knn_topk_kernel")

def our_knn(N, D, A, X, K):
    # Step 1: Calculate the distances using the first kernel
    distances = cp.zeros(N, dtype=cp.float32)
    indices = cp.zeros(K, dtype=cp.int32)  # Store K nearest indices

    # Configure block and grid sizes for the distance calculation kernel
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    threads_per_block = 256  # Can experiment with 128-1024
    blocks_per_grid = N
    shared_mem_size = threads_per_block * cp.dtype(cp.float32).itemsize
    # Run the kernel to calculate the distances
    l2_b(
        (blocks_per_grid,),       # One block per vector
        (threads_per_block,),     # Threads per block
        (A, X, distances, N, D),     # Arguments
        shared_mem=shared_mem_size
    )

    # Step 2: Apply the second kernel to find the K nearest neighbors by distance
    grid_size_topk = (K + block_size - 1) // block_size
    knn((grid_size_topk,), (block_size,), (distances, indices, K, N))

    return indices  # Return the indices of the K closest neighbors
# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

device = "cuda" if torch.cuda.is_available() else "cpu"

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

closest_kernel = cp.RawKernel(l2_closest_kernel, "closest_index")

def cupy_closest_index(x, y):
    N, D = x.shape
    M, D_y = y.shape
    assert D == D_y, "Dimensions of x and y must match"
    
    threads_per_block = 256  # Use a power of two for optimal reduction
    blocks_per_grid = N
    
    output = cp.zeros(N, dtype=cp.int32)
    
    # Calculate shared memory size: floats + ints per thread
    shared_mem_size = (threads_per_block * cp.dtype(cp.float32).itemsize +
                       threads_per_block * cp.dtype(cp.int32).itemsize)
    
    closest_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (x, y, output, N, M, D),
        shared_mem=shared_mem_size
    )
    
    return output

def our_kmeans(N,D,A,K, max_iters=100, tol=1e-4, random_state=None):
    """
    K-Means clustering using CuPy and a custom CUDA kernel for the assignment step.
    
    Args:
        data (cupy.ndarray): Input data of shape (N, D).
        n_clusters (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence checking.
        random_state (int): Seed for random number generation.
    
    Returns:
        centroids (cupy.ndarray): Final centroids of shape (n_clusters, D).
        assignments (cupy.ndarray): Indices of closest clusters for each data point.
    """
    # Ensure data is float32 for CUDA kernel compatibility

    # Initialize centroids by randomly selecting data points
    rng = cp.random.RandomState(seed=random_state)
    indices = rng.choice(N, size=K, replace=False)
    centroids = A[indices].copy()
    assignments = cp.zeros(N, dtype=cp.int32)

    for _ in range(max_iters):
        assignments = cupy_closest_index(A, centroids)

        mask = cp.zeros((N, K), cp.float32)
        mask[cp.arange(N), assignments] = 1.0
        counts = mask.sum(axis=0)
        sums = cp.matmul(mask.T, A)
        new_centroids = sums/cp.expand_dims(counts, 1)
        centroid_shift = cp.linalg.norm(new_centroids - centroids, axis=1).max()
        if centroid_shift <= tol:
            break
        
        centroids = new_centroids

    return centroids, assignments

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def cupy_l2_b(x,y):
    N, D = x.shape
    threads_per_block = 256  # Can experiment with 128-1024
    blocks_per_grid = N
    
    # Output array (one distance per vector)
    output = cp.zeros(N, dtype=cp.float32)
    
    # Calculate required shared memory (1 float per thread)
    shared_mem_size = threads_per_block * cp.dtype(cp.float32).itemsize
    
    # Launch kernel
    l2_b(
        (blocks_per_grid,),       # One block per vector
        (threads_per_block,),     # Threads per block
        (x, y, output, N, D),     # Arguments
        shared_mem=shared_mem_size
    )
    
    return output

def our_ann(N, D, A, X, K, assignments=None, means=None):
    if assignments is None or means is None:
        means, assignments = our_kmeans(N, D, A, 5)
    
    # Compute distances from means to X (shape: n_means x N)
    distances = cupy_l2_b(X, means)
    
    # Determine the nearest 2 means for each data point
    nearest_k = 2
    indices = cp.empty((N, nearest_k), dtype=cp.int32)
    block_size = 256
    grid_size_topk = (N + block_size - 1) // block_size
    knn((grid_size_topk,), (block_size,), (distances, indices, nearest_k, N))
    
    # Create mask where assignment is one of the top 2 nearest means
    mask = indices[assignments]
    original_indices = cp.nonzero(mask)[0]
    filtered = A[original_indices]
    
    # Compute approximate KNN on the filtered data
    approx_knn = our_knn(filtered.shape[0], D, filtered, X, K)
    
    # Map indices back to the original dataset
    return original_indices[approx_knn]

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
