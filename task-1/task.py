import torch
import cupy as cp
import numpy as np
import time
import json
import sys
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# L2 Distance
# ------------------------------------------------------------------------------------------------

def l2_distance_cpu(X, Y):
    """
    Computes the Euclidean (L2) distance between two vectors using NumPy on the CPU.

    Parameters
    ----------
    X : np.ndarray
        Input vector of shape (D,), must reside in CPU memory.
    Y : np.ndarray
        Input vector of shape (D,), must reside in CPU memory.

    Returns
    -------
    dist : float
        Scalar value representing the L2 distance between X and Y.
        Returned on CPU as Python float.
    """
    start_time = time.time()

    # Compute L2 distance: sqrt(sum((X - Y)^2))
    dist = np.sqrt(np.sum((X - Y) ** 2))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return dist

def l2_distance_cupy(X, Y):
    """
    Computes the Euclidean (L2) distance between two vectors using CuPy on the GPU.

    Parameters
    ----------
    X : np.ndarray or cp.ndarray
        Input vector of shape (D,), can be NumPy or CuPy array.
        Will be transferred to GPU if not already on GPU.
    Y : np.ndarray or cp.ndarray
        Input vector of shape (D,), same conditions as X.

    Returns
    -------
    dist : float
        Scalar L2 distance as a NumPy float.
        Computed on GPU, then transferred back to CPU.

    """
    # Start timer to measure execution time
    start_time = time.time()

    # Move the input arrays X and Y to the GPU memory (Cupy arrays)
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

    # Calculate the L2 distance in parallel on the GPU
    dist = cp.sqrt(cp.sum((X_gpu - Y_gpu) ** 2))

    # End timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")
    
    # Return the result back to the CPU as a numpy array
    return cp.asnumpy(dist)

l2_kernel = '''
extern "C" __global__
void calculate_diff_kernel(const float* X, const float* Y, float* diff, int D) {

    // Inputs:
    // - X, Y: float* (device pointers), shape (D,), GPU memory
    // - diff: float* (device pointer), shape (D,), output array (GPU memory)
    // - D: int, length of vectors
    //
    // Action: Performs element-wise squared difference: diff[i] = (X[i] - Y[i])^2 and writes to diff[i] in parallel using threads.



    // Each thread processes a single element of the vectors
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < D) {
        // Calculate the squared difference for each element
        float diff_val = X[idx] - Y[idx];
        diff[idx] = diff_val * diff_val;
    }
}

extern "C" __global__
void reduce_kernel(float* diff, int D) {

    // Inputs:
    // - diff: float* (device pointer), shape (D,), updated in-place
    // - D: int, number of elements
    //
    // Action: Performs in-place block-level parallel reduction to sum squared differences.
    // After multiple calls (externally looped), final result will be diff[0].


    __shared__ float cache[256];  // Shared memory to store partial sums for reduction within a block

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    float sum = 0.0f;

    // Each thread loads a value from the global memory and stores it in shared memory
    if (idx < D) {
        sum = diff[idx];
    }

    // Store the partial sum in shared memory
    cache[tid] = sum;
    __syncthreads();

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // After reduction, the thread with idx 0 writes the final sum of the block to global memory
    if (tid == 0) {
        diff[blockIdx.x] = cache[0];
    }
}
'''

def l2_distance_kernel(X, Y):

    """
    Computes the Euclidean (L2) distance between two vectors using custom CUDA kernels with CuPy.

    Parameters
    ----------
    X : np.ndarray or cp.ndarray
        Input vector of shape (D,), either on CPU or GPU.
        Will be transferred to GPU if not already.
    Y : np.ndarray or cp.ndarray
        Same as X.

    Returns
    -------
    dist : float
        Final L2 distance between X and Y.
        Computed on GPU and returned to CPU as a NumPy float.

    Internal Representation
    -----------------------
    - X and Y are converted to `cp.ndarray` of dtype `float32`
    - Output intermediate arrays (`diff_gpu`) live on GPU.
    - Final output is moved back to CPU with `cp.asnumpy`.

    """

    # Start timer to measure execution time
    start_time = time.time()

    # Compile the CUDA kernel from the raw code
    mod = cp.RawModule(code=l2_kernel_code)
    func_diff_square = mod.get_function('calculate_diff_kernel')
    func_sum = mod.get_function('reduce_kernel')

    # Get the dimension of the input vectors
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)
    D = X_gpu.size 

    # Allocate GPU memory for the squared differences
    diff_gpu = cp.empty(D, dtype=cp.float32)

    # Set the block size (number of threads per block) and grid size (number of blocks)
    block_size = 256
    grid_size = (D + block_size - 1) // block_size

    # Launch the kernel to calculate the squared differences between the two vectors
    func_diff_square(
        (grid_size,), (block_size,),  # grid and block dimensions
        (X_gpu, Y_gpu, diff_gpu, D)  # kernel arguments
    )

    # Perform parallel reduction to sum the squared differences
    while D > 1:
        grid_size = (D + block_size - 1) // block_size
        func_sum(
            (grid_size,), (block_size,),  # grid and block dimensions
            (diff_gpu, D)  # kernel arguments
        )
        D = grid_size  # Reduce D to the number of remaining blocks

    # End timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    # The final result is stored in diff_gpu[0] after the reduction
    return cp.asnumpy(cp.sqrt(diff_gpu[0]))  # Return the square root of the sum of squared differences


# ------------------------------------------------------------------------------------------------
# Manhattan Distance
# ------------------------------------------------------------------------------------------------

def manhattan_distance_cpu(X, Y):

    """
    Computes the Manhattan (L1) distance between two vectors using NumPy on the CPU.

    Parameters
    ----------
    X : np.ndarray
        Input vector of shape (D,), must reside in CPU memory.
    Y : np.ndarray
        Input vector of shape (D,), must reside in CPU memory.

    Returns
    -------
    dist : float
        Scalar value representing the L1 distance between X and Y.
        Returned as a Python float (CPU-resident).

    """

    # Start timer to measure execution time
    start_time = time.time()

    # Compute the Manhattan distance between vectors X and Y as the sum of absolute differences
    dist = np.sum(np.abs(X - Y))

    # End timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return dist

def manhattan_distance_cupy(X, Y):

    """
    Computes the Manhattan (L1) distance between two vectors using CuPy on the GPU.

    Parameters
    ----------
    X : np.ndarray or cp.ndarray
        Input vector of shape (D,), either CPU or GPU memory.
        Will be transferred to GPU if needed.
    Y : np.ndarray or cp.ndarray
        Same as X.

    Returns
    -------
    dist : cp.ndarray (scalar)
        Manhattan distance computed on GPU.
        Computed on GPU, then transferred back to CPU.

    """

    # Start timer to measure execution time
    start_time = time.time()

    # Transfer input vectors to GPU memory (Cupy arrays)
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

    # Compute the Manhattan distance on the GPU
    dist = cp.sum(cp.abs(X_gpu - Y_gpu))

    # End timer and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return dist.get()  # Transfer the result back to CPU as a NumPy float

manhattan_kernel = cp.RawKernel(r'''
extern "C" __global__
void manhattan_distance_optimized(const float* X, const float* Y, float* result, int size) {
                                
    // Parameters:
    // - X: const float* (device pointer), shape (D,), input vector (GPU memory)
    // - Y: const float* (device pointer), shape (D,), input vector (GPU memory)
    // - result: float* (device pointer), scalar output (GPU memory)
    // - size: int, number of elements D
    // Action:
    // Each thread computes abs(X[i] - Y[i]), stores it in shared memory,
    // and a parallel reduction is performed per block. Only thread 0 writes the block result.


    // Shared memory for parallel reduction within each block
    extern __shared__ float sdata[];

    int tid = threadIdx.x;  // Thread index within a block
    int idx = blockIdx.x * blockDim.x + tid;  // Global index for the data array
    
    // Load data into shared memory (handle boundary conditions)
    sdata[tid] = (idx < size) ? fabsf(X[idx] - Y[idx]) : 0.0f;
    __syncthreads();

    // Parallel reduction within a block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];  // Sum pairs of values
        }
        __syncthreads();
    }

    // Only thread 0 of each block writes the partial sum to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]); // Store the sum of this block using atomic operation to avoid race conditions if num_blocks > 1
    }
}
''', 'manhattan_distance_optimized')

def manhattan_distance_kernel(X, Y):

    """
    Computes the Manhattan (L1) distance between two vectors using a custom CUDA kernel on GPU.

    Parameters
    ----------
    X : np.ndarray or cp.ndarray
        Input vector of shape (D,), float32.
        Will be converted and moved to GPU if not already.
    Y : np.ndarray or cp.ndarray
        Same as X.

    Returns
    -------
    result : float
        Manhattan distance as a scalar float.
        Computed on GPU and returned to CPU.

    """

    # Start timer to measure execution time
    start_time = time.time()

    # Transfer input vectors to GPU memory as float32
    X_gpu = cp.asarray(X, dtype=cp.float32)
    Y_gpu = cp.asarray(Y, dtype=cp.float32)

    # Allocate memory for the result on the GPU
    result_gpu = cp.zeros(1, dtype=cp.float32)

    # Define CUDA kernel launch parameters
    threads_per_block = 256
    blocks_per_grid = (X_gpu.size + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel using shared memory
    manhattan_kernel(
        (blocks_per_grid,),  # Number of blocks in the grid
        (threads_per_block,),  # Number of threads per block
        (X_gpu, Y_gpu, result_gpu, X_gpu.size),  # Kernel arguments
        shared_mem=threads_per_block * 4  # Allocate shared memory (4 bytes per float)
    )

    # Transfer the result back to the CPU
    result = result_gpu[0].get()

    # End timer and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return result


# ------------------------------------------------------------------------------------------------
# Dot Distance
# ------------------------------------------------------------------------------------------------

def dot_cpu(X, Y):

    """
    Computes the dot product similarity distance between two vectors using NumPy on the CPU.

    Parameters
    ----------
    X : np.ndarray
        Input vector of shape (D,), must reside in CPU memory.
    Y : np.ndarray
        Input vector of shape (D,), must reside in CPU memory.

    Returns
    -------
    dist : float
        Scalar value representing the dot product similarity between X and Y.
        Returned as a Python float (CPU-resident).

    """

    # Start timer to measure execution time
    start_time = time.time()

    # Compute the Manhattan distance between vectors X and Y as the sum of absolute differences
    similarity = X @ Y

    # End timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return similarity

def dot_cupy(X, Y):

    """
    Computes the dot product similarity between two vectors using CuPy on the GPU.

    Parameters
    ----------
    X : np.ndarray or cp.ndarray
        Input vector of shape (D,), either CPU or GPU memory.
        Will be transferred to GPU if needed.
    Y : np.ndarray or cp.ndarray
        Same as X.

    Returns
    -------
    dist : cp.ndarray (scalar)
        Dot product similarity computed on GPU.
        Computed on GPU, then transferred back to CPU.

    """

    # Start timer to measure execution time
    start_time = time.time()

    # Transfer input vectors to GPU memory (Cupy arrays)
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

    # Compute the Manhattan distance on the GPU
    similarity = X_gpu @ Y_gpu

    # End timer and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return similarity.get()  # Transfer the result back to CPU as a NumPy float

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

def dot_kernel(X,Y):
    """
    Computes dot product similarity using a custom CUDA kernel.
    
    Args:
        X (cp.ndarray or np.ndarray): Shape (m, d)
        Y (cp.ndarray or np.ndarray): Shape (n, d)
    
    Returns:
        cp.ndarray: Dot product similarity (m, n)
    """
    # Start timer
    start_time = time.time()
    X_gpu = cp.asarray(X, dtype=cp.float32)
    Y_gpu = cp.asarray(Y, dtype=cp.float32)
    n = X_gpu.shape[0]

    # Launch configuration
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    result = cp.zeros(1, dtype=cp.float32)
    dot_product(
        (blocks_per_grid,), 
        (threads_per_block,), 
        (X_gpu, Y_gpu, result, n)
    )

    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")
    return result.get()[0]

# ------------------------------------------------------------------------------------------------
# Cosine Distance
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

# ------------------------------------------------------------------------------------------------
# TOP-K (RETRIEVE THE INDICES OF THE K SMALLEST VALUES OF A VECTOR)
# ------------------------------------------------------------------------------------------------

topK_radixSort_kernel = '''
extern "C" __global__
void topK_radix_sort(const float* arr, int* indices, float* values, int N, int K) {

    //Parameters:
    // - arr (float*): array containing float values where the K smallest elements are to be found. Shape: (N,). Memory Location: Global memory (GPU)
    // - indices (int*): Array to store the indices of the K smallest values in the input array `arr`. Shape: (N,).  Memory Location: Global memory (GPU)
    // - values (float*): Array to store the K smallest values found in the input array `arr`. Shape: (K,). Memory Location: Global memory (GPU)
    // - N (int): Total number of elements in the input array `arr`. Memory Location: Global memory (CPU)
    // - K (int): Number of smallest elements to find in the input array `arr`. Memory Location: Global memory (CPU)

    //Returns:
    // - values (float*): The K smallest values found in the input array `arr`. Shape: (K,). Memory Location: Global memory (GPU)
    // - indices (int*): The indices of the K smallest values in the input array `arr`. Shape: (K,). Memory Location: Global memory (GPU)


    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    #define INF 3.402823466e+38f  // Define an "infinity" value for padding

    // Pointers to shared memory
    float* shared_vals = shared_data;
    int* shared_idx = (int*)&shared_vals[blockDim.x];

    // Number of elements in this block
    int elements_in_block = min(blockDim.x, N - blockIdx.x * blockDim.x);

    // Load values into shared memory
    if (idx < N) {
        shared_vals[tid] = arr[idx];
        shared_idx[tid] = idx;
    } else {
        shared_vals[tid] = INF;  // Padding with infinity for out-of-bounds indices
        shared_idx[tid] = -1;
    }
    __syncthreads();

    // ========================
    // ** GPU RADIX SORT **
    // ========================
    int max_bits = 32;  // Max bits for floating-point values (as integers)
    for (int bit = 0; bit < max_bits; bit++) {
        int mask = 1 << bit;
        int prefix_sum = 0;

        // Count zeros in the current bit position
        for (int i = 0; i < elements_in_block; i++) {
            if (!((__float_as_int(shared_vals[i]) & mask) > 0)) {
                prefix_sum++;
            }
        }
        __syncthreads();

        // Temporary arrays for sorting
        float temp_vals[256];  
        int temp_idx[256];
        int zero_pos = 0, one_pos = prefix_sum;

        // Distribute elements based on the current bit value
        for (int i = 0; i < elements_in_block; i++) {
            int bit_val = (__float_as_int(shared_vals[i]) & mask) > 0;
            if (bit_val == 0) {
                temp_vals[zero_pos] = shared_vals[i];
                temp_idx[zero_pos] = shared_idx[i];
                zero_pos++;
            } else {
                temp_vals[one_pos] = shared_vals[i];
                temp_idx[one_pos] = shared_idx[i];
                one_pos++;
            }
        }
        __syncthreads();

        // Copy sorted values back to shared memory
        for (int i = 0; i < elements_in_block; i++) {
            shared_vals[i] = temp_vals[i];
            shared_idx[i] = temp_idx[i];
        }
        __syncthreads();
    }

    // Store the top-K smallest values in global memory
    if (tid < K && (blockIdx.x * K + tid) < N) {
        values[blockIdx.x * K + tid] = shared_vals[tid];
        indices[blockIdx.x * K + tid] = shared_idx[tid];
    }
}
'''

def our_topK(arr, K, all_indices=None, final=False):
    
    """
    Find the top-K smallest values in an array using a parallel radix sort on GPU.

    Parameters:
    - arr (cp.ndarray): Input array (Cupy array) of shape (N,). Must be in GPU memory.
    - K (int): Number of top-K smallest elements to find
    - all_indices (cp.ndarray , optional): Used to map indices to original positions in a multi-step KNN search. Must be set to the original indices if Final = True
    - final (bool, optional): If True, map the indices to the original input dataset

    Returns:
    - values (cp.ndarray): The K smallest values found. Shape (K,).
    - indices (cp.ndarray): Their corresponding indices. Shape (K,).
    """

    # Load the CUDA kernel
    mod = cp.RawModule(code=topK_radixSort_kernel)
    func_topk = mod.get_function('topK_radix_sort')

    # Configure CUDA execution
    threads_per_block = 256
    blocks_per_grid = (arr.shape[0] + threads_per_block - 1) // threads_per_block
    shared_mem_size = 2 * threads_per_block * arr.itemsize  # Shared memory for sorting

    # Allocate memory for results
    values = cp.empty((blocks_per_grid, K), dtype=arr.dtype)
    indices = cp.empty((blocks_per_grid, K), dtype=cp.int32)

    # First pass: Perform top-K selection on blocks
    func_topk(
        (blocks_per_grid,), (threads_per_block,),
        (arr, indices, values, arr.shape[0], K),
        shared_mem=shared_mem_size
    )

    # Map indices back to original dataset if final processing step
    if final:
        indices = all_indices[indices]

    values = values.flatten()
    indices = indices.flatten()

    # Reduce multiple blocks until only one block remains
    while blocks_per_grid > 1:
        values_1 = cp.empty((blocks_per_grid, K), dtype=values.dtype)
        indices_1 = cp.empty((blocks_per_grid, K), dtype=cp.int32)

        func_topk(
            (blocks_per_grid,), (threads_per_block,),
            (values, indices_1, values_1, values.shape[0], K),
            shared_mem=shared_mem_size
        )

        values = values_1
        indices = indices[indices_1]
        values = values.flatten()
        indices = indices.flatten()
        blocks_per_grid = (values.shape[0] + threads_per_block - 1) // threads_per_block

    # Final pass to get the top-K results from the last remaining block
    values_1 = cp.empty((blocks_per_grid, K), dtype=values.dtype)
    indices_1 = cp.empty((blocks_per_grid, K), dtype=cp.int32)

    func_topk(
        (blocks_per_grid,), (threads_per_block,),
        (values, indices_1, values_1, values.shape[0], K),
        shared_mem=shared_mem_size
    )

    values = values_1
    indices = indices[indices_1]
    values = values.flatten()
    indices = indices.flatten()

    return values, indices

# ------------------------------------------------------------------------------------------------
# K-NEAREST NEIGHBORS (RETRIEVE THE INDICES OF THE K NEAREST VECTORS OF A QUERY VECTOR WITHIN A SET) 
# ------------------------------------------------------------------------------------------------

def our_knn_cpu(N, D, A, X, K, distance = "l2"):

    """
    Computes the K nearest neighbors (KNN) using a CPU implementation.

    Args:
        N (int): Number of data points in the dataset A.
        D (int): Number of dimensions of each data point in A and X.
        A (numpy.ndarray): The dataset (NxD matrix) containing N data points, each with D features. Shape (N, D)
        X (numpy.ndarray): The query point. Shape (D,)
        K (int): The number of nearest neighbors to find.
        distance (str): The distance metric to use ('l2', 'manhattan', 'cosine', or 'dot').

    Returns:
        numpy.ndarray: The indices of the K nearest neighbors in A, based on the selected distance metric a (K,) numpy.ndarray.
    

    Memory considerations:
        The function assumes that the dataset A is loaded entirely in memory, e.g using np.mmap is A is too big.
    """

    #Start measuring the time
    start_time = time.time()
    
    memory_limit = 2**28 #We assume 2GB of RAM memory
    batch_size = min(N, memory_limit // (D * A.itemsize)) 

    #To store results of each batch
    all_distances = []
    all_indices = []
    
    # Process A with batches
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        
        # Extract the current batch of A
        batch_A = A[batch_start:batch_end]
        
        # Compute the distances from X to each point in the current batch of A based on the 'distance' argument
        if distance == 'l2':  # Euclidean distance (L2)
            distances = np.sqrt(np.sum((batch_A - X)**2, axis=1)).astype(np.float32)
        elif distance == 'manhattan':  # Manhattan distance
            distances = np.sum(np.abs(batch_A - X), axis=1).astype(np.float32)
        elif distance == 'cosine':  # Cosine similarity
            distances = 1 - (np.sum(batch_A * X, axis=1) / (np.linalg.norm(batch_A, axis=1) * np.linalg.norm(X))) 
        elif distance == 'dot':
            distances = np.sum(batch_A * X, axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
        
        # Get the K nearest neighbors (indices and distances)
        indices = np.argpartition(distances,K)[:K]
        values = distances[indices]
        
        # Append results from the current batch
        all_distances.append(values)
        all_indices.append(indices + batch_start)  # Add the offset of the batch
    
    # Concatenate results from all batches
    all_distances = np.concatenate(all_distances)
    all_indices = np.concatenate(all_indices)
    
    # Select the K nearest neighbors globally
    top_k_global_indices = np.argpartition(all_distances,K)[:K]

    # Stop measuring the time when we have found the indices
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total Time: {execution_time:.6f} seconds")
    
    return all_indices[top_k_global_indices]

def our_knn_cupy_batching(N, D, A, X, K, distance="l2"):

    """
    Computes the K nearest neighbors (KNN) using a batching strategy on GPU and streams to compute batches in parallel.

    Args:
        N (int): Number of data points in the dataset A.
        D (int): Number of dimensions of each data point in A and X.
        A (cupy.ndarray of shape (N,D)): The dataset (NxD matrix) stored on the GPU.
        X (cupy.ndarray of shape (D,)): The query point (1xD vector) stored on the GPU.
        K (int): The number of nearest neighbors to find.
        distance (str): The distance metric to use ('l2', 'manhattan', 'cosine', or 'dot').

    Returns:
        numpy.ndarray (K,): The indices of the K nearest neighbors in A, based on the selected distance metric.

    Memory considerations:
        The function assumes that A and X are loaded on the GPU.
    """

    start_time = time.time()

    # Choose batch_size based on available GPU memory (assuming 2GB RAM)
    memory_limit = 2**28 
    batch_size = min(N, memory_limit // (D * A.itemsize))  # Divide by the number of bytes per vector in A
    num_batches = (N + batch_size - 1) // batch_size  # Compute the number of batches

    # Most GPUs work well with 4 streams for parallel processing
    num_streams = 4
    streams = [cp.cuda.Stream() for _ in range(num_streams)]  # Create CUDA streams

    # Move input vector X to GPU
    X_gpu = cp.asarray(X).astype(cp.float32)
    
    # Lists to store distances and indices of the nearest neighbors
    all_distances = cp.empty((K*num_batches,), dtype=cp.float32)
    all_indices = cp.empty((K*num_batches,), dtype=cp.int32)

    # Process A in batches
    for batch_idx, batch_start in enumerate(range(0, N, batch_size)):
        batch_end = min(batch_start + batch_size, N)

        # Select appropriate stream for processing
        stream = streams[batch_idx % num_streams]

        with stream:

            #Move batch of A to GPU
            batch_A = cp.asarray(A[batch_start:batch_end])

            # Compute distances based on the selected metric
            if distance == 'l2':
                distances = cp.sqrt(cp.sum((X_gpu - batch_A) ** 2, axis=1))
            elif distance == 'manhattan':
                distances = cp.sum(cp.abs(X_gpu - batch_A), axis=1)
            elif distance == 'cosine':
                distances = 1 - (cp.sum(batch_A * X_gpu, axis=1) / 
                                    (cp.linalg.norm(batch_A, axis=1) * cp.linalg.norm(X_gpu)))
            elif distance == 'dot':
                distances = cp.sum(batch_A * X_gpu, axis=1).astype(cp.float32)
            else:
                raise ValueError(f"Unsupported distance metric: {distance}")

            
            # Find top K nearest neighbors within the batch
            indices = cp.argpartition(distances, K)[:K]
            values = distances[indices]
            
            # Store results
            all_indices[batch_idx*K : (batch_idx+1)*K] = indices + batch_start
            all_distances[batch_idx*K : (batch_idx+1)*K] = values

    # Synchronize all CUDA streams before further processing
    for stream in streams:
        stream.synchronize()

    # If multiple batches were processed, find the top-K nearest neighbors globally
    if num_batches > 1:
        top_k_global_indices = cp.argpartition(all_distances, K)[:K]
        result = all_indices[top_k_global_indices]
    else:
        result = indices  # If only one batch, return its results directly
    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")
    return result

def our_knn_chunks_manhattan(N, D, A, X, K):

    """
    Computes the K nearest neighbors (KNN) using chunking and Manhattan distance on GPU. Streams are used to compute distances of several chunks in parallel.

    Args:
        N (int): Number of data points in the dataset A.
        D (int): Number of dimensions of each data point in A and X.
        A (cupy.ndarray of shape (N,D)): The dataset (NxD matrix) stored on the GPU.
        X (cupy.ndarray of shape (D,)): The query point (1xD vector) stored on the GPU.
        K (int): The number of nearest neighbors to find.

    Returns:
        numpy.ndarray of shape (K,): The indices of the K nearest neighbors in A, based on Manhattan distance.

    Memory considerations:
        The function assumes that A and X are loaded on the GPU. It uses chunking to handle high-dimensional data and parallelizes the computation with multiple CUDA streams.
    """

    start_time = time.time()

    # Assume no more than 2GB of memory in GPU
    memory_limit = 2**28
    batch_size = min(N, memory_limit // (D * A.itemsize))  
    num_batches = (N + batch_size - 1) // batch_size  

    # If high dimensionality, then compute distances along chunks
    num_chunks = 1
    if(D > 32):
        num_chunks = 32

    chunk_size = D // num_chunks  

    # 4 streams to compute distances of different chunks in paralell
    num_streams = 4  
    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    # Move X to the GPU
    X_gpu = cp.asarray(X).astype(cp.float32)

    #To store the results of each batch of A
    all_distances = cp.empty((K*num_batches,), dtype=cp.float32)
    all_indices = cp.empty((K*num_batches,), dtype=cp.int32)

    # Process A  using batches
    for batch_idx, batch_start in enumerate(range(0, N, batch_size)):
        batch_end = min(batch_start + batch_size, N)

        #Move the batch to the GPU
        batch_A = cp.asarray(A[batch_start:batch_end]).astype(cp.float32)

        # To store the distances of each chunk
        partial_distances = cp.empty((num_chunks,batch_A.shape[0]), dtype=cp.float32)

        # Compute distances of each chunk
        for chunk_idx, chunk_start in enumerate(range(0, D, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, D)
            stream = streams[chunk_idx % num_streams]

            with stream:
                #Manhattan distance of the chunk
                diff = cp.abs(X_gpu[chunk_start:chunk_end] - batch_A[:, chunk_start:chunk_end])
                partial_distances[chunk_idx] = cp.sum(diff, axis=1)

        # Sincronizar los streams después de computar todas las distancias parciales
        for stream in streams:
            stream.synchronize()

        #Combine the results of each chunk to compute the total distance
        total_distances = cp.sum(partial_distances, axis = 0)

        #Get the K lowest values of the batch
        indices = cp.argpartition(total_distances, K)[:K]
        values = total_distances[indices]

        #Store the results of the batch
        all_indices[batch_idx*K : (batch_idx+1)*K] = indices + batch_start
        all_distances[batch_idx*K : (batch_idx+1)*K] = values

    # Select the K lowest values across batches
    if(num_batches > 1):
        top_k_global_indices = cp.argpartition(all_distances, K)[:K]
        result = all_indices[top_k_global_indices].get()
    else:
        result = indices.get()

    # Stop measuring the time when we have found the indices
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")

    return result

def our_knn_chunks_cosine(N, D, A, X, K):

    """
    Computes the K nearest neighbors (KNN) using chunking and Cosine distance on GPU. Streams are used to compute distances of several chunks in parallel.

    Args:
        N (int): Number of data points in the dataset A.
        D (int): Number of dimensions of each data point in A and X.
        A (cupy.ndarray of shape (N,D)): The dataset (NxD matrix) stored on the GPU.
        X (cupy.ndarray of shape (D,)): The query point (1xD vector) stored on the GPU.
        K (int): The number of nearest neighbors to find.

    Returns:
        numpy.ndarray of shape (K,): The indices of the K nearest neighbors in A, based on Cosine distance.

    Memory considerations:
        The function assumes that A and X are loaded on the GPU. It uses chunking to handle high-dimensional data and parallelizes the computation with multiple CUDA streams.
    """


    start_time = time.time()

    # Memory configuration and batch size calculation
    memory_limit = 2**28  #  2GB memory limit
    batch_size = min(N, memory_limit // (D * A.itemsize))  # Calculate the maximum batch size that fits in GPU memory
    num_batches = (N + batch_size - 1) // batch_size  # Compute the number of batches required

    # Determine the number of column chunks to split the data
    num_chunks = 1
    if D > 32:
        num_chunks = 32  # Use 32 chunks if the dimension is large
    chunk_size = D // num_chunks  # Divide columns into chunks

    # Set up CUDA streams for parallel processing
    num_streams = 4  # Use 4 streams to improve concurrency
    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    # Move the input vector X to the GPU
    X_gpu = cp.asarray(X).astype(cp.float32)

    # Allocate memory for storing the K nearest neighbors
    all_distances = cp.empty((K * num_batches,), dtype=cp.float32)
    all_indices = cp.empty((K * num_batches,), dtype=cp.int32)

    # Process the dataset A in batches
    for batch_idx, batch_start in enumerate(range(0, N, batch_size)):
        batch_end = min(batch_start + batch_size, N)

        # Load a batch of A into GPU memory
        batch_A = cp.asarray(A[batch_start:batch_end]).astype(cp.float32)

        # Initialize storage for partial distances (one row per chunk)
        partial_distances = cp.empty((num_chunks, batch_A.shape[0]), dtype=cp.float32)

        # Process data in column chunks
        for chunk_idx, chunk_start in enumerate(range(0, D, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, D)
            stream = streams[chunk_idx % num_streams]  # Assign stream in a round-robin fashion

            with stream:
                # Compute the dot product for the cosine similarity
                dot_product = cp.sum(X_gpu[chunk_start:chunk_end] * batch_A[:, chunk_start:chunk_end], axis=1)

                # Compute the L2 norm for both X and A (required for cosine similarity)
                norm_X = cp.linalg.norm(X_gpu[chunk_start:chunk_end])
                norm_A = cp.linalg.norm(batch_A[:, chunk_start:chunk_end], axis=1)

                # Store the cosine distance (1 - cosine similarity)
                partial_distances[chunk_idx] = 1 - (dot_product / (norm_X * norm_A))

        # Synchronize all streams to ensure computations are complete before proceeding
        for stream in streams:
            stream.synchronize()

        # Sum partial distances across chunks to obtain the final cosine distances
        total_distances = cp.sum(partial_distances, axis=0)

        # Find the K nearest neighbors within the batch
        indices = cp.argpartition(total_distances, K)[:K]  # Get the indices of the K smallest distances
        values = total_distances[indices]  # Retrieve the corresponding distances

        # Store results in global arrays
        all_indices[batch_idx * K : (batch_idx + 1) * K] = indices + batch_start  # Adjust index to match the full dataset
        all_distances[batch_idx * K : (batch_idx + 1) * K] = values

    # If multiple batches were processed, find the top-K nearest neighbors globally
    if num_batches > 1:
        top_k_global_indices = cp.argpartition(all_distances, K)[:K]
        result = all_indices[top_k_global_indices]
    else:
        result = indices  # If only one batch, return its results directly
    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")
    return result

def our_knn_chunks_dot(N, D, A, X, K):

    """
    Computes the K nearest neighbors (KNN) using chunking and Dot distance on GPU. Streams are used to compute distances of several chunks in parallel.

    Args:
        N (int): Number of data points in the dataset A.
        D (int): Number of dimensions of each data point in A and X.
        A (cupy.ndarray of shape (N,D)): The dataset (NxD matrix) stored on the GPU.
        X (cupy.ndarray of shape (D,)): The query point (1xD vector) stored on the GPU.
        K (int): The number of nearest neighbors to find.

    Returns:
        numpy.ndarray of shape (K,): The indices of the K nearest neighbors in A, based on Dot distance.

    Memory considerations:
        The function assumes that A and X are loaded on the GPU. It uses chunking to handle high-dimensional data and parallelizes the computation with multiple CUDA streams.
    """
    start_time = time.time()

    # Memory configuration and batch size calculation
    memory_limit = 2**28  # 256 MiB memory limit
    batch_size = min(N, memory_limit // (D * A.itemsize))  # Maximum batch size that fits in GPU memory
    num_batches = (N + batch_size - 1) // batch_size  # Compute the number of batches needed

    # Determine the number of column chunks to split the data
    num_chunks = 1
    if D > 32:
        num_chunks = 32  # Use 32 chunks if the dimension is large
    chunk_size = D // num_chunks  # Divide columns into chunks

    # Set up CUDA streams for parallel processing
    num_streams = 4  # Use 4 streams to improve concurrency
    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    # Move the input vector X to the GPU
    X_gpu = cp.asarray(X).astype(cp.float32)

    # Allocate memory for storing the K nearest neighbors
    all_distances = cp.empty((K * num_batches,), dtype=cp.float32)
    all_indices = cp.empty((K * num_batches,), dtype=cp.int32)

    # Process the dataset A in batches
    for batch_idx, batch_start in enumerate(range(0, N, batch_size)):
        batch_end = min(batch_start + batch_size, N)

        # Load a batch of A into GPU memory
        batch_A = cp.asarray(A[batch_start:batch_end]).astype(cp.float32)

        # Initialize storage for partial dot product calculations (one row per chunk)
        partial_distances = cp.empty((num_chunks, batch_A.shape[0]), dtype=cp.float32)

        # Process data in column chunks
        for chunk_idx, chunk_start in enumerate(range(0, D, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, D)
            stream = streams[chunk_idx % num_streams]  # Assign stream in a round-robin fashion

            with stream:
                # Compute the dot product for the given chunk
                dot_product = cp.sum(X_gpu[chunk_start:chunk_end] * batch_A[:, chunk_start:chunk_end], axis=1)

                # Store the partial results
                partial_distances[chunk_idx] = dot_product

        # Synchronize all streams to ensure computations are complete before proceeding
        for stream in streams:
            stream.synchronize()

        # Sum partial dot products across chunks to get the final similarity score
        total_distances = cp.sum(partial_distances, axis=0)

        # Find the K highest dot product values (since a higher dot product means higher similarity)
        indices = cp.argpartition(total_distances, -K)[-K:]  # Get indices of the top-K highest values
        values = total_distances[indices]  # Retrieve the corresponding values

        # Store results in global arrays
        all_indices[batch_idx * K : (batch_idx + 1) * K] = indices + batch_start  # Adjust index to match full dataset
        all_distances[batch_idx * K : (batch_idx + 1) * K] = values

    # If multiple batches were processed, find the top-K nearest neighbors globally
    if num_batches > 1:
        top_k_global_indices = cp.argpartition(all_distances, K)[:K]
        result = all_indices[top_k_global_indices]
    else:
        result = indices  # If only one batch, return its results directly
    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")
    return result

def our_knn_chunks_l2(N, D, A, X, K):

    """
    Computes the K nearest neighbors (KNN) using chunking and L2 distance on GPU. Streams are used to compute distances of several chunks in parallel.

    Args:
        N (int): Number of data points in the dataset A.
        D (int): Number of dimensions of each data point in A and X.
        A (cupy.ndarray of shape (N,D)): The dataset (NxD matrix) stored on the GPU.
        X (cupy.ndarray of shape (D,)): The query point (1xD vector) stored on the GPU.
        K (int): The number of nearest neighbors to find.

    Returns:
        numpy.ndarray of shape (K,): The indices of the K nearest neighbors in A, based on L2 distance.

    Memory considerations:
        The function assumes that A and X are loaded on the GPU. It uses chunking to handle high-dimensional data and parallelizes the computation with multiple CUDA streams.
    """
    start_time = time.time()

    # Memory configuration and batch size calculation
    memory_limit = 2**28  # 256 MiB memory limit
    batch_size = min(N, memory_limit // (D * A.itemsize))  # Maximum batch size that fits in GPU memory
    num_batches = (N + batch_size - 1) // batch_size  # Compute the number of batches

    # Determine the number of column chunks to split the data
    num_chunks = 1
    if D > 32:
        num_chunks = 32  # Use 32 chunks if the dimension is large
    chunk_size = D // num_chunks  # Divide columns into chunks

    # Set up CUDA streams for parallel processing
    num_streams = 4  # Use 4 streams for better parallelization
    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    # Move the input vector X to the GPU
    X_gpu = cp.asarray(X).astype(cp.float32)

    # Allocate memory for storing the K nearest neighbors
    all_distances = cp.empty((K * num_batches,), dtype=cp.float32)
    all_indices = cp.empty((K * num_batches,), dtype=cp.int32)

    # Process dataset A in batches
    for batch_idx, batch_start in enumerate(range(0, N, batch_size)):
        batch_end = min(batch_start + batch_size, N)

        # Load a batch of A into GPU memory
        batch_A = cp.asarray(A[batch_start:batch_end]).astype(cp.float32)

        # Initialize storage for partial L2 distance calculations
        partial_distances = cp.empty((num_chunks, batch_A.shape[0]), dtype=cp.float32)

        # Process data in column chunks
        for chunk_idx, chunk_start in enumerate(range(0, D, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, D)
            stream = streams[chunk_idx % num_streams]  # Assign stream in a round-robin fashion

            with stream:
                # Compute squared difference for the chunk
                diff = X_gpu[chunk_start:chunk_end] - batch_A[:, chunk_start:chunk_end]
                partial_distances[chunk_idx] = cp.sum(diff ** 2, axis=1)  # Sum squared differences

        # Synchronize all streams to ensure computations are complete before proceeding
        for stream in streams:
            stream.synchronize()

        # Sum all partial squared distances and take the square root to get final L2 distance
        total_distances = cp.sqrt(cp.sum(partial_distances, axis=0))

        # Find the K smallest L2 distances (lower values indicate greater similarity)
        indices = cp.argpartition(total_distances, K)[:K]  # Get indices of the top-K lowest values
        values = total_distances[indices]  # Retrieve corresponding values

        # Store results in global arrays
        all_indices[batch_idx * K : (batch_idx + 1) * K] = indices + batch_start  # Adjust index to match full dataset
        all_distances[batch_idx * K : (batch_idx + 1) * K] = values

    # If multiple batches were processed, find the top-K nearest neighbors globally
    if num_batches > 1:
        top_k_global_indices = cp.argpartition(all_distances, K)[:K]
        result = all_indices[top_k_global_indices]
    else:
        result = indices  # If only one batch, return its results directly
    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")
    return result


# ------------------------------------------------------------------------------------------------
# K-MEANS
# ------------------------------------------------------------------------------------------------

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

def our_kmeans_numpy(N, D, A, K, max_iters=100, tol=1e-4, random_state=None, distance_metric="l2"):
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

    memory_limit = 2**28 # assume 2GB of RAM
    batch_size = min(N, memory_limit // (D * A.itemsize))

    # Initialise centroids by randomly selecting K data points from A
    rng = np.random.RandomState(seed=random_state)
    indices = rng.choice(N, size=K, replace=False)
    centroids = cp.asarray(A[indices.get()])  # Initial centroids
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
    print(f"Total Time: {execution_time:.6f} seconds")

    return centroids, assignments, execution_time

# CUPY DESIGN 1 ----------------------------------------------------------------------------------
def our_kmeans_cupy_1(N, D, A, K, max_iters=100, tol=1e-4, distance_metric="l2"):
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

    memory_limit = 2**28 # assume 2GB of RAM
    batch_size = min(N, memory_limit // (D * A.itemsize))
    
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
    print(f"Total Time: {execution_time:.6f} seconds")
    
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

def our_kmeans_cupy_2(N,D,A,K, max_iters=100, tol=1e-4, random_state=None, batch_size=1000, distance_metric="l2"):
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
                new_centroids[k] = A[cp.random.randint(0, N, 1)].squeeze(0)


        # Check for convergence
        centroid_shift = cp.linalg.norm(new_centroids - centroids, axis=1).max() # max shift in centroids between iterations
        if centroid_shift <= tol:
            break
        
        centroids = new_centroids

    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total Time: {execution_time:.6f} seconds")
    
    return centroids, assignments, execution_time

# ------------------------------------------------------------------------------------------------
# ANN (APPROXIMATE NEAREST NEIGHBORS)
# ------------------------------------------------------------------------------------------------

#TODO

# ------------------------------------------------------------------------------------------------
# TESTING FUNCTIONS
# ------------------------------------------------------------------------------------------------

def test_l2(D):
    """
    Tests the L2 (Euclidean) distance calculation between two random vectors using different CPU, Cupy kernels and custom CUDA kernels.
    Args:
        D (list of int): A list of integers where each integer represents the dimensionality of the random vectors X and Y.
    Returns:
        None: This function prints the results of the distance calculations and the time spent, but does not return any values.
    """
    for d in D:
        print(d)
        for i in range(0,3):
            np.random.seed(42)
            X = np.random.randn(d).astype(np.float32)
            Y = np.random.randn(d).astype(np.float32)
            print(l2_distance_cpu(X,Y))
            print(l2_distance_cupy(X,Y))
            print(l2_distance_kernel(X,Y))

def test_manhattan(D):
    """
    Tests the L1 (Manhattan) distance calculation between two random vectors using different CPU, Cupy kernels and custom CUDA kernels.
    Args:
        D (list of int): A list of integers where each integer represents the dimensionality of the random vectors X and Y.
    Returns:
        None: This function prints the results of the distance calculations and the time spent, but does not return any values.
    """

    for d in D:
        print(d)
        for i in range(0,3):
            np.random.seed(42)
            X = np.random.randn(d).astype(np.float32)
            Y = np.random.randn(d).astype(np.float32)
            print(manhattan_distance_cpu(X,Y))
            print(manhattan_distance_cupy(X,Y))
            print(manhattan_distance_kernel(X,Y))

def test_dot(D):
    """
    Tests the dot product similarity calculation between two random vectors using different CPU, Cupy kernels and custom CUDA kernels.
    Args:
        D (list of int): A list of integers where each integer represents the dimensionality of the random vectors X and Y.
    Returns:
        None: This function prints the results of the similarity calculations and the time spent, but does not return any values.
    """
    for d in D:
        print(d)
        for i in range(0,3):
            np.random.seed(42)
            X = np.random.randn(d).astype(np.float32)
            Y = np.random.randn(d).astype(np.float32)
            print(dot_cpu(X,Y))
            print(dot_cupy(X,Y))
            print(dot_kernel(X,Y))

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

def test_knn(random = False):

    """
    Tests K-Nearest Neighbors algorithms using different distance metrics (Cosine, Dot, Manhattan, and L2) and different implementations (batching, chunks, CPU).
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
        None: This function prints the results of the KNN and the time spent by each implementatios, but does not return any values.
    """

    if random:
        N, D, A, X, K = testdata_knn("")
    else:
        N, D, A, X, K = testdata_knn("Data/test_data.json")
    
    print(N)
    print(D)

    print("COSINE")
    for i in range(0,3):
        print(i)
        print("BATCHING")
        print(our_knn_cupy_batching(N, D, A, X, K, "cosine"))
        print("CHUNKS")
        print(our_knn_chunks_cosine(N, D, A, X, K))
        print("CPU")
        print(our_knn_cpu(N, D, A, X, K, "cosine"))

    print("DOT")
    for i in range(0,3):
        print(i)
        print("BATCHING")
        print(our_knn_cupy_batching(N, D, A, X, K, "dot"))
        print("CHUNKS")
        print(our_knn_chunks_dot(N, D, A, X, K))
        print("CPU")
        print(our_knn_cpu(N, D, A, X, K, "dot"))

    print("MANHATTAN")
    for i in range(0,2):
        print(i)
        print("BATCHING")
        print(our_knn_cupy_batching(N, D, A, X, K, "manhattan"))
        print("CHUNKS")
        print(our_knn_chunks_manhattan(N, D, A, X, K))
        print("CPU")
        print(our_knn_cpu(N, D, A, X, K, "manhattan"))

    print("L2")
    for i in range(0,2):
        print(i)
        print("BATCHING")
        print(our_knn_cupy_batching(N, D, A, X, K, "l2"))
        print("CHUNKS")
        print(our_knn_chunks_l2(N, D, A, X, K))
        print("CPU")
        print(our_knn_cpu(N, D, A, X, K, "l2"))

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

    print("\nL2")
    for i in range(0, 2):
        print("Iteration: ", i)
        print("CPU (Numpy)")
        our_kmeans_numpy(N, D, A, K, distance_metric="l2")
        print("GPU (Native CuPy Functions)")
        our_kmeans_cupy_1(N, D, A, K, distance_metric="l2")
        print("GPU (Custom Cuda Kernels)\n")
        our_kmeans_cupy_2(N, D, A, K, distance_metric="l2")

    print("\nCOSINE")
    for i in range(0, 2):
        print("Iteration: ", i)
        print("CPU (Numpy)")
        our_kmeans_numpy(N, D, A, K, distance_metric="cosine")
        print("GPU (Native CuPy Functions)")
        our_kmeans_cupy_1(N, D, A, K, distance_metric="cosine")
        print("GPU (Custom Cuda Kernels)\n")
        our_kmeans_cupy_2(N, D, A, K, distance_metric="cosine")


#TODO: Add testing functions for dot distance. Add tests for ANN.

if __name__ == "__main__":
    test_knn(random=False)
    test_kmeans(random=False)

    test_manhattan(D = [2, 1024, 2**15,2**20])
    test_l2(D = [2, 1024, 2**15,2**20])
    test_cosine(D = [2, 1024, 2**15,2**20])