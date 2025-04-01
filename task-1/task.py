import torch
import cupy as cp
import triton
import numpy as np
import time
import json
import sys
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# L2
# ------------------------------------------------------------------------------------------------

# CPU version of L2 distance
def distance_l2_cpu(X, Y):
    # Start timer to measure execution time
    start_time = time.time()

    # Calculate the L2 distance between vectors X and Y by computing the sum of squared differences, then taking the square root
    dist = np.sqrt(np.sum((X - Y) ** 2))

    # End timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return dist

# GPU version of L2 distance calculation using Cupy
def distance_l2_cp(X, Y):
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

# Raw CUDA kernel code for calculating L2 distance between two vectors X and Y
l2_kernel_code = '''
extern "C" __global__
void calculate_diff_kernel(const float* X, const float* Y, float* diff, int D) {
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

# GPU version of L2 distance calculation using custom CUDA kernels
def distance_l2_kernel(X, Y):
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
# Manhattan
# ------------------------------------------------------------------------------------------------

# CPU version of Manhattan distance calculation using NumPy
def manhattan_distance_numpy(X, Y):
    # Start timer to measure execution time
    start_time = time.time()

    # Compute the Manhattan distance between vectors X and Y as the sum of absolute differences
    dist = np.sum(np.abs(X - Y))

    # End timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return dist

# GPU version of Manhattan distance calculation using CuPy
def manhattan_distance_cupy(X, Y):
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

    return dist

# CUDA kernel for Manhattan distance calculation
manhattan_kernel_optimized = cp.RawKernel(r'''
extern "C" __global__
void manhattan_distance_optimized(const float* X, const float* Y, float* result, int size) {
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
        result = sdata[0];  // Store the sum of this block
    }
}
''', 'manhattan_distance_optimized')

# Optimized GPU function using a custom CUDA kernel
def manhattan_distance_cupy_optimized(X, Y):
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
    manhattan_kernel_optimized(
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
# KNN
# ------------------------------------------------------------------------------------------------

def our_knn_cpu(N, D, A, X, K, distance = "l2"):
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

#---- FIRST DESIGN (BATCHING)----
def our_knn_batching(N, D, A, X, K, distance="l2"):
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

    # Select the K nearest neighbors globally
    if(num_batches > 1):
        top_k_global_indices = cp.argpartition(all_distances, K)[:K]
    else:
        top_k_global_indices = indices

    # Stop measuring the time when we have found the indices
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total Time: {execution_time:.6f} seconds")

    return all_indices[top_k_global_indices]

#---- SECOND DESIGN (BATCHING + CHUNK) -----
def our_knn_chunks_manhattan(N, D, A, X, K):
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
    else:
        top_k_global_indices = indices

    # Stop measuring the time when we have found the indices
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")

    return all_indices[top_k_global_indices]

def our_knn_chunks_cosine(N, D, A, X, K):
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

    # If multiple batches were processed, find the K nearest neighbors globally
    if num_batches > 1:
        top_k_global_indices = cp.argpartition(all_distances, K)[:K]
    else:
        top_k_global_indices = indices  # If only one batch, we can directly return the batch's results

    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")

    return all_indices[top_k_global_indices]

def our_knn_chunks_dot(N, D, A, X, K):
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
        top_k_global_indices = cp.argpartition(all_distances, -K)[-K:]
    else:
        top_k_global_indices = indices  # If only one batch, return its results directly

    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")

    return all_indices[top_k_global_indices]

def our_knn_chunks_l2(N, D, A, X, K):
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
    else:
        top_k_global_indices = indices  # If only one batch, return its results directly

    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Total Time: {execution_time:.6f} seconds")

    return all_indices[top_k_global_indices]

# ------------------------------------------------------------------------------------------------
# TOP K
# ------------------------------------------------------------------------------------------------
# CUDA Kernel for performing top-K selection using radix sort
kernel_code = '''
extern "C" __global__
void topK_radix_sort(const float* arr, int* indices, float* values, int N, int K) {
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
    - arr: Input array (Cupy array)
    - K: Number of top-K smallest elements to find
    - all_indices: (Optional) Used to map indices to original positions in a multi-step KNN search
    - final: (Optional) If True, map the indices to the original input dataset

    Returns:
    - values: The K smallest values found
    - indices: Their corresponding indices
    """

    # Load the CUDA kernel
    mod = cp.RawModule(code=kernel_code)
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
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_l2(D):
    for d in D:
        print(d)
        for i in range(0,3):
            np.random.seed(42)
            X = np.random.randn(d).astype(np.float32)
            Y = np.random.randn(d).astype(np.float32)
            print(distance_l2_cpu(X,Y))
            print(distance_l2_cp(X,Y))
            print(distance_l2_kernel(X,Y))

def test_manhattan(D):

    for d in D:
        print(d)
        for i in range(0,3):
            np.random.seed(42)
            X = np.random.randn(d).astype(np.float32)
            Y = np.random.randn(d).astype(np.float32)
            print(manhattan_distance_cupy(X,Y))
            print(manhattan_distance_numpy(X,Y))
            print(manhattan_distance_cupy_optimized(X,Y))

def test_knn():
    N, D, A, X, K = testdata_knn("Data/test_data.json")
    #N, D, A, X, K = testdata_knn("")
    sys.stdout.flush()
    print(N)
    sys.stdout.flush()
    print(D)
    sys.stdout.flush()


    for i in range(0,2):
        print(i)
        sys.stdout.flush()
        print("COSINE")
        print("DESIGN 1")
        sys.stdout.flush()
        print(our_knn_batching(N, D, A, X, K, "cosine"))
        sys.stdout.flush()
        print("DESIGN 2")
        sys.stdout.flush()
        print(our_knn_chunks_cosine(N, D, A, X, K))
        sys.stdout.flush()
        print("CPU")
        sys.stdout.flush()
        print(our_knn_cpu(N, D, A, X, K, "cosine"))

    for i in range(0,2):
        print(i)
        sys.stdout.flush()
        print("DOT")
        print("DESIGN 1")
        sys.stdout.flush()
        print(our_knn_batching(N, D, A, X, K, "dot"))
        sys.stdout.flush()
        print("DESIGN 2")
        sys.stdout.flush()
        print(our_knn_chunks_dot(N, D, A, X, K))
        sys.stdout.flush()
        print("CPU")
        sys.stdout.flush()
        print(our_knn_cpu(N, D, A, X, K, "dot"))

    for i in range(0,2):
        print(i)
        sys.stdout.flush()
        print("manhattan")
        print("DESIGN 1")
        sys.stdout.flush()
        print(our_knn_batching(N, D, A, X, K, "manhattan"))
        sys.stdout.flush()
        print("DESIGN 2")
        sys.stdout.flush()
        print(our_knn_chunks_manhattan(N, D, A, X, K))
        sys.stdout.flush()
        print("CPU")
        sys.stdout.flush()
        print(our_knn_cpu(N, D, A, X, K, "manhattan"))

    for i in range(0,2):
        print(i)
        sys.stdout.flush()
        print("l2")
        print("DESIGN 1")
        sys.stdout.flush()
        print(our_knn_batching(N, D, A, X, K, "l2"))
        sys.stdout.flush()
        print("DESIGN 2")
        sys.stdout.flush()
        print(our_knn_chunks_l2(N, D, A, X, K))
        sys.stdout.flush()
        print("CPU")
        sys.stdout.flush()
        print(our_knn_cpu(N, D, A, X, K, "l2"))


if __name__ == "__main__":
    test_knn()
    test_manhattan(D = [2, 1024, 2**15,2**20])
    test_l2(D = [2, 1024, 2**15,2**20])
