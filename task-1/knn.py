import torch
import cupy as cp
from cupyx import jit
import time

# 1. Using torch (topK)
def knn_torch(N, D, A, X, K, device=torch.device("cuda")):
    A = torch.tensor(A, dtype=torch.float32, device=device)
    X = torch.tensor(X, dtype=torch.float32, device=device)

    # Calculate the L2 distance (Euclidean distance)
    distances = torch.sqrt(torch.sum((A - X) ** 2, axis=1))

    # Get the K nearest neighbors using topK
    _, indices = torch.topk(distances, K, largest=False)

    return indices.cpu().numpy()

# 2. Using a simple CuPy kernel
def knn_cupy_simple_kernel(N, D, A, X, K, device=torch.device("cuda")):
    A = cp.asarray(A)
    X = cp.asarray(X)

    distances = cp.linalg.norm(A - X, axis=1)

    # Get the K nearest neighbors
    return cp.argsort(distances)[:K]


def knn_cupy_naive_topK(N, D, A, X, K, device=torch.device("cuda")):
    kernel_code = '''
    extern "C" __global__
    void knn_kernel(const float* A, const float* X, float* distances, int N, int D) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = A[idx * D + d] - X[d];
                sum += diff * diff;
            }
            distances[idx] = sqrtf(sum);  // L2 distance
        }
    }
    
    extern "C" __global__
    void topK_naive_kernel(float* distances, int* indices, int K, int N) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < N) {
            for (int j = 0; j < K; j++) {
                for (int k = j + 1; k < N; k++) {
                    if (distances[k] < distances[j]) {
                        // Swap distances
                        float temp = distances[k];
                        distances[k] = distances[j];
                        distances[j] = temp;

                        // Swap indices
                        int temp_idx = indices[k];
                        indices[k] = indices[j];
                        indices[j] = temp_idx;
                    }
                }
            }
        }
    }
    '''
    
    mod = cp.RawModule(code=kernel_code)
    func_knn = mod.get_function('knn_kernel')
    func_topk_naive = mod.get_function('topK_naive_kernel')

    distances = cp.zeros(N, dtype=cp.float32)
    indices = cp.arange(N, dtype=cp.int32)  # Índices originales

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    func_knn((grid_size,), (block_size,), (A, X, distances, N, D))

    # Aplicamos el kernel naive para Top-K
    func_topk_naive((grid_size,), (block_size,), (distances, indices, K, N))

    return indices[:K]


# 3. Using an optimized CuPy kernel
def knn_cupy_optimized_kernel(N, D, A, X, K, device=torch.device("cuda")):
    kernel_code = '''
    extern "C" __global__
    void knn_kernel(const float* A, const float* X, float* distances, int N, int D) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = A[idx * D + d] - X[d];
                sum += diff * diff;
            }
            distances[idx] = sqrtf(sum);  // L2 distance
        }
    }
    extern "C" __global__
    void knn_topk_kernel(const float* distances, int* indices, int K, int N) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (tid == 0) {  // Solo un hilo realiza la selección
            float* temp_dist = new float[N];  // Copy of the distances
            for (int i = 0; i < N; i++) {
                temp_dist[i] = distances[i];
            }

            for (int k = 0; k < K; k++) {
                float min_dist = 1e10f;
                int min_idx = -1;

                // Buscar el mínimo
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
    
    mod = cp.RawModule(code=kernel_code)
    func_knn = mod.get_function('knn_kernel')
    func_topk = mod.get_function('knn_topk_kernel')

    # Step 1: Calculate the distances using the first kernel
    distances = cp.zeros(N, dtype=cp.float32)
    indices = cp.zeros(K, dtype=cp.int32)  # Store K nearest indices

    # Configure block and grid sizes for the distance calculation kernel
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    
    # Run the kernel to calculate the distances
    func_knn((grid_size,), (block_size,), (A, X, distances, N, D))

    # Step 2: Apply the second kernel to find the K nearest neighbors by distance
    grid_size_topk = (K + block_size - 1) // block_size
    func_topk((grid_size_topk,), (block_size,), (distances, indices, K, N))

    return indices  # Return the indices of the K closest neighbors

# Function to measure execution time (average over 10 executions)
def measure_time(func, *args, num_runs=10, device=torch.device("cuda")):
    total_time = 0.0
    for _ in range(num_runs):
        start = time.time()
        func(*args, device = device)
        end = time.time()
        total_time += (end - start)
    average_time = total_time / num_runs
    return average_time

# Main function to compare performance
def compare_performance():
    # Create vectors of different sizes and dimensions
    D = [2, 1024, 2**15]  # N = Number of vectors, D = Dimension of vectors
    N = [4000]
    K = 5  # Number of nearest neighbors to find

    # Prepare a list to store the results
    results = []

    for d in D:
        X_cpu = torch.rand(d)
        X_gpu = cp.asarray(X_cpu.numpy())

        for n in N:
            print(f"\nComparing times for vectors of size {n}:")

            A_cpu = torch.rand(n, d)
            A_gpu = cp.asarray(A_cpu.numpy())

            # 1. Execution time using torch.topk (on GPU)
            torch_time_gpu = measure_time(knn_torch, n, d, A_cpu.numpy(), X_cpu.numpy(), K)

            # 2. Execution time using CuPy simple kernel
            cupy_simple_kernel_time = measure_time(knn_cupy_simple_kernel, n, d, A_gpu, X_gpu, K)

            # 3. Execution time using CuPy optimized kernel
            cupy_optimized_kernel_time = measure_time(knn_cupy_optimized_kernel, n, d, A_gpu, X_gpu, K)

            cupy_optimized_naive_kernel_time = measure_time(knn_cupy_naive_topK, n, d, A_gpu, X_gpu, K)


            # 4. Execution time using torch (on CPU)
            torch_time_cpu = measure_time(knn_torch, n, d, A_cpu.numpy(), X_cpu.numpy(), K, device=torch.device("cpu"))

            #Speedups
            speedup_torch_gpu = torch_time_cpu / torch_time_gpu if torch_time_gpu != 0 else float('inf')
            speedup_cupy_simple = torch_time_cpu / cupy_simple_kernel_time if cupy_simple_kernel_time != 0 else float('inf')
            speedup_cupy_raw_kernel = torch_time_cpu / cupy_optimized_kernel_time if cupy_optimized_kernel_time != 0 else float('inf')
            speedup_cupy_naive_kernel = torch_time_cpu / cupy_optimized_naive_kernel_time if cupy_optimized_kernel_time != 0 else float('inf')



            # Append results to the list
            results.append((n,d, 'Torch (CPU)', torch_time_cpu, 1))
            results.append((n,d, 'Torch (GPU)', torch_time_gpu, speedup_torch_gpu))
            results.append((n,d, 'Cupy (Simple Kernel)', cupy_simple_kernel_time, speedup_cupy_simple))
            results.append((n,d, 'Cupy (Optimized Kernel)', cupy_optimized_kernel_time, speedup_cupy_raw_kernel))
            results.append((n,d, 'Cupy (Naive Kernel)', cupy_optimized_naive_kernel_time, speedup_cupy_naive_kernel))



    # Print results in a table format
    print(f"\n{'Size':<12}{'Dimension':<12}{'Function Used':<25}{'Time Taken (s)':20}{'Speedup':<10}")
    print('-' * 50)
    for result in results:
        size, dimension, function_used, time_taken, speedup = result
        print(f"{size:<12}{dimension:<12}{function_used:<25}{time_taken:<20.6f}{speedup:<10.2f}")

# Run the comparison
torch.manual_seed(42)
compare_performance()

"""
Size        Dimension   Function Used            Time Taken (s)      Speedup   
--------------------------------------------------
4000        2           Torch (CPU)              0.003425            1.00      
4000        2           Torch (GPU)              0.097266            0.04      
4000        2           Cupy (Simple Kernel)     0.119963            0.03      
4000        2           Cupy (Optimized Kernel)  0.001328            2.58      
4000        2           Cupy (Naive Kernel)      0.001728            1.98      
4000        1024        Torch (CPU)              0.023574            1.00      
4000        1024        Torch (GPU)              0.003630            6.49      
4000        1024        Cupy (Simple Kernel)     0.001788            13.18     
4000        1024        Cupy (Optimized Kernel)  0.000109            216.93    
4000        1024        Cupy (Naive Kernel)      0.000191            123.33    
4000        32768       Torch (CPU)              1.478767            1.00      
4000        32768       Torch (GPU)              0.159718            9.26      
4000        32768       Cupy (Simple Kernel)     0.011554            127.99    
4000        32768       Cupy (Optimized Kernel)  0.000106            13928.59  
4000        32768       Cupy (Naive Kernel)      0.000177            8339.92   
"""