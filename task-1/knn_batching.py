import torch
import cupy as cp
import time
import numpy as np

def knn_torch(N, D, X, K, batch_size=50, device=torch.device("cuda")):
    X = torch.tensor(X, dtype=torch.float32, device=device)  # Pasamos X a la GPU

    all_knn_distances = []
    all_knn_indices = []

    # Procesamos por batches sin cargar toda `A` en memoria
    for i in range(0, N, batch_size):
        batch_size_actual = min(batch_size, N - i)  # Ajustar si estamos en el último batch

        # Generamos el batch de A en cada iteración sin almacenarlo completamente
        batch_A = torch.rand(batch_size_actual, D, dtype=torch.float32, device=device)

        # Calculamos la distancia L2 (Euclidiana)
        distances = torch.sqrt(torch.sum((batch_A - X) ** 2, dim=1))

        # Obtenemos los K vecinos más cercanos dentro de este batch
        knn_distances, knn_indices = torch.topk(distances, K, largest=False)

        # Convertimos los índices locales a índices globales en A
        knn_indices += i

        # Guardamos los resultados de este batch
        all_knn_distances.append(knn_distances)
        all_knn_indices.append(knn_indices)

    # Concatenamos todas las distancias e índices obtenidos de los batches
    all_knn_distances = torch.cat(all_knn_distances)
    all_knn_indices = torch.cat(all_knn_indices)

    # Seleccionamos los K vecinos más cercanos globalmente
    _, final_indices = torch.topk(all_knn_distances, K, largest=False)

    return all_knn_indices[final_indices].cpu().numpy()  # Convertimos a numpy y devolvemos



def knn_cupy_simple_kernel(N, D, X, K, batch_size=50, device=torch.device("cuda")):
    X = cp.asarray(X, dtype=cp.float32)  # Convertimos X a CuPy

    all_knn_distances = []
    all_knn_indices = []

    # Procesamos A por batches sin almacenarlo completamente
    for i in range(0, N, batch_size):
        batch_size_actual = min(batch_size, N - i)  # Último batch puede ser menor

        # Generamos el batch de A en cada iteración
        batch_A = cp.random.rand(batch_size_actual, D, dtype=cp.float32)

        # Calculamos la distancia L2 (Euclidiana)
        distances = cp.linalg.norm(batch_A - X, axis=1)

        # Obtenemos los K vecinos más cercanos dentro de este batch
        knn_indices = cp.argsort(distances)[:K]  # Índices locales dentro del batch
        knn_distances = distances[knn_indices]

        # Convertimos los índices locales a índices globales en A
        knn_indices += i  

        # Guardamos los resultados de este batch
        all_knn_distances.append(knn_distances)
        all_knn_indices.append(knn_indices)

    # Concatenamos todas las distancias e índices
    all_knn_distances = cp.concatenate(all_knn_distances)
    all_knn_indices = cp.concatenate(all_knn_indices)

    # Seleccionamos los K vecinos más cercanos globalmente
    final_indices = cp.argsort(all_knn_distances)[:K]

    return all_knn_indices[final_indices]  # Devolvemos los índices globales correctos

def knn_cupy_optimized_kernel(N, D, X, K, batch_size=50, device=torch.device("cuda")):
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
    void knn_topk_kernel(float* distances, int* indices, int K, int N) {
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
    func_topk = mod.get_function('knn_topk_kernel')

    distances = cp.zeros(N, dtype=cp.float32)  # Array to store all distances
    indices = cp.zeros(N, dtype=cp.int32)  # To store indices globally
            for (int d = 0; d < D; d++) {
                float diff = A[idx * D + d] - X[d];
                sum += diff * diff;
            }
            distances[idx] = sqrtf(sum);  // L2 distance
        }
    for i in range(0, N, batch_size):
        batch_size_actual = min(batch_size, N - i)  # Adjust batch size for the last batch

        # Generate a batch of A in each iteration
        batch_A = cp.random.rand(batch_size_actual, D, dtype=cp.float32)

        # Distances for this batch
        batch_distances = cp.zeros(batch_size_actual, dtype=cp.float32)
        batch_indices = cp.zeros(K, dtype=cp.int32)  # Store K indices per batch

        # Configure block and grid sizes for the distance calculation kernel
        block_size = 256
            for (int d = 0; d < D; d++) {
                float diff = A[idx * D + d] - X[d];
                sum += diff * diff;
            }
            distances[idx] = sqrtf(sum);  // L2 distance
        }
        grid_size = (batch_A.shape[0] + block_size - 1) // block_size
        
        # Run the kernel to calculate the distances for the batch
        func_knn((grid_size,), (block_size,), (batch_A, X, batch_distances, batch_A.shape[0], D))
        
        # Run the second kernel to find the K nearest neighbors within this batch
        func_topk((1,), (block_size,), (batch_distances, batch_indices, K, batch_size_actual))

        # Store the K closest indices for this batch (with global indices)
        batch_indices += i  # Add global offset to indices
        all_indices.append(batch_indices)
        all_distances.append(batch_distances)

    # Concatenate all the indices and distances from the batches into one array
    all_indices = cp.concatenate(all_indices)
    all_distances = cp.concatenate(all_distances)

    # Apply the second kernel to find the K nearest neighbors globally (across all batches)
    final_indices = cp.zeros(K, dtype=cp.int32)
    func_topk((1,), (block_size,), (all_distances, final_indices, K, N))

    return final_indices  # Return the indices of the K closest neighbors globally


# Function to measure execution time (average over 10 executions)
def measure_time(func, *args, num_runs=1, batch_size=50, device=torch.device("cuda")):
    total_time = 0.0
    for _ in range(num_runs):
        start = time.time()
        func(*args, device = device, batch_size=batch_size)
        end = time.time()
        total_time += (end - start)
    average_time = total_time / num_runs
    return average_time

# Main function to compare performance
def compare_performance():
    # Create vectors of different sizes and dimensions
    D = [2, 1024, 2**15]  # N = Number of vectors, D = Dimension of vectors
    N = [4000, 4000000]  # Number of vectors
    K = 5  # Number of nearest neighbors to find
    batch_size = 50  # Batch size for processing in batches

    # Prepare a list to store the results
    results = []

    for d in D:
        # Create random vectors X (1 x D)
        X_cpu = torch.rand(d)
        X_gpu = cp.asarray(X_cpu.numpy())

        for n in N:
            print(f"\nComparing times for vectors of size {n}, dimension {d}:")

            # 1. Execution time using torch.topk (on GPU)
            torch_time_gpu = measure_time(knn_torch, n, d, X_cpu.numpy(), K, batch_size=batch_size)

            # 2. Execution time using CuPy simple kernel
            cupy_simple_kernel_time = measure_time(knn_cupy_simple_kernel, n, d, X_gpu, K, batch_size=batch_size)

            # 3. Execution time using CuPy optimized kernel
            cupy_optimized_kernel_time = measure_time(knn_cupy_optimized_kernel, n, d, X_gpu, K, batch_size=batch_size)
            #cupy_optimized_kernel_time = 1

            # 4. Execution time using torch (on CPU)
            torch_time_cpu = measure_time(knn_torch, n, d, X_cpu.numpy(), K, batch_size=batch_size, device=torch.device("cpu"))
            #torch_time_cpu = 1

            # Speedups
            speedup_torch_gpu = torch_time_cpu / torch_time_gpu if torch_time_gpu != 0 else float('inf')
            speedup_cupy_simple = torch_time_cpu / cupy_simple_kernel_time if cupy_simple_kernel_time != 0 else float('inf')
            speedup_cupy_raw_kernel = torch_time_cpu / cupy_optimized_kernel_time if cupy_optimized_kernel_time != 0 else float('inf')

            # Append results to the list
            results.append((n, d, 'Torch (CPU)', torch_time_cpu, 1))
            results.append((n, d, 'Torch (GPU)', torch_time_gpu, speedup_torch_gpu))
            results.append((n, d, 'Cupy (Simple Kernel)', cupy_simple_kernel_time, speedup_cupy_simple))
            results.append((n, d, 'Cupy (Optimized Kernel)', cupy_optimized_kernel_time, speedup_cupy_raw_kernel))

    # Print results in a table format
    print(f"\n{'Size':<12}{'Dimension':<12}{'Function Used':<25}{'Time Taken (s)':20}{'Speedup':<10}")
    print('-' * 50)
    for result in results:
        size, dimension, function_used, time_taken, speedup = result
        print(f"{size:<12}{dimension:<12}{function_used:<25}{time_taken:<20.6f}{speedup:<10.2f}")

# Run the comparison

torch.manual_seed(42)
cp.random.seed(42)
compare_performance()

#5k
"""
Size        Dimension   Function Used            Time Taken (s)      Speedup   
--------------------------------------------------
4000        2           Torch (CPU)              0.100571            1.00      
4000        2           Torch (GPU)              0.207590            0.48      
4000        2           Cupy (Simple Kernel)     0.550400            0.18      
4000        2           Cupy (Optimized Kernel)  0.016129            6.24      
4000000     2           Torch (CPU)              0.318912            1.00      
4000000     2           Torch (GPU)              0.168743            1.89      
4000000     2           Cupy (Simple Kernel)     0.545132            0.59      
4000000     2           Cupy (Optimized Kernel)  4.268028            0.07      
4000        1024        Torch (CPU)              0.092590            1.00      
4000        1024        Torch (GPU)              4.070247            0.02      
4000        1024        Cupy (Simple Kernel)     0.014208            6.52      
4000        1024        Cupy (Optimized Kernel)  0.000783            118.29    
4000000     1024        Torch (CPU)              77.099050           1.00      
4000000     1024        Torch (GPU)              0.701269            109.94    
4000000     1024        Cupy (Simple Kernel)     0.899115            85.75     
4000000     1024        Cupy (Optimized Kernel)  3.619627            21.30     
4000        32768       Torch (CPU)              2.718827            1.00      
4000        32768       Torch (GPU)              0.319704            8.50      
4000        32768       Cupy (Simple Kernel)     0.544685            4.99      
4000        32768       Cupy (Optimized Kernel)  0.017127            158.74    
4000000     32768       Torch (CPU)              3006.187364         1.00      
4000000     32768       Torch (GPU)              20.799752           144.53    
4000000     32768       Cupy (Simple Kernel)     20.847348           144.20    
4000000     32768       Cupy (Optimized Kernel)  23.233212           129.39 "
"""

#50
"""
Size        Dimension   Function Used            Time Taken (s)      Speedup   
--------------------------------------------------
4000        2           Torch (CPU)              0.023595            1.00      
4000        2           Torch (GPU)              1.209336            0.02      
4000        2           Cupy (Simple Kernel)     1.432323            0.02      
4000        2           Cupy (Optimized Kernel)  0.036963            0.64      
4000000     2           Torch (CPU)              6.153474            1.00      
4000000     2           Torch (GPU)              16.511610           0.37      
4000000     2           Cupy (Simple Kernel)     42.128921           0.15      
4000000     2           Cupy (Optimized Kernel)  15.820445           0.39      
4000        1024        Torch (CPU)              0.057419            1.00      
4000        1024        Torch (GPU)              0.018997            3.02      
4000        1024        Cupy (Simple Kernel)     0.057195            1.00      
4000        1024        Cupy (Optimized Kernel)  0.016116            3.56      
4000000     1024        Torch (CPU)              57.473203           1.00      
4000000     1024        Torch (GPU)              16.714437           3.44      
4000000     1024        Cupy (Simple Kernel)     39.609991           1.45      
4000000     1024        Cupy (Optimized Kernel)  16.001380           3.59      
4000        32768       Torch (CPU)              2.186296            1.00      
4000        32768       Torch (GPU)              0.027029            80.89     
4000        32768       Cupy (Simple Kernel)     0.061870            35.34     
4000        32768       Cupy (Optimized Kernel)  0.016227            134.74    
4000000     32768       Torch (CPU)              1764.108840         1.00      
4000000     32768       Torch (GPU)              25.467585           69.27     
4000000     32768       Cupy (Simple Kernel)     42.682925           41.33     
4000000     32768       Cupy (Optimized Kernel)  159.666184          11.05  """