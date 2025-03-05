import torch
import cupy as cp
import time

# 1. Using torch (on GPU)
def distance_manhattan_torch(X, Y, device=torch.device("cuda")):
    X = X.to(device)
    Y = Y.to(device)
    return torch.sum(torch.abs(X - Y), dim=-1)

# 2. Using an element-wise kernel in cupy
def distance_manhattan_cupy_elementwise(X, Y, device=torch.device("cuda")):
    return cp.sum(cp.abs(X - Y), axis=-1)

# 3. Using a raw kernel in cupy
def distance_manhattan_cupy_raw_kernel(X, Y, device=torch.device("cuda")):
    kernel_code = '''
    extern "C" __global__
    void manhattan_kernel(const float* X, const float* Y, float* result, int N) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) {
            result[idx] = fabsf(X[idx] - Y[idx]);
        }
    }
    '''
    mod = cp.RawModule(code=kernel_code)
    func = mod.get_function('manhattan_kernel')

    N = X.size
    result = cp.zeros_like(X)

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    func((grid_size,), (block_size,), (X, Y, result, N))

    return cp.sum(result)



# Function to measure execution time (average over 10 executions)
def measure_time(func, *args, device=torch.device("cuda"), num_runs=10):
    total_time = 0.0
    for _ in range(num_runs):
        start = time.time()
        print(func(*args, device=device))
        end = time.time()
        total_time += (end - start)
    average_time = total_time / num_runs
    return average_time

# Main function to compare performance
def compare_performance():
    # Create vectors of different sizes
    sizes = [2, 2**15, 2**20]

    # Prepare a list to store the results
    results = []

    for size in sizes:
        print(f"\nComparing times for vectors of size {size}:")

        # Create the input vectors
        X_cpu = torch.rand(size)
        Y_cpu = torch.rand(size)
        X_gpu = cp.asarray(X_cpu.numpy())
        Y_gpu = cp.asarray(Y_cpu.numpy())

        # 1. Execution time using torch on GPU
        torch_time = measure_time(distance_manhattan_torch, X_cpu, Y_cpu)

        # 2. Execution time using cupy (element-wise)
        cupy_elementwise_time = measure_time(distance_manhattan_cupy_elementwise, X_gpu, Y_gpu)

        # 3. Execution time using raw kernel in cupy
        cupy_raw_kernel_time = measure_time(distance_manhattan_cupy_raw_kernel, X_gpu, Y_gpu)


        # Execution time on CPU (for the torch version on CPU)
        torch_cpu_time = measure_time(distance_manhattan_torch, X_cpu, Y_cpu, device=torch.device("cpu"))

        # Speedup calculation
        speedup_torch_gpu = torch_cpu_time / torch_time if torch_time != 0 else float('inf')
        speedup_cupy_elementwise = torch_cpu_time / cupy_elementwise_time if cupy_elementwise_time != 0 else float('inf')
        speedup_cupy_raw_kernel = torch_cpu_time / cupy_raw_kernel_time if cupy_raw_kernel_time != 0 else float('inf')

        # Append results to the list
        results.append((size, 'Torch (CPU)', torch_cpu_time, 1.0))  # CPU has speedup of 1.0
        results.append((size, 'Torch (GPU)', torch_time, speedup_torch_gpu))
        results.append((size, 'Cupy (Element-wise)', cupy_elementwise_time, speedup_cupy_elementwise))
        results.append((size, 'Cupy (Raw Kernel)', cupy_raw_kernel_time, speedup_cupy_raw_kernel))

    # Print results in a table format
    print(f"\n{'Dimension':<12}{'Function Used':<25}{'Time Taken (s)':<20}{'Speedup':<10}")
    print('-' * 70)
    for result in results:
        dimension, function_used, time_taken, speedup = result
        print(f"{dimension:<12}{function_used:<25}{time_taken:<20.6f}{speedup:<10.2f}")

# Run the comparison
torch.manual_seed(42)
compare_performance()


"""
Dimension   Function Used            Time Taken (s)      Speedup   
----------------------------------------------------------------------
2           Torch (CPU)              0.000068            1.00      
2           Torch (GPU)              0.012224            0.01      
2           Cupy (Element-wise)      0.017121            0.00      
2           Cupy (Raw Kernel)        0.000633            0.11      
32768       Torch (CPU)              0.000131            1.00      
32768       Torch (GPU)              0.000281            0.47      
32768       Cupy (Element-wise)      0.000199            0.66      
32768       Cupy (Raw Kernel)        0.000178            0.74      
1048576     Torch (CPU)              0.003142            1.00      
1048576     Torch (GPU)              0.001483            2.12      
1048576     Cupy (Element-wise)      0.000261            12.03     
1048576     Cupy (Raw Kernel)        0.000162            19.40    
"""