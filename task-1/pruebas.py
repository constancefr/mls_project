import torch
import numpy as np
import time

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================
# K-Nearest Neighbors (KNN) Optimized with Batching
# ================================================
def our_knn(N, D, X, K, batch_size=500000, use_gpu=False):
    indices_list = []
    for i in range(0, N, batch_size):
        batch_size = min(batch_size, N - i)

        #Generamos `A_batch` dinámicamente, sin cargar todo `A`
        A_batch = torch.rand((batch_size, D), dtype=torch.float32, device=device if use_gpu else "cpu")

        #Computamos distancias Manhattan
        distances = torch.sum(torch.abs(A_batch - X), dim=1)

        # Obtenemos los índices de los `K` vecinos más cercanos
        _, batch_indices = torch.topk(distances, K, largest=False)

        indices_list.append(batch_indices)

    return torch.cat(indices_list)


# ================================================
# Benchmark Functions
# ================================================
def benchmark_function(func, N, D, X, K, num_repeats=5, batch_size=500000):
    """Generic benchmarking function for CPU vs GPU using batch processing."""

    #CPU Benchmark
    start_cpu = time.time()
    for _ in range(num_repeats):
        func(N, D, X["cpu"], K, batch_size, use_gpu=False)
    cpu_time = (time.time() - start_cpu) / num_repeats

    #GPU Benchmark
    gpu_time = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_gpu = time.time()
        for _ in range(num_repeats):
            func(N, D, X["gpu"], K, batch_size, use_gpu=True)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_gpu) / num_repeats

    speedup = cpu_time / gpu_time if gpu_time else None
    return cpu_time, gpu_time, speedup


def run_benchmark(N, D, K, batch_size=500000):
    """Ejecuta todos los benchmarks con procesamiento por lotes."""

    #Solo almacenamos `X`
    X_cpu = torch.rand((D,), dtype=torch.float32, device="cpu")
    X_gpu = X_cpu.to(device) if torch.cuda.is_available() else None

    #Run benchmarks sin almacenar `A`
    results = {
        "KNN": benchmark_function(our_knn, N, D, {"cpu": X_cpu, "gpu": X_gpu}, K, batch_size=batch_size)
    }
    return results


# ================================================
#Execute Benchmarks
# ================================================
if __name__ == "__main__":
    dimensions = [10, 1024, 32768]
    sizes = [4000, 4000000]
    K = 5

    print("\nBenchmark - CPU vs GPU Performance")
    print(" N       | D        | Function              | CPU Time (s)  | GPU Time (s)  | Speedup ")
    print("-" * 100)

    for D in dimensions:
        for N in sizes:
            batch_size = 5000  # Procesamos en lotes
            results = run_benchmark(N, D, K, batch_size=batch_size)

            for func_name, (cpu_time, gpu_time, speedup) in results.items():
                print(f"{N:<8} | {D:<8} | {func_name:<20} | {cpu_time:<12.6f} | {gpu_time if gpu_time else 'N/A':<12} | {speedup if speedup else 'N/A'}")
