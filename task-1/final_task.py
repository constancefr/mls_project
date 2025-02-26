import torch
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def distance_manhattan(X, Y):
    return torch.sum(torch.abs(X - Y))

def distance_l2(X, Y):
    return (X-Y).norm(2)

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------


def our_knn(N, D, A, X, K):
    def f(Y):
        return distance_l2(X,Y)
    distance = torch.vmap(f)(A)
    _, indices = distance.sort()
    return indices[:K]

def our_knn_batching(N, D, X, K, batch_size=500000, use_gpu=False):
    indices_list = []
    for i in range(0, N, batch_size):

        # Generate A by using small batches
        A_batch = torch.rand((batch_size, D), dtype=torch.float32, device=device if use_gpu else "cpu")

        # Compute Manhattan distances
        distances = torch.sum(torch.abs(A_batch - X), dim=1)

        # Get K nearest neighbors
        _, batch_indices = torch.topk(distances, K, largest=False)

        indices_list.append(batch_indices)

    return torch.cat(indices_list)



# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K):
    indices = torch.randperm(N, device=A.device)[:K]
    centroids = A[indices]
    def f(Y):
        return torch.vmap(lambda x: distance_l2(x, Y))(centroids)
    while True:
        distances = torch.vmap(f)(A)

        # Assign each point to the nearest centroid
        assignments = torch.argmin(distances, dim=1)

        # Update centroids using vectorized operations
        mask = torch.zeros(N, K, dtype=A.dtype, device=A.device)
        mask[torch.arange(N, device=A.device), assignments] = 1.0
        counts = mask.sum(dim=0)  # Shape (K,)
        sums = torch.mm(mask.T, A)  # Shape (K, D)
        new_centroids = sums / counts.unsqueeze(1)

        # Check for centroid convergence
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
        if centroid_shift < 1e-4:
            break
        centroids = new_centroids
    return assignments, centroids

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_knn_filter(A,X,K):        
    def f(Y):
        return distance_l2(X,Y)
    distance = torch.vmap(f)(A)
    _, indices = distance.sort()
    return indices < K

def our_ann(N, D, A, X, K):

    assignments, means = our_kmeans(N, D, X, 5)
    nearest_means = our_knn_filter(means, X, 2)
    mask = nearest_means[assignments]
    filtered = A[mask]
    return our_knn(N, D, filtered, X, K)

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def benchmark_function(func, N, D, A, X, K, num_repeats=5):
    """
    Generic benchmarking function for CPU vs GPU.
    """
    # CPU Benchmark
    if(func == distance_manhattan):
        Y_np = np.random.rand(D).astype(np.float32)
        Y = torch.tensor(Y_np, device="cpu")
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(X["cpu"],Y)
        cpu_time = (time.time() - start_cpu) / num_repeats
    elif(func == our_knn):
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(N, D, A["cpu"], X["cpu"], K)
        cpu_time = (time.time() - start_cpu) / num_repeats
    elif(func == our_kmeans):
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(N, D, A["cpu"], K)
        cpu_time = (time.time() - start_cpu) / num_repeats
    elif(func == our_ann):
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(N, D, A["cpu"], X["cpu"], K)
        cpu_time = (time.time() - start_cpu) / num_repeats


    # GPU Benchmark
    gpu_time = None
    if torch.cuda.is_available():    
        if(func == distance_manhattan):
            Y_np = np.random.rand(D).astype(np.float32)
            Y = torch.tensor(Y_np, device=device)
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(X["gpu"], Y)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_gpu) / num_repeats
        elif(func == our_knn):
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(N, D, A["gpu"], X["gpu"], K)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_gpu) / num_repeats
        elif(func == our_kmeans):
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(N, D, A["gpu"], K)
            gpu_time = (time.time() - start_gpu) / num_repeats
        elif(func == our_ann):
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(N, D, A["gpu"], X["gpu"], K)
            gpu_time = (time.time() - start_gpu) / num_repeats

    speedup = cpu_time / gpu_time if gpu_time else None
    return cpu_time, gpu_time, speedup

def benchmark_knn_batching(func, N, D, X, K, num_repeats=5, batch_size=500000):
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

def run_benchmark(N, D, K):
    """
    Executes all benchmarks for a given dataset size and dimension.
    """
    # Generate random dataset and query vector
    A_np = np.random.rand(N, D).astype(np.float32)
    X_np = np.random.rand(D).astype(np.float32)

    # Convert to PyTorch tensors for CPU and GPU
    A = {"cpu": torch.tensor(A_np, device="cpu"), "gpu": torch.tensor(A_np, device=device)}
    X = {"cpu": torch.tensor(X_np, device="cpu"), "gpu": torch.tensor(X_np, device=device)}

    # Run benchmarks
    results = {
        "Manhattan Distance": benchmark_function(distance_manhattan,N, D, A, X, K),
        "KNN": benchmark_function(our_knn, N, D, A, X, K),
        "K-Means": benchmark_function(our_kmeans, N, D, A, X, K),
        "ANN": benchmark_function(our_ann, N, D, A, X, K)
    }
    recall_value = compute_recall(N, D, A["cpu"], X["cpu"], K)
    print(f"Recall ANN vs KNN: {recall_value:.2%}")

    return results

def run_benchmark_batching(N, D, K, batch_size=500000):
    """Ejecuta todos los benchmarks con procesamiento por lotes."""

    #Solo almacenamos `X`
    X_cpu = torch.rand((D,), dtype=torch.float32, device="cpu")
    X_gpu = X_cpu.to(device) if torch.cuda.is_available() else None

    #Run benchmarks sin almacenar `A`
    results = {
        "KNN": benchmark_knn_batching(our_knn_batching, N, D, {"cpu": X_cpu, "gpu": X_gpu}, K, batch_size=batch_size)
    }
    return results

def compute_recall(N, D, A, X, K):
    knn_indices = our_knn(N, D, A, X, K).tolist()
    ann_indices = our_ann(N, D, A, X, K).tolist()

    knn_set = set(knn_indices)
    ann_set = set(ann_indices)

    recall = len(knn_set & ann_set) / K  
    return recall


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimensions = [2, 1024, 32768]
    sizes = [4000, 4000000]
    K = 5

    print("\nBenchmark - CPU vs GPU Performance")
    print(" N       | D        | Function              | CPU Time (s)  | GPU Time (s)  | Speedup ")
    print("-" * 100)

    for D in dimensions:
        if(D <= K):
            K = D-1
        for N in sizes:
            if(N > 1000000):
                batch_size = 5000
                results = run_benchmark_batching(N, D, K, batch_size=batch_size)
            else:
                results = run_benchmark(N, D, K)

            for func_name, (cpu_time, gpu_time, speedup) in results.items():
                print(f"{N:<8} | {D:<8} | {func_name:<20} | {cpu_time:<12.6f} | {gpu_time if gpu_time else 'N/A':<12} | {speedup if speedup else 'N/A'}")

