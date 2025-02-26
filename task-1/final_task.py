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

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------


def our_knn(N, D, A, X, K):
    A = A.to(device)
    X = X.to(device)

    indices_list = []
    distances = torch.sum(torch.abs(A - X), dim=1)  # Compute Manhattan distances
    _, indices = torch.topk(distances, K, largest=False)  # Get K nearest neighbors

    return indices

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

def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4):
    A = A.to(device)

    # Initialize centroids
    indices = torch.randperm(N, device=device)[:K]
    centroids = A[indices].clone()

    for _ in range(max_iters):
        cluster_ids = torch.empty(N, device=device, dtype=torch.long)

        distances = torch.cdist(A, centroids, p=2)
        cluster_ids = torch.argmin(distances, dim=1)

        # Compute new centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device)

        for i in range(K):
            mask = (cluster_ids == i)
            if mask.any():
                new_centroids[i] = A[mask].mean(dim=0)
                counts[i] = mask.sum()
            else:
                new_centroids[i] = A[torch.randint(0, N, (1,), device=device)].clone()

        if torch.norm(centroids - new_centroids) < tol:
            break

        centroids = new_centroids

    return cluster_ids

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K, K1=5, K2=10):
    if(D <= K1):
        K1 = D-1
    if(D <= K2):
        K2 = D-1
    cluster_ids = our_kmeans(N, D, A, K1)

    # Find closest clusters to X
    cluster_distances = torch.cdist(X.view(1, -1), A, p=2)
    nearest_clusters = torch.topk(cluster_distances, K1, largest=False).indices

    # Find K2 nearest neighbors within those clusters
    filtered_vectors = A[nearest_clusters.view(-1)]
    K2 = min(K2, len(filtered_vectors)) 
    knn_indices = our_knn(len(filtered_vectors), D, filtered_vectors, X, K2)

    return knn_indices[:K]

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

