import torch
import numpy as np
import time

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================================
# Manhattan Distance Function
# ================================================


# ================================================
# K-Nearest Neighbors (KNN or TopK)
# ================================================
def our_knn(N, D, A, X, K):
    A = A.to(device)
    X = X.to(device)

    indices_list = []
    distances = torch.sum(torch.abs(A - X), dim=1)  # Compute Manhattan distances
    _, indices = torch.topk(distances, K, largest=False)  # Get K nearest neighbors

    return indices

# ================================================
# K-Means Clustering Optimized
# ================================================
def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4):
    A = A.to(device)
    K = min(K, N // 1000)  # Prevent excessive clusters

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

# ================================================
# Approximate Nearest Neighbors (ANN)
# ================================================
def our_ann(N, D, A, X, K, K1=5, K2=10):
    cluster_ids = our_kmeans(N, D, A, K1)

    # Find closest clusters to X
    cluster_distances = torch.cdist(X.view(1, -1), A, p=2)
    nearest_clusters = torch.topk(cluster_distances, K1, largest=False).indices

    # Find K2 nearest neighbors within those clusters
    filtered_vectors = A[nearest_clusters.view(-1)]
    knn_indices = our_knn(len(filtered_vectors), D, filtered_vectors, X, K2)

    return knn_indices[:K]

# ================================================
# Benchmark Functions
# ================================================
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
        "K-Means": benchmark_function(our_kmeans,N, D, A, X, K)
    }
    return results

# ================================================
# Execute Benchmarks
# ================================================
if __name__ == "__main__":
    dimensions = [10, 1024, 32768]
    sizes = [4000]
    K = 5

    print("\nBenchmark - CPU vs GPU Performance")
    print(" N       | D        | Function              | CPU Time (s)  | GPU Time (s)  | Speedup ")
    print("-" * 100)

    for D in dimensions:
        for N in sizes:
            results = run_benchmark(N, D, K)

            for func_name, (cpu_time, gpu_time, speedup) in results.items():
                print(f"{N:<8} | {D:<8} | {func_name:<20} | {cpu_time:<12.6f} | {gpu_time if gpu_time else 'N/A':<12} | {speedup if speedup else 'N/A'}")
