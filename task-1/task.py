import torch
#import cupy as cp
#import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# Task 1.1: Distance Functions (GPU implementations)
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    """
    Cosine distance:
       d(X, Y) = 1 - (X ⋅ Y) / (||X|| ||Y||)
    """
    dot = torch.dot(X, Y)
    normX = torch.norm(X)
    normY = torch.norm(Y)
    # Add a small epsilon to avoid division by zero
    return 1 - (dot / (normX * normY + 1e-8))

def distance_l2(X, Y):
    """
    L2 distance:
       d(X, Y) = sqrt(sum((X_i - Y_i)^2))
    """
    diff = X - Y
    diff_squared = diff ** 2
    sum_diff_squared = torch.sum(diff_squared)
    l2_distance = torch.sqrt(sum_diff_squared)
    return l2_distance

def distance_dot(X, Y):
    """
    Dot-product:
       d(X, Y) = X ⋅ Y
    """
    return torch.dot(X, Y)

def distance_manhattan(X, Y):
    """
    Manhattan (L1) distance:
       d(X, Y) = sum(|X_i - Y_i|)
    """
    return torch.sum(torch.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Task 1.2: Top-K with GPU (KNN search without torch.topk)
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K):
    """
    Custom KNN search using torch.sort to obtain the K nearest neighbors (using L2 distance).
    """
    distances = torch.sqrt(torch.sum((A - X) ** 2, dim=1))
    sorted_distances, sorted_indices = torch.sort(distances)
    return sorted_indices[:K]

# (Optional) Baseline KNN using torch.topk (for performance comparison)
def baseline_knn(N, D, A, X, K):
    """
    Baseline KNN that uses torch.topk.
    Note: Since torch.topk returns the largest values by default, we invert distances.
    """
    distances = torch.sqrt(torch.sum((A - X) ** 2, dim=1))
    neg_distances = -distances
    values, indices = torch.topk(neg_distances, K)
    return indices

# ------------------------------------------------------------------------------------------------
# Task 2.1: KMeans clustering on the GPU
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, max_iter=100, tol=1e-4):
    """
    KMeans clustering.
    """
    indices = torch.randperm(N)[:K]
    centroids = A[indices].clone()

    for i in range(max_iter):
        A_norm = torch.sum(A * A, dim=1, keepdim=True)
        centroids_norm = torch.sum(centroids * centroids, dim=1)
        distances = A_norm + centroids_norm - 2 * torch.matmul(A, centroids.t())
        cluster_assignments = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(centroids)
        for j in range(K):
            mask = (cluster_assignments == j)
            if torch.sum(mask) > 0:
                new_centroids[j] = torch.mean(A[mask], dim=0)
            else:
                new_centroids[j] = A[torch.randint(0, N, (1,))]
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
        centroids = new_centroids.clone()
        if centroid_shift < tol:
            break
    return cluster_assignments

# ------------------------------------------------------------------------------------------------
# Task 2.2: Approximate Nearest Neighbors (ANN)
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K, num_clusters=8, K1=3):
    """
    ANN search using a clustering-based approach.
    """
    # Cluster the data into num_clusters clusters
    cluster_assignments = our_kmeans(N, D, A, num_clusters)
    
    # Compute centroids for each cluster
    centroids = torch.zeros((num_clusters, D), device=A.device, dtype=A.dtype)
    for j in range(num_clusters):
        mask = (cluster_assignments == j)
        if torch.sum(mask) > 0:
            centroids[j] = torch.mean(A[mask], dim=0)
        else:
            centroids[j] = A[torch.randint(0, N, (1,))]
    
    # Compute distances from query X to each centroid and select nearest K1 clusters
    centroid_distances = torch.norm(centroids - X, dim=1)
    _, nearest_centroid_idx = torch.sort(centroid_distances)
    nearest_centroid_idx = nearest_centroid_idx[:K1]
    
    # Gather candidate indices from the selected clusters
    candidate_mask = torch.zeros(N, dtype=torch.bool, device=A.device)
    for idx in nearest_centroid_idx:
        candidate_mask |= (cluster_assignments == idx)
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze()
    
    if candidate_indices.numel() == 0:
        candidate_indices = torch.arange(N, device=A.device)
    
    # Compute distances from X to candidate vectors and select the top K
    candidates = A[candidate_indices]
    candidate_distances = torch.norm(candidates - X, dim=1)
    _, sorted_order = torch.sort(candidate_distances)
    top_candidates = candidate_indices[sorted_order][:K]
    return top_candidates

# ------------------------------------------------------------------------------------------------
# Utility function: Recall Rate
# ------------------------------------------------------------------------------------------------

def recall_rate(list1, list2):
    """
    Calculate the recall rate between two lists of K indices.
    Recall rate = (# common elements) / K
    """
    return len(set(list1) & set(list2)) / len(list1)

# ------------------------------------------------------------------------------------------------
# Test functions
# ------------------------------------------------------------------------------------------------

def test_kmeans():
    N, D, A, K = testdata_kmeans("")
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    kmeans_result = our_kmeans(N, D, A, K)
    print("KMeans cluster assignments:")
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("")
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    knn_result = our_knn(N, D, A, X, K)
    print("KNN result (our_knn):")
    print(knn_result)

def test_ann():
    N, D, A, X, K = testdata_ann("")
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    ann_result = our_ann(N, D, A, X, K)
    print("ANN result (our_ann):")
    print(ann_result)

def test_recall_rate_knn():
    """
    Compute and print the recall rate between our_knn and baseline_knn.
    """
    N, D, A, X, K = testdata_knn("")
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    result_our = our_knn(N, D, A, X, K)
    result_baseline = baseline_knn(N, D, A, X, K)
    recall = recall_rate(result_our.tolist(), result_baseline.tolist())
    print(f"Recall rate (our_knn vs baseline_knn): {recall:.2f}")

def test_recall_rate_ann():
    """
    Compute and print the recall rate between baseline_knn and our_ann.
    """
    # For this test, we use testdata_ann.
    N, D, A, X, K = testdata_ann("")
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    result_baseline = baseline_knn(N, D, A, X, K)
    result_ann = our_ann(N, D, A, X, K)
    recall = recall_rate(result_baseline.tolist(), result_ann.tolist())
    print(f"Recall rate (baseline_knn vs our_ann): {recall:.2f}")

# ------------------------------------------------------------------------------------------------
# Benchmarking routines for CPU vs GPU performance
# ------------------------------------------------------------------------------------------------

def benchmark_distance_functions():
    functions = [
        ("distance_l2", distance_l2),
        ("distance_cosine", distance_cosine),
        ("distance_dot", distance_dot),
        ("distance_manhattan", distance_manhattan)
    ]
    dims = [2, 2**15]
    iterations = 1000
    results = []
    
    for D in dims:
        X_cpu = torch.randn(D)
        Y_cpu = torch.randn(D)
        if torch.cuda.is_available():
            X_gpu = X_cpu.cuda()
            Y_gpu = Y_cpu.cuda()
        else:
            X_gpu, Y_gpu = X_cpu, Y_cpu
        
        for fname, func in functions:
            start = time.time()
            for _ in range(iterations):
                _ = func(X_cpu, Y_cpu)
            cpu_time = (time.time() - start) / iterations
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iterations):
                    _ = func(X_gpu, Y_gpu)
                torch.cuda.synchronize()
                gpu_time = (time.time() - start) / iterations
            else:
                gpu_time = float('nan')
            
            speedup = cpu_time / gpu_time if torch.cuda.is_available() else float('nan')
            results.append((1, D, fname, cpu_time, gpu_time, speedup))
    
    header = " N       | D        | Function              | CPU Time (s)  | GPU Time (s)  | Speedup "
    print("\nBenchmark - CPU vs GPU Performance (Distance Functions)")
    print(header)
    print("-" * len(header))
    for row in results:
        N_val, D_val, fname, cpu_time, gpu_time, speedup = row
        if torch.cuda.is_available():
            print(f"{N_val:8d} | {D_val:8d} | {fname:20s} | {cpu_time:12.6f} | {gpu_time:12.6f} | {speedup:7.2f}")
        else:
            print(f"{N_val:8d} | {D_val:8d} | {fname:20s} | {cpu_time:12.6f} | {'N/A':12s} | {'N/A':7s}")

def benchmark_knn_comparison(N, D, iterations=100, K=10):
    """
    Benchmark both our_knn and baseline_knn on CPU and GPU.
    """
    A_cpu = torch.randn(N, D)
    X_cpu = torch.randn(D)
    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
        X_gpu = X_cpu.cuda()
    else:
        A_gpu, X_gpu = A_cpu, X_cpu

    start = time.time()
    for _ in range(iterations):
        _ = our_knn(N, D, A_cpu, X_cpu, K)
    our_cpu_time = (time.time() - start) / iterations

    start = time.time()
    for _ in range(iterations):
        _ = baseline_knn(N, D, A_cpu, X_cpu, K)
    baseline_cpu_time = (time.time() - start) / iterations

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = our_knn(N, D, A_gpu, X_gpu, K)
        torch.cuda.synchronize()
        our_gpu_time = (time.time() - start) / iterations
    else:
        our_gpu_time = float('nan')

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = baseline_knn(N, D, A_gpu, X_gpu, K)
        torch.cuda.synchronize()
        baseline_gpu_time = (time.time() - start) / iterations
    else:
        baseline_gpu_time = float('nan')

    print(f"\nKNN Comparison Benchmark (N={N}, D={D}, K={K}):")
    print(f"{'Version':12s} | {'CPU Time (s)':>12s} | {'GPU Time (s)':>12s} | {'Speedup (CPU/GPU)':>18s}")
    if torch.cuda.is_available():
        our_speedup = our_cpu_time / our_gpu_time
        baseline_speedup = baseline_cpu_time / baseline_gpu_time
        print(f"{'our_knn':12s} | {our_cpu_time:12.6f} | {our_gpu_time:12.6f} | {our_speedup:18.2f}")
        print(f"{'baseline_knn':12s} | {baseline_cpu_time:12.6f} | {baseline_gpu_time:12.6f} | {baseline_speedup:18.2f}")
    else:
        print(f"{'our_knn':12s} | {our_cpu_time:12.6f} | {'N/A':12s} | {'N/A':>18s}")
        print(f"{'baseline_knn':12s} | {baseline_cpu_time:12.6f} | {'N/A':12s} | {'N/A':>18s}")

def benchmark_kmeans(N, D, iterations=10, clusters=8):
    A_cpu = torch.randn(N, D)
    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
    else:
        A_gpu = A_cpu

    start = time.time()
    for _ in range(iterations):
        _ = our_kmeans(N, D, A_cpu, clusters)
    cpu_time = (time.time() - start) / iterations

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = our_kmeans(N, D, A_gpu, clusters)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / iterations
    else:
        gpu_time = float('nan')

    speedup = cpu_time / gpu_time if torch.cuda.is_available() else float('nan')
    print(f"\nKMeans Benchmark (N={N}, D={D}): CPU: {cpu_time:.6f}s, GPU: {gpu_time:.6f}s, Speedup: {speedup:.2f}")

def benchmark_ann(N, D, iterations=10, topK=10):
    A_cpu = torch.randn(N, D)
    X_cpu = torch.randn(D)
    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
        X_gpu = X_cpu.cuda()
    else:
        A_gpu, X_gpu = A_cpu, X_cpu

    start = time.time()
    for _ in range(iterations):
        _ = our_ann(N, D, A_cpu, X_cpu, topK)
    cpu_time = (time.time() - start) / iterations

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = our_ann(N, D, A_gpu, X_gpu, topK)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / iterations
    else:
        gpu_time = float('nan')

    speedup = cpu_time / gpu_time if torch.cuda.is_available() else float('nan')
    print(f"\nANN Benchmark (N={N}, D={D}): CPU: {cpu_time:.6f}s, GPU: {gpu_time:.6f}s, Speedup: {speedup:.2f}")

# ------------------------------------------------------------------------------------------------
# Main block to run tests and benchmarks
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    test_kmeans()
    test_knn()
    test_ann()
    test_recall_rate_knn()  # Recall rate between our_knn and baseline_knn
    test_recall_rate_ann()  # Recall rate between baseline_knn and our_ann
    
    benchmark_distance_functions()
    benchmark_knn_comparison(N=4000, D=128, iterations=100, K=10)
    benchmark_kmeans(N=4000, D=128, iterations=10, clusters=8)
    benchmark_ann(N=4000, D=128, iterations=10, topK=10)
