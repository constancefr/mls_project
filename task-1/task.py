import torch
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
import cupy as cp

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------
def distance_cosine(X, Y):

    X_norm = torch.nn.functional.normalize(X, p=2, dim=1) 
    Y_norm = torch.nn.functional.normalize(Y, p=2, dim=1)
    cosine_similarity = torch.mm(X_norm, Y_norm.T)  
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

def distance_l2(X, Y):
    return torch.sqrt(torch.sum((X - Y)**2))

def distance_dot(X, Y):
    return X@Y

def distance_manhattan(X, Y, device = torch.device("cuda")):
    if not isinstance(Y,torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
    if not isinstance(X,torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    return torch.sum(torch.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------


def our_knn(N, D, A, X, K, device = torch.device("cuda")):
    if not isinstance(A,torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    if not isinstance(X,torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=device)

    distances = torch.sqrt(torch.sum((A - X)**2, axis = 1))
    _, indices = torch.topk(distances, K, largest=False)  # Get K nearest neighbors

    return indices.cpu().numpy()


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


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4, device = torch.device("cuda")):
    if not isinstance(A,torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32, device=device)

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

        centroids = new_centroids

        if torch.norm(centroids - new_centroids) < tol:
            break

    return cluster_ids.cpu().numpy(), centroids.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here


def our_ann(N, D, A, X, K, C = 10, K1=5, K2=10, device = torch.device("cuda")):
    """
    Approximate Nearest Neighbor search optimized for GPU using PyTorch.
    
    Parameters:
      N (int): Number of vectors.
      D (int): Dimension of vectors.
      A (numpy.ndarray): Data matrix of shape (N, D).
      X (numpy.ndarray): Query vector of shape (D,).
      K (int): Number of top nearest neighbors to return (as indices).

    Returns:
      numpy.ndarray: An array of shape (K,) containing the indices of the top K nearest vectors.
    """
    # Convert A and X into torch tensors
    if not isinstance(A,torch.Tensor):
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
    if not isinstance(X,torch.Tensor):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # (1, D)

    # Step 1: Choose the number of clusters (C)
    C = min(C, N)  

    # Step 2: Use our_kmeans to cluster A into C clusters
    assignments, centroids = our_kmeans(N, D, A, C)  # Returns cluster labels & centroids
    assignments = torch.tensor(assignments, dtype=torch.long, device=device)
    centroids = torch.tensor(centroids, dtype=torch.float32, device=device)

    # Step 3: Find the K1 nearest clusters to X
    K1 = min(K1, C)
    cluster_distances = torch.sqrt(torch.sum((X_tensor - centroids)**2, axis = 1))
    nearest_cluster_indices = torch.topk(cluster_distances, k=K1, largest=False).indices

    # Step 4: Find the K2 nearest neighbors within those clusters
    K2 = 10  # Number of candidates per cluster
    candidate_indices = []
    
    for c in nearest_cluster_indices:
        cluster_mask = (assignments == c)  # Boolean mask for cluster
        cluster_indices = torch.nonzero(cluster_mask, as_tuple=True)[0]  # Indices of points in this cluster

        if len(cluster_indices) > 0:
            # Compute distances within the cluster
            cluster_points = A_tensor[cluster_indices]
            K2 = min(K2, len(cluster_indices))
            local_knn_indices = torch.tensor(our_knn(N, D, cluster_points, X_tensor, K2))
            # Convert local indices to global indices
            candidate_indices.extend(cluster_indices[local_knn_indices].tolist())

    # Step 5: Select the overall top K nearest neighbors
    candidate_points = A_tensor[candidate_indices]
    K = min(K, len(candidate_indices))
    final_top_indices = our_knn(N, D, candidate_points, X_tensor, K2)
 
    
    return np.array(candidate_indices)[final_top_indices]



# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_kmeans():
    N, D, A, K = testdata_kmeans("")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

def test_ann_recall():
    N, D, A, X, K = testdata_ann("")
    ann_result = our_ann(N, D, A, X, K)
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    print(ann_result)
    print(recall_rate(ann_result,knn_result))


def compute_recall(N, D, A, X, K):
    knn_indices = our_knn(N, D, A, X, K).tolist()
    ann_indices = our_ann(N, D, A, X, K).tolist()

    knn_set = set(knn_indices)
    ann_set = set(ann_indices)

    recall = len(knn_set & ann_set) / K  
    return recall

# ------------------------------------------------------------------------------------------------
# Benchmarking functions
# ------------------------------------------------------------------------------------------------

def run_benchmark(N, D, K):
    """
    Executes all benchmarks for a given dataset size and dimension.
    """
    # Generate random dataset and query vector
    A = np.random.rand(N, D).astype(np.float32)
    X = np.random.rand(D).astype(np.float32)

    # Run benchmarks
    results = {
        "Manhattan Distance": benchmark_function(distance_manhattan,N, D, A, X, K),
        "KNN": benchmark_function(our_knn, N, D, A, X, K),
        "K-Means": benchmark_function(our_kmeans, N, D, A, X, K),
        "ANN": benchmark_function(our_ann, N, D, A, X, K)
    }
    recall_value = compute_recall(N, D, A, X, K)
    print(f"Recall ANN vs KNN: {recall_value:.2%}")

    return results

def run_benchmark_batching(N, D, K, batch_size=500000):
    """Ejecuta todos los benchmarks con procesamiento por lotes."""

    #Solo almacenamos `X`
    X = np.random.rand(D).astype(np.float32)

    #Run benchmarks sin almacenar `A`
    results = {
        "KNN": benchmark_knn_batching(our_knn_batching, N, D, X, K, batch_size=batch_size)
    }
    return results

def benchmark_function(func, N, D, A, X, K, num_repeats=10):
    """
    Generic benchmarking function for CPU vs GPU.
    """
    # CPU Benchmark
    if(func == distance_manhattan):
        Y = np.random.rand(D).astype(np.float32)
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(X , Y, device = "cpu")
        cpu_time = (time.time() - start_cpu) / num_repeats
    elif(func == our_knn):
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(N, D, A, X, K, device = "cpu")
        cpu_time = (time.time() - start_cpu) / num_repeats
    elif(func == our_kmeans):
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(N, D, A, K, device = "cpu")
        cpu_time = (time.time() - start_cpu) / num_repeats
    elif(func == our_ann):
        start_cpu = time.time()
        for _ in range(num_repeats):
            func(N, D, A, X, K, device = "cpu")
        cpu_time = (time.time() - start_cpu) / num_repeats


    # GPU Benchmark
    gpu_time = None
    if torch.cuda.is_available():    
        if(func == distance_manhattan):
            Y = np.random.rand(D).astype(np.float32)
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(X, Y)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_gpu) / num_repeats
        elif(func == our_knn):
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(N, D, A, X, K)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_gpu) / num_repeats
        elif(func == our_kmeans):
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(N, D, A, K)
            gpu_time = (time.time() - start_gpu) / num_repeats
        elif(func == our_ann):
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(num_repeats):
                func(N, D, A, X, K)
            gpu_time = (time.time() - start_gpu) / num_repeats

    speedup = cpu_time / gpu_time if gpu_time else None
    return cpu_time, gpu_time, speedup

def benchmark_knn_batching(func, N, D, X, K, num_repeats=1, batch_size=500000):
    """Generic benchmarking function for CPU vs GPU using batch processing."""

    #CPU Benchmark
    """start_cpu = time.time()
    for _ in range(num_repeats):
        func(N, D, X, K, batch_size, device = "cpu")
    cpu_time = (time.time() - start_cpu) / num_repeats"""
    cpu_time = 0
    #GPU Benchmark
    gpu_time = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_gpu = time.time()
        for _ in range(num_repeats):
            func(N, D, X, K, batch_size, device="cuda")
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_gpu) / num_repeats

    speedup = cpu_time / gpu_time if gpu_time else None
    return cpu_time, gpu_time, speedup


if __name__ == "__main__":
    torch.manual_seed(42) 
    #test_knn()
    #test_kmeans()
    #test_ann()
    #test_ann_recall()

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
            """if(N > 1000000):
                batch_size = 5000
                results = run_benchmark_batching(N, D, K, batch_size=batch_size)
            else:
                results = run_benchmark(N, D, K)"""
            batch_size = 50
            results = run_benchmark_batching(N, D, K, batch_size=batch_size)
            for func_name, (cpu_time, gpu_time, speedup) in results.items():
                print(f"{N:<8} | {D:<8} | {func_name:<20} | {cpu_time:<12.6f} | {gpu_time if gpu_time else 'N/A':<12} | {speedup if speedup else 'N/A'}")




    
