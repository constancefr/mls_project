import torch
#import cupy as cp
#import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):
    pass

def distance_l2(X, Y):
    # All components from Y are subtracted from their respective components of X
    diff = X - Y

    # Square each component of the resultant vector
    diff_squared = diff ** 2

    # Calculates the total of all the components
    sum_diff_squared = torch.sum(diff_squared)

    # Square roots the total
    l2_distance = torch.sqrt(sum_diff_squared)

    return l2_distance


def distance_dot(X, Y):
    pass

def distance_manhattan(X, Y):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4):
    """
    Perform KMeans clustering on dataset A using L2 distance (via distance_l2).

    Parameters:
      N (int): Number of vectors.
      D (int): Dimension of vectors.
      A (numpy.ndarray): Data matrix of shape (N, D).
      K (int): Number of clusters.
      max_iters (int): Maximum number of iterations.
      tol (float): Convergence tolerance based on centroid movement.

    Returns:
      numpy.ndarray: An array of shape (N,) containing the cluster assignment (ID)
                     for each vector.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)  # (N, D)

    # Randomly select K points from A as initial centroids.
    indices = torch.randperm(N)[:K]
    centroids = A_tensor[indices]  # (K, D)

    for iteration in range(max_iters):
        # Assignment step: for each point, compute its distance to every centroid using distance_l2.
        assignments = torch.empty(N, dtype=torch.long, device=device)
        for i in range(N):
            dists = []
            for j in range(K):
                d = distance_l2(A_tensor[i], centroids[j])
                dists.append(d)
            dists = torch.stack(dists)  # (K,)
            assignments[i] = torch.argmin(dists)

        # Update step: recalc centroids as the mean of points assigned to each cluster.
        new_centroids = []
        for j in range(K):
            # Get indices of all points assigned to cluster j.
            cluster_indices = (assignments == j).nonzero(as_tuple=False).squeeze()
            if cluster_indices.numel() == 0:
                # If a cluster gets no points, reinitialize its centroid randomly.
                new_centroid = A_tensor[torch.randint(0, N, (1,))]
            else:
                cluster_points = A_tensor[cluster_indices]
                new_centroid = torch.mean(cluster_points, dim=0, keepdim=True)  # (1, D)
            new_centroids.append(new_centroid)
        new_centroids = torch.cat(new_centroids, dim=0)  # (K, D)

        # Check for convergence: if centroids move less than tol, stop.
        if torch.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return assignments.cpu().numpy()


def baseline_knn(N, D, A, X, K):
    """
    Baseline KNN implementation using NumPy for true nearest neighbor search.
    
    Parameters:
      N (int): Number of vectors.
      D (int): Dimension of vectors.
      A (numpy.ndarray): Data matrix of shape (N, D).
      X (numpy.ndarray): Query vector of shape (D,).
      K (int): Number of nearest neighbors to return.
    
    Returns:
      numpy.ndarray: An array of shape (K,) containing the indices of the top K nearest vectors.
    """
    # Compute L2 distances between X and every row in A.
    distances = np.linalg.norm(A - X, axis=1)
    # Return the indices of the K smallest distances.
    return np.argsort(distances)[:K]


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4):
    """
    Perform KMeans clustering on dataset A using L2 distance.
    
    Parameters:
      N (int): Number of vectors.
      D (int): Dimension of vectors.
      A (numpy.ndarray): Data matrix of shape (N, D).
      K (int): Number of clusters.
      max_iters (int): Maximum number of iterations.
      tol (float): Convergence tolerance based on centroid movement.
      
    Returns:
      numpy.ndarray: An array of shape (N,) containing the cluster assignment (ID)
                     for each vector.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert A to a torch tensor.
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)  # Shape: (N, D)
    
    # Initialise centroids by randomly selecting K points from A.
    indices = torch.randperm(N)[:K]
    centroids = A_tensor[indices]  # Shape: (K, D)
    
    for i in range(max_iters):
        # Assignment step: compute squared L2 distances from each point to each centroid.
        # Use broadcasting to compute distances: result shape (N, K)
        distances = torch.sum((A_tensor.unsqueeze(1) - centroids.unsqueeze(0)) ** 2, dim=2)
        # For each point, choose the closest centroid.
        assignments = torch.argmin(distances, dim=1)  # Shape: (N,)
        
        # Update step: recalc centroids as the mean of points assigned to each cluster.
        new_centroids = []
        for k in range(K):
            cluster_points = A_tensor[assignments == k]
            if cluster_points.shape[0] == 0:
                # If a cluster gets no points, reinitialize its centroid randomly.
                new_centroid = A_tensor[torch.randint(0, N, (1,))]
            else:
                new_centroid = torch.mean(cluster_points, dim=0, keepdim=True)  # Shape: (1, D)
            new_centroids.append(new_centroid)
        new_centroids = torch.cat(new_centroids, dim=0)  # Shape: (K, D)
        
        # Check for convergence: if centroids move less than tol, stop.
        if torch.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        
        centroids = new_centroids
    
    # Return assignments as a numpy array.
    return assignments.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    """
    Approximate Nearest Neighbor search using clustering and local KNN, reusing distance_l2.

    Steps:
      1. Cluster the data into C clusters using our_kmeans.
      2. Compute centroids and select the nearest K1 clusters to query X.
      3. Within each selected cluster, compute distances using distance_l2 to get K2 candidate points.
      4. Merge candidates and select the overall top K nearest vectors.

    Parameters:
      N (int): Number of vectors.
      D (int): Dimension of vectors.
      A (numpy.ndarray): Data matrix of shape (N, D).
      X (numpy.ndarray): Query vector of shape (D,).
      K (int): Number of top nearest neighbors to return (as indices).

    Returns:
      numpy.ndarray: An array of shape (K,) containing the indices of the top K nearest vectors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Choose a number of clusters (C)
    C = 10 if N >= 10 else N

    # Use our_kmeans to cluster A into C clusters.
    assignments = our_kmeans(N, D, A, C)

    # Convert A and X into torch tensors.
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    # Compute centroids from assignments.
    centroids = []
    for c in range(C):
        indices = np.where(assignments == c)[0]
        if len(indices) == 0:
            centroid = torch.zeros(D, device=device)
        else:
            cluster_points = A_tensor[indices]
            centroid = torch.mean(cluster_points, dim=0)
        centroids.append(centroid)
    centroids = torch.stack(centroids)  # (C, D)

    # Step 2: Find the nearest K1 clusters.
    K1 = min(3, C)  # Choose 3 nearest clusters (or fewer if C < 3)
    centroid_dists = []
    for c in range(C):
        centroid_dists.append(distance_l2(X_tensor, centroids[c]))
    centroid_dists = torch.stack(centroid_dists)  # (C,)
    _, nearest_cluster_indices = torch.topk(centroid_dists, k=K1, largest=False)
    nearest_cluster_indices = nearest_cluster_indices.cpu().numpy()

    # Step 3: For each selected cluster, perform a local KNN search.
    candidate_indices = []
    K2 = 10  # Number of candidates per cluster.
    for c in nearest_cluster_indices:
        cluster_indices = np.where(assignments == c)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = A_tensor[cluster_indices]
        local_dists = []
        for i in range(cluster_points.shape[0]):
            local_dists.append(distance_l2(X_tensor, cluster_points[i]))
        local_dists = torch.stack(local_dists)
        k2 = min(K2, len(cluster_indices))
        _, local_top_indices = torch.topk(local_dists, k=k2, largest=False)
        local_top_indices = local_top_indices.cpu().numpy()
        # Map local indices back to original indices.
        candidate_indices.extend(np.array(cluster_indices)[local_top_indices].tolist())

    # Remove duplicate candidate indices.
    candidate_indices = list(set(candidate_indices))

    # Step 4: Merge candidates and select the overall top K.
    candidate_points = A_tensor[candidate_indices]
    final_dists = []
    for i in range(candidate_points.shape[0]):
        final_dists.append(distance_l2(X_tensor, candidate_points[i]))
    final_dists = torch.stack(final_dists)
    final_k = K if len(candidate_indices) >= K else len(candidate_indices)
    _, final_top_indices = torch.topk(final_dists, k=final_k, largest=False)
    final_top_indices = final_top_indices.cpu().numpy()
    final_candidate_indices = np.array(candidate_indices)[final_top_indices]

    return final_candidate_indices


# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
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


if __name__ == "__main__":
    # Generate synthetic test data using testdata_ann (which creates random data)
    N, D, A, X, K = testdata_ann("")
    
    # Compute baseline KNN result
    baseline_result = baseline_knn(N, D, A, X, K)
    print("Baseline KNN indices:", baseline_result)
    
    # Compute ANN result
    ann_result = our_ann(N, D, A, X, K)
    print("ANN indices:", ann_result)
    
    # Calculate and print the recall rate
    recall = recall_rate(baseline_result, ann_result)
    print("Recall Rate:", recall)



