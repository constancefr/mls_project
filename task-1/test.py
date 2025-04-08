import numpy as np
import json

def read_data(file_path=""):
    """
    Read data from a file
    """
    if file_path == "":
        return None
    if file_path.endswith(".npy"):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def testdata_kmeans(test_file):
    if test_file == "":
        # use random data
        N = 100000
        D = pow(2,1)
        A = np.random.randn(N, D)
        K = 10
        return N, D, A, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            K = data["k"]
            A = np.memmap(A_file, dtype=np.float32, mode='r', shape=(N, D))
        return N, D, A, K

def testdata_knn(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            X = np.memmap(X_file, dtype=np.float32, mode='r', shape=(D,))
            A = np.memmap(A_file, dtype=np.float32, mode='r', shape=(N, D))
        return N, D, A, X, K
    
def testdata_ann(test_file):
    if test_file == "":
        # use random data
        N = 100000
        D = pow(2,10)
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            X = np.memmap(X_file, dtype=np.float32, mode='r', shape=(D,))
            A = np.memmap(A_file, dtype=np.float32, mode='r', shape=(N, D))
        return N, D, A, X, K
