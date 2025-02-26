import numpy as np
import json
import os

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
            A = np.loadtxt(A_file)
        return N, D, A, K


def testdata_knn(N, D, K, data_dir="data"):
    # Generate a unique filename for the JSON metadata based on N, D, K
    json_filename = os.path.join(data_dir, f"N{N}_D{D}_K{K}.json")
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if the metadata file exists for this experiment
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            data = json.load(f)
            # Load the corresponding data files
            A = np.loadtxt(data["a_file"])
            X = np.loadtxt(data["x_file"])
            print(f"Loaded existing data from {json_filename}")
    else:
        # If not, generate new test data
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        
        # Generate filenames for the data files
        A_file = os.path.join(data_dir, f"N{N}_D{D}_K{K}_A_data.txt")
        X_file = os.path.join(data_dir, f"N{N}_D{D}_K{K}_X_data.txt")
        
        # Save the data files
        np.savetxt(A_file, A)
        np.savetxt(X_file, X)
        
        # Save metadata to the JSON file
        data = {
            "n": N,
            "d": D,
            "k": K,
            "a_file": A_file,
            "x_file": X_file
        }
        with open(json_filename, "w") as f:
            json.dump(data, f)
        
        print(f"Generated and saved new data for N={N}, D={D}, K={K}")

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
            A = np.loadtxt(A_file)
            X = np.loadtxt(X_file)
        return N, D, A, X, K
