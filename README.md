# Task1 - GPU Optimization for Nearest Neighbor Search

In this task, we focus on implementing GPU-based kernels to optimize various operations related to Nearest Neighbor (NN) Search tasks, which are essential for a wide range of machine learning algorithms, especially in clustering and classification tasks. The goal of this task is to accelerate operations like distance computations, TopK searches, K-Nearest Neighbors (KNN), Approximate Nearest Neighbors (ANN), and K-means clustering by leveraging GPU power. All GPU kernels are designed to run on an NVIDIA GTX 1060, optimizing parameters such as batch size, chunk size, and the number of streams, blocks and threads for parallel computation based on its capabilities. This means that more powerful GPUs may be limited by the values assigned to these variables in the code.

### Folder Structure

The `task1` folder contains the following files:

1. **task.py**  
   This file contains the main implementations for various algorithms and benchmarks. It includes:
   - **Distance Kernels**: Functions to compute various distance metrics (L2, Cosine, Manhattan, Dot).
   - **TopK Search**: Functions to perform the top-k nearest neighbors search.
   - **KNN (K-Nearest Neighbors)**: Implementations of KNN algorithms.
   - **ANN (Approximate Nearest Neighbors)**: Functions for approximate nearest neighbor search.
   - **KMeans**: KMeans clustering algorithm implementation.
   - **Benchmarking Functions**: Functions to benchmark the performance of the algorithms on CPU vs. GPU and measure their efficiency.

   All functions are fully documented inside the file to provide details about their usage, inputs, outputs, and expected behavior.

2. **test.py**  
   This file contains functions to generate random datasets or load datasets from `.npy` files to test the algorithms implemented in `task.py`. Key functionalities include:
   - **Random Data Generation**: Functions to generate synthetic data with a specified number of samples and features.
   - **Dataset Loading**: Functions to load data from pre-existing `.npy` files, useful for testing with large datasets or previously generated data.
   
   The functions inside `test.py` allow users to test the algorithms in `task.py` on various datasets and ensure correctness and performance.

3. **generate_data.py**  
   This file contains a utility function for generating large datasets of random vectors. The function `generate_large_npy_file` allows you to create large `.npy` files (e.g., containing 4 million vectors with 2^15 dimensions) and store them efficiently using memory-mapped files to prevent memory overload. This file can be used to generate datasets for benchmarking or testing the algorithms in `task.py` on large-scale data. 


4. **test_data.json**  
    This JSON file is used to specify the characteristics of the dataset that is to be loaded inside test.py. It contains the following parameters:

   - **n**: number of vectors.
   - **d**: dimension of each vector.
   - **a_file**: path to the `.npy` file containing the dataset.
   - **x_file**: path to the `.npy` file containing the query vector.
   - **k**: number of vectors to retrieve.

5. **requirements_2.py** 
   List of Python dependencies required to run the code of task 1.

### How to use

1. Install the `requirements_1.txt` file using `pip install -r requirements_1.txt`.

2. If needed, generate a large dataset using the generate_data.py file, especially useful for testing on large-scale data. To try with small datasets, you can choose to use `random = True` in `test_knn()` from `task.py`. 

3. Use the task.py functions to run KNN, ANN, or KMeans on the dataset.

4. Run benchmarking functions to compare the performance of CPU-based vs GPU-based implementations.

# Task 2 – RAG Service with Queue and Batcher Performance Evaluation

In this task, we evaluate the performance of two different implementations of a Retrieval-Augmented Generation (RAG) service. One corresponds to a simple baseline provided to us, and the other is our optimized version that incorporates request queuing and batching to improve efficiency under load. We simulate different request rates and analyze key performance metrics like throughput and latency. Additionally, we provide tooling to visualize the results.

### Folder Structure

The `task2` folder contains the following files:

1. **serving_rag.py**  
Baseline RAG service provided in the study

2. **our_serving_rag.py**  
Our optimized RAG service using a queue and batch processing

3. **test_service.py**  
Script to load test any RAG service. Reports latency and throughput at various request rates

4. **generate_plots.py**  
Script to visualize the experiment results (throughput vs. rate and latency at rate=100)

5. **requirements_2.py** 
List of Python dependencies required to run the services and test scripts

### How to Use

### 1. Install Required Packages

Make sure you are in the project directory and

1. **Install Required Packages** 
Install the `requirements_2.txt` file using `pip install -r requirements_1.txt`.
2. **Download Models Locally**
This project assumes that both the embedding model and the language model are downloaded and available locally.
Update the paths to these models in the scripts (`our_serving_rag.py` and/or `serving_rag.py`) accordingly.
3. **Start the service**
In one terminal, run either the baseline or your optimized RAG service. Take note of the IP address of this terminal.
4. **Update the IP for testing**
Ensure the IP in the `test_service.py` matches the address where the service is running
5. **Run the test**
In a different terminal, run `test_service.py`. This will test the service under different request rates and output metrics like latency and throughput.
6. **Generate Plots**
If you wish to visualize the results, manually input them into `generate_plots.py` and run it. This will generate comparison plots for throughput and latency.




   

   
