import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------
def distance_l2_cpu(X, Y):
    start_time = time.time()

    dist = np.sqrt(np.sum((X - Y) ** 2))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    return dist

def distance_l2_cp(X, Y):
    start_time = time.time()

    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

    dist = cp.sqrt(cp.sum((X_gpu - Y_gpu) ** 2))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")
    
    return cp.asnumpy(dist)

# Raw kernel para calcular la distancia L2 entre dos vectores X y Y
l2_kernel_code = '''
extern "C" __global__
void calculate_diff_kernel(const float* X, const float* Y, float* diff, int D) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < D) {
        // Calculamos la diferencia al cuadrado
        float diff_val = X[idx] - Y[idx];
        diff[idx] = diff_val * diff_val;
    }
}
extern "C" __global__
void reduce_kernel(float* diff, int D) {
    __shared__ float cache[256];  // Memoria compartida para la reducción por bloque

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    float sum = 0.0f;

    // Calculamos las sumas parciales
    if (idx < D) {
        sum = diff[idx];
    }

    // Almacenamos la suma parcial en memoria compartida
    cache[tid] = sum;
    __syncthreads();

    // Reducción dentro del bloque
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Al final, el hilo con idx 0 guarda el valor final en global memory
    if (tid == 0) {
        diff[blockIdx.x] = cache[0];
    }
}
'''
def distance_l2_kernel(X, Y):
    start_time = time.time()

    mod = cp.RawModule(code=l2_kernel_code)
    func_diff_square = mod.get_function('calculate_diff_kernel')
    func_sum = mod.get_function('reduce_kernel')

    # Calcular las diferencias cuadradas
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

    D = X_gpu.size  # Dimensiones de los vectores

    # Alocar memoria para las diferencias cuadradas
    diff_gpu = cp.empty(D, dtype=cp.float32)

    # Establecer tamaño de bloque y rejilla
    block_size = 256
    grid_size = (D + block_size - 1) // block_size

    # Ejecutar el kernel de cálculo de diferencias cuadradas
    func_diff_square(
        (grid_size,), (block_size,), 
        (X_gpu, Y_gpu, diff_gpu, D)
    )

    # Ejecutar la reducción en paralelo hasta que quede solo un bloque
    while D > 1:
        grid_size = (D + block_size - 1) // block_size
        func_sum(
            (grid_size,), (block_size,), 
            (diff_gpu, D)
        )
        D = grid_size  # Reducir D a la cantidad de bloques restantes

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time:.10f} secs")

    # El valor final está en diff_gpu[0]
    return cp.asnumpy(cp.sqrt(diff_gpu[0]))


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------


def our_knn_cpu(N, D, A, X, K):

    start_time = time.time()
    
    memory_limit = 2**28
    batch_size = min(N, memory_limit // (D * A.itemsize)) 

    all_distances = []
    all_indices = []
    
    # Process A with batches
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        
        # Extract the current batch of A
        batch_A = A[batch_start:batch_end]
        
        # Compute the distances from X to each point in the current batch of A
        distances = np.sqrt(np.sum((batch_A - X)**2, axis=1)).astype(np.float32)
        
        # Get the K nearest neighbors (indices and distances)
        indices = np.argsort(distances)[:K]
        values = distances[indices]
        
        # Append results from the current batch
        all_distances.append(values)
        all_indices.append(indices + batch_start)  # Add the offset of the batch
    
    # Concatenate results from all batches
    all_distances = np.concatenate(all_distances)
    all_indices = np.concatenate(all_indices)
    
    # Select the K nearest neighbors globally
    top_k_global_indices = np.argsort(all_distances)[:K]

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Total Time: {execution_time:.2f} seconds")
    
    return all_indices[top_k_global_indices]

def our_knn_cp(N, D, A, X, K):
    start_time = time.time()

    # Choose batch_size in terms of the amount of free mem of the GPU (we assume 2GB RAM)
    memory_limit = 2**28
    batch_size = min(N, memory_limit // (D * A.itemsize))  #Divide by the number of bytes of each vector of A

    #Most GPUs usually works fine with 4 streams
    num_streams = 4
    stream_batch_size = batch_size // num_streams #Divide the batch size by the number of streams
    streams = [cp.cuda.Stream() for _ in range(num_streams)] #Generate streams
    
    #Move X to the GPU
    X_gpu = cp.asarray(X).astype(cp.float32)
    #Lists to stores distances and indices of the knn
    all_distances = []
    all_indices = []

    #Process A with batches
    for batch_start in range(0, N, stream_batch_size):
        batch_end = min(batch_start + stream_batch_size, N)

        #Move the batch of A to the GPU
        batch_A = cp.asarray(A[batch_start:batch_end])

        #Apply streams
        stream = streams[batch_start % num_streams]
        

        with stream:
            #One stream to compute the distances
            distances = cp.sqrt(cp.sum((X_gpu - batch_A)**2, axis=1))

        
        with stream:
            #One stream to compute the knn of each batch of A

            indices = cp.argsort(distances)[:K]
            values= distances[indices]
            all_distances.append(values)
            all_indices.append(indices + batch_start)  

    #Synchroize stream
    for stream in streams:
        stream.synchronize()

    #Concatenate results from all batches
    all_distances = cp.concatenate(all_distances)
    all_indices = cp.concatenate(all_indices)

    #Select the KNN globally
    top_k_global_indices = cp.argsort(all_distances)[:K] 

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Total Time: {execution_time:.2f} seconds")

    return all_indices[top_k_global_indices]


kernel_code = '''
extern "C" __global__
void topK_oddeven_sort(const float* arr, int* indices, float* values, int N, int K) {
    extern __shared__ float shared_data[];

    #define INF 3.402823466e+38f

    
    //To identify each block
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    //Pointers to share memory region
    float* shared_vals = shared_data;
    int* shared_idx = (int*)&shared_vals[blockDim.x];

    //Compute the number of elements in the block
    int elements_in_block = min(blockDim.x, N - blockIdx.x * blockDim.x);

    //Intialize share memory
    if (idx < N) {

        shared_vals[tid] = arr[idx];
        shared_idx[tid] = idx;
    } else {
        shared_vals[tid] = INF;  // Rellenar con infinito para que siempre quede al final
        shared_idx[tid] = -1;
    }

    __syncthreads();

    // **Aplicar Odd-Even Sort paralelo en el bloque**
    for (int i = 0; i < elements_in_block; i++) {
        // Pasos pares (tid = 0, 2, 4...)
        if (tid % 2 == 0 && tid + 1 < elements_in_block) {
            if (shared_vals[tid] > shared_vals[tid + 1]) {
                // Intercambiar valores
                float temp_val = shared_vals[tid];
                int temp_idx = shared_idx[tid];

                shared_vals[tid] = shared_vals[tid + 1];
                shared_idx[tid] = shared_idx[tid + 1];
                shared_vals[tid + 1] = temp_val;
                shared_idx[tid + 1] = temp_idx;
            }
        }
        __syncthreads();

        // Pasos impares (tid = 1, 3, 5...)
        if (tid % 2 == 1 && tid + 1 < elements_in_block) {
            if (shared_vals[tid] > shared_vals[tid + 1]) {
                // Intercambiar valores
                float temp_val = shared_vals[tid];
                int temp_idx = shared_idx[tid];

                shared_vals[tid] = shared_vals[tid + 1];
                shared_idx[tid] = shared_idx[tid + 1];
                shared_vals[tid + 1] = temp_val;
                shared_idx[tid + 1] = temp_idx;
            }
        }
        __syncthreads();
    }

    // Guardar los K menores valores en memoria global
    if (tid < K && (blockIdx.x * K + tid) < N) {
        values[blockIdx.x * K + tid] = shared_vals[tid];
        indices[blockIdx.x * K + tid] = shared_idx[tid];
    }
}
extern "C" __global__
void topK_radix_sort(const float* arr, int* indices, float* values, int N, int K) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    #define INF 3.402823466e+38f

    // Punteros a memoria compartida
    float* shared_vals = shared_data;
    int* shared_idx = (int*)&shared_vals[blockDim.x];

    int elements_in_block = min(blockDim.x, N - blockIdx.x * blockDim.x);

    // Cargar valores en memoria compartida
    if (idx < N) {
        shared_vals[tid] = arr[idx];
        shared_idx[tid] = idx;
    } else {
        shared_vals[tid] = INF;  // Relleno con infinito
        shared_idx[tid] = -1;
    }
    __syncthreads();

    // ========================
    // ** RADIX SORT PARA GPU **
    // ========================
    int max_bits = 32;  // Para flotantes reinterpretados como enteros
    for (int bit = 0; bit < max_bits; bit++) {
        int mask = 1 << bit;
        int prefix_sum = 0;

        // Cuenta cuántos ceros hay en este bit
        for (int i = 0; i < elements_in_block; i++) {
            if (!((__float_as_int(shared_vals[i]) & mask) > 0)) {
                prefix_sum++;
            }
        }
        __syncthreads();

        // Posicionar elementos en el nuevo orden
        float temp_vals[256];  // Temporal en registros
        int temp_idx[256];
        int zero_pos = 0, one_pos = prefix_sum;

        for (int i = 0; i < elements_in_block; i++) {
            int bit_val = (__float_as_int(shared_vals[i]) & mask) > 0;
            if (bit_val == 0) {
                temp_vals[zero_pos] = shared_vals[i];
                temp_idx[zero_pos] = shared_idx[i];
                zero_pos++;
            } else {
                temp_vals[one_pos] = shared_vals[i];
                temp_idx[one_pos] = shared_idx[i];
                one_pos++;
            }
        }
        __syncthreads();

        // Copiar de vuelta a memoria compartida
        for (int i = 0; i < elements_in_block; i++) {
            shared_vals[i] = temp_vals[i];
            shared_idx[i] = temp_idx[i];
        }
        __syncthreads();
    }

    // Guardar los K menores valores en memoria global
    if (tid < K && (blockIdx.x * K + tid) < N) {
        values[blockIdx.x * K + tid] = shared_vals[tid];
        indices[blockIdx.x * K + tid] = shared_idx[tid];
    }
}
extern "C" __global__
void l2_kernel(const float* X, const float* A, float* diff, int D, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < D; i++) {
            float diff_val = X[i] - A[idx * D + i];
            sum += diff_val * diff_val;
        }
        diff[idx] = sum;
    }
}
'''

def our_knn_kernel(N, D, A, X, K):
    start_time = time.time()

    mod = cp.RawModule(code=kernel_code)
    func_topk = mod.get_function('topK_radix_sort')
    func_l2 = mod.get_function('l2_kernel')


    # Choose batch_size in terms of the amount of free mem of the GPU (we assume 2GB RAM)
    memory_limit = 2**28
    batch_size = min(N, memory_limit // (D * A.itemsize))  #Divide by the number of bytes of each vector of A

    #Most GPUs usually works fine with 4 streams
    num_streams = 4
    stream_batch_size = batch_size // num_streams #Divide the batch size by the number of streams
    streams = [cp.cuda.Stream() for _ in range(num_streams)] #Generate streams
    
    #Move X to the GPU
    X_gpu = cp.asarray(X).astype(cp.float32)
    #Lists to stores distances and indices of the knn
    all_distances = []
    all_indices = []

    #Process A with batches
    for batch_start in range(0, N, stream_batch_size):
        batch_end = min(batch_start + stream_batch_size, N)

        #Move the batch of A to the GPU
        batch_A = cp.asarray(A[batch_start:batch_end])

        #Apply streams
        stream = streams[batch_start % num_streams]
        

        with stream:
            #One stream to compute the distances
            distances = cp.sqrt(cp.sum((X_gpu - batch_A)**2, axis=1)).astype(cp.float32)

        
        with stream:
            #One stream to compute the knn of each batch of A

            #The kernel gives us the knn of each block
            threads_per_block = 256
            blocks_per_grid = (batch_A.shape[0] + threads_per_block - 1) // threads_per_block
            shared_mem_size = 2 * threads_per_block * distances.itemsize
            values = cp.empty((blocks_per_grid, K), dtype=distances.dtype)
            indices = cp.empty((blocks_per_grid, K), dtype=cp.int32)

            
            func_topk(
                (blocks_per_grid,), (threads_per_block,),
                (distances, indices, values, batch_A.shape[0], K),
                shared_mem=shared_mem_size
            )

            # Flatten the results
            values = values.flatten()
            indices = indices.flatten()

            # Sort the values to get the K smallest
            top_k_global = cp.argsort(values)[:K]

            #Now we can compute the knn of each batch
            all_distances.append(values[top_k_global])
            all_indices.append(indices[top_k_global] + batch_start)  # Agregar el índice global
  

    #Synchroize stream
    for stream in streams:
        stream.synchronize()

    #Concatenate results from all batches
    all_distances = cp.concatenate(all_distances)
    all_indices = cp.concatenate(all_indices)

    #Select the KNN globally
    top_k_global_indices = cp.argsort(all_distances)[:K] 

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Total Time: {execution_time:.2f} seconds")
    return all_indices[top_k_global_indices]
# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_l2(D):
    for d in D:
        np.random.seed(42)
        X = np.random.randn(d).astype(np.float32)
        Y = np.random.randn(d).astype(np.float32)
        print(d)
        print(distance_l2_cpu(X,Y))
        print(distance_l2_cp(X,Y))
        print(distance_l2_kernel(X,Y))



def test_knn():
    #N, D, A, X, K = testdata_knn("Data/test_data.json")
    A = np.random.randn(4000, 1000)
    X = np.random.randn(1000)
    knn_result_cpu = our_knn_cpu(4000, 1000, A, X, 10)
    #knn_result_cp = our_knn_cp(N, D, A, X, K)
    knn_result_kernel = our_knn_kernel(4000, 1000, A, X, 10)
    print(knn_result_cpu)
    #print(knn_result_cp)
    print(knn_result_kernel)
    

if __name__ == "__main__":
    test_knn()
    #test_l2(D = [2, 1024, 2**15,2**20])


    """# Obtener información de la GPU
    device = cp.cuda.Device(0)  # Seleccionar la primera GPU (ajusta si tienes más GPUs)
    
    # Obtener la capacidad de la GPU
    max_threads_per_block = device.attributes.get('MAX_THREADS_PER_BLOCK', 1024)
    max_shared_memory_per_block = device.attributes.get('MAX_SHARED_MEMORY_PER_BLOCK', 49152)
    
    print(f"Max threads per block: {max_threads_per_block}")
    print(f"Max shared memory per block (bytes): {max_shared_memory_per_block}")

    # Supongamos que un hilo utiliza aproximadamente 4 bytes de memoria compartida (dependiendo de la operación)
    estimated_memory_per_thread = 8  # bytes por hilo, ajusta según el cálculo
    max_threads_based_on_memory = max_shared_memory_per_block // estimated_memory_per_thread
    
    # Seleccionamos el menor número de hilos entre los basados en memoria y los hilos máximos permitidos
    optimal_threads_per_block = min(max_threads_per_block, max_threads_based_on_memory)
    
    print(optimal_threads_per_block)"""