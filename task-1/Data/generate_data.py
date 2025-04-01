import numpy as np

def generate_large_npy_file(n, d, file_name="X_1024.npy", batch_size=1000):

    
    data_memmap = np.lib.format.open_memmap(file_name, mode='w+', dtype=np.float32, shape=(n, d))

    # Escribir en bloques
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # Generate a random block of vectors
        data_memmap[start:end] = np.random.rand(end - start, d).astype(np.float32)

        print(f"Storing block {start // batch_size + 1} out of {n // batch_size + 1}...")

    print(f"File generated: {file_name}")


generate_large_npy_file(n=1, d=32768)



