import numpy as np

def generate_large_npy_file(n, d, file_name="A.npy", batch_size=1000):
    """
    Generates a large `.npy` file containing random vectors. The data is stored using memory-mapped files to handle large datasets efficiently.

    The function generates `n` random vectors, each with `d` dimensions, and stores them in a NumPy binary file. The data is written in batches to optimize memory usage, especially for large files.

    Args:
        n (int): The number of vectors (rows) to generate. The total size of the dataset will be `n * d` elements.
        d (int): The number of dimensions (columns) for each vector.
        file_name (str, optional): The name of the output file where the data will be saved. Default is "A.npy".
        batch_size (int, optional): The number of vectors to write at a time. The default is 1000.

    Notes:
        - The function uses `np.lib.format.open_memmap` to store the data in a memory-mapped file, which allows working with large arrays that may not fit in memory.
        - To generate the array `X` with shape `(d,)` for a single vector, set `n=1` and reshape the result to `(d,)`.

    Returns:
        None: This function does not return anything but instead saves the generated data to the specified `.npy` file.


    """
    
    # Create a memory-mapped file to store the data
    data_memmap = np.lib.format.open_memmap(file_name, mode='w+', dtype=np.float32, shape=(n, d))

    # Write data in blocks to avoid memory overload
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # Generate a random block of vectors and store it in the memory-mapped file
        data_memmap[start:end] = np.random.rand(end - start, d).astype(np.float32)

        # Print progress of data storage
        print(f"Storing block {start // batch_size + 1} out of {n // batch_size + 1}...")

    # Indicate that the file generation is complete
    print(f"File generated: {file_name}")

# Example usage
generate_large_npy_file(n=4000000, d=32768)
