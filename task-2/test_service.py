import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor

# === Configuration ===

# Base URL of the RAG service to be tested
BASE_URL = "http://192.168.47.132:8000/rag"

# Sample query and retrieval configuration
QUERY = "What is the capital of France?"
K = 2  # Number of top-k documents to retrieve from the backend

# Request rates to test (requests per second)
RATES = [5, 10, 20, 50, 100]  

# === Request Function ===

def send_request(rate: int) -> float | None:
    """
    Send a single POST request to the RAG service and measure latency.

    Args:
        rate (int): The current request rate (used only for logging/debugging, not functionally).

    Returns:
        float | None: The response time in seconds if the request is successful; None otherwise.
    """
    payload = {"query": QUERY, "k": K}
    start_time = time.time()

    try:
        response = requests.post(BASE_URL, json=payload)
        end_time = time.time()
        if response.status_code == 200:
            return end_time - start_time
    except Exception as e:
        print(f"Request failed: {e}")

    return None

# === Load Testing Function ===

def load_test(rate: int):
    """
    Perform a load test by sending multiple parallel requests to simulate a given request rate.

    Args:
        rate (int): Number of requests to send simultaneously.
    """
    total_requests = 0
    total_time = 0.0
    successful_requests = 0
    failed_requests = 0
    latencies = []

    start_time = time.time()

    # Use a thread pool to send requests in parallel
    with ThreadPoolExecutor(max_workers=rate) as executor:
        futures = [executor.submit(send_request, rate) for _ in range(rate)]

        # Collect the results as they complete
        for future in futures:
            response_time = future.result()
            if response_time is not None:
                successful_requests += 1
                total_time += response_time
                latencies.append(response_time)
            else:
                failed_requests += 1

        total_requests += rate

    # Calculate throughput (successful requests per second)
    throughput = successful_requests / total_time if total_time > 0 else 0.0

    # Print performance results for this rate
    print(f"Rate: {rate} req/s")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"Latencies: {[round(l, 4) for l in latencies]} seconds")
    print(f"Throughput: {throughput:.4f} requests/second")
    print("-" * 50)

# === Entry Point ===

def main():
    """
    Main function to run the load tests across different request rates defined in RATES.
    """
    for rate in RATES:
        load_test(rate)

# Run the tests if this script is executed directly
if __name__ == "__main__":
    main()
