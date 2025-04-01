import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor

# URL of your service
BASE_URL = "http://192.168.47.132:8000/rag"

# Parameters for the tests
QUERY = "What is the capital of France?"
K = 2  # Number of documents to retrieve
RATES = [5, 10, 20, 50, 100]  # Different request rates per second
#DURATION = 10

# Function to send a request and measure the response time
def send_request(rate):
    payload = {"query": QUERY, "k": K}
    
    start_time = time.time()  # Start measuring the request time
    
    # Send the POST request
    response = requests.post(BASE_URL, json=payload)
    
    end_time = time.time()  # End time
    
    if response.status_code == 200:
        response_time = end_time - start_time
        return response_time  # Return the response time in seconds
    else:
        return None

# Function to perform the load test by simulating different request rates
def load_test(rate):
    total_requests = 0
    total_time = 0.0
    successful_requests = 0
    failed_requests = 0
    latencies = []

    start_time = time.time()

    # Use a ThreadPoolExecutor to send multiple requests simultaneously
    with ThreadPoolExecutor(max_workers=rate) as executor:
        futures = []
        for _ in range(rate):
            futures.append(executor.submit(send_request, rate))
        
        for future in futures:
            response_time = future.result()
            if response_time is not None:
                successful_requests += 1
                total_time += response_time
                latencies.append(response_time)
            else:
                failed_requests += 1
        
        total_requests += rate  # Increment the total request counter
    

    # Calculate the metrics
    throughput = successful_requests / total_time

    print(f"Rate: {rate}")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"Latencies: {latencies} seconds")
    print(f"Throughput: {throughput:.4f} requests/second")
    print("-" * 50)

# Main function to run the test with different request rates
def main():
    for rate in RATES:
        load_test(rate)

if __name__ == "__main__":
    main()
