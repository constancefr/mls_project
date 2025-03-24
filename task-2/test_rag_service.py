import requests
import time
# import threading
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configuration
SERVICE_URL = "http://192.168.16.21:8000/rag"
QUERIES = [
    "Tell me about cats",
    "What are dogs like?",
    "How do hummingbirds fly?",
    "What pets do people keep?",
    "Describe small animals"
]
TEST_DURATION = 30  # seconds for each rate test
REQUEST_RATES = [1, 2, 5, 10, 20]  # requests per second to test
MAX_WORKERS = 50  # maximum concurrent threads

# Statistics tracking
results = {
    'rates': [],
    'throughput': [],
    'latency_avg': [],
    'latency_p95': [],
    'success_rate': []
}

def send_request():
    """Send a single request to the RAG service"""
    query = random.choice(QUERIES)
    payload = {"query": query, "k": 2}
    
    start_time = time.time()
    try:
        response = requests.post(SERVICE_URL, json=payload, timeout=10)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            return latency, True
        else:
            return latency, False
    except Exception as e:
        return time.time() - start_time, False

def run_test(rate):
    """Run a test at a specific request rate"""
    print(f"\nStarting test at {rate} requests/sec...")
    
    request_interval = 1.0 / rate
    total_requests = 0
    successful_requests = 0
    latencies = []
    
    start_time = time.time()
    end_time = start_time + TEST_DURATION
    
    def handle_result(future):
        nonlocal total_requests, successful_requests
        latency, success = future.result()
        latencies.append(latency)
        total_requests += 1
        if success:
            successful_requests += 1
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while time.time() < end_time:
            request_time = time.time()
            
            # Submit request
            future = executor.submit(send_request)
            future.add_done_callback(handle_result)
            
            # Sleep to maintain rate
            elapsed = time.time() - request_time
            sleep_time = max(0, request_interval - elapsed)
            time.sleep(sleep_time)
    
    # Calculate statistics
    actual_duration = time.time() - start_time
    actual_rate = total_requests / actual_duration
    success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = np.percentile(latencies, 95)
    else:
        avg_latency = 0
        p95_latency = 0
    
    print(f"Test completed at target rate {rate} req/sec")
    print(f"  Actual rate: {actual_rate:.2f} req/sec")
    print(f"  Success rate: {success_rate:.2f}%")
    print(f"  Avg latency: {avg_latency:.3f} sec")
    print(f"  95th percentile latency: {p95_latency:.3f} sec")
    
    # Store results
    results['rates'].append(rate)
    results['throughput'].append(actual_rate)
    results['latency_avg'].append(avg_latency)
    results['latency_p95'].append(p95_latency)
    results['success_rate'].append(success_rate)

def plot_results():
    """Plot the test results"""
    plt.figure(figsize=(15, 10))
    
    # Throughput plot
    plt.subplot(2, 2, 1)
    plt.plot(results['rates'], results['throughput'], 'bo-')
    plt.xlabel('Target Request Rate (req/sec)')
    plt.ylabel('Actual Throughput (req/sec)')
    plt.title('Throughput vs Request Rate')
    
    # Latency plot
    plt.subplot(2, 2, 2)
    plt.plot(results['rates'], results['latency_avg'], 'ro-', label='Average')
    plt.plot(results['rates'], results['latency_p95'], 'go-', label='95th Percentile')
    plt.xlabel('Target Request Rate (req/sec)')
    plt.ylabel('Latency (sec)')
    plt.title('Latency vs Request Rate')
    plt.legend()
    
    # Success rate plot
    plt.subplot(2, 2, 3)
    plt.plot(results['rates'], results['success_rate'], 'mo-')
    plt.xlabel('Target Request Rate (req/sec)')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Request Rate')
    
    plt.tight_layout()
    
    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"rag_service_test_{timestamp}.png")
    print(f"\nResults plot saved as rag_service_test_{timestamp}.png")

if __name__ == "__main__":
    print("Starting RAG service load testing...")
    
    # Run tests for each rate
    for rate in REQUEST_RATES:
        run_test(rate)
    
    # Plot results
    plot_results()
    
    print("\nLoad testing completed!")