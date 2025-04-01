import os
import requests
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configuration
DEFAULT_PORT_ORIGINAL = 8000
DEFAULT_PORT_ENHANCED = 8001
QUERIES = [
    "Tell me about cats",
    "What are dogs like?",
    "How do hummingbirds fly?",
    "What pets do people keep?",
    "Describe small animals"
]
TEST_DURATION = 30  # seconds for each rate test
REQUEST_RATES = [1, 2, 5, 10, 20]  # requests per second to test
# REQUEST_RATES = [1]  # requests per second to test
MAX_WORKERS = 50  # maximum concurrent threads

def parse_args():
    parser = argparse.ArgumentParser(description='Test RAG service performance')
    parser.add_argument('--service', choices=['original', 'enhanced'], required=True,
                       help='Which service to test (original or enhanced)')
    parser.add_argument('--host', default='192.168.16.27',
                       help='Host address of the service')
    parser.add_argument('--port', type=int,
                       help='Custom port number (overrides default)')
    return parser.parse_args()

def initialize_results():
    return {
        'rates': [],
        'throughput': [],
        'latency_avg': [],
        'latency_p95': [],
        'success_rate': []
    }

def send_request(service_url):
    """Send a single request to the RAG service"""
    query = random.choice(QUERIES)
    payload = {"query": query, "k": 2}
    
    start_time = time.time()
    try:
        response = requests.post(service_url, json=payload, timeout=20)
        latency = time.time() - start_time
        
        # Check for both status code and proper response structure
        if response.status_code == 200:
            json_response = response.json()
            # Success conditions for both services:
            # Original service returns {'query':..., 'result':...}
            # Enhanced service returns {'query':..., 'result':..., 'latency':...}
            if 'result' in json_response:
                return latency, True
        return latency, False
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return time.time() - start_time, False

def run_test(rate, service_url, results):
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
            
            future = executor.submit(send_request, service_url)
            future.add_done_callback(handle_result)
            
            elapsed = time.time() - request_time
            time.sleep(max(0, request_interval - elapsed))
    
    # Calculate statistics
    actual_duration = time.time() - start_time
    actual_rate = total_requests / actual_duration
    success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
    
    if latencies:
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
    else:
        avg_latency = p95_latency = 0
    
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

def plot_results(results, service_type):
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(FIGURES_DIR, f"{service_type}_service_test_{timestamp}.png")
    
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nResults plot saved to {filename}")
    except Exception as e:
        print(f"\nError saving plot: {str(e)}")
        # Fallback to current directory if figures/ fails
        fallback_filename = f"{service_type}_service_test_{timestamp}.png"
        plt.savefig(fallback_filename)
        print(f"Saved plot to current directory as {fallback_filename}")

def main():
    args = parse_args()
    
    # Set default port if not specified
    if args.port is None:
        args.port = DEFAULT_PORT_ENHANCED if args.service == 'enhanced' else DEFAULT_PORT_ORIGINAL
    
    service_url = f"http://{args.host}:{args.port}/rag"
    print(f"Testing {args.service} service at {service_url}")
    
    results = initialize_results()
    
    for rate in REQUEST_RATES:
        run_test(rate, service_url, results)
    
    plot_results(results, args.service)
    print("\nLoad testing completed!")

if __name__ == "__main__":
    main()