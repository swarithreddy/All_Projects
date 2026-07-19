import threading
import numpy as np
import os

def heavy_math():
    size = 1000
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    while True:
        # Matrix multiplication (CPU-heavy)
        C = np.dot(A, B)
        
        # Trig loop (more floating-point math)
        np.sin(C)
        np.cos(C)
        np.tan(C)

def max_cpu_with_math():
    num_threads = os.cpu_count() or 4
    print(f"Launching {num_threads} threads of heavy math...")
    for _ in range(num_threads):
        t = threading.Thread(target=heavy_math)
        t.daemon = True
        t.start()

    while True:
        pass  # Keep the main thread alive

if __name__ == "__main__":
    max_cpu_with_math()
