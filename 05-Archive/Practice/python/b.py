import torch
import threading
import os

def heavy_gpu_math(device):
    size = 4096  # Large tensor size to increase GPU load
    A = torch.randn((size, size), device=device)
    B = torch.randn((size, size), device=device)

    while True:
        # Matrix multiplication (extremely GPU intensive)
        C = torch.matmul(A, B)

        # Additional floating-point math
        torch.sin(C)
        torch.cos(C)
        torch.exp(C)

def max_gpu_with_math():
    if not torch.cuda.is_available():
        print("CUDA GPU not available.")
        return

    device = torch.device("cuda")
    num_threads = min(4, torch.cuda.device_count())  # use a few threads per GPU

    print(f"Launching {num_threads} heavy GPU threads on {device}...")
    for _ in range(num_threads):
        t = threading.Thread(target=heavy_gpu_math, args=(device,))
        t.daemon = True
        t.start()

    while True:
        pass  # Keep main thread alive

if __name__ == "__main__":
    max_gpu_with_math()
