#!/usr/bin/env python3
"""
Garbage Collection Script for PyTorch

This script provides utilities to free up GPU memory by calling garbage collection
and clearing the PyTorch CUDA cache. Run this script when you're experiencing memory
issues with PyTorch models.

Usage:
    python garbage.py          # Basic cleanup
    python garbage.py --loop   # Continuous monitoring and cleanup
"""

import gc
import torch
import argparse
import time
import os
import psutil

def display_memory_info():
    """Display system and GPU memory information"""
    print("\n=== MEMORY INFORMATION ===")
    
    # System memory
    system_mem = psutil.virtual_memory()
    print(f"System Memory: {system_mem.used / 1e9:.2f} GB used / {system_mem.total / 1e9:.2f} GB total ({system_mem.percent}%)")
    
    # CUDA memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem_alloc = torch.cuda.memory_allocated(i) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(i) / 1e9
            prop = torch.cuda.get_device_properties(i)
            print(f"GPU {i} ({prop.name}): {gpu_mem_alloc:.2f} GB allocated, {gpu_mem_reserved:.2f} GB reserved / {prop.total_memory / 1e9:.2f} GB total")
    else:
        print("No CUDA devices available")
    
    print("==========================\n")

def clear_memory():
    """Run garbage collection and clear CUDA cache if available"""
    print("Clearing memory...")
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        print("Emptying CUDA cache...")
        torch.cuda.empty_cache()
        # Optional: Reset peak memory stats
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.reset_accumulated_memory_stats(i)
    
    print("Memory cleared!")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Memory Cleanup Script")
    parser.add_argument("--loop", action="store_true", help="Run in continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=60, help="Interval between checks in seconds (for loop mode)")
    args = parser.parse_args()
    
    if args.loop:
        print(f"Starting memory monitoring loop (Ctrl+C to exit, checking every {args.interval} seconds)")
        try:
            while True:
                display_memory_info()
                clear_memory()
                print(f"Waiting {args.interval} seconds until next cleanup...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting memory monitoring loop")
    else:
        # One-time cleanup
        display_memory_info()
        clear_memory()
        display_memory_info()
        print("One-time memory cleanup complete")

if __name__ == "__main__":
    main() 
