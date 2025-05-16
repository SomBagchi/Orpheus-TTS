#!/bin/bash

# Clean up memory before starting
echo "Running garbage collection..."
python garbage.py

# Set PyTorch CUDA memory allocation configuration
# max_split_size_mb:128 - Limits individual allocations to 128MB to reduce fragmentation
# garbage_collection_threshold:0.8 - Start garbage collection when 80% of reserved memory is unused
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

echo "Starting training with optimized memory settings..."
/usr/bin/python3 voice_cloning/voice_clone.py $@

echo "Training complete or interrupted." 
