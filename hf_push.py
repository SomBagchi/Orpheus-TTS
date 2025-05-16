from huggingface_hub import HfApi, create_repo
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def push_to_huggingface(
    train_data_path,
    val_data_path,
    repo_id,
    private=True
):
    # Create repository
    create_repo(repo_id, private=private)
    api = HfApi()
    
    # Push datasets
    print(f"Pushing training data from {train_data_path}...")
    train_ds = load_from_disk(train_data_path)
    train_ds.push_to_hub(repo_id, "voice_cloning_train")
    
    print(f"Pushing validation data from {val_data_path}...")
    val_ds = load_from_disk(val_data_path)
    val_ds.push_to_hub(repo_id, "voice_cloning_val")
    
    print(f"Successfully pushed all data to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    train_ds = 'voice_cloning/voice_cloning_train'
    val_ds = 'voice_cloning/voice_cloning_val'
    repo_id = 'SomBagchi/voice_cloning_pairs'
    
    push_to_huggingface(
        train_ds,
        val_ds,
        repo_id,
        private=True
    )
