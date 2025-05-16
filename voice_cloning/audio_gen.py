#!/usr/bin/env python3
"""
Audio Generation Script for Voice Cloning

This script:
1. Loads samples from the voice cloning validation dataset
2. Generates 4 different audio files for each sample:
   - Reference audio (original reference voice)
   - Real sample (actual target speech)
   - Base model generation
   - Finetuned model generation
3. Saves MP3 files in samples/{episode_id}/ directories

Example Usage:
    python voice_cloning/audio_gen.py
"""

import os
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import shutil
import wave
import logging
from pathlib import Path

# Add orpheus_tts to path so we can import the decoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from orpheus_tts_pypi.orpheus_tts.decoder import convert_to_audio

# Constants
BASE_MODEL_NAME = "canopylabs/orpheus-tts-0.1-pretrained"
FINETUNED_MODEL_DIR = "voice_cloning/models"
DATASET_NAME = "SomBagchi/voice_cloning_pairs"
VAL_SUBSET = "voice_cloning_val"
NUM_SAMPLES = 3
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
END_OF_AI_TOKEN = 128265
SAMPLES_DIR = "samples"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_audio_to_mp3(audio_bytes, file_path):
    """
    Convert audio bytes to mp3 and save to file_path
    """
    # Save as WAV first
    wav_path = file_path.replace(".mp3", ".wav")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Save as WAV
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(24000)  # Assuming 24kHz sample rate
        wf.writeframes(audio_bytes)
    
    # Convert to MP3 using ffmpeg if available
    try:
        import subprocess
        subprocess.call(['ffmpeg', '-y', '-i', wav_path, file_path])
        # Remove temporary WAV file
        os.remove(wav_path)
    except Exception as e:
        logger.warning(f"Failed to convert to MP3, keeping WAV file: {e}")
        # Rename WAV to MP3 as fallback
        shutil.move(wav_path, file_path)
    
    logger.info(f"Audio saved to {file_path}")

def extract_tokens_between(tokens, start_token, end_token, start_pos=0):
    """
    Extract tokens between start_token and end_token, 
    starting from start_pos.
    """
    tokens = tokens[start_pos:]
    start_idx = tokens.index(start_token) if start_token in tokens else 0
    end_idx = tokens.index(end_token) if end_token in tokens else len(tokens)
    return tokens[start_idx:end_idx+1]

def find_token_positions(tokens, token_id):
    """
    Find all positions of a specific token in a sequence
    """
    return [i for i, t in enumerate(tokens) if t == token_id]

def process_tokens_to_audio(tokens):
    """
    Process tokens to audio using the SNAC decoder
    """
    audio_bytes = convert_to_audio(tokens, 0)
    return audio_bytes

def run_model_inference(model, tokenizer, input_tokens, max_new_tokens=200):
    """
    Run model inference starting from the input tokens
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert tokens to tensor
    input_ids = torch.tensor([input_tokens]).to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Generate tokens
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False  # Deterministic generation
        )
    
    # Extract the generated tokens (without the input tokens)
    generated_tokens = output[0][len(input_tokens):].tolist()
    
    # Append special token if needed
    if END_OF_SPEECH_TOKEN not in generated_tokens and END_OF_AI_TOKEN not in generated_tokens:
        generated_tokens.append(END_OF_SPEECH_TOKEN)
    
    # Return the full sequence (input + generated)
    return input_tokens + generated_tokens

def main():
    # Create samples directory
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    logger.info("Loading dataset...")
    dataset_dict = load_dataset(DATASET_NAME, VAL_SUBSET)
    
    # Get the appropriate split from the dataset dictionary
    # Since we only have 'train' available, use that
    dataset = dataset_dict['train']
    
    logger.info(f"Loaded {len(dataset)} validation samples")
    
    # Select random samples (ensure we don't request more than available)
    num_samples_to_use = min(NUM_SAMPLES, len(dataset))
    random_indices = random.sample(range(len(dataset)), num_samples_to_use)
    selected_samples = [dataset[i] for i in random_indices]
    
    logger.info(f"Selected sample indices: {random_indices}")
    
    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load finetuned model
    logger.info(f"Loading finetuned model from: {FINETUNED_MODEL_DIR}")
    try:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL_DIR,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        finetuned_tokenizer = base_tokenizer  # Reuse base tokenizer
    except Exception as e:
        logger.error(f"Failed to load finetuned model: {e}")
        logger.info("Will proceed with only the base model")
        finetuned_model = None
    
    # Check if using GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    base_model = base_model.to(device)
    if finetuned_model:
        finetuned_model = finetuned_model.to(device)
    
    # Process each selected sample
    for i, sample in enumerate(selected_samples):
        input_ids = sample["input_ids"]
        episode_id = sample["episode_id"]
        start_of_speech_pos = sample["start_of_speech_pos"]
        
        logger.info(f"Processing sample {i+1}/{NUM_SAMPLES}, episode_id: {episode_id}")
        
        # Create directory for this sample
        sample_dir = os.path.join(SAMPLES_DIR, episode_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Find positions of start_of_speech tokens
        start_speech_positions = find_token_positions(input_ids, START_OF_SPEECH_TOKEN)
        
        if len(start_speech_positions) < 2:
            logger.warning(f"Sample {i} doesn't have enough start_of_speech tokens, skipping")
            continue
        
        # 1. Extract reference audio (before first start_of_speech)
        ref_end_pos = start_speech_positions[0]
        reference_tokens = input_ids[:ref_end_pos+1]
        
        # Find the end_of_speech token after the first start_of_speech
        ref_end_speech_pos = None
        for j in range(ref_end_pos + 1, len(input_ids)):
            if input_ids[j] == END_OF_SPEECH_TOKEN:
                ref_end_speech_pos = j
                break
        
        if ref_end_speech_pos:
            reference_speech_tokens = input_ids[ref_end_pos:ref_end_speech_pos+1]
            reference_audio = process_tokens_to_audio(reference_speech_tokens)
            if reference_audio:
                save_audio_to_mp3(
                    reference_audio, 
                    os.path.join(sample_dir, "reference.mp3")
                )
        
        # 2. Extract real sample audio (after second start_of_speech)
        sample_start_pos = start_speech_positions[1]
        
        # Find the end_of_speech token after the second start_of_speech
        sample_end_speech_pos = None
        for j in range(sample_start_pos + 1, len(input_ids)):
            if input_ids[j] == END_OF_SPEECH_TOKEN:
                sample_end_speech_pos = j
                break
        
        if sample_end_speech_pos:
            real_sample_tokens = input_ids[sample_start_pos:sample_end_speech_pos+1]
            real_sample_audio = process_tokens_to_audio(real_sample_tokens)
            if real_sample_audio:
                save_audio_to_mp3(
                    real_sample_audio, 
                    os.path.join(sample_dir, "real_sample.mp3")
                )
        
        # 3. Generate with base model
        # Crop input_ids to the second start_of_speech token
        crop_pos = start_speech_positions[1]
        cropped_input = input_ids[:crop_pos+1]
        
        logger.info("Generating with base model...")
        base_generated = run_model_inference(base_model, base_tokenizer, cropped_input)
        
        # Extract speech tokens from generation
        base_speech_start = crop_pos  # Second start_of_speech position
        
        # Find the end_of_speech token in the generated sequence
        base_end_positions = find_token_positions(base_generated, END_OF_SPEECH_TOKEN)
        base_speech_end = base_end_positions[0] if base_end_positions else len(base_generated) - 1
        
        base_speech_tokens = base_generated[base_speech_start:base_speech_end+1]
        base_audio = process_tokens_to_audio(base_speech_tokens)
        
        if base_audio:
            save_audio_to_mp3(
                base_audio, 
                os.path.join(sample_dir, "base_model.mp3")
            )
        
        # 4. Generate with finetuned model (if available)
        if finetuned_model:
            logger.info("Generating with finetuned model...")
            finetuned_generated = run_model_inference(
                finetuned_model, finetuned_tokenizer, cropped_input
            )
            
            # Extract speech tokens from generation
            finetuned_speech_start = crop_pos  # Second start_of_speech position
            
            # Find the end_of_speech token in the generated sequence
            finetuned_end_positions = find_token_positions(finetuned_generated, END_OF_SPEECH_TOKEN)
            finetuned_speech_end = finetuned_end_positions[0] if finetuned_end_positions else len(finetuned_generated) - 1
            
            finetuned_speech_tokens = finetuned_generated[finetuned_speech_start:finetuned_speech_end+1]
            finetuned_audio = process_tokens_to_audio(finetuned_speech_tokens)
            
            if finetuned_audio:
                save_audio_to_mp3(
                    finetuned_audio, 
                    os.path.join(sample_dir, "finetuned_model.mp3")
                )
        
        logger.info(f"Completed processing sample {i+1}, files saved to {sample_dir}")
    
    logger.info(f"Audio generation complete. Results saved in {SAMPLES_DIR} directory")

if __name__ == "__main__":
    main()
