"""
Voice Cloning Data Processing Script
===================================

This script processes the "SomBagchi/orpheus_voice_cloning" dataset, creating paired examples
from the same speaker for voice cloning tasks.

Example Usage:
-------------

# Process full dataset (WARNING: This will take a long time and requires significant compute resources)
python data_processing.py --output-dir ./processed_data

# Test mode with 5 speakers (default)
python data_processing.py --test --output-dir ./test_data

# Test mode with specific number of speakers
python data_processing.py --test --speakers 10 --output-dir ./test_data

# Debug mode (shows additional processing information)
python data_processing.py --test --debug --output-dir ./test_debug

# Show prompt format example (useful for checking the tokenization structure)
python data_processing.py --show-prompt-format

# Test mode + debug + viewing prompt format
python data_processing.py --test --debug --show-prompt-format --output-dir ./test_data

# Force CPU usage (use when CUDA compatibility issues occur)
python data_processing.py --test --debug --force-cpu --output-dir ./test_data
"""

import torch
import numpy as np
import torchaudio.transforms as T
from datasets import load_dataset
from transformers import AutoTokenizer
from snac import SNAC
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
import sys
from datasets import Dataset

def get_device(force_cpu=False):
    """Determine device to use (CPU or CUDA)"""
    if force_cpu:
        print("Forcing CPU usage as requested")
        return "cpu"
    
    if torch.cuda.is_available():
        try:
            # Test if CUDA is actually working
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            print("Using CUDA for audio processing")
            return "cuda"
        except Exception as e:
            print(f"CUDA error detected during initialization: {e}")
            print("Falling back to CPU")
            return "cpu"
    else:
        print("CUDA not available, using CPU")
        return "cpu"

# Load the text tokenizer
tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

# Function to tokenize audio waveforms
def tokenise_audio(waveform, ds_sample_rate=16000, device="cpu"):
    # Load the SNAC model for audio tokenization if not already loaded
    global model
    if 'model' not in globals():
        try:
            model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            model = model.to(device)
        except Exception as e:
            print(f"Error loading SNAC model: {e}")
            print("If this is a CUDA error, try using --force-cpu")
            sys.exit(1)
            
    # Ensure the input is a numpy array
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    
    # Resample to 24kHz as required by SNAC
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to(device)

    # Generate the codes from SNAC
    with torch.inference_mode():
        codes = model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+128266)
        all_codes.append(codes[1][0][2*i].item()+128266+4096)
        all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

    return all_codes

# Function to create a tokenized prompt
def create_tokenized_prompt(ref_text, ref_audio_tokens, sample_text, sample_audio_tokens):
    # Special token IDs
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    
    start_of_speech = tokeniser_length + 1  # 128257
    end_of_speech = tokeniser_length + 2    # 128258
    
    start_of_human = tokeniser_length + 3   # 128259
    end_of_human = tokeniser_length + 4     # 128260
    
    start_of_ai = tokeniser_length + 5      # 128261
    end_of_ai = tokeniser_length + 6        # 128262
    
    # Tokenize the reference and sample text
    ref_text_with_prefix = f"Reference text: {ref_text}\nReference speech: "
    ref_text_tokens = tokenizer.encode(ref_text_with_prefix, add_special_tokens=False)
    
    sample_text_with_prefix = f"\nSample text: {sample_text}\nSample speech: "
    sample_text_tokens = tokenizer.encode(sample_text_with_prefix, add_special_tokens=False)
    
    # Build the full token sequence according to the specified template
    all_tokens = [
        start_of_human, start_of_text,
        *ref_text_tokens,
        end_of_text, end_of_human,
        
        start_of_ai, start_of_speech,
        *ref_audio_tokens,
        end_of_speech, end_of_ai,
        
        start_of_human, start_of_text,
        *sample_text_tokens,
        end_of_text, end_of_human,
        
        start_of_ai, start_of_speech,
        *sample_audio_tokens,
        end_of_speech, end_of_ai
    ]
    
    return all_tokens

# Function to print tokenized prompt in a readable format for debugging
def print_tokenized_prompt(tokens):
    # Special token names for readability
    special_tokens = {
        128000: "<START_OF_TEXT>",
        128009: "<END_OF_TEXT>",
        128257: "<START_OF_SPEECH>",
        128258: "<END_OF_SPEECH>",
        128259: "<START_OF_HUMAN>",
        128260: "<END_OF_HUMAN>",
        128261: "<START_OF_AI>",
        128262: "<END_OF_AI>"
    }
    
    print("\n===== TOKENIZED PROMPT (FULL) =====")
    print(f"Total tokens: {len(tokens)}")
    
    # Print all tokens with their values and meanings
    print("\nToken-by-token breakdown:")
    
    section_name = "unknown"
    current_section_tokens = []
    text_tokens = []
    
    for i, token in enumerate(tokens):
        # Handle special tokens that mark sections
        if token in special_tokens:
            # If we're ending a text section, decode and print the collected tokens
            if token == 128009 and text_tokens:  # END_OF_TEXT
                try:
                    decoded_text = tokenizer.decode(text_tokens)
                    print(f"   Text content: \"{decoded_text}\"")
                except:
                    print(f"   Text content: <could not decode>")
                text_tokens = []
            
            # Print the special token
            print(f"\n[{i}] {token} - {special_tokens[token]}")
            
            # Update section tracking
            if token == 128259:  # START_OF_HUMAN
                section_name = "HUMAN"
                current_section_tokens = []
            elif token == 128261:  # START_OF_AI
                section_name = "AI"
                current_section_tokens = []
            elif token == 128000:  # START_OF_TEXT
                section_name = "TEXT"
                text_tokens = []
            elif token == 128257:  # START_OF_SPEECH
                section_name = "SPEECH"
                print("   Speech tokens:")
        
        # Handle regular tokens
        else:
            if section_name == "TEXT":
                # For text sections, collect tokens for later decoding
                text_tokens.append(token)
                print(f"   [{i}] {token} - text token")
            elif section_name == "SPEECH":
                # For speech sections, just print the token
                print(f"   [{i}] {token} - audio token")
            else:
                # For other sections
                print(f"   [{i}] {token} - {section_name} section token")
    
    # Print summary of sections
    sections = [
        (0, "Start of prompt"),
        (tokens.index(128260) if 128260 in tokens else -1, "End of human reference text"),
        (tokens.index(128258) if 128258 in tokens else -1, "End of AI reference speech"),
        (tokens.index(128260, tokens.index(128258) + 1) if 128258 in tokens and 128260 in tokens[tokens.index(128258) + 1:] else -1, "End of human sample text"),
        (len(tokens) - 1, "End of prompt")
    ]
    
    print("\n\nSection lengths:")
    for i, (pos, label) in enumerate(sections):
        if i > 0 and pos > 0 and sections[i-1][0] >= 0:
            prev_pos = sections[i-1][0]
            section_len = pos - prev_pos
            print(f"- {label}: {section_len} tokens")
            
    print("\n==============================")

# Function to create a test/debug prompt with limited tokens to check format
def create_debug_prompt():
    # Special token IDs
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    
    start_of_speech = tokeniser_length + 1  # 128257
    end_of_speech = tokeniser_length + 2    # 128258
    
    start_of_human = tokeniser_length + 3   # 128259
    end_of_human = tokeniser_length + 4     # 128260
    
    start_of_ai = tokeniser_length + 5      # 128261
    end_of_ai = tokeniser_length + 6        # 128262
    
    # Short example texts
    ref_text = "Hello world"
    sample_text = "Testing the voice cloning"
    
    # Tokenize texts
    ref_text_tokens = tokenizer.encode(f"Reference text: {ref_text}\nReference speech: ", add_special_tokens=False)
    sample_text_tokens = tokenizer.encode(f"\nSample text: {sample_text}\nSample speech: ", add_special_tokens=False)
    
    # Create mock audio tokens (just a few for demonstration)
    ref_audio_tokens = [128266 + i for i in range(10)]
    sample_audio_tokens = [128266 + 100 + i for i in range(10)]
    
    # Build the full token sequence
    all_tokens = [
        start_of_human, start_of_text,
        *ref_text_tokens,
        end_of_text, end_of_human,
        
        start_of_ai, start_of_speech,
        *ref_audio_tokens,
        end_of_speech, end_of_ai,
        
        start_of_human, start_of_text,
        *sample_text_tokens,
        end_of_text, end_of_human,
        
        start_of_ai, start_of_speech,
        *sample_audio_tokens,
        end_of_speech, end_of_ai
    ]
    
    return all_tokens

def parse_args():
    parser = argparse.ArgumentParser(description='Process dataset for voice cloning')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited number of speakers')
    parser.add_argument('--speakers', type=int, default=5, help='Number of speakers to process in test mode')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save processed files')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--show-prompt-format', action='store_true', help='Show a debug prompt format example')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage (for CUDA compatibility issues)')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio (0.1 = 10%)')
    parser.add_argument('--split-by-speaker', action='store_true', help='Split train/val by speaker rather than by samples')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir} (created if it didn't exist)")
    
    # Determine device to use (CPU or CUDA)
    device = get_device(force_cpu=args.force_cpu)
    
    # Show debug prompt format example if requested
    if args.show_prompt_format and args.test:
        print("\n=== Testing prompt format with debug example ===")
        debug_prompt = create_debug_prompt()
        print_tokenized_prompt(debug_prompt)
    elif args.show_prompt_format:
        print("\n=== Skipping prompt format example (only shown in test mode) ===")
        print("Use --test flag together with --show-prompt-format to see tokenized prompts")
    
    print("Loading dataset...")
    # Load the dataset
    dataset = load_dataset("SomBagchi/orpheus_voice_cloning")
    
    print(f"Dataset loaded with {len(dataset['train'])} samples")
    
    # Check the structure of the first example to understand the audio format
    if args.debug:
        first_example = dataset['train'][0]
        print("\nFirst example keys:", first_example.keys())
        print("\nAudio field type:", type(first_example['audio']))
        print("\nAudio field keys:", first_example['audio'].keys() if isinstance(first_example['audio'], dict) else "Not a dictionary")
        
        # Print the full audio details
        print("\nFull audio details:", first_example['audio'])
        
        if 'array' in first_example['audio']:
            print("\nAudio array shape:", first_example['audio']['array'].shape)
            print("\nAudio array type:", type(first_example['audio']['array']))
        if 'sampling_rate' in first_example['audio']:
            print("\nAudio sampling rate:", first_example['audio']['sampling_rate'])
    
    # Group examples by episode_id (same speaker) and sort by episode_id
    print("Grouping and sorting examples by speaker (episode_id)...")
    speaker_examples = defaultdict(list)
    
    for i, example in enumerate(dataset['train']):
        speaker_examples[example['episode_id']].append(i)
    
    # Sort speakers by episode_id
    sorted_speakers = sorted(speaker_examples.keys())
    
    # Calculate speaker statistics
    speaker_count = len(sorted_speakers)
    samples_per_speaker = [len(speaker_examples[spk]) for spk in sorted_speakers]
    avg_samples = sum(samples_per_speaker) / speaker_count if speaker_count else 0
    min_samples = min(samples_per_speaker) if samples_per_speaker else 0
    max_samples = max(samples_per_speaker) if samples_per_speaker else 0
    
    print(f"Found {speaker_count} speakers in the dataset")
    print(f"Average samples per speaker: {avg_samples:.2f}")
    print(f"Min samples per speaker: {min_samples}")
    print(f"Max samples per speaker: {max_samples}")
    
    # Limit number of speakers in test mode
    if args.test:
        num_speakers = min(args.speakers, speaker_count)
        print(f"Test mode: Processing {num_speakers} speakers")
        sorted_speakers = sorted_speakers[:num_speakers]
    else:
        print("Processing all speakers")
    
    # Create pairs with examples from the same speaker
    pairs_examples = []
    total_pairs_created = 0
    speakers_used = set()
    first_prompt_printed = False
    
    # Process each speaker to create pairs
    for speaker in tqdm(sorted_speakers, desc="Creating pairs"):
        example_indices = speaker_examples[speaker]
        
        # Skip speakers with less than 2 examples
        if len(example_indices) < 2:
            continue
            
        # Maximum pairs for this speaker is half the number of examples
        max_pairs_for_speaker = len(example_indices) // 2
        print(f"Speaker {speaker}: Creating {max_pairs_for_speaker} pairs from {len(example_indices)} examples")
        
        # Create pairs for this speaker
        pairs_created = 0
        
        # Create all possible pairs (up to max_pairs_for_speaker)
        # We'll use all examples without replacement until we've used them all or reached our limit
        available_indices = example_indices.copy()
        random.shuffle(available_indices)
        
        while len(available_indices) >= 2 and pairs_created < max_pairs_for_speaker:
            try:
                # Take two examples from the available indices
                idx1 = available_indices.pop()
                idx2 = available_indices.pop()
                
                ref_example = dataset['train'][idx1]
                sample_example = dataset['train'][idx2]
                
                # Debug information
                if args.debug:
                    print(f"\nCreating pair from idx1={idx1}, idx2={idx2}")
                    if isinstance(ref_example['audio'], dict) and 'array' in ref_example['audio']:
                        print(f"Ref audio array shape: {ref_example['audio']['array'].shape}")
                    if isinstance(sample_example['audio'], dict) and 'array' in sample_example['audio']:
                        print(f"Sample audio array shape: {sample_example['audio']['array'].shape}")
                
                # Verify they're from the same speaker
                assert ref_example['episode_id'] == sample_example['episode_id']
                
                # Get reference data directly from the array
                ref_waveform = ref_example['audio']['array']
                ref_sample_rate = ref_example['audio']['sampling_rate']
                ref_transcript = ref_example['transcript']
                
                # Get sample data directly from the array
                sample_waveform = sample_example['audio']['array']
                sample_sample_rate = sample_example['audio']['sampling_rate']
                sample_transcript = sample_example['transcript']
                
                # Tokenize
                ref_audio_tokens = tokenise_audio(ref_waveform, ref_sample_rate, device)
                sample_audio_tokens = tokenise_audio(sample_waveform, sample_sample_rate, device)
                
                # Create tokenized prompt
                tokenized_prompt = create_tokenized_prompt(
                    ref_transcript, ref_audio_tokens, sample_transcript, sample_audio_tokens
                )
                
                # Print tokenized prompt for debugging in test mode
                if args.test and not first_prompt_printed:
                    print(f"\n--- FULL EXAMPLE TOKENIZED PROMPT FOR SPEAKER {speaker} ---")
                    print(f"Reference text: {ref_transcript}")
                    print(f"Sample text: {sample_transcript}")
                    print_tokenized_prompt(tokenized_prompt)
                    first_prompt_printed = True
                
                pairs_examples.append({
                    'tokens': tokenized_prompt,
                    'ref_uuid': ref_example['uuid'],
                    'sample_uuid': sample_example['uuid'],
                    'episode_id': speaker  # Store the speaker ID
                })
                
                pairs_created += 1
                speakers_used.add(speaker)
                
            except Exception as e:
                print(f"Error processing pair from speaker {speaker}: {e}")
                continue
                
        total_pairs_created += pairs_created
        print(f"Created {pairs_created} pairs for speaker {speaker}")
    
    print(f"Created {total_pairs_created} different same-speaker reference-sample pairs")
    print(f"Used {len(speakers_used)} unique speakers for pairs")
    
    # Save the pairs examples
    if total_pairs_created > 0:
        # Split data into training and validation sets
        val_split = min(max(0.0, args.val_split), 0.5)  # Ensure val_split is between 0 and 0.5
        print(f"Using validation split of {val_split:.1%}")
        
        if args.split_by_speaker:
            # Split by speaker
            print("Splitting train/val by speaker")
            speakers_list = list(speakers_used)
            random.shuffle(speakers_list)
            
            val_speakers_count = max(1, int(len(speakers_list) * val_split))
            val_speakers = set(speakers_list[:val_speakers_count])
            
            train_pairs = [ex for ex in pairs_examples if ex['episode_id'] not in val_speakers]
            val_pairs = [ex for ex in pairs_examples if ex['episode_id'] in val_speakers]
            
            print(f"Split by speaker: {len(val_speakers)} validation speakers, {len(speakers_used) - len(val_speakers)} training speakers")
        else:
            # Split by samples
            print("Splitting train/val by samples")
            random.shuffle(pairs_examples)
            val_count = max(1, int(len(pairs_examples) * val_split))
            
            val_pairs = pairs_examples[:val_count]
            train_pairs = pairs_examples[val_count:]
        
        print(f"Created train split with {len(train_pairs)} samples")
        print(f"Created validation split with {len(val_pairs)} samples")
        
        # Function to convert pairs to HuggingFace dataset format
        def pairs_to_hf_dataset(pairs):
            hf_data = {
                'input_ids': [],
                'labels': [],
                'attention_mask': [],
                'episode_id': [],
                'start_of_speech_pos': []  # New field for start of speech position
            }
            
            # Special token ID for start_of_speech
            start_of_speech = 128257
            
            for example in pairs:
                tokens = example['tokens']
                
                # Find the position of the second start_of_speech token (for the sample speech)
                # The first one is for reference speech, the second one is for sample speech
                speech_positions = [i for i, t in enumerate(tokens) if t == start_of_speech]
                sample_speech_pos = speech_positions[1] if len(speech_positions) >= 2 else -1
                
                hf_data['input_ids'].append(tokens)
                hf_data['labels'].append(tokens)  # Same as input_ids
                hf_data['attention_mask'].append([1] * len(tokens))
                hf_data['episode_id'].append(example['episode_id'])
                hf_data['start_of_speech_pos'].append(sample_speech_pos)
            
            return Dataset.from_dict(hf_data)
        
        # Create and save training dataset
        train_dataset = pairs_to_hf_dataset(train_pairs)
        train_output_path = f"{args.output_dir}/voice_cloning_train"
        train_dataset.save_to_disk(train_output_path)
        print(f"Training dataset saved to {train_output_path}")
        
        # Create and save validation dataset
        val_dataset = pairs_to_hf_dataset(val_pairs)
        val_output_path = f"{args.output_dir}/voice_cloning_val"
        val_dataset.save_to_disk(val_output_path)
        print(f"Validation dataset saved to {val_output_path}")
        
        # Save the original pairs data for reference
        torch_output_path = f"{args.output_dir}/voice_cloning_pairs.pt"
        torch.save({
            'train': train_pairs,
            'val': val_pairs
        }, torch_output_path)
        print(f"Original paired data also saved to {torch_output_path}")
    else:
        print("No pairs were created, skipping file save")

if __name__ == "__main__":
    main()
