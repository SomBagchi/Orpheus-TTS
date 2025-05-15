import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset
from transformers import AutoTokenizer
from snac import SNAC
import random
from tqdm import tqdm
from collections import defaultdict
import argparse

# Load the SNAC model for audio tokenization
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the text tokenizer
tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

# Function to tokenize audio waveforms
def tokenise_audio(waveform, ds_sample_rate=16000):
    # Ensure the input is a numpy array
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    
    # Resample to 24kHz as required by SNAC
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

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
    # Tokenize reference text
    ref_text_tokens = tokenizer.encode(f"Reference text: {ref_text}", add_special_tokens=False)
    
    # Create the reference audio string
    ref_audio_str = "Reference audio:"
    ref_audio_tokens_str = tokenizer.encode(ref_audio_str, add_special_tokens=False)
    
    # Tokenize sample text
    sample_text_tokens = tokenizer.encode(f"Sample text: {sample_text}", add_special_tokens=False)
    
    # Create the sample audio string
    sample_audio_str = "Sample audio:"
    sample_audio_tokens_str = tokenizer.encode(sample_audio_str, add_special_tokens=False)
    
    # Combine all tokens - no need to tokenize the audio tokens, they're already tokens
    all_tokens = ref_text_tokens + [tokenizer.convert_tokens_to_ids("\n")] + \
                 ref_audio_tokens_str + ref_audio_tokens + [tokenizer.convert_tokens_to_ids("\n")] + \
                 sample_text_tokens + [tokenizer.convert_tokens_to_ids("\n")] + \
                 sample_audio_tokens_str + sample_audio_tokens
    
    return all_tokens

def parse_args():
    parser = argparse.ArgumentParser(description='Process dataset for voice cloning')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited number of speakers')
    parser.add_argument('--speakers', type=int, default=5, help='Number of speakers to process in test mode')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save processed files')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    return parser.parse_args()

def main():
    args = parse_args()
    
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
    
    # Create a list to store our processed examples
    processed_examples = []
    
    # Process each selected speaker's examples
    for speaker in tqdm(sorted_speakers, desc="Processing speakers"):
        example_indices = speaker_examples[speaker]
        
        # Process each example for this speaker
        for idx in example_indices:
            try:
                example = dataset['train'][idx]
                
                # Debug information about the audio
                if args.debug:
                    print(f"\nProcessing example idx={idx}, speaker={speaker}, uuid={example['uuid']}")
                    print(f"Audio type: {type(example['audio'])}")
                    print(f"Audio keys: {example['audio'].keys() if isinstance(example['audio'], dict) else 'Not a dict'}")
                    if isinstance(example['audio'], dict) and 'array' in example['audio']:
                        print(f"Audio array type: {type(example['audio']['array'])}")
                        print(f"Audio array shape: {example['audio']['array'].shape if hasattr(example['audio']['array'], 'shape') else 'No shape attribute'}")
                    if isinstance(example['audio'], dict) and 'sampling_rate' in example['audio']:
                        print(f"Audio sampling rate: {example['audio']['sampling_rate']}")
                
                # Get the audio waveform and sample rate directly from the array and sampling_rate fields
                waveform = example['audio']['array']
                sample_rate = example['audio']['sampling_rate']
                transcript = example['transcript']
                
                # Tokenize the audio
                audio_tokens = tokenise_audio(waveform, sample_rate)
                
                # Create self-reference example (same example as both reference and sample)
                tokenized_prompt = create_tokenized_prompt(
                    transcript, audio_tokens, transcript, audio_tokens
                )
                
                processed_examples.append({
                    'tokens': tokenized_prompt,
                    'uuid': example['uuid'],
                    'episode_id': speaker
                })
                
            except Exception as e:
                print(f"Error processing example from speaker {speaker}, index {idx}: {e}")
                continue
    
    print(f"Processed {len(processed_examples)} examples")
    
    # Save the processed examples
    output_path = f"{args.output_dir}/voice_cloning_processed.pt"
    torch.save(processed_examples, output_path)
    print(f"Processed data saved to {output_path}")
    
    # Let's create pairs with examples from the same speaker
    pairs_examples = []
    total_pairs_created = 0
    speakers_used = set()
    
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
                ref_audio_tokens = tokenise_audio(ref_waveform, ref_sample_rate)
                sample_audio_tokens = tokenise_audio(sample_waveform, sample_sample_rate)
                
                # Create tokenized prompt
                tokenized_prompt = create_tokenized_prompt(
                    ref_transcript, ref_audio_tokens, sample_transcript, sample_audio_tokens
                )
                
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
    pairs_output_path = f"{args.output_dir}/voice_cloning_pairs.pt"
    torch.save(pairs_examples, pairs_output_path)
    print(f"Paired data saved to {pairs_output_path}")

if __name__ == "__main__":
    main()
