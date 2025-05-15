# Voice Cloning System

This directory contains the implementation for fine-tuning the Orpheus TTS model for voice cloning tasks using a custom loss function that focuses on learning from specific parts of the sequences.

## Overview

The system consists of two main components:

1. **Data Processing**: `data_processing.py` - Processes audio data and creates paired examples for voice cloning.
2. **Model Training**: `voice_clone.py` - Implements a custom trainer with masked loss for fine-tuning the model.

## How It Works

The voice cloning process works by:

1. Taking a reference audio sample and text from a speaker
2. Learning to transfer that voice to new text passages
3. Using a custom loss function that focuses only on the speech generation part of the sequence

## Usage

### 1. Process the data

First, process your audio data using the data processing script:

```bash
# For testing with a small subset of speakers
python data_processing.py --test --output-dir ./voice_cloning/processed_data

# For processing the full dataset
python data_processing.py --output-dir ./voice_cloning/processed_data
```

This will create the necessary dataset files in the specified output directory.

### 2. Configure the training

Edit the `config.yaml` file to set your training parameters:

```yaml
# Key parameters to consider:
model_name: "canopylabs/orpheus-tts-0.1-pretrained"
train_dataset_path: "voice_cloning/processed_data/voice_cloning_train"
eval_dataset_path: "voice_cloning/processed_data/voice_cloning_val"
weighted_loss: false  # Set to true for weighted loss
```

### 3. Train the model

Run the voice cloning training script:

```bash
python voice_cloning/voice_clone.py --config voice_cloning/config.yaml
```

## Custom Loss Function

The implementation uses a specialized loss function that:

1. Only calculates loss on tokens after the second `<start_of_speech>` token (masking all previous tokens)
2. Optionally weights the loss differently for the first set of speech tokens vs. later tokens

## Requirements

- PyTorch
- Transformers
- Datasets
- YAML
- SNAC (for audio tokenization)

## Model and Data

The system is designed to work with the Orpheus TTS model and the voice cloning dataset that contains paired examples of speech from the same speaker. 
