from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
import numpy as np
import yaml
import wandb
import torch
import argparse
import os
import glob
import shutil

# Add argument parsing
parser = argparse.ArgumentParser(description="Voice cloning script")
parser.add_argument("--test", action="store_true", help="Run in test mode")
args = parser.parse_args()

config_file = "voice_cloning/config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dataset_name = config["dataset_name"]
train_subset = config["train_subset"]
val_subset = config["val_subset"]
split = config["split"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
eval_batch_size = config["eval_batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
keep_checkpoints = config.get("keep_checkpoints", 1)  # Default to 1 if not specified

# Load optimizer and scheduler config
optimizer = config["optimizer"]
betas = tuple(config["betas"])
epsilon = config["epsilon"]
weight_decay = config["weight_decay"]
lr_scheduler_type = config["lr_scheduler_type"]
warmup_steps = config["warmup_steps"]
max_grad_norm = config["max_grad_norm"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
logging_steps = config["logging_steps"]

# Loss masking parameters from config.yaml
boosted_loss_weight = config["boosted_loss_weight"]
boosted_token_count = config["boosted_token_count"]

# Custom callback to clean up old checkpoints
class CheckpointCleanupCallback(TrainerCallback):
    def __init__(self, output_dir, keep_checkpoints=1):
        self.output_dir = output_dir
        self.keep_checkpoints = keep_checkpoints
    
    def on_save(self, args, state, control, **kwargs):
        if self.keep_checkpoints <= 0:
            return
        
        # Get all checkpoint directories
        checkpoints = sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1])
        )
        
        # If we have more checkpoints than we want to keep, remove the oldest ones
        if len(checkpoints) > self.keep_checkpoints:
            for checkpoint in checkpoints[:-self.keep_checkpoints]:
                print(f"Removing old checkpoint: {checkpoint}")
                try:
                    shutil.rmtree(checkpoint)
                except Exception as e:
                    print(f"Error removing checkpoint {checkpoint}: {e}")
        
        # Print disk usage after cleanup
        try:
            import subprocess
            result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            print(f"Current disk usage after checkpoint cleanup:\n{result.stdout}")
        except Exception as e:
            print(f"Error checking disk usage: {e}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# Custom data collator to preserve start_of_speech_pos
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, examples):
        # Extract start_of_speech_pos before creating the batch
        start_of_speech_positions = [example.pop("start_of_speech_pos") for example in examples]
        
        # Basic collation for input_ids, labels, and attention_mask
        batch = {
            "input_ids": torch.stack([torch.tensor(example["input_ids"]) for example in examples]),
            "labels": torch.stack([torch.tensor(example["labels"]) for example in examples]),
            "attention_mask": torch.stack([torch.tensor(example["attention_mask"]) for example in examples]),
        }
        
        # Add back the start_of_speech_pos as a tensor
        batch["start_of_speech_pos"] = torch.tensor(start_of_speech_positions)
        
        return batch

# Subclass Trainer to implement masked loss
class SpeechMaskTrainer(Trainer):
    def __init__(self, *args, boosted_weight, boosted_count, **kwargs):
        super().__init__(*args, **kwargs)
        self.boosted_weight = boosted_weight
        self.boosted_count = boosted_count
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract start of speech positions
        start_pos = inputs.pop("start_of_speech_pos")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Create loss mask with default zeros
        loss_mask = torch.zeros_like(inputs["labels"], dtype=torch.float)
        
        # For each item in the batch
        for i, pos in enumerate(start_pos):
            # Position after start_of_speech token
            start_idx = pos + 1
            
            # Apply boosted weight to specified number of tokens after start position
            end_boosted_idx = min(start_idx + self.boosted_count, loss_mask.size(1))
            if start_idx < loss_mask.size(1):
                loss_mask[i, start_idx:end_boosted_idx] = self.boosted_weight
            
            # Set weight to 1.0 for all remaining tokens
            if end_boosted_idx < loss_mask.size(1):
                loss_mask[i, end_boosted_idx:] = 1.0
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        shift_loss_mask = loss_mask[..., 1:].contiguous()
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Apply mask and calculate mean of masked values
        loss = loss.view(shift_labels.size())
        masked_loss = loss * shift_loss_mask
        
        # Average loss over all tokens with non-zero mask
        # Using sum of masked loss divided by sum of mask values
        loss = masked_loss.sum() / (shift_loss_mask.sum() + 1e-8)  # Small epsilon to avoid division by zero
        
        return (loss, outputs) if return_outputs else loss

# Load dataset from Hugging Face with the correct config name
train_ds = load_dataset(dataset_name, train_subset, split=split)
val_ds = load_dataset(dataset_name, val_subset, split=split)

wandb.init(project=project_name, name=run_name)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=eval_batch_size,
    logging_steps=logging_steps,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb", 
    save_steps=save_steps,
    eval_strategy="steps",
    eval_steps=500,
    remove_unused_columns=False,  # Important: Keep all columns including start_of_speech_pos
    learning_rate=learning_rate,
    optim=optimizer,
    adam_beta1=betas[0],
    adam_beta2=betas[1],
    adam_epsilon=epsilon,
    weight_decay=weight_decay,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=warmup_steps,
    max_grad_norm=max_grad_norm,
    gradient_accumulation_steps=gradient_accumulation_steps,
)

# Initialize our custom trainer with the custom data collator
trainer = SpeechMaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=CustomDataCollator(tokenizer),
    boosted_weight=boosted_loss_weight,  
    boosted_count=boosted_token_count,   
    callbacks=[CheckpointCleanupCallback(output_dir=f"./{base_repo_id}", keep_checkpoints=keep_checkpoints)]
)

# Print disk space before training
try:
    import subprocess
    print("Disk space before training:")
    subprocess.run(['df', '-h', '/'], check=True)
except Exception as e:
    print(f"Error checking disk space: {e}")

# Move model to GPU before training
model.to("cuda")

trainer.train()
trainer.evaluate()

print("Training complete")

print(f"Peak CUDA memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
