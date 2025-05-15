import torch
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_from_disk
import yaml
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

class VoiceCloningTrainer(Trainer):
    """
    Custom Trainer class for voice cloning that implements:
    1. Masked loss calculation - only computes loss on tokens after the start_of_speech_pos
    2. Optional weighted loss - applies higher weights to the first h tokens after start_of_speech_pos
    """
    def __init__(
        self,
        model=None,
        args=None,
        weighted_loss=False,
        higher_weight_tokens=50,
        higher_weight_factor=2.0,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.weighted_loss = weighted_loss
        self.higher_weight_tokens = higher_weight_tokens
        self.higher_weight_factor = higher_weight_factor
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override the compute_loss method to implement masked loss
        that only considers tokens after start_of_speech_pos.
        """
        # Get the start position markers for each example in the batch
        # This is the position of the second <start_of_speech> token (for sample speech)
        speech_positions = inputs.pop("start_of_speech_pos")
        
        # Forward pass through the model
        outputs = model(**inputs)
        logits = outputs.logits  # shape: [batch_size, sequence_length, vocab_size]
        
        # Get labels from inputs
        labels = inputs.get("labels")
        
        # Create a mask that ignores positions before the speech position
        # We'll use this to mask out the loss for tokens we don't want to predict
        loss_mask = torch.zeros_like(labels, dtype=torch.float)
        
        # For each example in the batch, set the mask to 1 for positions after speech_position
        for i, pos in enumerate(speech_positions):
            # Add +1 because we want to predict the token after the start_of_speech token
            start_pos = pos + 1 if pos >= 0 else 0
            
            if self.weighted_loss:
                # Apply higher weight to the first h tokens after start_pos
                end_higher_weight = min(start_pos + self.higher_weight_tokens, labels.size(1))
                loss_mask[i, start_pos:end_higher_weight] = self.higher_weight_factor
                # Apply regular weight to remaining tokens
                loss_mask[i, end_higher_weight:] = 1.0
            else:
                # Apply uniform weight to all tokens after start_pos
                loss_mask[i, start_pos:] = 1.0
        
        # Shift logits and labels for next token prediction
        # (standard for causal language modeling training)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = loss_mask[..., 1:].contiguous()
        
        # Calculate the loss with CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        
        # Calculate per-token loss first
        # flatten the tensors to match the expected shapes
        shift_logits_view = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_view = shift_labels.view(-1)
        per_token_loss = loss_fct(shift_logits_view, shift_labels_view).view(shift_labels.size())
        
        # Apply the mask to the loss
        masked_loss = per_token_loss * shift_mask
        
        # Calculate the final loss (average over non-zero elements)
        # Add small epsilon to avoid division by zero
        loss = masked_loss.sum() / (shift_mask.sum() + 1e-8)
        
        if return_outputs:
            return loss, outputs
        return loss

def train_voice_cloning_model(config_path):
    """
    Train a model for voice cloning using the VoiceCloningTrainer
    
    Args:
        config_path: Path to the configuration YAML file
    """
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load the dataset
    train_dataset = load_from_disk(config["train_dataset_path"])
    eval_dataset = load_from_disk(config["eval_dataset_path"]) if "eval_dataset_path" in config else None
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], 
        attn_implementation="flash_attention_2" if config.get("use_flash_attention", True) else "eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=config.get("overwrite_output_dir", True),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        save_steps=config["save_steps"],
        save_total_limit=config.get("save_total_limit", 3),
        learning_rate=config["learning_rate"],
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=True,
    )
    
    # Create the VoiceCloningTrainer
    trainer = VoiceCloningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        weighted_loss=config.get("weighted_loss", False),
        higher_weight_tokens=config.get("higher_weight_tokens", 50),
        higher_weight_factor=config.get("higher_weight_factor", 2.0),
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    model.save_pretrained(os.path.join(config["output_dir"], "final_model"))
    tokenizer.save_pretrained(os.path.join(config["output_dir"], "final_model"))

    # After training
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model for voice cloning")
    parser.add_argument("--config", type=str, default="voice_cloning/config.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    train_voice_cloning_model(args.config)
