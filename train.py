#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py - Training script for Small Language Models
Modified to work with existing cs_slm.py implementation
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Import classes from cs_slm module
try:
    from cs_slm import (
        ContextualTextDataset,
        ModelSelection,
        set_seed,
        AdapterModel,
        ModelPruning,
        ModelQuantization,
        CustomTokenizer
    )
except ImportError as e:
    print(f"Warning: Partial import from cs_slm.py: {e}")
    print("Make sure cs_slm.py is in the current directory or Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class CS_SLM_Pipeline:
    """Pipeline for training small language models"""

    def __init__(self, domain_texts, use_t5=False, context_size=512):
        """
        Initialize the pipeline

        Args:
            domain_texts: List of domain-specific texts for training
            use_t5: Whether to use T5 (True) or BERT (False) as the base model
            context_size: Maximum context size
        """
        self.domain_texts = domain_texts
        self.use_t5 = use_t5
        self.context_size = context_size
        self.tokenizer = None
        self.model = None

        # Set seed for reproducibility
        set_seed(42)

        logger.info(f"Initialized CS-SLM pipeline with model type: {'T5' if use_t5 else 'BERT'}")
        logger.info(f"Context size: {context_size}")
        logger.info(f"Number of training texts: {len(domain_texts)}")

    def load_base_model(self):
        """Load the base model"""
        logger.info("Loading base model...")
        tokenizer, model = ModelSelection.load_base_model(
            use_t5=self.use_t5,
            context_size=self.context_size
        )

        self.tokenizer = tokenizer
        self.model = model

        return tokenizer, model

    def prepare_data(self, test_size=0.1, val_size=0.1, batch_size=8):
        """Prepare training, validation, and test datasets"""
        logger.info("Preparing datasets...")

        # Split data
        train_texts, test_texts = train_test_split(
            self.domain_texts,
            test_size=test_size,
            random_state=42
        )

        train_texts, val_texts = train_test_split(
            train_texts,
            test_size=val_size / (1 - test_size),
            random_state=42
        )

        logger.info(f"Data split: {len(train_texts)} train, {len(val_texts)} validation, {len(test_texts)} test")

        # Create datasets using the ContextualTextDataset class
        train_dataset = ContextualTextDataset(
            train_texts,
            self.tokenizer,
            max_length=self.context_size,
            is_t5=self.use_t5
        )

        val_dataset = ContextualTextDataset(
            val_texts,
            self.tokenizer,
            max_length=self.context_size,
            is_t5=self.use_t5
        )

        test_dataset = ContextualTextDataset(
            test_texts,
            self.tokenizer,
            max_length=self.context_size,
            is_t5=self.use_t5
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return {
            "train_texts": train_texts,
            "val_texts": val_texts,
            "test_texts": test_texts,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader
        }

    def train_model(self, train_dataloader, val_dataloader, epochs=3, lr=5e-5, use_adapters=False):
        """Train the model with options for using adapters"""
        from transformers import get_linear_schedule_with_warmup

        logger.info(f"Training model for {epochs} epochs, using adapters: {use_adapters}")

        # Move model to device
        self.model.to(device)

        if use_adapters:
            # Initialize and add adapters
            adapter_model = AdapterModel(self.model, is_t5=self.use_t5)
            self.model = adapter_model.add_adapters(adapter_size=64)

            # Train only the adapters
            return adapter_model.train_adapters(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=epochs,
                lr=lr
            )
        else:
            # Regular full model training
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            total_steps = len(train_dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

            self.model.train()
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                epoch_loss = 0

                for batch in train_dataloader:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )

                    loss = outputs.loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_dataloader)
                train_losses.append(avg_loss)

                # Validate
                self.model.eval()
                val_loss = 0

                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]
                        )
                        val_loss += outputs.loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)

                logger.info(f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_loss:.4f}, Val loss: {avg_val_loss:.4f}")
                self.model.train()

            # Plot training progress
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Training Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Training Progress')
                plt.savefig('training_progress.png')
                logger.info("Saved training progress plot to training_progress.png")
            except Exception as e:
                logger.warning(f"Could not create training plot: {e}")

            return self.model

    def optimize_model(self, quantize=False, prune=False, pruning_ratio=0.3, quantization_bits=8):
        """Apply optimization techniques to the model"""
        optimized_model = self.model

        if prune:
            logger.info(f"Pruning model with ratio {pruning_ratio}...")
            pruner = ModelPruning(optimized_model, is_t5=self.use_t5)
            optimized_model = pruner.magnitude_pruning(pruning_ratio=pruning_ratio)

        if quantize:
            logger.info(f"Quantizing model to {quantization_bits} bits...")
            quantizer = ModelQuantization(optimized_model, self.tokenizer, is_t5=self.use_t5)
            optimized_model = quantizer.quantize_model(quantization_bits=quantization_bits)

        return optimized_model

    def save_model(self, path, model=None):
        """Save the trained model"""
        if model is None:
            model = self.model

        logger.info(f"Saving model to {path}")
        os.makedirs(path, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save model info
        model_info = {
            "use_t5": self.use_t5,
            "context_size": self.context_size,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(path, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        return path

    def train_domain_tokenizer(self, vocab_size=1000):
        """Train a domain-specific tokenizer"""
        logger.info(f"Training domain-specific tokenizer with {vocab_size} new tokens")
        custom_tokenizer = CustomTokenizer(self.tokenizer, self.domain_texts)
        domain_tokenizer = custom_tokenizer.train_domain_tokenizer(vocab_size=vocab_size)

        # Update the tokenizer
        self.tokenizer = domain_tokenizer

        return domain_tokenizer


def load_text_data(file_path: str) -> List[str]:
    """
    Load text data from a file. Supports txt and csv formats.

    Args:
        file_path: Path to the text file

    Returns:
        List of text strings
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            # Split by double newlines (assuming paragraphs)
            texts = f.read().split('\n\n')
            # Filter out empty strings
            texts = [text.strip() for text in texts if text.strip()]

    elif file_path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(file_path)
            # Assume the first text column is our data
            text_column = None
            for col in df.columns:
                if df[col].dtype == 'object':  # string column
                    text_column = col
                    break

            if text_column is None:
                raise ValueError("No text column found in CSV file")

            texts = df[text_column].dropna().tolist()

        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(texts)} text samples from {file_path}")
    return texts


def extract_qa_pairs(file_path: str) -> List[str]:
    """
    Extract Q&A pairs from a formatted text file

    Args:
        file_path: Path to the file

    Returns:
        List of formatted Q&A pairs
    """
    logger.info(f"Extracting Q&A pairs from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for Q&A section markers
    qa_sections = []
    if "Q&A" in content or "Questions" in content:
        # Try to find Q&A section
        sections = content.split('##')
        for section in sections:
            if "Q&A" in section or "Questions" in section:
                qa_sections.append(section)

    # Process Q&A pairs
    qa_pairs = []

    # Extract from typical Q&A format
    for section in qa_sections:
        lines = section.split('\n')
        current_q = None
        current_a = None

        for line in lines:
            line = line.strip()
            # Look for question markers (e.g., "1. Q:")
            if 'Q:' in line or line.lstrip().startswith('Q '):
                # Save previous QA pair if exists
                if current_q and current_a:
                    qa_pairs.append(f"Question: {current_q}\nAnswer: {current_a}")

                # Extract new question
                if 'Q:' in line:
                    current_q = line.split('Q:')[1].strip()
                else:
                    current_q = line.split('Q ')[1].strip()
                current_a = None

            # Look for answer markers
            elif 'A:' in line or line.lstrip().startswith('A '):
                if 'A:' in line:
                    current_a = line.split('A:')[1].strip()
                else:
                    current_a = line.split('A ')[1].strip()

                # If we have both Q and A, add to pairs
                if current_q and current_a:
                    qa_pairs.append(f"Question: {current_q}\nAnswer: {current_a}")
                    current_q = None
                    current_a = None

    logger.info(f"Extracted {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the training script"""
    parser = argparse.ArgumentParser(
        description='Train a Small Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data file (txt or csv)')

    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for the trained model')

    # Model configuration
    parser.add_argument('--model_type', type=str, choices=['t5', 'bert'], default='bert',
                        help='Base model type')

    parser.add_argument('--context_size', type=int, default=512,
                        help='Maximum context size')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')

    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')

    # Optimization options
    parser.add_argument('--use_adapters', action='store_true',
                        help='Use adapter modules instead of full fine-tuning')

    parser.add_argument('--quantize', action='store_true',
                        help='Apply model quantization')

    parser.add_argument('--prune', action='store_true',
                        help='Apply model pruning')

    parser.add_argument('--pruning_ratio', type=float, default=0.3,
                        help='Ratio for pruning weights')

    parser.add_argument('--custom_tokenizer', action='store_true',
                        help='Train a domain-specific tokenizer')

    parser.add_argument('--extract_qa', action='store_true',
                        help='Extract Q&A pairs from the data for better training')

    # GPU options
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')

    return parser


def train_model(args):
    """
    Train the model with the specified configuration

    Args:
        args: Command-line arguments
    """
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training (this may be slow)")

    # Load training data
    domain_texts = load_text_data(args.data)

    # Extract Q&A pairs if requested
    if args.extract_qa:
        qa_pairs = extract_qa_pairs(args.data)
        if qa_pairs:
            domain_texts.extend(qa_pairs)
            logger.info(f"Added {len(qa_pairs)} Q&A pairs to training data")
            logger.info(f"Total training samples: {len(domain_texts)}")

    # Initialize pipeline
    pipeline = CS_SLM_Pipeline(
        domain_texts=domain_texts,
        use_t5=(args.model_type.lower() == 't5'),
        context_size=args.context_size
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save training arguments
    with open(os.path.join(args.output, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load base model
    logger.info("Loading base model...")
    tokenizer, model = pipeline.load_base_model()

    # Train custom tokenizer if requested
    if args.custom_tokenizer:
        logger.info("Training domain-specific tokenizer...")
        tokenizer = pipeline.train_domain_tokenizer(vocab_size=1000)

    # Prepare data
    logger.info("Preparing datasets...")
    data = pipeline.prepare_data(batch_size=args.batch_size)

    # Train model
    logger.info("Training model...")
    model = pipeline.train_model(
        data["train_dataloader"],
        data["val_dataloader"],
        epochs=args.epochs,
        lr=args.learning_rate,
        use_adapters=args.use_adapters
    )

    # Apply optimizations if requested
    if args.quantize or args.prune:
        logger.info("Applying model optimizations...")
        model = pipeline.optimize_model(
            quantize=args.quantize,
            prune=args.prune,
            pruning_ratio=args.pruning_ratio
        )

    # Save the model
    logger.info(f"Saving model to {args.output}...")
    pipeline.save_model(args.output, model)

    logger.info("Training complete!")
    return pipeline


def main():
    """Main function to handle training"""
    parser = create_arg_parser()
    args = parser.parse_args()

    try:
        # Print training configuration
        logger.info("=== Training Configuration ===")
        for arg, value in sorted(vars(args).items()):
            logger.info(f"{arg}: {value}")
        logger.info("============================")

        # Start timer
        start_time = datetime.now()
        logger.info(f"Training started at {start_time}")

        # Train model
        pipeline = train_model(args)

        # End timer
        end_time = datetime.now()
        training_time = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training time: {training_time}")

        return pipeline

    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
