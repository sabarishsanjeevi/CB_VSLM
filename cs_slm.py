#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cs_slm.py - Context-Specialized Small Language Model (CS-SLM) Implementation

This module provides the implementation for creating, training, and using
context-specialized small language models with various optimization techniques.
"""

import os
import sys
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForMaskedLM, AutoModelForCausalLM,
    T5ForConditionalGeneration, T5Tokenizer,
    BertTokenizer, BertForMaskedLM, BertConfig,
    Trainer, TrainingArguments,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple, Union
import time
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################################
# Dataset Classes
#################################################

class ContextualTextDataset(Dataset):
    """Dataset for training context-specialized language models"""

    def __init__(self, texts, tokenizer, max_length=512, is_t5=True):
        """
        Initialize the dataset

        Args:
            texts: List of text samples
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            is_t5: Whether using T5 (True) or BERT (False)
        """
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.is_t5 = is_t5

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.is_t5:
            # For T5, we'll use a denoising objective
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Create a corrupted version of input as target
            target_text = self.corrupt_text(text)
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": targets["input_ids"].squeeze()
            }
        else:
            # For BERT, we'll use masked language modeling
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].clone()
            # Mask 15% of tokens for MLM task
            rand = torch.rand(input_ids.shape)
            mask_arr = (rand < 0.15) * (input_ids != self.tokenizer.cls_token_id) * (
                    input_ids != self.tokenizer.sep_token_id)

            # Get indices of masked tokens
            selection = torch.flatten(mask_arr.nonzero()).tolist()

            # Create labels: set all to -100 (ignore) except masked tokens
            labels = torch.ones_like(input_ids) * -100

            # Set masked tokens in input_ids to mask token id
            input_ids[mask_arr] = self.tokenizer.mask_token_id

            # Set labels for masked tokens to their original value
            labels[mask_arr] = inputs["input_ids"][mask_arr]

            return {
                "input_ids": input_ids.squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }

    def corrupt_text(self, text, corruption_prob=0.15):
        """Corrupt text for T5 denoising objective"""
        tokens = text.split()
        corrupted_tokens = []

        for token in tokens:
            p = random.random()
            if p < corruption_prob:
                corrupted_tokens.append("<mask>")
            else:
                corrupted_tokens.append(token)

        return " ".join(corrupted_tokens)


#################################################
# Model Selection
#################################################

class ModelSelection:
    """Utility class for model selection and initialization"""

    @staticmethod
    def load_base_model(use_t5=True, context_size=512, model_directory="models"):
        """
        Load the base model from the models directory

        Args:
            use_t5: Whether to use T5 (True) or BERT (False)
            context_size: Maximum context size
            model_directory: Directory containing pre-trained models

        Returns:
            tuple: (tokenizer, model)
        """
        model_path = Path(model_directory)

        # Create models directory if it doesn't exist
        if not model_path.exists():
            logger.info(f"Creating models directory: {model_path}")
            model_path.mkdir(parents=True)

        if use_t5:
            model_name = "t5-small"
            model_path_local = model_path / model_name

            # Check if model exists locally
            if model_path_local.exists() and (model_path_local / "pytorch_model.bin").exists():
                logger.info(f"Loading T5-small from local path: {model_path_local}")
                tokenizer = T5Tokenizer.from_pretrained(str(model_path_local))
                model = T5ForConditionalGeneration.from_pretrained(str(model_path_local))
            else:
                logger.info(f"Downloading T5-small from Hugging Face")
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)

                # Save model locally
                logger.info(f"Saving T5-small to {model_path_local}")
                tokenizer.save_pretrained(str(model_path_local))
                model.save_pretrained(str(model_path_local))

            logger.info(f"T5-small loaded: {model.config.d_model} hidden dim, {model.config.num_layers} layers")
        else:
            model_name = "bert-base-cased"
            model_path_local = model_path / model_name

            # Check if model exists locally
            if model_path_local.exists() and (model_path_local / "pytorch_model.bin").exists():
                logger.info(f"Loading BERT-base-cased from local path: {model_path_local}")
                tokenizer = BertTokenizer.from_pretrained(str(model_path_local))
                model = BertForMaskedLM.from_pretrained(str(model_path_local))
            else:
                logger.info(f"Downloading BERT-base-cased from Hugging Face")
                tokenizer = BertTokenizer.from_pretrained(model_name)
                model = BertForMaskedLM.from_pretrained(model_name)

                # Save model locally
                logger.info(f"Saving BERT-base-cased to {model_path_local}")
                tokenizer.save_pretrained(str(model_path_local))
                model.save_pretrained(str(model_path_local))

            logger.info(f"BERT loaded: {model.config.hidden_size} hidden dim, {model.config.num_hidden_layers} layers")

        return tokenizer, model


#################################################
# Knowledge Distillation
#################################################

class KnowledgeDistillation:
    """Knowledge distillation for model compression"""

    def __init__(self, teacher_model, student_config, tokenizer, is_t5=True):
        """
        Initialize knowledge distillation

        Args:
            teacher_model: The larger teacher model
            student_config: Configuration for the smaller student model
            tokenizer: Tokenizer
            is_t5: Whether using T5 (True) or BERT (False)
        """
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.is_t5 = is_t5

        # Create student model with reduced parameters
        if is_t5:
            # Reduce the size of T5 model
            from transformers import T5Config
            self.student_config = T5Config(
                d_model=student_config.get("hidden_size", 256),
                d_ff=student_config.get("intermediate_size", 512),
                num_layers=student_config.get("num_layers", 4),
                num_heads=student_config.get("num_attention_heads", 4),
                vocab_size=self.teacher_model.config.vocab_size
            )
            self.student_model = T5ForConditionalGeneration(self.student_config)
        else:
            # Reduce the size of BERT model
            self.student_config = BertConfig(
                hidden_size=student_config.get("hidden_size", 256),
                intermediate_size=student_config.get("intermediate_size", 512),
                num_hidden_layers=student_config.get("num_layers", 4),
                num_attention_heads=student_config.get("num_attention_heads", 4),
                vocab_size=self.teacher_model.config.vocab_size
            )
            self.student_model = BertForMaskedLM(self.student_config)

        logger.info(
            f"Created student model with {self.student_config.hidden_size if not is_t5 else self.student_config.d_model} hidden dim, {self.student_config.num_hidden_layers if not is_t5 else self.student_config.num_layers} layers")

    def distill(self, train_dataloader, val_dataloader, epochs=3, lr=5e-5, temp=2.0, alpha=0.5):
        """
        Perform knowledge distillation from teacher to student

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate
            temp: Temperature for softening logits
            alpha: Weight for distillation loss vs task loss

        Returns:
            Student model after distillation
        """
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=lr)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Put models in training/eval mode
        self.teacher_model.eval()
        self.student_model.train()

        # Move models to device
        self.teacher_model.to(device)
        self.student_model.to(device)

        distillation_losses = []
        task_losses = []
        total_losses = []
        val_losses = []

        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            epoch_distill_loss = 0
            epoch_task_loss = 0
            epoch_total_loss = 0

            for batch in train_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass through teacher (no grad)
                with torch.no_grad():
                    if self.is_t5:
                        teacher_outputs = self.teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            output_hidden_states=True
                        )
                    else:
                        teacher_outputs = self.teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            output_hidden_states=True
                        )

                # Forward pass through student
                if self.is_t5:
                    student_outputs = self.student_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        output_hidden_states=True
                    )
                else:
                    student_outputs = self.student_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        output_hidden_states=True
                    )

                # Task-specific loss (already calculated)
                task_loss = student_outputs.loss

                # Knowledge distillation loss (from hidden states)
                # We'll use the last hidden state
                teacher_hidden = teacher_outputs.hidden_states[-1]
                student_hidden = student_outputs.hidden_states[-1]

                # If dimensions differ, project student hidden states
                if teacher_hidden.shape[-1] != student_hidden.shape[-1]:
                    projection = nn.Linear(student_hidden.shape[-1], teacher_hidden.shape[-1]).to(device)
                    student_hidden = projection(student_hidden)

                # Calculate MSE loss between hidden states
                distill_loss = F.mse_loss(student_hidden, teacher_hidden)

                # Combined loss
                loss = alpha * distill_loss + (1 - alpha) * task_loss

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_distill_loss += distill_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_total_loss += loss.item()

            # Calculate average losses for the epoch
            avg_distill_loss = epoch_distill_loss / len(train_dataloader)
            avg_task_loss = epoch_task_loss / len(train_dataloader)
            avg_total_loss = epoch_total_loss / len(train_dataloader)

            distillation_losses.append(avg_distill_loss)
            task_losses.append(avg_task_loss)
            total_losses.append(avg_total_loss)

            # Validate
            self.student_model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.student_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

            logger.info(f"Epoch {epoch + 1} - Train loss: {avg_total_loss:.4f}, Val loss: {avg_val_loss:.4f}")
            self.student_model.train()

        # Create and save a plot of the training progress
        plt.figure(figsize=(10, 6))
        plt.plot(distillation_losses, label='Distillation Loss')
        plt.plot(task_losses, label='Task Loss')
        plt.plot(total_losses, label='Total Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Knowledge Distillation Training Progress')
        plt.savefig('distillation_progress.png')

        return self.student_model


#################################################
# Model Pruning
#################################################

class ModelPruning:
    """Model weight pruning for size reduction"""

    def __init__(self, model, is_t5=True):
        """
        Initialize model pruning

        Args:
            model: Model to prune
            is_t5: Whether using T5 (True) or BERT (False)
        """
        self.model = model
        self.is_t5 = is_t5

    def magnitude_pruning(self, pruning_ratio=0.3):
        """
        Prune model weights based on magnitude

        Args:
            pruning_ratio: Ratio of weights to prune (0.0 to 1.0)

        Returns:
            Pruned model
        """
        logger.info(f"Pruning {pruning_ratio * 100:.1f}% of model weights")

        # Get all parameters
        state_dict = self.model.state_dict()
        total_params = 0
        zero_params = 0

        # Iterate through each parameter tensor
        for name, param in list(state_dict.items()):
            # Skip non-weight parameters like bias and layer norm
            if 'weight' not in name or 'norm' in name or 'embedding' in name:
                continue

            # Get tensor values and shape
            tensor = param.data.cpu().numpy()
            total_params += tensor.size

            # Determine threshold for pruning
            abs_tensor = np.abs(tensor)
            threshold = np.percentile(abs_tensor, pruning_ratio * 100)

            # Create mask for values to keep (above threshold)
            mask = abs_tensor > threshold

            # Apply mask (set values below threshold to zero)
            pruned_tensor = tensor * mask
            zero_params += tensor.size - np.count_nonzero(mask)

            # Update model weight
            state_dict[name] = torch.tensor(pruned_tensor, dtype=param.dtype, device=param.device)

        # Load pruned state dict
        self.model.load_state_dict(state_dict)

        logger.info(
            f"Model pruned: {zero_params}/{total_params} parameters set to zero ({zero_params / total_params * 100:.2f}%)")

        return self.model


#################################################
# Model Quantization
#################################################

class ModelQuantization:
    """Model quantization for size reduction"""

    def __init__(self, model, tokenizer, is_t5=True):
        """
        Initialize model quantization

        Args:
            model: Model to quantize
            tokenizer: Tokenizer
            is_t5: Whether using T5 (True) or BERT (False)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.is_t5 = is_t5

    def quantize_model(self, quantization_bits=8):
        """
        Quantize model to reduced precision

        Args:
            quantization_bits: Bit precision for quantization (8 or 16)

        Returns:
            Quantized model
        """
        if quantization_bits not in [8, 16]:
            raise ValueError("Only 8-bit and 16-bit quantization is supported")

        logger.info(f"Quantizing model to {quantization_bits}-bit precision")

        # We'll use PyTorch's quantization functionality
        if quantization_bits == 8:
            # For 8-bit quantization
            try:
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as e:
                logger.warning(f"Dynamic quantization failed: {e}. Using manual quantization.")
                # Fallback: Manual weight quantization
                state_dict = self.model.state_dict()
                for name, param in list(state_dict.items()):
                    if 'weight' in name and 'embedding' not in name:
                        # Apply simple 8-bit quantization
                        tensor = param.data.cpu()
                        min_val = torch.min(tensor)
                        max_val = torch.max(tensor)
                        scale = (max_val - min_val) / 255.0
                        zero_point = min_val

                        # Quantize to int8
                        quantized = torch.round((tensor - zero_point) / scale).clamp(0, 255).to(torch.uint8)

                        # Dequantize (for inference)
                        dequantized = scale * quantized.float() + zero_point

                        # Update model weight
                        state_dict[name] = dequantized.to(param.dtype).to(param.device)

                # Create a new model with quantized weights
                if self.is_t5:
                    quantized_model = T5ForConditionalGeneration.from_pretrained(None, config=self.model.config,
                                                                                 state_dict=state_dict)
                else:
                    quantized_model = BertForMaskedLM.from_pretrained(None, config=self.model.config,
                                                                      state_dict=state_dict)
        else:
            # For 16-bit quantization (half precision)
            quantized_model = self.model.half()

        # Calculate model size before and after
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)  # MB
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)  # MB

        logger.info(f"Original model size: {original_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")

        return quantized_model


#################################################
# Custom Tokenizer
#################################################

class CustomTokenizer:
    """Domain-specific tokenizer training"""

    def __init__(self, base_tokenizer, domain_texts):
        """
        Create a domain-specific tokenizer based on a base tokenizer

        Args:
            base_tokenizer: Base tokenizer to extend
            domain_texts: List of domain-specific texts for vocabulary learning
        """
        self.base_tokenizer = base_tokenizer
        self.domain_texts = domain_texts

    def train_domain_tokenizer(self, vocab_size=1000):
        """
        Train a domain-specific tokenizer

        Args:
            vocab_size: Number of new tokens to add to the vocabulary

        Returns:
            Domain-specific tokenizer
        """
        logger.info("Training domain-specific tokenizer")

        try:
            # Create a tokenizer with the same settings as the base one
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers

            # Initialize with base vocabulary
            base_vocab = self.base_tokenizer.get_vocab()

            # Create a BPE tokenizer
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            # Create a trainer
            trainer = trainers.BpeTrainer(
                vocab_size=len(base_vocab) + vocab_size,
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )

            # Prepare training data
            files = []
            if not os.path.exists("domain_texts"):
                os.makedirs("domain_texts")

            # Write texts to files
            for i, text in enumerate(self.domain_texts):
                filename = f"domain_texts/text_{i}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
                files.append(filename)

            # Train the tokenizer
            tokenizer.train(files, trainer)

            # Create a transformers tokenizer from the trained one
            from transformers import PreTrainedTokenizerFast

            fast_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]"
            )

            # Add special tokens to match base tokenizer
            special_tokens = {}
            if hasattr(self.base_tokenizer, 'additional_special_tokens'):
                special_tokens["additional_special_tokens"] = self.base_tokenizer.additional_special_tokens

            # Add any other special tokens from the base tokenizer
            for token_name in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token",
                               "mask_token"]:
                if hasattr(self.base_tokenizer, token_name):
                    token_value = getattr(self.base_tokenizer, token_name)
                    if token_value is not None:
                        special_tokens[token_name] = token_value

            # Apply special tokens
            fast_tokenizer.add_special_tokens(special_tokens)

            logger.info(f"Domain tokenizer trained with {len(fast_tokenizer)} tokens")

            return fast_tokenizer

        except ImportError:
            logger.warning("Tokenizers library not found. Falling back to base tokenizer.")
            return self.base_tokenizer


#################################################
# Context Adapters
#################################################

class ContextAdapter(nn.Module):
    """Adapter module for parameter-efficient fine-tuning"""

    def __init__(self, hidden_size, adapter_size):
        """
        Initialize adapter module

        Args:
            hidden_size: Size of the hidden representations
            adapter_size: Size of the adapter bottleneck
        """
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        """Forward pass through the adapter"""
        residual = hidden_states
        x = self.down_project(hidden_states)
        x = self.activation(x)
        x = self.up_project(x)
        output = self.layer_norm(x + residual)
        return output


class AdapterModel:
    """Adapter management for domain adaptation"""

    def __init__(self, model, is_t5=True):
        """
        Initialize adapter model

        Args:
            model: Base model to add adapters to
            is_t5: Whether using T5 (True) or BERT (False)
        """
        self.model = model
        self.is_t5 = is_t5
        self.adapters = {}

    def add_adapters(self, adapter_size=64):
        """
        Add adapter modules to all transformer layers

        Args:
            adapter_size: Size of the adapter bottleneck

        Returns:
            Model with adapters
        """
        logger.info(f"Adding adapters with bottleneck size {adapter_size}")

        # Get hidden size from model config
        if self.is_t5:
            hidden_size = self.model.config.d_model
        else:
            hidden_size = self.model.config.hidden_size

        # Create and attach adapters
        adapters = nn.ModuleDict()

        try:
            if self.is_t5:
                # For T5, add adapters to encoder and decoder
                for i in range(self.model.config.num_layers):
                    # Encoder adapters
                    encoder_adapter = ContextAdapter(hidden_size, adapter_size)
                    self.model.encoder.block[i].layer[1].add_module("adapter", encoder_adapter)
                    adapters[f"encoder_layer_{i}"] = encoder_adapter

                    # Decoder adapters
                    decoder_adapter = ContextAdapter(hidden_size, adapter_size)
                    self.model.decoder.block[i].layer[2].add_module("adapter", decoder_adapter)
                    adapters[f"decoder_layer_{i}"] = decoder_adapter
            else:
                # For BERT, add adapters to each layer
                for i in range(self.model.config.num_hidden_layers):
                    adapter = ContextAdapter(hidden_size, adapter_size)
                    self.model.bert.encoder.layer[i].output.add_module("adapter", adapter)
                    adapters[f"layer_{i}"] = adapter
        except AttributeError as e:
            logger.warning(f"Error adding adapters: {e}")
            logger.warning("Model structure doesn't match expected T5/BERT architecture")
            logger.warning("Using alternative adapter insertion method")

            # Alternative adapter insertion (less intrusive)
            # Create wrapper modules that will be inserted during forward pass
            if self.is_t5:
                # For T5, we'll wrap the encoder and decoder modules
                class T5AdapterWrapper(nn.Module):
                    def __init__(self, module, adapter):
                        super().__init__()
                        self.module = module
                        self.adapter = adapter

                    def forward(self, *args, **kwargs):
                        output = self.module(*args, **kwargs)
                        if isinstance(output, tuple):
                            # Apply adapter to the hidden states
                            hidden_states = output[0]
                            adapted_hidden_states = self.adapter(hidden_states)
                            return (adapted_hidden_states,) + output[1:]
                        else:
                            return self.adapter(output)

                # Add adapters to last layers of encoder and decoder
                encoder_adapter = ContextAdapter(hidden_size, adapter_size)
                decoder_adapter = ContextAdapter(hidden_size, adapter_size)

                # Store original forward methods
                original_encoder_forward = self.model.encoder.forward
                original_decoder_forward = self.model.decoder.forward

                # Define new forward methods with adapters
                def encoder_forward_with_adapter(*args, **kwargs):
                    output = original_encoder_forward(*args, **kwargs)
                    last_hidden_state = output[0]
                    adapted_hidden_state = encoder_adapter(last_hidden_state)
                    return (adapted_hidden_state,) + output[1:]

                def decoder_forward_with_adapter(*args, **kwargs):
                    output = original_decoder_forward(*args, **kwargs)
                    last_hidden_state = output[0]
                    adapted_hidden_state = decoder_adapter(last_hidden_state)
                    return (adapted_hidden_state,) + output[1:]

                # Replace forward methods
                self.model.encoder.forward = encoder_forward_with_adapter
                self.model.decoder.forward = decoder_forward_with_adapter

                # Add adapters to dict
                adapters["encoder"] = encoder_adapter
                adapters["decoder"] = decoder_adapter

            else:
                # For BERT, we'll wrap the output layer
                bert_adapter = ContextAdapter(hidden_size, adapter_size)

                # Store original forward method
                original_bert_forward = self.model.bert.forward

                # Define new forward method with adapter
                def bert_forward_with_adapter(*args, **kwargs):
                    output = original_bert_forward(*args, **kwargs)
                    last_hidden_state = output[0]
                    adapted_hidden_state = bert_adapter(last_hidden_state)
                    return (adapted_hidden_state,) + output[1:]

                # Replace forward method
                self.model.bert.forward = bert_forward_with_adapter

                # Add adapter to dict
                adapters["bert"] = bert_adapter

        # Store the adapter modules
        self.adapters = adapters

        # Freeze the base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the adapter parameters
        for adapter in adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True

        logger.info(f"Added {len(adapters)} adapter modules to the model")

        return self.model

    def train_adapters(self, train_dataloader, val_dataloader, epochs=3, lr=1e-4):
        """
        Train only the adapter modules while keeping the base model frozen

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Model with trained adapters
        """
        # Get only the adapter parameters for optimization
        optimizer_params = []
        for adapter in self.adapters.values():
            optimizer_params.extend(adapter.parameters())

        optimizer = torch.optim.AdamW(optimizer_params, lr=lr)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Put model in training mode
        self.model.train()

        # Move model to device
        self.model.to(device)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
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

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optimizer_params, 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            # Calculate average loss for the epoch
            avg_train_loss = epoch_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

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

            logger.info(f"Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
            self.model.train()

        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Adapter Training Progress')
        plt.savefig('adapter_training_progress.png')

        return self.model

    def save_adapters(self, path):
        """Save adapter weights only"""
        os.makedirs(path, exist_ok=True)

        # Save adapter state dict
        adapter_state = {k: v.state_dict() for k, v in self.adapters.items()}
        torch.save(adapter_state, os.path.join(path, "adapters.pt"))

        logger.info(f"Saved adapters to {path}")

    def load_adapters(self, path):
        """Load adapter weights"""
        adapter_state = torch.load(os.path.join(path, "adapters.pt"))

        for name, state in adapter_state.items():
            self.adapters[name].load_state_dict(state)

        logger.info(f"Loaded adapters from {path}")


#################################################
# Response Generator
#################################################

class ResponseGenerator:
    """Response generation for context-specialized language models"""

    def __init__(self, model, tokenizer, is_t5=True, max_length=100, device=None):
        """
        Initialize response generator

        Args:
            model: Language model
            tokenizer: Tokenizer
            is_t5: Whether model is T5 (True) or BERT (False)
            max_length: Maximum response length
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.is_t5 = is_t5
        self.max_length = max_length
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Put model in eval mode
        self.model.eval()
        self.model.to(self.device)

        # Initialize conversation history
        self.conversation_history = []

    def _format_response_for_bert(self, prompt, predicted_tokens, num_tokens=5):
        """
        Create a coherent response using BERT's masked prediction capabilities

        Args:
            prompt: The user's input
            predicted_tokens: List of predicted tokens from BERT
            num_tokens: Number of tokens to generate for the response

        Returns:
            A formulated response
        """
        # Map common question words to response templates
        question_templates = {
            "who": ["I am", "This is", "That's", "It's"],
            "what": ["It's", "That's", "This is", "It refers to"],
            "when": ["In", "During", "At", "Around"],
            "where": ["In", "At", "Near", "Around"],
            "why": ["Because", "The reason is", "It's due to"],
            "how": ["By", "Through", "Using", "With"],
            "is": ["Yes,", "No,", "Indeed,", "Actually,"],
            "are": ["Yes, they are", "No, they aren't", "Indeed, they are", "Actually,"],
            "can": ["Yes, it can", "No, it cannot", "It depends on"],
            "do": ["Yes,", "No,", "Sometimes,", "It depends,"]
        }

        # Common response starters
        general_starters = [
            "I think", "Based on context,", "From what I know,",
            "According to information,", "In my understanding,",
            "Let me explain:", "That's a good question.",
            "To answer that,"
        ]

        # Extract the first word of the prompt (lowercase)
        first_word = prompt.lower().strip().split()[0] if prompt.strip() else ""

        # Choose template based on first word or use general starter
        if first_word in question_templates:
            starter = random.choice(question_templates[first_word])
        else:
            starter = random.choice(general_starters)

        # Generate a coherent response using the context and predicted tokens
        response_words = [starter]

        # Add predicted tokens, interspersed with connecting words
        connecting_words = ["and", "also", "additionally", "moreover", "furthermore", "", "", ""]

        # Use only unique tokens from predictions
        unique_tokens = []
        for token in predicted_tokens:
            # Clean token and only use meaningful ones
            cleaned_token = token.replace("##", "").strip()
            if (len(cleaned_token) > 2 and
                    cleaned_token not in unique_tokens and
                    cleaned_token.lower() not in ["the", "and", "for", "that", "this"]):
                unique_tokens.append(cleaned_token)

        # Use a subset of unique tokens to build response
        used_tokens = unique_tokens[:min(num_tokens, len(unique_tokens))]

        # Craft response with context awareness
        for i, token in enumerate(used_tokens):
            # Add the token
            response_words.append(token)

            # Add connecting word (except for the last token)
            if i < len(used_tokens) - 1:
                response_words.append(random.choice(connecting_words))

        # Make sure the response isn't too short
        if len(response_words) < 3:
            response_words.append("is the appropriate response based on the context.")

        # Join the words to form a response
        response = " ".join(response_words)

        # Do some basic cleanup
        response = response.replace(" ,", ",").replace("  ", " ")

        # Make it sentence-case if it doesn't start with "I"
        if not response.startswith("I "):
            response = response[0].upper() + response[1:]

        # End with a period if necessary
        if not response.endswith((".", "!", "?")):
            response += "."

        return response

    def generate_response(self, prompt):
        """
        Generate a response to the user's prompt

        Args:
            prompt: User's input text

        Returns:
            Generated response
        """
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": prompt})

            if self.is_t5:
                # T5 response generation
                # Format prompt for T5
                input_text = f"respond: {prompt}"

                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.model.config.max_length if hasattr(self.model.config, 'max_length') else 512,
                    truncation=True,
                    padding="max_length"
                ).to(self.device)

                with torch.no_grad():
                    output_sequences = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        num_return_sequences=1
                    )

                response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

            else:
                # BERT response generation
                # For BERT, we'll use masked tokens to craft a response

                # 1. Create a template with masked tokens
                template = f"{prompt} [SEP] [MASK] [MASK] [MASK] [MASK] [MASK]"

                # 2. Tokenize the template
                inputs = self.tokenizer(
                    template,
                    return_tensors="pt",
                    max_length=self.model.config.max_position_embeddings if hasattr(self.model.config,
                                                                                    'max_position_embeddings') else 512,
                    truncation=True,
                    padding="max_length"
                ).to(self.device)

                # 3. Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 4. Find masked token positions
                mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]

                # 5. Get predictions for each masked position
                predicted_tokens = []

                for idx in mask_token_index:
                    # Get logits for this position
                    logits = outputs.logits[0, idx, :]

                    # Get top 5 predictions
                    top_tokens = torch.topk(logits, k=5)
                    top_token_ids = top_tokens.indices.tolist()

                    # Decode token IDs to strings
                    for token_id in top_token_ids:
                        token = self.tokenizer.decode([token_id])
                        predicted_tokens.append(token)

                # 6. Format a meaningful response using the predicted tokens
                response = self._format_response_for_bert(prompt, predicted_tokens)

            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"


#################################################
# Interface Class
#################################################

class ModelInterface:
    """Interface for context-specialized language models"""

    def __init__(self, model_path, max_length=100):
        """
        Initialize model interface

        Args:
            model_path: Path to the model
            max_length: Maximum response length
        """
        self.model_path = Path(model_path)
        self.max_length = max_length

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model info
        try:
            with open(self.model_path / "model_info.json", "r") as f:
                self.model_info = json.load(f)
            self.is_t5 = self.model_info.get("use_t5", True)
        except:
            # Default to BERT if info not available
            logger.warning("No model_info.json found, defaulting to BERT model")
            self.is_t5 = False

        # Load model
        logger.info(f"Loading model from {self.model_path}")

        if self.is_t5:
            self.tokenizer = T5Tokenizer.from_pretrained(str(self.model_path))
            self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_path))
        else:
            self.tokenizer = BertTokenizer.from_pretrained(str(self.model_path))
            self.model = BertForMaskedLM.from_pretrained(str(self.model_path))

        # Initialize response generator
        self.generator = ResponseGenerator(
            self.model,
            self.tokenizer,
            is_t5=self.is_t5,
            max_length=self.max_length,
            device=self.device
        )

        logger.info("Model loaded successfully")

    def generate_response(self, prompt):
        """
        Generate a response to the user's prompt

        Args:
            prompt: User's input

        Returns:
            Generated response
        """
        return self.generator.generate_response(prompt)
