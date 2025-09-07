#!/usr/bin/env python3
"""
Advanced TRL training script with controller integration and distributed coordination.
Combines controller-managed checkpointing with production-grade distributed training.
"""

import os
import json
import time
import signal
import torch
import random
import numpy
from numpy.core.multiarray import _reconstruct
import torch.serialization
import torch.distributed as dist
import logging
from datetime import datetime
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

# Safe tensor loading configuration
torch.serialization.add_safe_globals([_reconstruct, numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType])

# Patch torch.load to handle weights_only parameter and device mapping
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    # Add map_location for CPU-only environments or device compatibility
    if 'map_location' not in kwargs:
        if torch.cuda.is_available():
            kwargs['map_location'] = 'cuda'
        else:
            kwargs['map_location'] = 'cpu'
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

class AdvancedDistributedCheckpointCallback(TrainerCallback):
    """
    Production-grade distributed SIGTERM handling with tensor-based coordination.
    Combines the example's advanced features with controller integration.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_requested = False
        self.save_triggered = False
        self.checkpoint_stream = None
        self.sigterm_tensor = None
        
        # Use controller-injected checkpoint configuration
        self.checkpoint_enabled = os.environ.get('CHECKPOINT_ENABLED', 'false').lower() == 'true'
        self.checkpoint_uri = os.environ.get('CHECKPOINT_URI', '/workspace/checkpoints')
        
        # Controller progress file (simple)
        self.progress_file = os.environ.get('TRAINING_PROGRESS_FILE', '/workspace/training_progress.json')
        
        # Advanced distributed checkpointing initialized

    def _log_message(self, message: str):
        """Helper to print messages with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def _write_progress(self, state: TrainerState):
        """Simple progress file writer for controller consumption"""
        # Only write from rank 0
        rank = int(os.environ.get('RANK', '0'))
        if rank != 0:
            return
            
        try:
            # Extract metrics from trainer state
            latest_loss = 0.0
            latest_lr = 0.0
            if state.log_history:
                latest_log = state.log_history[-1]
                latest_loss = latest_log.get('loss', latest_log.get('train_loss', latest_log.get('training_loss', 0.0)))
                latest_lr = latest_log.get('learning_rate', latest_log.get('lr', latest_log.get('train_lr', 0.0)))
            
            progress_data = {
                "epoch": int(state.epoch) if state.epoch else 1,
                "totalEpochs": int(state.num_train_epochs) if state.num_train_epochs else 1,
                "step": state.global_step,
                "totalSteps": state.max_steps,
                "loss": f"{latest_loss:.4f}",
                "learningRate": f"{latest_lr:.6f}",
                "percentComplete": f"{(state.global_step / state.max_steps * 100):.1f}" if state.max_steps > 0 else "0.0",
                "lastUpdateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            
            # Atomic write
            temp_file = self.progress_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            os.rename(temp_file, self.progress_file)
            os.chmod(self.progress_file, 0o644)
            
        except Exception as e:
            pass  # Silent fail - controller handles missing files gracefully

    def _init_distributed_signal_tensor(self):
        """Initialize tensor for distributed SIGTERM signaling."""
        try:
            if dist.is_initialized():
                device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
                self.sigterm_tensor = torch.zeros(1, dtype=torch.float32, device=device)
                self._log_message(f"Initialized distributed SIGTERM tensor on device: {device}")
            else:
                self._log_message("Distributed training not initialized - using local SIGTERM handling only")
        except Exception as e:
            self._log_message(f"Failed to initialize distributed SIGTERM tensor: {e}. Using local handling only.")

    def _check_distributed_sigterm(self):
        """Check if any rank has received SIGTERM."""
        try:
            if dist.is_initialized() and self.sigterm_tensor is not None:
                dist.all_reduce(self.sigterm_tensor, op=dist.ReduceOp.MAX)
                return self.sigterm_tensor.item() > 0.5
        except Exception as e:
            self._log_message(f"Distributed SIGTERM check failed: {e}. Using local signal only.")
        return self.checkpoint_requested

    def _sigterm_handler(self, signum, frame):
        """Sets a flag and updates the tensor to indicate that a SIGTERM signal was received."""
        rank = os.environ.get("RANK", "-1")
        self._log_message(f"Rank {rank}: SIGTERM received, flagging for distributed checkpoint.")
        self.checkpoint_requested = True
        if self.sigterm_tensor is not None:
            self.sigterm_tensor.fill_(1.0)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        rank = os.environ.get("RANK", "-1")
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_distributed_signal_tensor()
        
        if torch.cuda.is_available():
            self.checkpoint_stream = torch.cuda.Stream()
            self._log_message(f"Rank {rank}: Created dedicated CUDA stream for checkpointing.")

        signal.signal(signal.SIGTERM, self._sigterm_handler)
        self._log_message(f"Rank {rank}: Advanced distributed SIGTERM handler registered.")

        try:
            if dist.is_initialized():
                dist.barrier()
                self._log_message(f"Rank {rank}: Distributed coordination setup synchronized across all ranks")
        except Exception as e:
            self._log_message(f"Rank {rank}: Failed to synchronize distributed setup: {e}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Write progress for controller (every logging step)
        if state.global_step % args.logging_steps == 0:
            self._write_progress(state)
            
        if self._check_distributed_sigterm() and not self.save_triggered:
            rank = os.environ.get("RANK", "-1")
            self._log_message(f"Rank {rank}: Distributed SIGTERM detected, initiating checkpoint at step {state.global_step}.")
            self.save_triggered = True
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Write final progress
        self._write_progress(state)
        
        rank = os.environ.get("RANK", "-1")
        if rank == "0" and self.checkpoint_requested:
            self._log_message(f"Rank {rank}: Training ended due to distributed SIGTERM checkpoint request.")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        rank = os.environ.get("RANK", "-1")
        if rank == "0":
            self._log_message(f"Rank {rank}: Checkpoint save completed.")
            if self.checkpoint_requested:
                self._log_message(f"Rank {rank}: Distributed SIGTERM-triggered checkpoint save finished successfully.")

def setup_distributed():
    """Initialize distributed training using operator-injected PET environment variables"""
    # Use PET_* environment variables injected by the training operator
    node_rank = int(os.getenv('PET_NODE_RANK', '0'))
    num_nodes = int(os.getenv('PET_NNODES', '1'))
    nproc_per_node = int(os.getenv('PET_NPROC_PER_NODE', '1'))
    master_addr = os.getenv('PET_MASTER_ADDR', 'localhost')
    master_port = os.getenv('PET_MASTER_PORT', '29500')
    
    # Calculate standard PyTorch distributed variables
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = num_nodes * nproc_per_node
    global_rank = node_rank * nproc_per_node + local_rank
    
    # Set standard PyTorch environment variables for compatibility
    os.environ['RANK'] = str(global_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize distributed training if world_size > 1
    if world_size > 1:
        try:
            torch.distributed.init_process_group(
                backend='gloo',
                rank=global_rank,
                world_size=world_size
            )
            torch.distributed.barrier()
        except Exception as e:
            print(f"Warning: Failed to initialize distributed training: {e}")
    
    return local_rank, global_rank, world_size

def load_dataset_from_initializer():
    """Load dataset from V2 initializer or fallback to download"""
    dataset_dir = Path("/workspace/dataset")
    
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        try:
            full_dataset = load_from_disk(str(dataset_dir))
            if isinstance(full_dataset, dict):
                train_dataset = full_dataset.get('train', full_dataset.get('train[:100]'))
                test_dataset = full_dataset.get('test', full_dataset.get('test[:20]'))
            else:
                # Split dataset if it's not already split
                train_size = min(100, len(full_dataset) - 20)
                train_dataset = full_dataset.select(range(train_size))
                test_dataset = full_dataset.select(range(train_size, min(train_size + 20, len(full_dataset))))
            
            return train_dataset, test_dataset
        except Exception as e:
            print(f"Failed to load from initializer: {e}")
    
    # Fallback to direct download
    dataset_name = os.getenv('DATASET_NAME', 'tatsu-lab/alpaca')
    train_split = os.getenv('DATASET_TRAIN_SPLIT', 'train[:100]')
    test_split = os.getenv('DATASET_TEST_SPLIT', 'train[100:120]')
    
    train_dataset = load_dataset(dataset_name, split=train_split)
    test_dataset = load_dataset(dataset_name, split=test_split)
    
    return train_dataset, test_dataset

def load_model_from_initializer():
    """Load model and tokenizer from V2 initializer or fallback to download"""
    model_dir = Path("/workspace/model")
    
    if model_dir.exists() and any(model_dir.iterdir()):
        model_path = str(model_dir)
    else:
        model_path = os.getenv('MODEL_NAME', 'gpt2')
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set up chat template for instruction following
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "### Instruction:\n{{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}"
                "### Response:\n{{ message['content'] }}{{ eos_token }}\n"
                "{% endif %}"
                "{% endfor %}"
            )
        
        return model_path, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to gpt2
        model_path = 'gpt2'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model_path, tokenizer

def prepare_datasets(train_dataset, test_dataset, tokenizer):
    """Prepare datasets for training with proper formatting"""
    def template_dataset(sample):
        # Handle different dataset formats
        if 'instruction' in sample and 'output' in sample:
            # Alpaca format
            messages = [
                {"role": "user", "content": sample['instruction']},
                {"role": "assistant", "content": sample['output']},
            ]
        elif 'question' in sample and 'answer' in sample:
            # GSM8K format
            messages = [
                {"role": "user", "content": sample['question']},
                {"role": "assistant", "content": sample['answer']},
            ]
        else:
            # Fallback format
            content = str(sample.get('text', sample.get('content', 'Sample text')))
            messages = [
                {"role": "user", "content": "Complete this text:"},
                {"role": "assistant", "content": content},
            ]
        
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    
    # Get original column names to remove
    train_columns = list(train_dataset.features.keys())
    train_columns.remove('text') if 'text' in train_columns else None
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=train_columns)
    
    if test_dataset is not None:
        test_columns = list(test_dataset.features.keys())
        test_columns.remove('text') if 'text' in test_columns else None
        test_dataset = test_dataset.map(template_dataset, remove_columns=test_columns)
    
    return train_dataset, test_dataset

def get_training_parameters():
    """Get training parameters from controller and environment variables"""
    # Use controller-injected checkpoint configuration
    checkpoint_dir = Path(os.getenv('CHECKPOINT_URI', '/workspace/checkpoints'))
    checkpoint_enabled = os.getenv('CHECKPOINT_ENABLED', 'false').lower() == 'true'
    checkpoint_interval = os.getenv('CHECKPOINT_INTERVAL', '30s')
    max_checkpoints = int(os.getenv('CHECKPOINT_MAX_RETAIN', '5'))
    
    # Training hyperparameters from environment (with sensible defaults)
    parameters = {
        'model_name_or_path': os.getenv('MODEL_NAME', 'gpt2'),
        'model_revision': 'main',
        'torch_dtype': 'bfloat16',
        'use_peft': True,
        'lora_r': int(os.getenv('LORA_R', '16')),
        'lora_alpha': int(os.getenv('LORA_ALPHA', '32')),
        'lora_dropout': float(os.getenv('LORA_DROPOUT', '0.1')),
        'lora_target_modules': ['c_attn', 'c_proj'],  # GPT-2 specific
        'dataset_name': os.getenv('DATASET_NAME', 'tatsu-lab/alpaca'),
        'dataset_config': 'main',
        'dataset_train_split': os.getenv('DATASET_TRAIN_SPLIT', 'train[:100]'),
        'dataset_test_split': os.getenv('DATASET_TEST_SPLIT', 'train[100:120]'),
        'max_seq_length': int(os.getenv('MAX_SEQ_LENGTH', '512')),
        'num_train_epochs': int(os.getenv('MAX_EPOCHS', '3')),
        'per_device_train_batch_size': int(os.getenv('BATCH_SIZE', '2')),
        'per_device_eval_batch_size': int(os.getenv('BATCH_SIZE', '2')),
        'eval_strategy': 'steps',
        'eval_steps': int(os.getenv('EVAL_STEPS', '25')),
        'bf16': torch.cuda.is_available(),  # Only use bf16 if CUDA is available
        'fp16': not torch.cuda.is_available(),  # Use fp16 for CPU training
        'learning_rate': float(os.getenv('LEARNING_RATE', '5e-5')),
        'warmup_steps': int(os.getenv('WARMUP_STEPS', '10')),
        'lr_scheduler_type': 'cosine',
        'optim': 'adamw_torch',
        'max_grad_norm': 1.0,
        'seed': 42,
        'gradient_accumulation_steps': int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4')),
        'save_strategy': 'steps',
        'save_steps': int(os.getenv('SAVE_STEPS', '20')),
        'save_total_limit': max_checkpoints if checkpoint_enabled else None,
        'logging_strategy': 'steps',
        'logging_steps': int(os.getenv('LOGGING_STEPS', '5')),
        'report_to': [],
        'output_dir': str(checkpoint_dir),
    }
    
    # Controller checkpointing configuration loaded
    
    return parameters

def main():
    """Main training function with controller integration and distributed coordination"""
    
    # Import os locally to avoid any scoping issues
    import os
    
    # Setup distributed training
    local_rank, global_rank, world_size = setup_distributed()
    
    # Create necessary directories
    os.makedirs("/workspace/cache/transformers", exist_ok=True)
    os.makedirs("/workspace/cache", exist_ok=True)
    os.makedirs("/workspace/cache/datasets", exist_ok=True)
    
    # Get training parameters
    parameters = get_training_parameters()
    checkpoint_dir = Path(parameters['output_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training configuration loaded
    
    # Parse configuration using TrlParser for robust handling
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(parameters)
    
    set_seed(training_args.seed)
    
    # Load components using V2 initializers
    model_path, tokenizer = load_model_from_initializer()
    train_dataset, test_dataset = load_dataset_from_initializer()
    train_dataset, test_dataset = prepare_datasets(train_dataset, test_dataset, tokenizer)
    
    # Initialize trainer with advanced callbacks
    callbacks = [
        AdvancedDistributedCheckpointCallback(str(checkpoint_dir))  # Advanced distributed coordination
    ]
    
    # Initialize SFTTrainer with controller integration
    trainer = SFTTrainer(
        model=model_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # Print trainable parameters info
    if trainer.accelerator.is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    
    # Check for resume from checkpoint (controller-managed)
    checkpoint = get_last_checkpoint(training_args.output_dir)
    if checkpoint is not None:
        # Validate checkpoint compatibility before resuming
        try:
            checkpoint_files = os.listdir(checkpoint)
            # Check if this is a valid checkpoint directory
            if 'trainer_state.json' not in checkpoint_files:
                checkpoint = None
        except Exception as e:
            print(f"Checkpoint validation failed: {e}")
            checkpoint = None
    
    # Start training
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as e:
        print(f"Training failed: {e}")
        # If checkpoint loading failed, try without checkpoint
        if checkpoint is not None:
            try:
                trainer.train(resume_from_checkpoint=None)
            except Exception as retry_e:
                print(f"Training failed even from scratch: {retry_e}")
                raise retry_e
        else:
            raise
    
    # Save final model
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()

