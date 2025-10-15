"""
Betfair Transformer Model - Main Training Script

This module contains the transformer architecture and training logic.
Tokenization functions have been moved to tokenization.py to avoid monolith.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import json
import gzip
import orjson

# Import tokenization functions
from tokenization import (
    create_token_from_bin, get_fixed_horse_ordering, create_token_from_bin_with_fixed_order,
    calculate_ev_targets_per_step, lay_price_predictions_to_ev
)

# Import chunked sampling for memory efficiency
from chunked_sampler import ChunkSampler, FixedValidationSet, collate_blocks

#hyperparameters
batch_size = 64
micro_batch_size = 16  # Smaller micro-batches to fit in memory
gradient_accumulation_steps = batch_size // micro_batch_size  # 4 steps
block_size = 256
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_other_horses = 23
token_size = 12 + max_other_horses * 5
prediction_offset=32

# Chunked sampling parameters
chunk_size_races = 500      # Load 500 races per chunk
blocks_per_chunk = 4000     # Generate 4000 blocks per chunk
val_blocks = 1000           # Fixed validation set size
# ---------------------------------------------------------

torch.manual_seed(1802)


def masked_huber(pred, target, mask, delta=0.1):
    """
    Masked Huber loss to ignore invalid/bad windows during training.
    
    Args:
        pred: Predictions tensor (batch_size, seq_len)
        target: Target tensor (batch_size, seq_len) 
        mask: Boolean mask tensor (batch_size, seq_len) - True for valid, False for invalid
        delta: Huber loss delta parameter
    
    Returns:
        Scalar loss tensor
    """
    import torch
    import torch.nn.functional as F
    
    # Only compute loss on valid (masked=True) elements
    valid_pred = pred[mask]
    valid_target = target[mask]
    
    if len(valid_pred) == 0:
        # No valid elements - return zero loss
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Compute Huber loss on valid elements only
    loss = F.huber_loss(valid_pred, valid_target, delta=delta)
    return loss


def masked_bce(logits, target, mask):
    """
    Masked Binary Cross Entropy loss to ignore invalid/bad windows during training.
    
    Args:
        logits: Prediction logits tensor (batch_size, seq_len)
        target: Target tensor (batch_size, seq_len) - binary labels (0 or 1)
        mask: Boolean mask tensor (batch_size, seq_len) - True for valid, False for invalid
    
    Returns:
        Scalar loss tensor
    """
    import torch
    import torch.nn.functional as F
    
    # Only compute loss on valid (masked=True) elements
    valid_logits = logits[mask]
    valid_target = target[mask]
    
    if len(valid_logits) == 0:
        # No valid elements - return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Compute BCE loss on valid elements only
    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_target)
    return loss


# --- Transformer Architecture ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #(B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) #(B,T,C)
        out = wei @ v #(B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerClassificationModel(nn.Module):
    def __init__(self):  # 12 + 23*5 = 127 features per token
        super().__init__()
        self.token_projection = nn.Linear(token_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.output_head = nn.Linear(n_embd, 1)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T, token_dim = idx.shape
        tok_embd = self.token_projection(idx)  # (B,T,n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,n_embd)
        x = tok_embd + pos_embd  # (B,T,n_embd)
        x = self.blocks(x)  # (B,T,n_embd)
        x = self.ln_f(x)  # (B,T,n_embd)
        
        # Predict binary classification logits for each time step
        logits = self.output_head(x)  # (B,T,1)
        logits = logits.squeeze(-1)  # (B,T)

        if targets is None:
            loss = None
        else:
            if loss_mask is not None:
                # Use masked binary cross entropy loss to ignore invalid windows
                loss = masked_bce(logits, targets, loss_mask)
            else:
                # Fallback to regular binary cross entropy loss
                loss = F.binary_cross_entropy_with_logits(logits, targets)

        return logits, loss


# --- Model Management Functions ---

def save_model(model, optimizer, step, train_loss, val_loss, run_dir="./checkpoints/run_003"):
    """Save model checkpoint with training state in run-specific directory."""
    import os
    os.makedirs(run_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'hyperparameters': {
            'batch_size': batch_size,
            'block_size': block_size,
            'max_iters': max_iters,
            'learning_rate': learning_rate,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
            'token_size': token_size,
            'max_other_horses': max_other_horses,
            'prediction_offset': prediction_offset,
            'tokenization_version': 'v2_normalized',  # Track tokenization improvements
            'prediction_target': 'binary_classification_edge'  # Track what we're predicting
        }
    }
    
    # Save both step-specific and latest checkpoints
    step_path = f"{run_dir}/checkpoint_step_{step}.pt"
    latest_path = f"{run_dir}/checkpoint_latest.pt"
    
    torch.save(checkpoint, step_path)
    torch.save(checkpoint, latest_path)
    
    print(f"💾 Saved checkpoint: {step_path}")
    print(f"💾 Updated latest: {latest_path}")
    
    return step_path


def load_model(model, optimizer, checkpoint_path):
    """Load model checkpoint and resume training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint.get('step', 0)
    train_loss = checkpoint.get('train_loss', 0.0)
    val_loss = checkpoint.get('val_loss', 0.0)
    
    print(f"📚 Loaded checkpoint from step {step}")
    print(f"    Train loss: {train_loss:.6f}")
    print(f"    Val loss: {val_loss:.6f}")
    
    return step, train_loss, val_loss


# --- Training Functions ---

# Global variables for chunked training
train_sampler = None
val_set = None
current_blocks = []
block_index = 0


@torch.no_grad()
def estimate_loss(model):
    """Estimate loss on train and validation sets using chunked sampling."""
    global val_set, current_blocks, block_index
    
    out = {}
    
    # Validation loss using fixed validation set
    if val_set is not None:
        val_losses = torch.zeros(min(eval_iters, 100))  # Limit val eval for speed
        for k in range(len(val_losses)):
            X, Y, mask = val_set.get_batch(micro_batch_size, device)
            log_lay_price_predictions, loss = model(X, Y, mask)
            val_losses[k] = loss.item() if loss is not None else 0.0
        out['val'] = val_losses.mean()
    else:
        out['val'] = torch.tensor(0.0)
    
    # Train loss using current blocks
    if current_blocks:
        train_losses = torch.zeros(min(eval_iters // 4, len(current_blocks) // micro_batch_size))
        for k in range(len(train_losses)):
            # Get micro-batch from current blocks
            start_idx = (block_index + k * micro_batch_size) % len(current_blocks)
            end_idx = start_idx + micro_batch_size
            if end_idx > len(current_blocks):
                batch_blocks = current_blocks[start_idx:] + current_blocks[:end_idx - len(current_blocks)]
            else:
                batch_blocks = current_blocks[start_idx:end_idx]
            
            if batch_blocks:
                X, Y, mask = collate_blocks(batch_blocks, device, max_batch_size=micro_batch_size)
                log_lay_price_predictions, loss = model(X, Y, mask)
                train_losses[k] = loss.item() if loss is not None else 0.0
        out['train'] = train_losses.mean()
    else:
        out['train'] = torch.tensor(0.0)
    
    return out


def start_training(model=None, optimizer=None):
    """Start chunked training loop with memory-efficient sampling."""
    global train_sampler, val_set, current_blocks, block_index
    
    if model is None or optimizer is None:
        print("❌ Error: Model and optimizer must be provided")
        return
    
    model.train()
    
    # Initialize chunked samplers
    print("📂 Initializing chunked samplers...")
    train_sampler = ChunkSampler(
        "datasets/races_train_split_with_bins.jsonl.gz",
        chunk_size_races=chunk_size_races,
        blocks_per_chunk=blocks_per_chunk,
        seed=1337
    )
    
    val_set = FixedValidationSet(
        "datasets/races_validation_split.jsonl.gz",
        num_blocks=val_blocks,
        block_size=block_size,
        seed=42
    )
    
    # Training loop with chunked sampling
    print("🚀 Starting chunked training...")
    best_val_loss = float('inf')
    train_steps = 0
    
    while train_steps < max_iters:
        # Load new chunk of training blocks
        print(f"\n📦 Loading training chunk (step {train_steps})...")
        current_blocks = train_sampler.next_chunk_blocks(blocks_per_chunk, block_size)
        
        if not current_blocks:
            print("⚠️ No blocks generated, skipping chunk")
            continue
        
        block_index = 0
        
        # Train on current chunk with gradient accumulation
        chunk_steps = 0
        max_chunk_steps = len(current_blocks) // (micro_batch_size * gradient_accumulation_steps)
        
        while chunk_steps < max_chunk_steps and train_steps < max_iters:
            # Validation and logging
            if train_steps % eval_interval == 0 or train_steps == max_iters - 1:
                model.eval()
                losses = estimate_loss(model)
                model.train()
                
                train_loss = losses['train']
                val_loss = losses['val']
                
                print(f"step {train_steps}: train_loss {train_loss:.6f}, val_loss {val_loss:.6f}")
                
                # Save model if validation improves or at regular intervals
                if val_loss < best_val_loss or train_steps % (eval_interval * 4) == 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"🎯 New best validation loss: {val_loss:.6f}")
                    
                    save_model(model, optimizer, train_steps, train_loss, val_loss)
        
            # Gradient accumulation step
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            
            for accum_step in range(gradient_accumulation_steps):
                # Get micro-batch
                start_idx = block_index
                end_idx = block_index + micro_batch_size
                
                if end_idx > len(current_blocks):
                    # Not enough blocks left, break
                    break
                
                micro_batch = current_blocks[start_idx:end_idx]
                block_index = end_idx
                
                if not micro_batch:
                    break
                
                # Forward pass
                try:
                    xb, yb, mask = collate_blocks(micro_batch, device, max_batch_size=micro_batch_size)
                    log_lay_price_predictions, loss = model(xb, yb, mask)
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / gradient_accumulation_steps
                    scaled_loss.backward()
                    accumulated_loss += loss.item()
                    
                except Exception as e:
                    print(f"⚠️ Training error at step {train_steps}: {e}")
                    continue
            
            # Optimization step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_steps += 1
            chunk_steps += 1
            
            # Detailed logging
            if train_steps % 100 == 0:
                avg_loss = accumulated_loss / gradient_accumulation_steps
                print(f"    step {train_steps}: loss={avg_loss:.6f}, chunk_progress={chunk_steps}/{max_chunk_steps}")
            
            # Check if we've exhausted the current chunk
            if block_index >= len(current_blocks) - micro_batch_size * gradient_accumulation_steps:
                print(f"🔄 Exhausted chunk after {chunk_steps} steps, loading new chunk...")
                break
        
        # Free memory
        del current_blocks
        import gc
        gc.collect()
    
    print("✅ Training completed!")


def main():
    """Main training function when run as script."""
    print("🤖 Initializing Betfair Transformer v3.0")
    print("🔧 Improvements: normalized tokens, binary classification (2% edge detection)")
    print(f"📊 Device: {device}")
    
    # Initialize model and optimizer
    model = TransformerClassificationModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Start training
    start_training(model, optimizer)


if __name__ == '__main__':
    main()
