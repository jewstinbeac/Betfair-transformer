"""
Chunked Sampling for Memory-Efficient Training

Loads only K races at a time, generates M blocks from them, trains for a while,
then drops and loads the next K races. This prevents OOM issues with large datasets.
"""

import gzip
import orjson
import numpy as np
import torch
import gc
from typing import List, Tuple, Optional
from tokenization import (
    get_eligible_races_and_weights, sample_training_block,
    calculate_classification_targets_per_step
)


def trim_race(race_data):
    """
    Trim race to only essential fields to save RAM.
    Keep only what's needed for tokenization and training.
    """
    trimmed = {
        'marketId': race_data.get('marketId'),
        'marketTime_ms': race_data.get('marketTime_ms'),
        'bins': []
    }
    
    # Keep market definition events for validity checking
    if 'marketDefinition_events' in race_data:
        trimmed['marketDefinition_events'] = race_data['marketDefinition_events']
    
    # Trim each bin to essential fields
    for bin_data in race_data.get('bins', []):
        trimmed_bin = {
            't_ms': bin_data.get('t_ms'),
            'number_active_runners': bin_data.get('number_active_runners', 0),
            'market_total_matched': bin_data.get('market_total_matched', 0.0),
            'runners': {}
        }
        
        # Keep only essential runner fields
        for horse_id, runner in bin_data.get('runners', {}).items():
            trimmed_runner = {
                'active': runner.get('active', True),
                'ltp': runner.get('ltp'),
                'back': runner.get('back', [])[:2],  # Keep top 2 levels only
                'lay': runner.get('lay', [])[:2]     # Keep top 2 levels only
            }
            trimmed_bin['runners'][horse_id] = trimmed_runner
        
        trimmed['bins'].append(trimmed_bin)
    
    return trimmed


class ChunkSampler:
    """
    Memory-efficient sampler that loads races in chunks and generates training blocks.
    """
    
    def __init__(self, gz_path: str, chunk_size_races: int = 500, 
                 blocks_per_chunk: int = 4000, seed: int = 1337):
        self.gz_path = gz_path
        self.chunk_size = chunk_size_races
        self.blocks_per_chunk = blocks_per_chunk
        self.rng = np.random.RandomState(seed)
        self.current_seed = seed
        
    def load_chunk(self) -> List[dict]:
        """
        Stream next chunk of races from random starting position.
        Trims races to essential fields to save RAM.
        """
        races = []
        
        with gzip.open(self.gz_path, 'rt') as f:
            # Skip a random number of lines so chunks come from different regions
            # Estimate file has ~50k races, so skip up to 10k lines for variety
            skip = int(self.rng.rand() * 10000)
            for _ in range(skip):
                line = f.readline()
                if not line:  # Hit EOF, wrap around
                    f.seek(0)
                    break
            
            # Load chunk_size races
            for _ in range(self.chunk_size):
                line = f.readline()
                if not line:  # Hit EOF, wrap around
                    f.seek(0)
                    line = f.readline()
                    if not line:  # Empty file
                        break
                
                try:
                    race = orjson.loads(line)
                    trimmed_race = trim_race(race)
                    races.append(trimmed_race)
                except Exception as e:
                    print(f"⚠️ Failed to parse race line: {e}")
                    continue
        
        print(f"📦 Loaded chunk: {len(races)} races")
        return races
    
    def make_blocks(self, races: List[dict], blocks_needed: int, 
                   block_size: int = 256) -> List[Tuple]:
        """
        Generate training blocks from the current chunk of races.
        """
        if not races:
            return []
        
        # Build weights once per chunk
        eligible_races, weights = get_eligible_races_and_weights(races)
        if not eligible_races:
            print("⚠️ No eligible races in chunk")
            return []
        
        print(f"🎯 Eligible races: {len(eligible_races)}/{len(races)}")
        
        blocks = []
        attempts = 0
        max_attempts = blocks_needed * 3  # Prevent infinite loops
        
        while len(blocks) < blocks_needed and attempts < max_attempts:
            attempts += 1
            sample = sample_training_block(eligible_races, weights, block_size=block_size)
            if sample is None:
                continue
            
            input_tokens, target_tokens, metadata = sample
            
            # Convert to targets and masks
            race_data = metadata['race_data']
            selected_horse = metadata['horse_id']
            binary_targets, loss_mask = calculate_classification_targets_per_step(
                input_tokens, target_tokens, race_data, selected_horse
            )
            
            # Store as numpy arrays to save memory
            block_data = (
                np.array(input_tokens, dtype=np.float32),
                np.array(binary_targets, dtype=np.float32),
                np.array(loss_mask, dtype=bool)
            )
            blocks.append(block_data)
        
        # Shuffle blocks for variety
        self.rng.shuffle(blocks)
        print(f"🔧 Generated {len(blocks)} blocks from {attempts} attempts")
        return blocks
    
    def next_chunk_blocks(self, blocks_needed: int, block_size: int = 256) -> List[Tuple]:
        """
        Load next chunk and generate blocks. Updates seed for variety.
        """
        # Rotate seed for variety across chunks
        self.current_seed += 1
        self.rng = np.random.RandomState(self.current_seed)
        
        # Load chunk
        races = self.load_chunk()
        
        # Generate blocks
        blocks = self.make_blocks(races, blocks_needed, block_size)
        
        # Free race memory
        del races
        gc.collect()
        
        return blocks


def collate_blocks(blocks: List[Tuple], device: str, max_batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of blocks into tensors for training.
    Limits batch size to prevent OOM.
    """
    # Limit batch size
    blocks = blocks[:max_batch_size]
    
    if not blocks:
        # Return dummy tensors
        dummy_size = (1, 256, 127)  # (batch, seq, features)
        return (
            torch.zeros(dummy_size, dtype=torch.float32, device=device),
            torch.zeros((1, 256), dtype=torch.float32, device=device),
            torch.ones((1, 256), dtype=torch.bool, device=device)
        )
    
    # Stack blocks into batches
    input_arrays = [block[0] for block in blocks]
    target_arrays = [block[1] for block in blocks]
    mask_arrays = [block[2] for block in blocks]
    
    # Convert to tensors
    input_tensor = torch.tensor(np.stack(input_arrays), dtype=torch.float32, device=device)
    target_tensor = torch.tensor(np.stack(target_arrays), dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(np.stack(mask_arrays), dtype=torch.bool, device=device)
    
    return input_tensor, target_tensor, mask_tensor


class FixedValidationSet:
    """
    Fixed validation set to ensure consistent validation metrics across chunks.
    """
    
    def __init__(self, val_path: str, num_blocks: int = 1000, block_size: int = 256, seed: int = 42):
        self.blocks = []
        self.block_size = block_size
        
        print(f"🔍 Building fixed validation set ({num_blocks} blocks)...")
        
        # Use fixed seed for reproducible validation set
        val_sampler = ChunkSampler(val_path, chunk_size_races=200, seed=seed)
        
        while len(self.blocks) < num_blocks:
            chunk_blocks = val_sampler.next_chunk_blocks(
                blocks_needed=min(500, num_blocks - len(self.blocks)),
                block_size=block_size
            )
            self.blocks.extend(chunk_blocks)
        
        # Trim to exact size
        self.blocks = self.blocks[:num_blocks]
        print(f"✅ Fixed validation set: {len(self.blocks)} blocks")
    
    def get_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a random batch from the fixed validation set.
        """
        if not self.blocks:
            # Return dummy batch
            dummy_size = (batch_size, self.block_size, 127)
            return (
                torch.zeros(dummy_size, dtype=torch.float32, device=device),
                torch.zeros((batch_size, self.block_size), dtype=torch.float32, device=device),
                torch.ones((batch_size, self.block_size), dtype=torch.bool, device=device)
            )
        
        # Sample random blocks
        indices = np.random.choice(len(self.blocks), size=min(batch_size, len(self.blocks)), replace=False)
        batch_blocks = [self.blocks[i] for i in indices]
        
        return collate_blocks(batch_blocks, device, max_batch_size=batch_size)
