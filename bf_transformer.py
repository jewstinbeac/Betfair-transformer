import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import json
import gzip
import orjson

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_other_horses = 23
token_size = 12 + max_other_horses * 5
prediction_offset=32
# ---------------------------------------------------------

torch.manual_seed(1337)


def create_token_from_bin(bin_data, subject_horse_id, race_metadata, max_other_horses=max_other_horses):
    """
    Create a token from a single bin for a specific subject horse.
    
    Args:
        bin_data: Single bin from race JSON
        subject_horse_id: The selectionId of the horse this token is about
        race_metadata: Race metadata (marketTime_ms, etc.)
        max_other_horses: Maximum number of other horses to include
    
    Returns:
        List of floats representing the token
    """
    token = []
    
    # Get subject horse data
    subject_runner = bin_data['runners'].get(str(subject_horse_id))
    if not subject_runner:
        # If subject horse not in this bin, return None or empty token
        return None
    
    # Time to off (seconds to marketTime)
    time_to_off = (race_metadata['marketTime_ms'] - bin_data['t_ms']) / 1000.0
    
    # Subject horse features
    # Best back price and size
    back_orders = subject_runner.get('back', [])
    best_back_price = back_orders[0][0] if back_orders else 0.0
    best_back_size = back_orders[0][1] if back_orders else 0.0
    
    # Best lay price and size
    lay_orders = subject_runner.get('lay', [])
    best_lay_price = lay_orders[0][0] if lay_orders else 0.0
    best_lay_size = lay_orders[0][1] if lay_orders else 0.0
    
    # Second back price and size
    second_back_price = back_orders[1][0] if len(back_orders) > 1 else 0.0
    second_back_size = back_orders[1][1] if len(back_orders) > 1 else 0.0
    
    # Second lay price and size
    second_lay_price = lay_orders[1][0] if len(lay_orders) > 1 else 0.0
    second_lay_size = lay_orders[1][1] if len(lay_orders) > 1 else 0.0
    
    # Last traded price
    ltp = subject_runner.get('ltp', 0.0) or 0.0
    
    # Market features
    number_active_runners = bin_data.get('number_active_runners', 0)
    market_total_matched = bin_data.get('market_total_matched', 0.0)
    
    # Add subject horse features to token
    token.extend([
        best_back_price, best_back_size,
        best_lay_price, best_lay_size,
        second_back_price, second_back_size,
        second_lay_price, second_lay_size,
        ltp,
        time_to_off,
        number_active_runners,
        market_total_matched
    ])
    
    # Other horses features
    other_horses = []
    for horse_id, runner_data in bin_data['runners'].items():
        if horse_id != str(subject_horse_id) and runner_data.get('active', True):
            other_horses.append((horse_id, runner_data))
    
    # Sort other horses with robust tiebreaking for deterministic ordering
    def sort_key(horse_runner_tuple):
        horse_id, runner_data = horse_runner_tuple
        
        # Primary: LTP (handle missing/NaN)
        ltp = runner_data.get('ltp')
        if ltp is None or ltp <= 0:
            ltp = 999999.0  # Put missing LTP horses at the end
        
        # Secondary: Best lay price (lowest first)
        lay_orders = runner_data.get('lay', [])
        best_lay = lay_orders[0][0] if lay_orders else 999999.0
        
        # Tertiary: Best back price (lowest first) 
        back_orders = runner_data.get('back', [])
        best_back = back_orders[0][0] if back_orders else 999999.0
        
        # Quaternary: Selection ID for final deterministic ordering
        selection_id = int(horse_id)
        
        return (ltp, best_lay, best_back, selection_id)
    
    other_horses.sort(key=sort_key)
    
    # Add other horses features (up to max_other_horses)
    for i in range(max_other_horses):
        if i < len(other_horses):
            # Horse exists in this slot
            _, runner_data = other_horses[i]
            
            # Horse presence indicator
            token.append(1.0)
            
            # Best back price and size for this other horse
            back_orders = runner_data.get('back', [])
            other_best_back_price = back_orders[0][0] if back_orders else 0.0
            other_best_back_size = back_orders[0][1] if back_orders else 0.0
            
            # Best lay price and size for this other horse
            lay_orders = runner_data.get('lay', [])
            other_best_lay_price = lay_orders[0][0] if lay_orders else 0.0
            other_best_lay_size = lay_orders[0][1] if lay_orders else 0.0
            
            token.extend([
                other_best_back_price, other_best_back_size,
                other_best_lay_price, other_best_lay_size
            ])
        else:
            # No horse in this slot - add zeros
            token.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # presence + 4 features
    
    return token


def get_fixed_horse_ordering(bin_data, subject_horse_id, max_other_horses):
    """
    Determine a fixed ordering of other horses for the entire block.
    This ensures horses maintain their token positions even if rankings change.
    
    Args:
        bin_data: First bin data to determine initial ordering
        subject_horse_id: The subject horse ID to exclude
        max_other_horses: Maximum number of other horses
    
    Returns:
        List of horse IDs in fixed order for this block
    """
    # Get other horses from the first bin
    other_horses = []
    for horse_id, runner_data in bin_data['runners'].items():
        if horse_id != str(subject_horse_id) and runner_data.get('active', True):
            other_horses.append((horse_id, runner_data))
    
    # Sort other horses with robust tiebreaking for deterministic ordering
    def sort_key(horse_runner_tuple):
        horse_id, runner_data = horse_runner_tuple
        
        # Primary: LTP (handle missing/NaN)
        ltp = runner_data.get('ltp')
        if ltp is None or ltp <= 0:
            ltp = 999999.0  # Put missing LTP horses at the end
        
        # Secondary: Best lay price (lowest first)
        lay_orders = runner_data.get('lay', [])
        best_lay = lay_orders[0][0] if lay_orders else 999999.0
        
        # Tertiary: Best back price (lowest first) 
        back_orders = runner_data.get('back', [])
        best_back = back_orders[0][0] if back_orders else 999999.0
        
        # Quaternary: Selection ID for final deterministic ordering
        selection_id = int(horse_id)
        
        return (ltp, best_lay, best_back, selection_id)
    
    other_horses.sort(key=sort_key)
    
    # Return just the horse IDs in fixed order, limited to max_other_horses
    return [horse_id for horse_id, _ in other_horses[:max_other_horses]]


def create_token_from_bin_with_fixed_order(bin_data, subject_horse_id, race_metadata, fixed_horse_order):
    """
    Create a token from a single bin for a specific subject horse with fixed horse ordering.
    
    Args:
        bin_data: Single bin from race JSON
        subject_horse_id: The selectionId of the horse this token is about
        race_metadata: Race metadata (marketTime_ms, etc.)
        fixed_horse_order: Pre-determined list of horse IDs in fixed positions
    
    Returns:
        List of floats representing the token
    """
    token = []
    
    # Get subject horse data
    subject_runner = bin_data['runners'].get(str(subject_horse_id))
    if not subject_runner:
        # If subject horse not in this bin, return None or empty token
        return None
    
    # Time to off (seconds to marketTime)
    time_to_off = (race_metadata['marketTime_ms'] - bin_data['t_ms']) / 1000.0
    
    # Subject horse features
    # Best back price and size
    back_orders = subject_runner.get('back', [])
    best_back_price = back_orders[0][0] if back_orders else 0.0
    best_back_size = back_orders[0][1] if back_orders else 0.0
    
    # Best lay price and size
    lay_orders = subject_runner.get('lay', [])
    best_lay_price = lay_orders[0][0] if lay_orders else 0.0
    best_lay_size = lay_orders[0][1] if lay_orders else 0.0
    
    # Second back price and size
    second_back_price = back_orders[1][0] if len(back_orders) > 1 else 0.0
    second_back_size = back_orders[1][1] if len(back_orders) > 1 else 0.0
    
    # Second lay price and size
    second_lay_price = lay_orders[1][0] if len(lay_orders) > 1 else 0.0
    second_lay_size = lay_orders[1][1] if len(lay_orders) > 1 else 0.0
    
    # Last traded price
    ltp = subject_runner.get('ltp', 0.0) or 0.0
    
    # Market features
    number_active_runners = bin_data.get('number_active_runners', 0)
    market_total_matched = bin_data.get('market_total_matched', 0.0)
    
    # Add subject horse features to token
    token.extend([
        best_back_price, best_back_size,
        best_lay_price, best_lay_size,
        second_back_price, second_back_size,
        second_lay_price, second_lay_size,
        ltp,
        time_to_off,
        number_active_runners,
        market_total_matched
    ])
    
    # Add other horses features using fixed ordering
    for i, horse_id in enumerate(fixed_horse_order):
        runner_data = bin_data['runners'].get(horse_id)
        
        if runner_data and runner_data.get('active', True):
            # Horse exists and is active in this bin
            token.append(1.0)  # Horse presence indicator
            
            # Best back price and size for this other horse
            back_orders = runner_data.get('back', [])
            other_best_back_price = back_orders[0][0] if back_orders else 0.0
            other_best_back_size = back_orders[0][1] if back_orders else 0.0
            
            # Best lay price and size for this other horse
            lay_orders = runner_data.get('lay', [])
            other_best_lay_price = lay_orders[0][0] if lay_orders else 0.0
            other_best_lay_size = lay_orders[0][1] if lay_orders else 0.0
            
            token.extend([
                other_best_back_price, other_best_back_size,
                other_best_lay_price, other_best_lay_size
            ])
        else:
            # Horse not in this bin or inactive - fill with zeros
            token.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # presence + 4 features
    
    # Fill remaining slots with zeros if we have fewer horses than max_other_horses
    remaining_slots = max_other_horses - len(fixed_horse_order)
    for _ in range(remaining_slots):
        token.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # presence + 4 features
    
    return token


def create_tokens_from_race(race_data, max_other_horses=max_other_horses):
    """
    Create all tokens for a race (one token per horse per bin).
    
    Args:
        race_data: Complete race JSON object
        max_other_horses: Maximum number of other horses to include in each token
    
    Returns:
        List of (horse_id, bin_index, token) tuples
    """
    tokens = []
    bins = race_data.get('bins', [])
    runners = race_data.get('runners', [])
    
    # Get all horse IDs from race metadata
    horse_ids = [runner['selectionId'] for runner in runners]
    
    for bin_index, bin_data in enumerate(bins):
        for horse_id in horse_ids:
            token = create_token_from_bin(bin_data, horse_id, race_data, max_other_horses)
            if token is not None:
                tokens.append((horse_id, bin_index, token))
    
    return tokens


def load_race_data(file_path, max_races=None):
    """
    Load race data from jsonl.gz file.
    
    Args:
        file_path: Path to races.jsonl.gz file
        max_races: Maximum number of races to load (None for all)
    
    Returns:
        List of race data dictionaries
    """
    races = []
    
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if max_races and i >= max_races:
                break
            races.append(json.loads(line))
    
    return races


def get_eligible_races_and_weights(races, min_consecutive_tokens=160):
    """
    Get races that have enough consecutive tokens and their sampling weights.
    
    Args:
        races: List of race data
        min_consecutive_tokens: Minimum tokens needed (default 160 for 80 seconds)
    
    Returns:
        Tuple of (eligible_races, weights) for weighted sampling
    """
    eligible_races = []
    weights = []
    
    for race in races:
        n_bins = race.get('n_bins', 0)
        if n_bins >= min_consecutive_tokens:
            eligible_races.append(race)
            # Weight by number of valid starting positions
            valid_starts = n_bins - min_consecutive_tokens + 1
            weights.append(valid_starts)
    
    return eligible_races, weights


def get_eligible_horses_in_block(race, start_bin, max_odds=15.0):
    """
    Get horses eligible for selection at the start of a block.
    
    Args:
        race: Race data
        start_bin: Starting bin index
        max_odds: Maximum odds allowed (default 15.0)
    
    Returns:
        List of horse IDs that meet the criteria
    """
    eligible_horses = []
    
    if start_bin >= len(race['bins']):
        return eligible_horses
    
    start_bin_data = race['bins'][start_bin]
    
    for horse_id, runner_data in start_bin_data['runners'].items():
        if not runner_data.get('active', True):
            continue
            
        # Check if horse has back price <= max_odds
        back_orders = runner_data.get('back', [])
        if back_orders:
            best_back_price = back_orders[0][0]
            if best_back_price <= max_odds:
                eligible_horses.append(int(horse_id))
    
    return eligible_horses


def ensure_market_time_ms(race):
    """
    Ensure race has marketTime_ms field, computing from ISO if needed.
    
    Args:
        race: Race data dictionary
    
    Returns:
        Race data with guaranteed marketTime_ms field
    """
    if 'marketTime_ms' not in race:
        from datetime import datetime
        
        # Get marketTime from race data (should be ISO format)
        market_time_iso = race.get('marketTime')
        if market_time_iso:
            try:
                # Parse ISO timestamp and convert to milliseconds
                dt = datetime.fromisoformat(market_time_iso.replace('Z', '+00:00'))
                race['marketTime_ms'] = int(dt.timestamp() * 1000)
            except Exception:
                # Fallback: use a reasonable default or derive from bins
                if race.get('bins'):
                    # Use last bin time + some buffer as fallback
                    race['marketTime_ms'] = race['bins'][-1]['t_ms'] + 60000  # +1 minute
                else:
                    race['marketTime_ms'] = 0
        else:
            race['marketTime_ms'] = 0
    
    return race


def sample_training_block(races, races_weights, block_size=128, max_odds=15.0):
    """
    Sample a training block (sequence of tokens for one horse).
    
    Args:
        races: List of eligible races
        races_weights: Weights for sampling races
        block_size: Number of tokens in sequence (default 128 = 64 seconds)
        prediction_offset: Tokens ahead for prediction target (default 32 = 16 seconds)
        max_odds: Maximum odds for horse selection
    
    Returns:
        Tuple of (input_tokens, target_tokens, metadata) or None if sampling fails
    """
    import random
    import numpy as np
    
    min_tokens_needed = block_size + prediction_offset
    
    # Weighted random selection of race
    race = random.choices(races, weights=races_weights, k=1)[0]
    
    # Ensure race has marketTime_ms field
    race = ensure_market_time_ms(race)
    
    # Random starting position within the race
    max_start = race['n_bins'] - min_tokens_needed
    if max_start < 0:
        return None
    
    start_bin = random.randint(0, max_start)
    
    # Get eligible horses at start of block
    eligible_horses = get_eligible_horses_in_block(race, start_bin, max_odds)
    if not eligible_horses:
        return None
    
    # Randomly select horse
    selected_horse = random.choice(eligible_horses)
    
    # Get fixed horse ordering for this block based on the first bin
    first_bin = race['bins'][start_bin]
    fixed_horse_order = get_fixed_horse_ordering(first_bin, selected_horse, max_other_horses)
    
    # Extract tokens for this horse and time range
    input_tokens = []
    target_tokens = []
    
    # Generate input sequence (block_size tokens)
    for bin_idx in range(start_bin, start_bin + block_size):
        if bin_idx >= len(race['bins']):
            return None
        
        token = create_token_from_bin_with_fixed_order(race['bins'][bin_idx], selected_horse, race, fixed_horse_order)
        if token is None:
            return None
        input_tokens.append(token)
    
    # Generate target sequence (same length, offset by prediction_offset)
    for bin_idx in range(start_bin + prediction_offset, start_bin + prediction_offset + block_size):
        if bin_idx >= len(race['bins']):
            return None
        
        token = create_token_from_bin_with_fixed_order(race['bins'][bin_idx], selected_horse, race, fixed_horse_order)
        if token is None:
            return None
        target_tokens.append(token)
    
    metadata = {
        'race_id': race['marketId'],
        'horse_id': selected_horse,
        'start_bin': start_bin,
        'start_time_ms': race['bins'][start_bin]['t_ms'],
        'target_time_ms': race['bins'][start_bin + prediction_offset]['t_ms'],
        'race_data': race  # Include full race data for mask calculation
    }
    
    return input_tokens, target_tokens, metadata


def calculate_ev_targets_per_step(input_tokens, target_tokens, race_data, selected_horse):
    """
    Calculate log-EV (Expected Value) targets and masks for each step in the sequence.
    log-EV = log(current_back_price / future_lay_price) for stability
    
    Args:
        input_tokens: Current token sequence (block_size,)
        target_tokens: Future token sequence (block_size,) offset by 32 steps
        race_data: Full race data for checking suspensions/removals
        selected_horse: Horse ID for checking activity status
    
    Returns:
        Tuple of (log_ev_targets, loss_mask) - lists of values and validity flags
    """
    import numpy as np
    
    log_ev_targets = []
    loss_mask = []
    
    # Get market definition events for checking suspensions/removals
    md_events = race_data.get('marketDefinition_events', [])
    
    for i in range(len(input_tokens)):
        # Current back price at step i
        current_back_price = input_tokens[i][0]  # Best back price at step i
        
        # Future lay price at corresponding step in target sequence
        future_lay_price = target_tokens[i][2] if i < len(target_tokens) else 0.0
        
        # Check validity conditions
        valid = True
        
        # Condition 1: Either price is missing/zero
        if future_lay_price <= 0 or current_back_price <= 0:
            valid = False
        
        # Condition 2: Subject horse is inactive at either current or future time
        if valid and i < len(race_data['bins']):
            # Check current time (input)
            current_bin = race_data['bins'][i] 
            current_runner = current_bin['runners'].get(str(selected_horse))
            if not current_runner or not current_runner.get('active', True):
                valid = False
            
            # Check future time (target) - need to find the actual future bin
            future_bin_idx = i + prediction_offset
            if future_bin_idx < len(race_data['bins']):
                future_bin = race_data['bins'][future_bin_idx]
                future_runner = future_bin['runners'].get(str(selected_horse))
                if not future_runner or not future_runner.get('active', True):
                    valid = False
        
        # Condition 3: 16s horizon crosses a suspension or removal
        if valid and i < len(race_data['bins']):
            current_time_ms = race_data['bins'][i]['t_ms']
            future_time_ms = race_data['bins'][min(i + prediction_offset, len(race_data['bins']) - 1)]['t_ms']
            
            # Check if any market definition events (suspensions/removals) occur in this window
            for event in md_events:
                event_time = event.get('pt_ms', 0)
                if current_time_ms <= event_time <= future_time_ms:
                    # Check if this is a suspension
                    if event.get('status') == 'SUSPENDED':
                        valid = False
                        break
                    
                    # Check if this is a removal affecting our horse
                    removed_horses = event.get('removed', [])
                    for removed in removed_horses:
                        if removed.get('selectionId') == selected_horse:
                            valid = False
                            break
                    
                    if not valid:
                        break
        
        # Calculate log-EV for stability
        if valid:
            log_ev = np.log(current_back_price / future_lay_price)
        else:
            log_ev = 0.0  # neutral log-EV (exp(0) = 1.0 EV)
        
        log_ev_targets.append(log_ev)
        loss_mask.append(valid)
    
    return log_ev_targets, loss_mask


def masked_huber(pred, target, mask, delta=0.1):
    """
    Masked Huber loss to ignore invalid/bad windows during training.
    
    Args:
        pred: Predictions tensor
        target: Target tensor  
        mask: Boolean mask (True for valid entries, False for invalid)
        delta: Huber loss threshold parameter
        
    Returns:
        Scalar loss value
    """
    import torch
    diff = pred - target
    absd = diff.abs()
    huber = torch.where(absd <= delta, 0.5 * diff**2 / delta, absd - 0.5 * delta)
    w = mask.float()
    return (huber * w).sum() / w.sum().clamp_min(1.0)


def log_ev_to_ev(log_ev_predictions):
    """
    Convert log-EV predictions back to EV for inference.
    
    Args:
        log_ev_predictions: Tensor of log-EV predictions
    
    Returns:
        Tensor of EV predictions (exp(log_ev))
    """
    import torch
    return torch.exp(log_ev_predictions)


def get_batch_from_races(races, races_weights, batch_size=64, block_size=128):
    """
    Generate a batch of training sequences from race data.
    
    Args:
        races: List of eligible races
        races_weights: Weights for sampling races
        batch_size: Number of sequences in batch
        block_size: Length of each sequence
    
    Returns:
        Tuple of (input_batch, log_ev_targets, loss_masks) as torch tensors
    """
    import torch
    import numpy as np
    
    input_batch = []
    log_ev_targets_batch = []
    loss_masks_batch = []
    
    attempts = 0
    max_attempts = batch_size * 10  # Prevent infinite loops
    
    while len(input_batch) < batch_size and attempts < max_attempts:
        attempts += 1
        
        sample = sample_training_block(races, races_weights, block_size)
        if sample is not None:
            input_tokens, target_tokens, metadata = sample
            
            # Convert to numpy arrays
            input_array = np.array(input_tokens, dtype=np.float32)
            
            # Calculate log-EV targets and masks for each step
            race_data = metadata['race_data']
            selected_horse = metadata['horse_id']
            log_ev_targets_sequence, loss_mask_sequence = calculate_ev_targets_per_step(
                input_tokens, target_tokens, race_data, selected_horse
            )
            
            input_batch.append(input_array)
            log_ev_targets_batch.append(log_ev_targets_sequence)
            loss_masks_batch.append(loss_mask_sequence)
    
    if len(input_batch) == 0:
        return None, None, None
    
    # Stack into tensors
    input_tensor = torch.tensor(np.stack(input_batch), dtype=torch.float32)
    log_ev_tensor = torch.tensor(np.array(log_ev_targets_batch), dtype=torch.float32)  # (batch_size, block_size)
    mask_tensor = torch.tensor(np.array(loss_masks_batch), dtype=torch.bool)  # (batch_size, block_size)
    
    return input_tensor, log_ev_tensor, mask_tensor


# Global variables for race data (to be initialized)
train_races = None
train_weights = None
val_races = None  
val_weights = None


def initialize_race_data(train_path, val_path, max_races=None):
    """
    Initialize global race data and weights for training.
    
    Args:
        train_path: Path to training races file
        val_path: Path to validation races file  
        max_races: Maximum races to load from each file
    """
    global train_races, train_weights, val_races, val_weights
    
    print("Loading training races...")
    print(f"  Reading from: {train_path}")
    all_train_races = load_race_data(train_path, max_races)
    print(f"  Raw races loaded: {len(all_train_races)}")
    
    # Ensure all races have marketTime_ms
    print("  Processing marketTime_ms...")
    all_train_races = [ensure_market_time_ms(race) for race in all_train_races]
    
    print("  Filtering eligible races...")
    train_races, train_weights = get_eligible_races_and_weights(all_train_races)
    print(f"✅ Loaded {len(train_races)} eligible training races (from {len(all_train_races)} total)")
    
    # Clear memory
    del all_train_races
    import gc
    gc.collect()
    
    print("Loading validation races...")
    print(f"  Reading from: {val_path}")
    all_val_races = load_race_data(val_path, max_races)
    print(f"  Raw races loaded: {len(all_val_races)}")
    
    # Ensure all races have marketTime_ms
    print("  Processing marketTime_ms...")
    all_val_races = [ensure_market_time_ms(race) for race in all_val_races]
    
    print("  Filtering eligible races...")
    val_races, val_weights = get_eligible_races_and_weights(all_val_races)
    print(f"✅ Loaded {len(val_races)} eligible validation races (from {len(all_val_races)} total)")
    
    # Clear memory
    del all_val_races
    gc.collect()
    
    print(f"📊 DATASET SUMMARY:")
    print(f"  Training races: {len(train_races)}")
    print(f"  Validation races: {len(val_races)}")
    print(f"  Total training weight: {sum(train_weights):,}")
    print(f"  Total validation weight: {sum(val_weights):,}")


# Example usage:
# initialize_race_data('datasets/races_train_split_with_bins.jsonl.gz', 'datasets/races_validation_split.jsonl.gz')
# 
# # Sample a single block
# sample = sample_training_block(train_races, train_weights)
# if sample:
#     inputs, targets, meta = sample
#     print(f"Block from race {meta['race_id']}, horse {meta['horse_id']}")
#     print(f"Input shape: {len(inputs)} x {len(inputs[0])}")
#     print(f"Target shape: {len(targets)} x {len(targets[0])}")
#
# # Get a training batch
# x_batch, y_batch = get_batch_from_races(train_races, train_weights, batch_size=8)
# if x_batch is not None:
#     print(f"Batch shape: {x_batch.shape} -> {y_batch.shape}")




# Initialize race data (comment out for now, enable when ready)
# initialize_race_data('datasets/races_train_split_with_bins.jsonl.gz', 'datasets/races_validation_split.jsonl.gz')

def get_batch(split):
    """
    Get a batch of race data for training or validation.
    
    Args:
        split: 'train' or 'val'
    
    Returns:
        Tuple of (x, y, mask) tensors where x is input sequences, y is log-EV targets, mask is validity
    """
    global train_races, train_weights, val_races, val_weights
    
    if split == 'train':
        if train_races is None:
            raise ValueError("Train races not loaded. Call initialize_race_data() first.")
        x, y, mask = get_batch_from_races(train_races, train_weights, batch_size, block_size)
    else:
        if val_races is None:
            raise ValueError("Validation races not loaded. Call initialize_race_data() first.")
        x, y, mask = get_batch_from_races(val_races, val_weights, batch_size, block_size)
    
    if x is None or y is None or mask is None:
        # Fallback: return empty tensors if sampling fails
        x = torch.zeros((batch_size, block_size, token_size), dtype=torch.float32)
        y = torch.zeros((batch_size, block_size), dtype=torch.float32)  # log-EV targets per step
        mask = torch.ones((batch_size, block_size), dtype=torch.bool)  # All valid by default
    
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask = get_batch(split)
            logits, loss = m(X, Y, mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


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


class TransformerRegressionModel(nn.Module):
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
        
        # Predict log-EV for each time step (more stable than raw EV)
        log_ev_predictions = self.output_head(x)  # (B,T,1)
        log_ev_predictions = log_ev_predictions.squeeze(-1)  # (B,T)

        if targets is None:
            loss = None
        else:
            if loss_mask is not None:
                # Use masked Huber loss to ignore invalid windows
                loss = masked_huber(log_ev_predictions, targets, loss_mask, delta=0.1)
            else:
                # Fallback to regular Huber loss
                loss = F.huber_loss(log_ev_predictions, targets, delta=1.0)

        return log_ev_predictions, loss


# Model initialization - only do this when running as main script
# When imported, these will be created by the importing script as needed

# Model saving functionality
def save_model(model, optimizer, step, train_loss, val_loss, save_dir="./checkpoints"):
    """Save model checkpoint with training state."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
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
            'prediction_offset': prediction_offset
        }
    }
    
    # Save with step number
    checkpoint_path = f"{save_dir}/checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = f"{save_dir}/checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"📀 Model saved: {checkpoint_path}")
    return checkpoint_path

def load_model(model, optimizer, checkpoint_path):
    """Load model checkpoint and resume training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint['step']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    print(f"📁 Model loaded from step {step}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
    return step, train_loss, val_loss

# Training loop with model saving (uncomment initialize_race_data above to run)
# print("🚀 Starting training...")
# best_val_loss = float('inf')
# 
# for iter in range(max_iters):
#     # Validation and logging every eval_interval steps
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         train_loss = losses['train']
#         val_loss = losses['val']
#         
#         print(f"step {iter}: train_loss {train_loss:.6f}, val_loss {val_loss:.6f}")
#         
#         # Save model if validation improves or at regular intervals
#         if val_loss < best_val_loss or iter % (eval_interval * 4) == 0:
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 print(f"🎯 New best validation loss: {val_loss:.6f}")
#             
#             save_model(m, optimizer, iter, train_loss, val_loss)
#     
#     try:
#         xb, yb, mask = get_batch('train')
#         log_ev_predictions, loss = m(xb, yb, mask)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#         
#         # Detailed logging every 100 steps
#         if iter % 100 == 0:
#             # Convert to EV for interpretable logging
#             ev_predictions = log_ev_to_ev(log_ev_predictions)
#             valid_frac = mask.float().mean().item()
#             print(f"  Step {iter}: loss={loss.item():.6f}, valid_frac={valid_frac:.3f}, log_ev_range=[{log_ev_predictions.min().item():.3f}, {log_ev_predictions.max().item():.3f}], ev_range=[{ev_predictions.min().item():.3f}, {ev_predictions.max().item():.3f}]")
#     
#     except Exception as e:
#         print(f"Error in training step {iter}: {e}")
#         continue
# 
# print("✅ Training completed!")

def test_token_generation():
    """
    Test function to print out token structure and EV calculations.
    """
    import numpy as np
    
    print("🔧 TEST: Token Generation and EV Calculation")
    print("=" * 60)
    
    # Initialize a small dataset for testing
    print("Loading test data...")
    try:
        test_races = load_race_data('datasets/races_train_split_with_bins.jsonl.gz', max_races=3)
        test_races = [ensure_market_time_ms(race) for race in test_races]
        eligible_races, weights = get_eligible_races_and_weights(test_races, min_consecutive_tokens=160)
        
        if not eligible_races:
            print("❌ No eligible races found for testing")
            return
            
        print(f"✅ Found {len(eligible_races)} eligible races")
        
        # Sample a training block
        sample = sample_training_block(eligible_races, weights, block_size=128)
        if sample is None:
            print("❌ Failed to sample training block")
            return
            
        input_tokens, target_tokens, metadata = sample
        
        print(f"\n📊 BLOCK METADATA:")
        print(f"  Race ID: {metadata['race_id']}")
        print(f"  Horse ID: {metadata['horse_id']}")
        print(f"  Start bin: {metadata['start_bin']}")
        print(f"  Block size: {len(input_tokens)} tokens")
        print(f"  Target size: {len(target_tokens)} tokens")
        
        # Show first 3 tokens structure
        print(f"\n🎯 FIRST 3 TOKENS STRUCTURE:")
        for i in range(min(3, len(input_tokens))):
            token = input_tokens[i]
            target_token = target_tokens[i] if i < len(target_tokens) else None
            
            print(f"\n  Token {i}:")
            print(f"    Subject horse features (12): {token[:12]}")
            print(f"      - Back price/size: {token[0]:.2f} / {token[1]:.2f}")
            print(f"      - Lay price/size:  {token[2]:.2f} / {token[3]:.2f}")
            print(f"      - LTP: {token[8]:.2f}")
            print(f"      - Time to off: {token[9]:.1f}s")
            print(f"      - Active runners: {token[10]:.0f}")
            print(f"      - Total matched: ${token[11]:.2f}")
            
            # Show first 2 other horses
            print(f"    Other horses (first 10 features):")
            other_start = 12
            for j in range(2):  # First 2 other horses
                start_idx = other_start + j * 5
                if start_idx + 4 < len(token):
                    presence = token[start_idx]
                    back_price = token[start_idx + 1]
                    back_size = token[start_idx + 2]
                    lay_price = token[start_idx + 3]
                    lay_size = token[start_idx + 4]
                    print(f"      Horse {j+1}: present={presence:.0f}, back={back_price:.2f}/{back_size:.2f}, lay={lay_price:.2f}/{lay_size:.2f}")
            
            # Show target and EV calculation
            if target_token is not None:
                current_back = token[0]  # Current back price
                future_lay = target_token[2]  # Future lay price
                
                if future_lay > 0 and current_back > 0:
                    ev_raw = current_back / future_lay
                    ev_log = float(np.log(ev_raw))
                    print(f"    EV Calculation:")
                    print(f"      Current back price: {current_back:.3f}")
                    print(f"      Future lay price:  {future_lay:.3f}")
                    print(f"      Raw EV: {ev_raw:.3f}")
                    print(f"      Log EV: {ev_log:.3f}")
                else:
                    print(f"    EV Calculation: INVALID (back={current_back:.3f}, lay={future_lay:.3f})")
        
        # Test the complete target calculation with masks
        print(f"\n🎭 MASK AND TARGET CALCULATION:")
        race_data = metadata['race_data']
        selected_horse = metadata['horse_id']
        log_ev_targets, loss_mask = calculate_ev_targets_per_step(
            input_tokens, target_tokens, race_data, selected_horse
        )
        
        valid_count = sum(loss_mask)
        total_count = len(loss_mask)
        print(f"  Valid targets: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        print(f"  First 5 log-EV targets: {log_ev_targets[:5]}")
        print(f"  First 5 masks: {loss_mask[:5]}")
        
        # Show some statistics
        valid_evs = [ev for ev, mask in zip(log_ev_targets, loss_mask) if mask]
        if valid_evs:
            print(f"  Valid log-EV range: [{min(valid_evs):.3f}, {max(valid_evs):.3f}]")
            print(f"  Valid EV range: [{np.exp(min(valid_evs)):.3f}, {np.exp(max(valid_evs)):.3f}]")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# Test the token generation
def start_training(model=None, optimizer=None):
    """
    Start the full training process with data loading and model training.
    
    Args:
        model: The model to train (uses global if None)
        optimizer: The optimizer to use (uses global if None)
    """
    # Use global variables if not provided
    if model is None:
        global m
        model = m
    if optimizer is None:
        # Use a different name to avoid shadowing the parameter
        global_optimizer = globals().get('optimizer')
        if global_optimizer is not None:
            optimizer = global_optimizer
    
    print("🚀 STARTING TRAINING PROCESS")
    print("=" * 60)
    
    # Initialize race data (start with very small subset to avoid memory issues)
    print("📂 Loading race datasets...")
    print("⚠️  Starting with 20 training races and 10 validation races")
    initialize_race_data('datasets/races_train_split_with_bins.jsonl.gz', 'datasets/races_validation_split.jsonl.gz', max_races=20)
    
    # Use part of training as validation for now
    global train_races, train_weights, val_races, val_weights
    if len(train_races) >= 10:
        split_point = len(train_races) // 2
        val_races = train_races[split_point:]
        val_weights = train_weights[split_point:]
        train_races = train_races[:split_point]
        train_weights = train_weights[:split_point]
        print(f"📊 Split dataset: {len(train_races)} train, {len(val_races)} validation")
    
    # Training loop
    print("🚀 Starting training...")
    best_val_loss = float('inf')

    for iter in range(max_iters):
        # Validation and logging every eval_interval steps
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            train_loss = losses['train']
            val_loss = losses['val']
            
            print(f"step {iter}: train_loss {train_loss:.6f}, val_loss {val_loss:.6f}")
            
            # Save model if validation improves or at regular intervals
            if val_loss < best_val_loss or iter % (eval_interval * 4) == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"🎯 New best validation loss: {val_loss:.6f}")
                
                save_model(model, optimizer, iter, train_loss, val_loss)
        
        try:
            xb, yb, mask = get_batch('train')
            log_ev_predictions, loss = model(xb, yb, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Detailed logging every 100 steps
            if iter % 100 == 0:
                # Convert to EV for interpretable logging
                ev_predictions = log_ev_to_ev(log_ev_predictions)
                valid_frac = mask.float().mean().item()
                print(f"  Step {iter}: loss={loss.item():.6f}, valid_frac={valid_frac:.3f}, log_ev_range=[{log_ev_predictions.min().item():.3f}, {log_ev_predictions.max().item():.3f}], ev_range=[{ev_predictions.min().item():.3f}, {ev_predictions.max().item():.3f}]")
        
        except Exception as e:
            print(f"Error in training step {iter}: {e}")
            continue
    
    print("✅ Training completed!")

def main():
    """Main function - only runs when script is executed directly."""
    # Initialize model when running as main script
    global model, m, optimizer
    
    model = TransformerRegressionModel()
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    
    print(f"Model initialized with token size: {token_size}")
    print(f"Model parameters: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
    print("\n" + "="*60)
    print("Token generation test...")
    test_token_generation()

    print("\n" + "="*60)
    print("🎯 TRAINING READY!")
    print("To start training, run: start_training()")
    print("Or uncomment the line below and run the script:")
    print("# start_training()")

    # Uncomment the line below to start training:
    start_training(m, optimizer)

if __name__ == '__main__':
    main()
