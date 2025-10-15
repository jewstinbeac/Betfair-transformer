"""
Tokenization and data processing functions for Betfair transformer model.

This module contains all functions related to:
- Token creation from market data
- Horse ordering and selection
- Race data loading and processing
- Target calculation for training
- Data sampling and batching
"""

import numpy as np
import json
import gzip
import orjson
from pathlib import Path


# Hyperparameters - these should match the main training script
max_other_horses = 23
prediction_offset = 32
token_size = 12 + max_other_horses * 5  # 12 subject + 23*5 others = 127


def create_token_from_bin(bin_data, subject_horse_id, race_metadata, max_other_horses=max_other_horses):
    """
    Create a token from a single bin for a specific subject horse.
    Uses improved normalization: log1p for sizes/volumes, 1/price for odds.
    
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
    
    # Add subject horse features to token with improved normalization
    # Convert prices to 1/price (odds space) for better scaling
    # Use log1p for sizes and volume for stability
    token.extend([
        1.0/best_back_price if best_back_price > 0 else 0.0,  # 1/price (odds space)
        np.log1p(best_back_size),  # log1p for size
        1.0/best_lay_price if best_lay_price > 0 else 0.0,   # 1/price (odds space)
        np.log1p(best_lay_size),   # log1p for size
        1.0/second_back_price if second_back_price > 0 else 0.0,  # 1/price (odds space)
        np.log1p(second_back_size),  # log1p for size
        1.0/second_lay_price if second_lay_price > 0 else 0.0,   # 1/price (odds space)
        np.log1p(second_lay_size),   # log1p for size
        1.0/ltp if ltp > 0 else 0.0,  # 1/price (odds space)
        np.log1p(time_to_off),       # log1p for time
        number_active_runners,       # Keep as-is (small integer)
        np.log1p(market_total_matched)  # log1p for volume
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
                1.0/other_best_back_price if other_best_back_price > 0 else 0.0,  # 1/price (odds space)
                np.log1p(other_best_back_size),  # log1p for size
                1.0/other_best_lay_price if other_best_lay_price > 0 else 0.0,   # 1/price (odds space)
                np.log1p(other_best_lay_size)    # log1p for size
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
    
    # Return just the horse IDs in order (up to max_other_horses)
    return [horse_id for horse_id, _ in other_horses[:max_other_horses]]


def create_token_from_bin_with_fixed_order(bin_data, subject_horse_id, race_metadata, fixed_horse_order):
    """
    Create a token from a single bin for a specific subject horse with fixed horse ordering.
    
    Args:
        bin_data: Single bin from race JSON
        subject_horse_id: The selectionId of the horse this token is about
        race_metadata: Race metadata (marketTime_ms, etc.)
        fixed_horse_order: Pre-determined ordering of other horses
    
    Returns:
        List of floats representing the token
    """
    token = []
    
    # Get subject horse data
    subject_runner = bin_data['runners'].get(str(subject_horse_id))
    if not subject_runner:
        return None
    
    # Time to off (seconds to marketTime)
    time_to_off = (race_metadata['marketTime_ms'] - bin_data['t_ms']) / 1000.0
    
    # Subject horse features
    back_orders = subject_runner.get('back', [])
    best_back_price = back_orders[0][0] if back_orders else 0.0
    best_back_size = back_orders[0][1] if back_orders else 0.0
    
    lay_orders = subject_runner.get('lay', [])
    best_lay_price = lay_orders[0][0] if lay_orders else 0.0
    best_lay_size = lay_orders[0][1] if lay_orders else 0.0
    
    second_back_price = back_orders[1][0] if len(back_orders) > 1 else 0.0
    second_back_size = back_orders[1][1] if len(back_orders) > 1 else 0.0
    
    second_lay_price = lay_orders[1][0] if len(lay_orders) > 1 else 0.0
    second_lay_size = lay_orders[1][1] if len(lay_orders) > 1 else 0.0
    
    ltp = subject_runner.get('ltp', 0.0) or 0.0
    
    # Market features
    number_active_runners = bin_data.get('number_active_runners', 0)
    market_total_matched = bin_data.get('market_total_matched', 0.0)
    
    # Add subject horse features with normalization
    token.extend([
        1.0/best_back_price if best_back_price > 0 else 0.0,
        np.log1p(best_back_size),
        1.0/best_lay_price if best_lay_price > 0 else 0.0,
        np.log1p(best_lay_size),
        1.0/second_back_price if second_back_price > 0 else 0.0,
        np.log1p(second_back_size),
        1.0/second_lay_price if second_lay_price > 0 else 0.0,
        np.log1p(second_lay_size),
        1.0/ltp if ltp > 0 else 0.0,
        np.log1p(time_to_off),
        number_active_runners,
        np.log1p(market_total_matched)
    ])
    
    # Add other horses features using fixed ordering
    for i, horse_id in enumerate(fixed_horse_order):
        if i >= max_other_horses:
            break
            
        runner_data = bin_data['runners'].get(horse_id)
        if runner_data and runner_data.get('active', True):
            # Horse is present and active
            token.append(1.0)  # presence indicator
            
            back_orders = runner_data.get('back', [])
            other_best_back_price = back_orders[0][0] if back_orders else 0.0
            other_best_back_size = back_orders[0][1] if back_orders else 0.0
            
            lay_orders = runner_data.get('lay', [])
            other_best_lay_price = lay_orders[0][0] if lay_orders else 0.0
            other_best_lay_size = lay_orders[0][1] if lay_orders else 0.0
            
            token.extend([
                1.0/other_best_back_price if other_best_back_price > 0 else 0.0,
                np.log1p(other_best_back_size),
                1.0/other_best_lay_price if other_best_lay_price > 0 else 0.0,
                np.log1p(other_best_lay_size)
            ])
        else:
            # Horse not present or inactive - add zeros
            token.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Fill remaining slots with zeros if needed
    remaining_slots = max_other_horses - len(fixed_horse_order)
    for _ in range(remaining_slots):
        token.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    return token


def create_tokens_from_race(race_data, max_other_horses=max_other_horses):
    """
    Create all tokens for a race (one token per horse per bin).
    
    Args:
        race_data: Complete race data JSON
        max_other_horses: Maximum number of other horses to include
    
    Returns:
        Dict mapping horse_id to list of tokens (one per bin)
    """
    race_tokens = {}
    
    # Get all horses that appear in any bin
    all_horses = set()
    for bin_data in race_data['bins']:
        all_horses.update(bin_data['runners'].keys())
    
    # Create tokens for each horse
    for horse_id in all_horses:
        horse_tokens = []
        for bin_data in race_data['bins']:
            token = create_token_from_bin(bin_data, horse_id, race_data, max_other_horses)
            horse_tokens.append(token)
        race_tokens[horse_id] = horse_tokens
    
    return race_tokens


def load_race_data(file_path, max_races=None):
    """
    Load race data from jsonl.gz file.
    
    Args:
        file_path: Path to the jsonl.gz file
        max_races: Maximum number of races to load (None for all)
    
    Returns:
        List of race data dictionaries
    """
    races = []
    
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if max_races and i >= max_races:
                break
            
            race = orjson.loads(line)
            races.append(race)
    
    return races


def get_eligible_races_and_weights(races, min_consecutive_tokens=160):
    """
    Get races that have enough consecutive tokens and their sampling weights.
    
    Args:
        races: List of race data
        min_consecutive_tokens: Minimum number of consecutive valid tokens required
    
    Returns:
        Tuple of (eligible_races, weights)
    """
    eligible_races = []
    weights = []
    
    for race in races:
        # Count bins with valid data
        valid_bins = 0
        max_consecutive = 0
        current_consecutive = 0
        
        for bin_data in race['bins']:
            if bin_data.get('runners') and len(bin_data['runners']) > 0:
                valid_bins += 1
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # Only include races with sufficient consecutive tokens
        if max_consecutive >= min_consecutive_tokens:
            eligible_races.append(race)
            # Weight by number of valid bins (more data = higher weight)
            weights.append(valid_bins)
    
    return eligible_races, weights


def get_eligible_horses_in_block(race, start_bin, max_odds=15.0):
    """
    Get horses eligible for selection at the start of a block.
    
    Args:
        race: Race data
        start_bin: Index of the starting bin
        max_odds: Maximum back odds allowed for eligibility
    
    Returns:
        List of eligible horse IDs
    """
    if start_bin >= len(race['bins']):
        return []
    
    bin_data = race['bins'][start_bin]
    eligible_horses = []
    
    for horse_id, runner_data in bin_data['runners'].items():
        if not runner_data.get('active', True):
            continue
        
        # Check back odds constraint
        back_orders = runner_data.get('back', [])
        if back_orders:
            best_back_price = back_orders[0][0]
            if best_back_price <= max_odds:
                eligible_horses.append(horse_id)
    
    return eligible_horses


def ensure_market_time_ms(race):
    """
    Ensure race has marketTime_ms field, computing from ISO if needed.
    
    Args:
        race: Race data dictionary
    
    Returns:
        Race data with marketTime_ms field
    """
    if 'marketTime_ms' not in race:
        if 'marketTime' in race:
            from datetime import datetime
            # Parse ISO format to timestamp
            market_time = datetime.fromisoformat(race['marketTime'].replace('Z', '+00:00'))
            race['marketTime_ms'] = int(market_time.timestamp() * 1000)
        else:
            # Fallback: use the last bin time + some offset
            if race['bins']:
                last_bin_time = race['bins'][-1]['t_ms']
                race['marketTime_ms'] = last_bin_time + 60000  # Assume race starts 1 min after last bin
    
    return race


def sample_training_block(races, races_weights, block_size=128, max_odds=15.0):
    """
    Sample a training block (sequence of tokens for one horse).
    
    Args:
        races: List of eligible races
        races_weights: Weights for sampling races
        block_size: Number of tokens in sequence
        max_odds: Maximum back odds for horse eligibility
    
    Returns:
        Tuple of (input_tokens, target_tokens, metadata) or None if sampling fails
    """
    import random
    import numpy as np
    
    min_tokens_needed = block_size + prediction_offset
    
    for attempt in range(10):  # Max 10 attempts per call
        # Sample a race
        race = random.choices(races, weights=races_weights)[0]
        
        # Ensure race has marketTime_ms
        race = ensure_market_time_ms(race)
        
        # Find valid starting positions
        max_start_bin = len(race['bins']) - min_tokens_needed
        if max_start_bin < 0:
            continue
        
        start_bin = random.randint(0, max_start_bin)
        
        # Get eligible horses at this starting point
        eligible_horses = get_eligible_horses_in_block(race, start_bin, max_odds)
        if not eligible_horses:
            continue
        
        # Sample a horse
        selected_horse = random.choice(eligible_horses)
        
        # Create token sequences
        input_tokens = []
        target_tokens = []
        
        # Get fixed horse ordering from the first bin
        first_bin = race['bins'][start_bin]
        horse_ordering = get_fixed_horse_ordering(first_bin, selected_horse, max_other_horses)
        
        # Generate input sequence
        valid_count = 0
        for i in range(block_size):
            bin_idx = start_bin + i
            if bin_idx >= len(race['bins']):
                break
            
            token = create_token_from_bin_with_fixed_order(
                race['bins'][bin_idx], selected_horse, race, horse_ordering
            )
            
            if token is not None:
                input_tokens.append(token)
                valid_count += 1
            else:
                # If we can't create a token, this block is invalid
                break
        
        # Generate target sequence (offset by prediction_offset)
        for i in range(block_size):
            target_bin_idx = start_bin + i + prediction_offset
            if target_bin_idx >= len(race['bins']):
                break
            
            target_token = create_token_from_bin_with_fixed_order(
                race['bins'][target_bin_idx], selected_horse, race, horse_ordering
            )
            
            if target_token is not None:
                target_tokens.append(target_token)
            else:
                break
        
        # Check if we have enough valid tokens
        if len(input_tokens) == block_size and len(target_tokens) == block_size:
            metadata = {
                'race_data': race,
                'horse_id': selected_horse,
                'start_bin': start_bin,
                'horse_ordering': horse_ordering
            }
            return input_tokens, target_tokens, metadata
    
    return None  # Failed to sample valid block


def calculate_future_lay_price_targets_per_step(input_tokens, target_tokens, race_data, selected_horse):
    """
    Calculate future log(lay_price) targets and masks for each step in the sequence.
    Now predicts future lay price directly rather than EV.
    
    Args:
        input_tokens: Current token sequence (block_size,)
        target_tokens: Future token sequence (block_size,) offset by 32 steps
        race_data: Full race data for checking suspensions/removals
        selected_horse: Horse ID for checking activity status
    
    Returns:
        Tuple of (log_lay_price_targets, loss_mask) - lists of values and validity flags
    """
    import numpy as np
    
    log_lay_price_targets = []
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
        
        # Calculate log(future_lay_price) for stability
        if valid:
            # Convert from 1/price back to price since we normalized it in tokenization
            # The target token has future_lay_price as 1/price, so we need to invert it
            future_lay_price_actual = 1.0 / future_lay_price if future_lay_price > 0 else 0.0
            log_lay_price = np.log(future_lay_price_actual)
        else:
            log_lay_price = 0.0  # Will be masked out anyway
        
        log_lay_price_targets.append(log_lay_price)
        loss_mask.append(valid)
    
    return log_lay_price_targets, loss_mask


def calculate_classification_targets_per_step(input_tokens, target_tokens, race_data, selected_horse, edge_threshold=1.02):
    """
    Calculate binary classification targets and masks for each step in the sequence.
    Predicts whether current_back_price < future_lay_price * edge_threshold (i.e., profitable trade).
    
    Args:
        input_tokens: Current token sequence (block_size,)
        target_tokens: Future token sequence (block_size,) offset by 32 steps
        race_data: Full race data for checking suspensions/removals
        selected_horse: Horse ID for checking activity status
        edge_threshold: Minimum edge required for positive classification (default 1.02 = 2% edge)
    
    Returns:
        Tuple of (binary_targets, loss_mask) - lists of binary labels (0/1) and validity flags
    """
    import numpy as np
    
    binary_targets = []
    loss_mask = []
    
    # Get market definition events for checking suspensions/removals
    md_events = race_data.get('marketDefinition_events', [])
    
    for i in range(len(input_tokens)):
        # Current back price at step i
        current_back_price = input_tokens[i][0]  # Best back price at step i
        
        # Future lay price at corresponding step in target sequence
        future_lay_price = target_tokens[i][2] if i < len(target_tokens) else 0.0
        
        # Check validity conditions (same logic as regression version)
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
        
        # Calculate binary classification target
        if valid:
            # Convert from 1/price back to price since we normalized it in tokenization
            current_back_price_actual = 1.0 / current_back_price if current_back_price > 0 else 0.0
            future_lay_price_actual = 1.0 / future_lay_price if future_lay_price > 0 else 0.0
            
            # Binary target: 1 if current_back_price < future_lay_price * edge_threshold, 0 otherwise
            # This means we can buy at current_back_price and sell at future_lay_price with at least edge_threshold profit
            binary_target = 1.0 if current_back_price_actual < future_lay_price_actual * edge_threshold else 0.0
        else:
            binary_target = 0.0  # Will be masked out anyway
        
        binary_targets.append(binary_target)
        loss_mask.append(valid)
    
    return binary_targets, loss_mask


def calculate_ev_targets_per_step(input_tokens, target_tokens, race_data, selected_horse):
    """
    Legacy function for backward compatibility with evaluation script.
    Converts future lay price predictions back to EV for evaluation.
    """
    # Get the new targets
    log_lay_price_targets, loss_mask = calculate_future_lay_price_targets_per_step(
        input_tokens, target_tokens, race_data, selected_horse
    )
    
    # Convert back to log-EV for compatibility
    log_ev_targets = []
    for i, (log_lay_price, mask) in enumerate(zip(log_lay_price_targets, loss_mask)):
        if mask:
            # Calculate EV = current_back_price / future_lay_price
            current_back_price_normalized = input_tokens[i][0]  # 1/price format
            current_back_price = 1.0 / current_back_price_normalized if current_back_price_normalized > 0 else 0.0
            future_lay_price = np.exp(log_lay_price)
            log_ev = np.log(current_back_price / future_lay_price) if future_lay_price > 0 else 0.0
        else:
            log_ev = 0.0
        log_ev_targets.append(log_ev)
    
    return log_ev_targets, loss_mask


def lay_price_predictions_to_ev(log_lay_price_predictions, current_back_prices):
    """
    Convert future lay price predictions to EV for inference.
    EV = current_back_price / future_lay_price
    
    Args:
        log_lay_price_predictions: Tensor of log(future_lay_price) predictions
        current_back_prices: Tensor of current back prices (in 1/price format)
    
    Returns:
        Tensor of EV predictions
    """
    import torch
    
    future_lay_prices = torch.exp(log_lay_price_predictions)
    # Convert current_back_prices from 1/price format back to actual prices
    current_back_actual = 1.0 / (current_back_prices + 1e-8)  # Add small epsilon to avoid division by zero
    ev_predictions = current_back_actual / (future_lay_prices + 1e-8)
    return ev_predictions


def get_batch_from_races(races, races_weights, batch_size=64, block_size=128):
    """
    Generate a batch of training sequences from race data.
    
    Args:
        races: List of eligible races
        races_weights: Weights for sampling races
        batch_size: Number of sequences in batch
        block_size: Length of each sequence
    
    Returns:
        Tuple of (input_batch, future_lay_price_targets, loss_masks) as torch tensors
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
            
            # Calculate future lay price targets and masks for each step
            race_data = metadata['race_data']
            selected_horse = metadata['horse_id']
            log_lay_price_targets_sequence, loss_mask_sequence = calculate_future_lay_price_targets_per_step(
                input_tokens, target_tokens, race_data, selected_horse
            )
            
            input_batch.append(input_array)
            log_ev_targets_batch.append(log_lay_price_targets_sequence)
            loss_masks_batch.append(loss_mask_sequence)
    
    if len(input_batch) == 0:
        return None, None, None
    
    # Stack into tensors
    input_tensor = torch.tensor(np.stack(input_batch), dtype=torch.float32)
    lay_price_tensor = torch.tensor(np.array(log_ev_targets_batch), dtype=torch.float32)  # (batch_size, block_size)
    mask_tensor = torch.tensor(np.array(loss_masks_batch), dtype=torch.bool)  # (batch_size, block_size)
    
    return input_tensor, lay_price_tensor, mask_tensor


def initialize_race_data(train_path, val_path, max_races=None):
    """
    Initialize global race data and weights for training.
    
    Args:
        train_path: Path to training data file
        val_path: Path to validation data file
        max_races: Maximum number of races to load per file
    
    Returns:
        Tuple of (train_races, train_weights, val_races, val_weights)
    """
    print(f"📂 Loading training data from {train_path}")
    train_races_all = load_race_data(train_path, max_races)
    train_races, train_weights = get_eligible_races_and_weights(train_races_all)
    
    print(f"📂 Loading validation data from {val_path}")
    val_races_all = load_race_data(val_path, max_races)
    val_races, val_weights = get_eligible_races_and_weights(val_races_all)
    
    print(f"✅ Loaded {len(train_races)} training races, {len(val_races)} validation races")
    
    return train_races, train_weights, val_races, val_weights


def test_token_generation():
    """
    Test function to print out token structure and future lay price prediction.
    """
    import numpy as np
    
    print("🔧 TEST: Token Generation and Future Lay Price Prediction")
    print("=" * 60)
    
    # Initialize a small dataset for testing
    print("Loading test data...")
    try:
        train_races, train_weights, _, _ = initialize_race_data(
            "datasets/races_train_split_with_bins.jsonl.gz",
            "datasets/races_validation_split.jsonl.gz",
            max_races=10
        )
        
        if not train_races:
            print("❌ No eligible races found!")
            return
        
        print(f"✅ Loaded {len(train_races)} test races")
        
        # Sample a few training blocks
        for test_num in range(3):
            print(f"\n🔍 TEST {test_num + 1}:")
            print("-" * 40)
            
            sample = sample_training_block(train_races, train_weights, block_size=5)
            if sample is None:
                print("❌ Failed to sample training block")
                continue
            
            input_tokens, target_tokens, metadata = sample
            race_data = metadata['race_data']
            selected_horse = metadata['horse_id']
            
            print(f"  Race: {race_data.get('marketId', 'Unknown')}")
            print(f"  Subject Horse: {selected_horse}")
            print(f"  Block size: {len(input_tokens)}")
            print(f"  Token size: {len(input_tokens[0]) if input_tokens else 0}")
            
            # Show first few tokens
            for i, (token, target_token) in enumerate(zip(input_tokens[:3], target_tokens[:3])):
                print(f"\n    Step {i+1}:")
                print(f"      Input token length: {len(token)}")
                print(f"      Subject horse features (first 12):")
                for j, val in enumerate(token[:12]):
                    feature_names = [
                        "best_back_1/price", "best_back_log_size", "best_lay_1/price", "best_lay_log_size",
                        "second_back_1/price", "second_back_log_size", "second_lay_1/price", "second_lay_log_size", 
                        "ltp_1/price", "log_time_to_off", "num_runners", "log_total_matched"
                    ]
                    print(f"        {feature_names[j]}: {val:.3f}")
                
                # Show a few other horses
                other_horse_start = 12
                for j in range(min(3, max_other_horses)):
                    start_idx = other_horse_start + j * 5
                    if start_idx + 4 < len(token):
                        presence = token[start_idx]
                        back_price = token[start_idx + 1]
                        back_size = token[start_idx + 2]
                        lay_price = token[start_idx + 3]
                        lay_size = token[start_idx + 4]
                        print(f"      Horse {j+1}: present={presence:.0f}, back_1/price={back_price:.3f}/log_size={back_size:.2f}, lay_1/price={lay_price:.3f}/log_size={lay_size:.2f}")
            
            # Show target and lay price calculation
            if target_token is not None:
                current_back_1_over_price = input_tokens[0][0]  # Current back 1/price
                future_lay_1_over_price = target_tokens[0][2]   # Future lay 1/price
                
                if current_back_1_over_price > 0 and future_lay_1_over_price > 0:
                    current_back_price = 1.0 / current_back_1_over_price
                    future_lay_price = 1.0 / future_lay_1_over_price
                    ev_raw = current_back_price / future_lay_price
                    log_lay_price = float(np.log(future_lay_price))
                    print(f"    Target Calculation:")
                    print(f"      Current back price: {current_back_price:.3f}")
                    print(f"      Future lay price:  {future_lay_price:.3f}")
                    print(f"      Log(future_lay_price): {log_lay_price:.3f}")
                    print(f"      Derived EV: {ev_raw:.3f}")
                else:
                    print(f"    Target Calculation: INVALID (back_1/price={current_back_1_over_price:.3f}, lay_1/price={future_lay_1_over_price:.3f})")
        
        # Test the complete target calculation with masks
        print(f"\n🧪 Testing target calculation...")
        if train_races:
            sample = sample_training_block(train_races, train_weights, block_size=10)
            if sample:
                input_tokens, target_tokens, metadata = sample
                log_lay_price_targets, loss_mask = calculate_future_lay_price_targets_per_step(
                    input_tokens, target_tokens, metadata['race_data'], metadata['horse_id']
                )
                
                valid_count = sum(loss_mask)
                total_count = len(loss_mask)
                print(f"  Valid targets: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                print(f"  First 5 log(lay_price) targets: {log_lay_price_targets[:5]}")
                print(f"  First 5 masks: {loss_mask[:5]}")
                
                # Show range of valid predictions
                valid_log_lay_prices = [target for target, mask in zip(log_lay_price_targets, loss_mask) if mask]
                if valid_log_lay_prices:
                    valid_lay_prices = [np.exp(log_price) for log_price in valid_log_lay_prices]
                    print(f"  Valid log(lay_price) range: [{min(valid_log_lay_prices):.3f}, {max(valid_log_lay_prices):.3f}]")
                    print(f"  Valid lay_price range: [{min(valid_lay_prices):.3f}, {max(valid_lay_prices):.3f}]")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
