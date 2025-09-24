#!/usr/bin/env python3
"""
Comprehensive EV evaluation script for transformer checkpoints.

Focus: When predicted EV > 1.0, what does realized EV actually end up being?
Delivers detailed CSV instances, JSON summaries, and performance analytics.
"""

import argparse
import json
import gzip
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import sys
import os
from datetime import datetime

# Add parent directory to path to import from bf_transformer
sys.path.append(str(Path(__file__).parent.parent))

# Import functions from bf_transformer (now safe since we fixed the import issue)
try:
    from bf_transformer import (
        TransformerRegressionModel, ensure_market_time_ms, 
        get_fixed_horse_ordering, create_token_from_bin_with_fixed_order,
        calculate_ev_targets_per_step
    )
except ImportError as e:
    print(f"❌ Error importing from bf_transformer: {e}")
    print("Make sure bf_transformer.py is in the parent directory")
    sys.exit(1)

# Device detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint and extract hyperparameters."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hps = checkpoint.get('hyperparameters', {})
    step = checkpoint.get('step', 'unknown')
    
    print(f"   Step: {step}")
    print(f"   Hyperparameters: {len(hps)} found")
    
    # Initialize model with correct architecture
    model = TransformerRegressionModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    
    return model, hps, step

def realized_ev_single(current_back, future_lay):
    """Calculate realized EV for a single instance."""
    if current_back <= 0 or future_lay <= 0:
        return np.nan
    return current_back / future_lay

def pnl_proxy(realized_ev, unit_stake, commission):
    """Calculate PnL proxy with optional commission."""
    if not np.isfinite(realized_ev):
        return 0.0
    
    gross = unit_stake * (realized_ev - 1.0)
    
    if gross <= 0 or commission <= 0:
        return gross
    
    # Apply commission only to positive gross (winning side)
    return gross * (1.0 - commission)

def get_bucketing_schemes():
    """Define bucketing schemes for analysis."""
    return {
        'pred_ev_bins': [
            ('≤0.95', 0.0, 0.95),
            ('0.95-1.00', 0.95, 1.00),
            ('1.00-1.02', 1.00, 1.02),
            ('1.02-1.05', 1.02, 1.05),
            ('1.05-1.10', 1.05, 1.10),
            ('1.10-1.20', 1.10, 1.20),
            ('1.20-1.50', 1.20, 1.50),
            ('>1.50', 1.50, float('inf'))
        ],
        'time_to_off_bins': [
            ('>600s', 600, float('inf')),
            ('300-600s', 300, 600),
            ('120-300s', 120, 300),
            ('60-120s', 60, 120),
            ('30-60s', 30, 60),
            ('10-30s', 10, 30),
            ('≤10s', 0, 10)
        ],
        'back_odds_bins': [
            ('≤2.0', 0.0, 2.0),
            ('2.0-3.0', 2.0, 3.0),
            ('3.0-5.0', 3.0, 5.0),
            ('5.0-8.0', 5.0, 8.0),
            ('8.0-12.0', 8.0, 12.0),
            ('12.0-15.0', 12.0, 15.0),
            ('>15.0', 15.0, float('inf'))
        ]
    }

def bucket_value(value, bins):
    """Assign a value to its bucket."""
    for label, min_val, max_val in bins:
        if min_val <= value < max_val:
            return label
    return bins[-1][0]  # Default to last bucket for overflow

def build_eval_instances_from_race(race, model, device, args, max_other_horses, expected_token_size):
    """Extract evaluation instances from a single race."""
    race_id = race.get('marketId', 'unknown')
    bins = race.get('bins', [])
    
    if len(bins) < args.block_size + args.prediction_offset:
        return []
    
    instances = []
    bucketing = get_bucketing_schemes()
    
    # Get all horses present in the race
    all_horse_ids = set()
    for bin_data in bins:
        runners = bin_data.get('runners', {})
        all_horse_ids.update(int(hid) for hid in runners.keys())
    
    # For each horse, build evaluation windows
    for horse_id in all_horse_ids:
        # Find valid window start positions - fix stride issue
        for t in range(args.block_size - 1, len(bins) - args.prediction_offset, args.stride):
            window_start = t - args.block_size + 1
            target_idx = t + args.prediction_offset
            
            if target_idx >= len(bins):
                continue
            
            # Get horse ordering from the first bin of the window - fix signature
            first_bin = bins[window_start]
            try:
                horse_ordering = get_fixed_horse_ordering(first_bin, horse_id, max_other_horses)
                # Remove incorrect check - horse_ordering is for OTHER horses, not subject
            except Exception:
                continue
            
            # Build input tokens for the window
            input_tokens = []
            target_tokens = []
            valid_window = True
            
            # Build input tokens for bins [window_start ... t]
            for bin_idx in range(window_start, t + 1):
                try:
                    token = create_token_from_bin_with_fixed_order(
                        bins[bin_idx], horse_id, race, horse_ordering
                    )
                    if token is None or len(token) != expected_token_size:
                        valid_window = False
                        break
                    input_tokens.append(token)
                except Exception:
                    valid_window = False
                    break
            
            if not valid_window or len(input_tokens) != args.block_size:
                continue
            
            # Build target tokens for bins [window_start + prediction_offset ... t + prediction_offset]
            for bin_idx in range(window_start + args.prediction_offset, t + 1 + args.prediction_offset):
                try:
                    token = create_token_from_bin_with_fixed_order(
                        bins[bin_idx], horse_id, race, horse_ordering
                    )
                    if token is None or len(token) != expected_token_size:
                        valid_window = False
                        break
                    target_tokens.append(token)
                except Exception:
                    valid_window = False
                    break
            
            if not valid_window or len(target_tokens) != args.block_size:
                continue
                
            # Use training-aligned validity criteria
            try:
                log_ev_targets, loss_mask = calculate_ev_targets_per_step(
                    input_tokens, target_tokens, race, horse_id
                )
                # Keep only if the last step is valid by training rules
                if not bool(loss_mask[-1]):
                    continue
                    
                # Enforce min_valid_seq_frac if specified
                if args.min_valid_seq_frac > 0:
                    valid_frac = float(np.mean(loss_mask))
                    if valid_frac < args.min_valid_seq_frac:
                        continue
            except Exception:
                continue
            
            # Get current market state (last step of window)
            current_bin = bins[t]
            current_runners = current_bin.get('runners', {})
            current_runner = current_runners.get(str(horse_id), {})
            
            if not current_runner.get('active', False):
                continue
            
            # Extract current market features
            current_back = current_runner.get('back', [])
            current_lay = current_runner.get('lay', [])
            
            if not current_back or not current_lay:
                continue
            
            current_back_price = float(current_back[0][0])
            current_back_size = float(current_back[0][1])
            current_lay_price = float(current_lay[0][0])
            current_lay_size = float(current_lay[0][1])
            current_ltp = current_runner.get('ltp')
            
            # Apply max back odds filter
            if current_back_price > args.max_back_odds:
                continue
            
            # Get target bin for realized EV
            target_bin = bins[target_idx]
            target_runners = target_bin.get('runners', {})
            target_runner = target_runners.get(str(horse_id), {})
            
            if not target_runner.get('active', False):
                continue
            
            target_lay = target_runner.get('lay', [])
            if not target_lay:
                continue
            
            future_lay_price = float(target_lay[0][0])
            
            # Calculate realized EV
            realized_ev = realized_ev_single(current_back_price, future_lay_price)
            if not np.isfinite(realized_ev):
                continue
            
            # Make prediction with model - fix return type handling
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_tokens).unsqueeze(0).to(device)
                pred_log_ev_all, _ = model(input_tensor)  # Model returns (predictions, loss)
                pred_log_ev_last = pred_log_ev_all[0, -1].cpu().item()  # Last step only
                pred_ev = math.exp(pred_log_ev_last)
            
            # Calculate PnL proxy
            pnl_gross = pnl_proxy(realized_ev, args.unit_stake, 0.0)
            pnl_after_fee = pnl_proxy(realized_ev, args.unit_stake, args.commission)
            
            # Get timing info
            current_t_ms = current_bin.get('t_ms', 0)
            target_t_ms = target_bin.get('t_ms', 0)
            market_time_ms = race.get('marketTime_ms', 0)
            time_to_off = max(0, (market_time_ms - current_t_ms) / 1000.0)
            
            # Create instance record
            instance = {
                # Identifiers
                'race_id': race_id,
                'horse_id': horse_id,
                'start_bin': window_start,
                'target_bin': target_idx,
                'start_time_ms': current_t_ms,
                'target_time_ms': target_t_ms,
                
                # Model predictions
                'pred_log_ev': pred_log_ev_last,
                'pred_ev': pred_ev,
                
                # Current market state
                'current_back_price': current_back_price,
                'current_back_size': current_back_size,
                'current_lay_price': current_lay_price,
                'current_lay_size': current_lay_size,
                'ltp': current_ltp,
                'time_to_off': time_to_off,
                'number_active_runners': current_bin.get('numberActiveRunners', 0),
                'market_total_matched': current_bin.get('market_total_matched', 0),
                
                # Target outcomes
                'future_lay_price': future_lay_price,
                'realized_ev': realized_ev,
                'edge_realized': realized_ev - 1.0,
                'valid_mask': True,
                
                # Policy proxy
                'unit_stake': args.unit_stake,
                'commission': args.commission,
                'pnl_proxy': pnl_gross,
                'pnl_proxy_after_fee': pnl_after_fee,
                
                # Buckets
                'bucket_pred_ev': bucket_value(pred_ev, bucketing['pred_ev_bins']),
                'bucket_time_to_off': bucket_value(time_to_off, bucketing['time_to_off_bins']),
                'bucket_back_odds': bucket_value(current_back_price, bucketing['back_odds_bins'])
            }
            
            instances.append(instance)
            
            # Use stride to avoid overlapping windows
            if args.stride > 1:
                t += args.stride - 1
    
    return instances

def compute_bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return np.nan, np.nan
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper

def summarize_results(df, args):
    """Compute comprehensive summary statistics."""
    print("\n📊 COMPUTING SUMMARY STATISTICS")
    print("=" * 50)
    
    # Clean NaNs and infinities before analysis
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pred_ev', 'realized_ev'])
    
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(args.checkpoint),
            'test_file': str(args.test_file),
            'total_instances': len(df),
            'commission_model': 'win-side only, simple model',
            'args': vars(args)
        }
    }
    
    if len(df) == 0:
        summary['error'] = "No valid instances found"
        return summary
    
    # Global statistics
    valid_df = df[df['valid_mask']]
    n_total = len(df)
    n_valid = len(valid_df)
    
    print(f"📈 Global Stats: {n_valid:,} valid instances from {n_total:,} total")
    
    summary['global'] = {
        'n_total': n_total,
        'n_valid': n_valid,
        'valid_rate': n_valid / n_total if n_total > 0 else 0,
    }
    
    if n_valid == 0:
        summary['error'] = "No valid instances found"
        return summary
    
    # Correlations
    pred_ev = valid_df['pred_ev'].values
    realized_ev = valid_df['realized_ev'].values
    
    if len(pred_ev) > 1:
        pearson_corr = np.corrcoef(pred_ev, realized_ev)[0, 1]
        spearman_corr = pd.Series(pred_ev).corr(pd.Series(realized_ev), method='spearman')
        
        summary['global']['pearson_correlation'] = float(pearson_corr)
        summary['global']['spearman_correlation'] = float(spearman_corr)
        
        print(f"🔗 Correlations: Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}")
    
    # Positive EV classification metrics
    positive_ev_target = (realized_ev > 1.0).astype(int)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = roc_auc_score(positive_ev_target, pred_ev)
        pr_auc = average_precision_score(positive_ev_target, pred_ev)
        
        summary['global']['roc_auc'] = float(roc_auc)
        summary['global']['pr_auc'] = float(pr_auc)
        
        print(f"🎯 Classification: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    except ImportError:
        print("⚠️  sklearn not available for ROC/PR metrics")
    
    # Predicted EV breakdown
    print(f"\n📊 PREDICTED EV BREAKDOWN")
    print("=" * 50)
    
    pred_ev_analysis = []
    bucketing = get_bucketing_schemes()
    
    for bucket_name, min_val, max_val in bucketing['pred_ev_bins']:
        if min_val == 0.0:
            mask = valid_df['pred_ev'] <= max_val
        elif max_val == float('inf'):
            mask = valid_df['pred_ev'] >= min_val
        else:
            mask = (valid_df['pred_ev'] >= min_val) & (valid_df['pred_ev'] < max_val)
        
        bucket_df = valid_df[mask]
        
        if len(bucket_df) == 0:
            continue
        
        realized_evs = bucket_df['realized_ev'].values
        pred_evs = bucket_df['pred_ev'].values
        pnl_values = bucket_df['pnl_proxy_after_fee'].values
        
        # Bootstrap CI for realized EV
        ci_lower, ci_upper = compute_bootstrap_ci(realized_evs)
        
        bucket_stats = {
            'bucket': bucket_name,
            'count': len(bucket_df),
            'coverage_pct': len(bucket_df) / len(valid_df) * 100,
            'mean_pred_ev': float(np.mean(pred_evs)),
            'mean_realized_ev': float(np.mean(realized_evs)),
            'median_realized_ev': float(np.median(realized_evs)),
            'frac_positive_ev': float(np.mean(realized_evs > 1.0)),
            'mean_edge_realized': float(np.mean(realized_evs - 1.0)),
            'mean_pnl_proxy': float(np.mean(pnl_values)),
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None
        }
        
        pred_ev_analysis.append(bucket_stats)
        
        print(f"{bucket_name:12s}: {len(bucket_df):4d} ({len(bucket_df)/len(valid_df)*100:5.1f}%) | "
              f"Pred: {np.mean(pred_evs):.3f} | Real: {np.mean(realized_evs):.3f} | "
              f"+EV: {np.mean(realized_evs > 1.0)*100:4.1f}% | PnL: {np.mean(pnl_values):+.4f}")
    
    summary['pred_ev_breakdown'] = pred_ev_analysis
    
    # Threshold analysis
    print(f"\n🎯 THRESHOLD POLICY ANALYSIS")
    print("=" * 70)
    
    thresholds = [1.00, 1.01, 1.02, 1.05, 1.10, 1.20, 1.30]
    threshold_analysis = []
    
    for threshold in thresholds:
        signals_df = valid_df[valid_df['pred_ev'] >= threshold].copy()
        
        if len(signals_df) == 0:
            continue
        
        # Sort by time for drawdown calculation
        signals_df = signals_df.sort_values('target_time_ms')
        
        realized_evs = signals_df['realized_ev'].values
        pnl_values = signals_df['pnl_proxy_after_fee'].values
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        threshold_stats = {
            'threshold': threshold,
            'n_signals': len(signals_df),
            'coverage_pct': len(signals_df) / len(valid_df) * 100,
            'mean_realized_ev': float(np.mean(realized_evs)),
            'frac_positive_ev': float(np.mean(realized_evs > 1.0)),
            'avg_pnl_proxy': float(np.mean(pnl_values)),
            'cum_pnl_proxy': float(np.sum(pnl_values)),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(np.mean(pnl_values) / np.std(pnl_values)) if np.std(pnl_values) > 0 else 0
        }
        
        threshold_analysis.append(threshold_stats)
        
        print(f"τ≥{threshold:.2f}: {len(signals_df):4d} signals ({len(signals_df)/len(valid_df)*100:5.1f}%) | "
              f"Real EV: {np.mean(realized_evs):.3f} | +EV: {np.mean(realized_evs > 1.0)*100:4.1f}% | "
              f"Avg PnL: {np.mean(pnl_values):+.4f} | Cum PnL: {np.sum(pnl_values):+.2f} | "
              f"MaxDD: {max_drawdown:+.2f}")
    
    summary['threshold_analysis'] = threshold_analysis
    
    # Context breakdowns
    summary['context_analysis'] = {}
    
    for context_type in ['bucket_time_to_off', 'bucket_back_odds']:
        context_analysis = []
        for bucket_name in valid_df[context_type].unique():
            bucket_df = valid_df[valid_df[context_type] == bucket_name]
            
            if len(bucket_df) == 0:
                continue
            
            realized_evs = bucket_df['realized_ev'].values
            pnl_values = bucket_df['pnl_proxy_after_fee'].values
            
            context_stats = {
                'bucket': bucket_name,
                'count': len(bucket_df),
                'coverage_pct': len(bucket_df) / len(valid_df) * 100,
                'mean_realized_ev': float(np.mean(realized_evs)),
                'frac_positive_ev': float(np.mean(realized_evs > 1.0)),
                'mean_pnl_proxy': float(np.mean(pnl_values))
            }
            
            context_analysis.append(context_stats)
        
        summary['context_analysis'][context_type] = context_analysis
    
    return summary

def create_summary_text(summary):
    """Create human-readable summary text."""
    if 'error' in summary:
        return f"❌ Error: {summary['error']}"
    
    text = []
    text.append("🎯 BETFAIR TRANSFORMER EV EVALUATION SUMMARY")
    text.append("=" * 60)
    text.append(f"Timestamp: {summary['metadata']['timestamp']}")
    text.append(f"Checkpoint: {summary['metadata']['checkpoint']}")
    text.append(f"Test file: {summary['metadata']['test_file']}")
    text.append("")
    
    # Global stats
    global_stats = summary['global']
    text.append("📊 GLOBAL STATISTICS")
    text.append("-" * 30)
    text.append(f"Total instances: {global_stats['n_total']:,}")
    text.append(f"Valid instances: {global_stats['n_valid']:,}")
    text.append(f"Valid rate: {global_stats['valid_rate']:.1%}")
    
    if 'pearson_correlation' in global_stats:
        text.append(f"Pearson correlation: {global_stats['pearson_correlation']:.4f}")
        text.append(f"Spearman correlation: {global_stats['spearman_correlation']:.4f}")
    
    if 'roc_auc' in global_stats:
        text.append(f"ROC-AUC (EV>1 classification): {global_stats['roc_auc']:.4f}")
        text.append(f"PR-AUC (EV>1 classification): {global_stats['pr_auc']:.4f}")
    
    text.append("")
    
    # Predicted EV breakdown
    text.append("📈 PREDICTED EV BREAKDOWN")
    text.append("-" * 40)
    text.append(f"{'Bucket':<12} {'Count':<6} {'Cov%':<6} {'MPredEV':<8} {'MRealEV':<8} {'+EV%':<6} {'AvgPnL':<8}")
    text.append("-" * 70)
    
    for bucket in summary['pred_ev_breakdown']:
        text.append(f"{bucket['bucket']:<12} {bucket['count']:<6} "
                   f"{bucket['coverage_pct']:<6.1f} {bucket['mean_pred_ev']:<8.3f} "
                   f"{bucket['mean_realized_ev']:<8.3f} {bucket['frac_positive_ev']:<6.1%} "
                   f"{bucket['mean_pnl_proxy']:<8.4f}")
    
    text.append("")
    
    # Threshold analysis
    text.append("🎯 THRESHOLD POLICY ANALYSIS")
    text.append("-" * 40)
    text.append(f"{'Thresh':<7} {'Signals':<8} {'Cov%':<6} {'MRealEV':<8} {'+EV%':<6} {'AvgPnL':<8} {'CumPnL':<8} {'MaxDD':<8}")
    text.append("-" * 70)
    
    for thresh in summary['threshold_analysis']:
        text.append(f"{thresh['threshold']:<7.2f} {thresh['n_signals']:<8} "
                   f"{thresh['coverage_pct']:<6.1f} {thresh['mean_realized_ev']:<8.3f} "
                   f"{thresh['frac_positive_ev']:<6.1%} {thresh['avg_pnl_proxy']:<8.4f} "
                   f"{thresh['cum_pnl_proxy']:<8.2f} {thresh['max_drawdown']:<8.2f}")
    
    text.append("")
    text.append("✅ Evaluation complete!")
    
    return "\n".join(text)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive EV evaluation for transformer checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pt file)')
    
    # Optional arguments
    parser.add_argument('--test-file', default='datasets/races_test_split.jsonl.gz',
                        help='Path to test dataset')
    parser.add_argument('--save-dir', default='./eval_artifacts',
                        help='Directory to save evaluation artifacts')
    parser.add_argument('--max-races', type=int, default=None,
                        help='Maximum races to process (None for all)')
    parser.add_argument('--block-size', type=int, default=None,
                        help='Block size (defaults from checkpoint)')
    parser.add_argument('--prediction-offset', type=int, default=None,
                        help='Prediction offset (defaults from checkpoint)')
    parser.add_argument('--max-back-odds', type=float, default=15.0,
                        help='Maximum back odds filter')
    parser.add_argument('--commission', type=float, default=0.0,
                        help='Commission rate for PnL calculation')
    parser.add_argument('--unit-stake', type=float, default=1.0,
                        help='Unit stake for PnL proxy')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride for window sampling (defaults to block_size)')
    parser.add_argument('--min-valid-seq-frac', type=float, default=0.8,
                        help='Minimum valid sequence fraction')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='auto',
                        help='Device to use for inference')
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🎯 BETFAIR TRANSFORMER EV EVALUATION")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file: {args.test_file}")
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Load model and checkpoint
    try:
        model, hps, step = load_checkpoint(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return 1
    
    # Set defaults from hyperparameters
    args.block_size = args.block_size or hps.get('block_size', 256)
    args.prediction_offset = args.prediction_offset or hps.get('prediction_offset', 32)
    args.stride = args.stride or args.block_size
    
    # Checkpoint sanity checks
    if args.block_size != hps.get('block_size', args.block_size):
        print(f"⚠️  block_size mismatch (ckpt {hps.get('block_size')} vs cli {args.block_size})")
    if args.prediction_offset != hps.get('prediction_offset', args.prediction_offset):
        print(f"⚠️  prediction_offset mismatch (ckpt {hps.get('prediction_offset')} vs cli {args.prediction_offset})")
    
    # Derive token parameters from checkpoint
    max_other_horses = int(hps.get('max_other_horses', 23))
    expected_token_size = int(hps.get('token_size', 12 + max_other_horses * 5))
    
    print(f"Block size: {args.block_size}")
    print(f"Prediction offset: {args.prediction_offset}")
    print(f"Stride: {args.stride}")
    print(f"Max other horses: {max_other_horses}")
    print(f"Expected token size: {expected_token_size}")
    
    # Process test data
    print(f"\n📂 PROCESSING TEST DATA")
    print("=" * 30)
    
    all_instances = []
    
    if not os.path.exists(args.test_file):
        print(f"❌ Test file not found: {args.test_file}")
        return 1
    
    with gzip.open(args.test_file, 'rt') as f:
        race_count = 0
        
        for line in tqdm(f, desc="Processing races"):
            if args.max_races and race_count >= args.max_races:
                break
            
            try:
                race = json.loads(line)
                race = ensure_market_time_ms(race)
                
                race_instances = build_eval_instances_from_race(
                    race, model, args.device, args, max_other_horses, expected_token_size
                )
                all_instances.extend(race_instances)
                
                race_count += 1
                
                if race_count % 100 == 0:
                    print(f"Processed {race_count} races, {len(all_instances)} instances so far")
                    
            except Exception as e:
                print(f"❌ Error processing race: {e}")
                continue
    
    print(f"\n✅ Processed {race_count} races")
    print(f"📊 Total instances: {len(all_instances)}")
    
    if not all_instances:
        print("❌ No valid instances found!")
        return 1
    
    # Convert to DataFrame
    df = pd.DataFrame(all_instances)
    
    # De-duplicate if stride < block_size to avoid double-counting
    if args.stride < args.block_size and len(df) > 0:
        print(f"🔄 De-duplicating instances (stride={args.stride} < block_size={args.block_size})")
        initial_count = len(df)
        # Keep first occurrence of each (race_id, horse_id, target_time_ms) combination
        df = df.drop_duplicates(subset=['race_id', 'horse_id', 'target_time_ms'], keep='first')
        final_count = len(df)
        if initial_count != final_count:
            print(f"   Removed {initial_count - final_count} duplicate instances")
    
    
    # Save instances CSV
    instances_file = save_dir / 'instances.csv'
    df.to_csv(instances_file, index=False)
    print(f"💾 Saved instances: {instances_file}")
    
    # Compute summary
    summary = summarize_results(df, args)
    
    # Save summary JSON
    summary_json_file = save_dir / 'summary.json'
    with open(summary_json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved JSON summary: {summary_json_file}")
    
    # Create and save summary text
    summary_text = create_summary_text(summary)
    summary_txt_file = save_dir / 'summary.txt'
    with open(summary_txt_file, 'w') as f:
        f.write(summary_text)
    print(f"💾 Saved text summary: {summary_txt_file}")
    
    # Print summary to console
    print(f"\n{summary_text}")
    
    print(f"\n🎉 Evaluation complete!")
    print(f"📁 Artifacts saved to: {save_dir}")
    
    return 0

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    exit_code = main()
    sys.exit(exit_code)
