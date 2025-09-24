#!/usr/bin/env python3
"""
Betfair Data Processing Pipeline

A deterministic, fast, streaming pipeline that processes data from raw .bz2 files 
directly to final training-ready outputs with optional multiprocessing.

No persistent intermediate files are saved - everything flows through temporary files.

Final outputs:
- races_train_split_with_bins.jsonl.gz (training set, ready for model)
- races_test_split.jsonl.gz (test set)  
- races_validation_split.jsonl.gz (validation set)

Usage:
    python process_data_pipeline.py --bz2-root /path/to/bz2/files
"""

import argparse
import os
import sys
import tempfile
import gzip
import json
import gc
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
import threading
from types import SimpleNamespace

from data_processing.build_race_json import BetfairRaceBuilder, list_bz2_files
from data_processing.split_races_streaming import write_split_line
from data_processing.split_scratchings import (
    find_scratching_events, create_exclusion_zones, create_race_segments, create_segmented_race
)

def _process_one_file(bz2_path, args_dict):
    """Subprocess-safe: reconstruct a Namespace-lite object and process one file."""
    class ArgsNamespace:
        pass
    
    args = ArgsNamespace()
    for k, v in args_dict.items():
        setattr(args, k, v)
    
    builder = BetfairRaceBuilder(args)
    return builder.process_file(bz2_path)


def pass_A_stream_and_split(args, tmp_dir, parallel=True):
    """
    Pass A: Build races from .bz2 files and stream to hash-based splits.
    
    Creates temporary split files:
    - train_raw.jsonl.gz
    - test_raw.jsonl.gz  
    - val_raw.jsonl.gz
    """
    print(f"\n{'='*60}")
    print(f"🚀 PASS A: Building races and streaming to hash-based splits")
    print(f"{'='*60}")
    
    # Open writers for raw (unscratched) splits
    train_raw = gzip.open(os.path.join(tmp_dir, "train_raw.jsonl.gz"), "wt")
    test_raw = gzip.open(os.path.join(tmp_dir, "test_raw.jsonl.gz"), "wt")
    val_raw = gzip.open(os.path.join(tmp_dir, "val_raw.jsonl.gz"), "wt")
    writers = {"train": train_raw, "test": test_raw, "val": val_raw}
    
    bz2_paths = list_bz2_files(args.bz2_root, args.max_files)
    print(f"📁 Found {len(bz2_paths)} .bz2 files to process")
    
    if not parallel:
        # Simple serial processing
        print("🔄 Processing files serially...")
        bargs = SimpleNamespace(
            root=args.bz2_root, out="", bin_ms=args.bin_ms, window_min=args.window_min,
            liquidity_min=args.liquidity_min, stop_criterion="marketTime",
            flat_book=False, omit_empty_runners=False, topk=getattr(args, "topk", 0),
            max_files=args.max_files
        )
        builder = BetfairRaceBuilder(bargs)
        
        race_count = 0
        for race in builder.iter_races_from_paths(bz2_paths):
            write_split_line(race, writers)
            race_count += 1
            
            if race_count % 100 == 0:
                print(f"  📊 Processed {race_count} races...")
                
        for f in writers.values():
            f.close()
        print(f"✅ Serial processing complete: {race_count} races processed")
        return
    
    # Parallel processing with worker threads
    print("⚡ Processing files in parallel...")
    
    def writer_thread(q):
        """Background thread to write races to split files."""
        while True:
            item = q.get()
            if item is None:
                break
            race = item
            write_split_line(race, writers)
            q.task_done()
    
    # Start writer thread
    q = Queue(maxsize=4096)
    wt = threading.Thread(target=writer_thread, args=(q,), daemon=True)
    wt.start()
    
    # Prepare args for subprocess
    args_dict = dict(
        root=args.bz2_root, out="", bin_ms=args.bin_ms, window_min=args.window_min,
        liquidity_min=args.liquidity_min, stop_criterion="marketTime",
        flat_book=False, omit_empty_runners=False, topk=getattr(args, "topk", 0),
        max_files=0  # handled by slicing earlier
    )
    
    # Process files in parallel
    race_count = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_one_file, p, args_dict) for p in bz2_paths]
        
        for future in as_completed(futures):
            race = future.result()
            if race:
                q.put(race)
                race_count += 1
                
                if race_count % 100 == 0:
                    print(f"  📊 Processed {race_count} races...")
    
    # Clean shutdown
    q.put(None)  # Signal writer thread to stop
    wt.join()
    
    for f in writers.values():
        f.close()
        
    print(f"✅ Parallel processing complete: {race_count} races processed")


def _segment_and_write(in_path, out_path, add_bins=False, buffer_seconds=10, min_segment_bins=10):
    """
    Process a split file: handle scratchings and optionally add bin counts.
    
    Returns:
        tuple: (total_input_races, total_output_races)
    """
    buf_ms = buffer_seconds * 1000
    total_in = 0
    total_out = 0
    
    with gzip.open(in_path, 'rt') as infile, gzip.open(out_path, 'wt') as out:
        for line in infile:
            total_in += 1
            race = json.loads(line)
            
            # Find scratchings
            scratching_events = find_scratching_events(race)
            
            if not scratching_events:
                # No scratchings - output race as-is
                if add_bins:
                    race['n_bins'] = len(race.get('bins', []))
                out.write(json.dumps(race, separators=(',', ':')) + '\n')
                total_out += 1
                continue
            
            # Handle scratchings by segmenting
            exclusion_zones = create_exclusion_zones(scratching_events, buf_ms)
            segments = create_race_segments(race, exclusion_zones, min_segment_bins)
            
            for segment in segments:
                seg_race = create_segmented_race(race, segment)
                if add_bins:
                    seg_race['n_bins'] = len(seg_race.get('bins', []))
                out.write(json.dumps(seg_race, separators=(',', ':')) + '\n')
                total_out += 1
    
    return total_in, total_out


def pass_B_finalize(args, tmp_dir, out_dir):
    """
    Pass B: Finalize each split with scratching segmentation and bin counts.
    
    Processes:
    - train_raw.jsonl.gz → races_train_split_with_bins.jsonl.gz (with n_bins)
    - test_raw.jsonl.gz → races_test_split.jsonl.gz  
    - val_raw.jsonl.gz → races_validation_split.jsonl.gz
    """
    print(f"\n{'='*60}")
    print(f"🚀 PASS B: Finalizing splits with scratching segmentation")
    print(f"{'='*60}")
    
    # Process training set (with bin counts)
    print("📈 Processing training set...")
    train_in, train_out = _segment_and_write(
        in_path=os.path.join(tmp_dir, "train_raw.jsonl.gz"),
        out_path=os.path.join(out_dir, "races_train_split_with_bins.jsonl.gz"),
        add_bins=True, 
        buffer_seconds=args.buffer_seconds, 
        min_segment_bins=args.min_segment_bins
    )
    print(f"  ✅ Training: {train_in:,} → {train_out:,} races")
    
    # Process test set
    print("🧪 Processing test set...")
    test_in, test_out = _segment_and_write(
        in_path=os.path.join(tmp_dir, "test_raw.jsonl.gz"),
        out_path=os.path.join(out_dir, "races_test_split.jsonl.gz"),
        add_bins=False,
        buffer_seconds=args.buffer_seconds, 
        min_segment_bins=args.min_segment_bins
    )
    print(f"  ✅ Test: {test_in:,} → {test_out:,} races")
    
    # Process validation set
    print("🔍 Processing validation set...")
    val_in, val_out = _segment_and_write(
        in_path=os.path.join(tmp_dir, "val_raw.jsonl.gz"),
        out_path=os.path.join(out_dir, "races_validation_split.jsonl.gz"),
        add_bins=False,
        buffer_seconds=args.buffer_seconds, 
        min_segment_bins=args.min_segment_bins
    )
    print(f"  ✅ Validation: {val_in:,} → {val_out:,} races")
    
    print(f"\n📊 Total: {train_in + test_in + val_in:,} input races → {train_out + test_out + val_out:,} output races")
    return (train_in + test_in + val_in, train_out + test_out + val_out)

def main():
    parser = argparse.ArgumentParser(
        description="Process Betfair data from .bz2 files directly to final training outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--bz2-root', required=True,
                        help='Root directory containing .bz2 stream files')
    parser.add_argument('--output-dir', default='datasets',
                        help='Directory for final output files')
    parser.add_argument('--bin-ms', type=int, default=500,
                        help='Bin size in milliseconds for race builder')
    parser.add_argument('--window-min', type=int, default=30,
                        help='Pre-off window in minutes for race builder')
    parser.add_argument('--liquidity-min', type=float, default=10000.0,
                        help='Minimum total matched threshold for race builder')
    parser.add_argument('--buffer-seconds', type=int, default=10,
                        help='Buffer time around scratchings in seconds')
    parser.add_argument('--min-segment-bins', type=int, default=10,
                        help='Minimum bins required for a valid segment')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Maximum files to process (0 = all files, useful for debugging)')
    parser.add_argument('--topk', type=int, default=0,
                        help='Keep top-k price levels per side (0=all)')
    parser.add_argument('--serial', action='store_true',
                        help='Use serial processing instead of parallel (for debugging)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.bz2_root):
        print(f"❌ Error: BZ2 root directory '{args.bz2_root}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎯 BETFAIR DATA PROCESSING PIPELINE")
    print("📦 Deterministic streaming pipeline with hash-based splits")
    print(f"📂 BZ2 root: {args.bz2_root}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"⚙️  Bin size: {args.bin_ms}ms")
    print(f"⏰ Window: {args.window_min} minutes")
    print(f"💰 Liquidity minimum: ${args.liquidity_min:,.2f}")
    print(f"📊 Book depth: {'All levels' if args.topk == 0 else f'Top {args.topk} levels'}")
    print(f"🔄 Processing: {'Serial' if args.serial else 'Parallel'}")
    print("🗂️  No persistent intermediate files will be saved")
    
    # Use temporary directory for intermediate files
    with tempfile.TemporaryDirectory(prefix="betfair_pipeline_") as temp_dir:
        print(f"🗂️  Using temporary directory: {temp_dir}")
        
        # Pass A: builder → streaming hash split
        pass_A_stream_and_split(args, temp_dir, parallel=not args.serial)
        
        # Pass B: finalize each split with scratch segmenting (and n_bins for train)
        total_in, total_out = pass_B_finalize(args, temp_dir, args.output_dir)
        
        print("🗑️  Cleaning up temporary files...")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"📁 Final output files in {args.output_dir}:")
    
    final_files = [
        ("races_train_split_with_bins.jsonl.gz", "Training set (~80%) - ready for model training"),
        ("races_test_split.jsonl.gz", "Test set (~10%) - for final evaluation"),
        ("races_validation_split.jsonl.gz", "Validation set (~10%) - for hyperparameter tuning")
    ]
    
    total_size_mb = 0
    for filename, description in final_files:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  ✅ {filename:<35} ({size_mb:6.2f} MB) - {description}")
        else:
            print(f"  ❌ {filename:<35} (missing) - {description}")
    
    print(f"\n📊 Total races processed: {total_in:,} → {total_out:,} output races")
    print(f"📊 Total dataset size: {total_size_mb:.2f} MB")
    print(f"🚀 Your data is ready for training!")
    print(f"📈 Main training file: races_train_split_with_bins.jsonl.gz")
    print(f"🔧 Deterministic split: Same marketId always goes to same set")
    print(f"🗑️  All intermediate files were automatically cleaned up")

if __name__ == '__main__':
    main()
