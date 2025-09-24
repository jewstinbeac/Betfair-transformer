#!/usr/bin/env python3
"""
Streaming hash-based race splitter for deterministic train/test/validation splits.

Uses MD5 hash of marketId to deterministically assign races to splits:
- Hash % 10 == 9: test (10%)
- Hash % 10 == 0: validation (10%) 
- Hash % 10 == 1-8: training (80%)

This ensures the same marketId always goes to the same split regardless of file order.
"""

import hashlib
import gzip
import json
import argparse
import os
from typing import Dict, TextIO


def _bucket_from_market_id(market_id: str) -> str:
    """Deterministically assign market to bucket based on hash."""
    h = int(hashlib.md5(market_id.encode()).hexdigest(), 16) % 10
    if h == 9:
        return "test"
    if h == 0:
        return "val"
    return "train"


def write_split_line(race_obj: dict, writers: Dict[str, TextIO]) -> str:
    """Write race to appropriate split file based on marketId hash."""
    bucket = _bucket_from_market_id(race_obj["marketId"])
    writers[bucket].write(json.dumps(race_obj, separators=(',', ':')) + '\n')
    return bucket


def split_races_streaming(input_file: str, output_prefix: str = "races"):
    """
    Split races file into train/test/validation sets using streaming hash-based split.
    
    Args:
        input_file: Path to races.jsonl.gz
        output_prefix: Prefix for output files (default: "races")
    """
    
    # Output file paths
    train_file = f"{output_prefix}_train.jsonl.gz"
    test_file = f"{output_prefix}_test.jsonl.gz"
    val_file = f"{output_prefix}_validation.jsonl.gz"
    
    print(f"🔧 STREAMING SPLIT PARAMETERS:")
    print(f"  • Input file: {input_file}")
    print(f"  • Training file: {train_file}")
    print(f"  • Test file: {test_file}")
    print(f"  • Validation file: {val_file}")
    print(f"  • Split method: Hash-based (deterministic)")
    print("============================================================")
    
    # Check input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        return
    
    # Initialize counters
    race_count = 0
    train_count = 0
    test_count = 0
    val_count = 0
    
    # Open all output files
    with gzip.open(input_file, 'rt') as infile, \
         gzip.open(train_file, 'wt') as train_out, \
         gzip.open(test_file, 'wt') as test_out, \
         gzip.open(val_file, 'wt') as val_out:
        
        writers = {
            "train": train_out,
            "test": test_out,
            "val": val_out
        }
        
        print("📂 Processing races...")
        
        for line in infile:
            race_count += 1
            race = json.loads(line)
            
            # Determine split based on hash
            bucket = write_split_line(race, writers)
            
            # Update counters
            if bucket == "train":
                train_count += 1
            elif bucket == "test":
                test_count += 1
            elif bucket == "val":
                val_count += 1
            
            # Progress logging
            if race_count % 1000 == 0:
                print(f"  📊 Processed {race_count:,} races...")
    
    print(f"\n✅ SPLITTING COMPLETE!")
    print(f"📈 FINAL STATISTICS:")
    print(f"  • Total races processed: {race_count:,}")
    print(f"  • Training races: {train_count:,} ({train_count/race_count*100:.1f}%)")
    print(f"  • Test races: {test_count:,} ({test_count/race_count*100:.1f}%)")
    print(f"  • Validation races: {val_count:,} ({val_count/race_count*100:.1f}%)")
    
    # File size information
    def get_file_size_mb(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"\n📁 FILE SIZES:")
    print(f"  • Training: {get_file_size_mb(train_file):.2f} MB")
    print(f"  • Test: {get_file_size_mb(test_file):.2f} MB")
    print(f"  • Validation: {get_file_size_mb(val_file):.2f} MB")
    
    # Show hash distribution verification
    print(f"\n🔍 HASH DISTRIBUTION VERIFICATION:")
    print("  Sample marketIds and their assignments:")
    
    # Re-read a few lines to show the distribution
    with gzip.open(input_file, 'rt') as infile:
        for i, line in enumerate(infile):
            if i >= 10:  # Show first 10 examples
                break
            race = json.loads(line)
            market_id = race["marketId"]
            bucket = _bucket_from_market_id(market_id)
            h = int(hashlib.md5(market_id.encode()).hexdigest(), 16) % 10
            print(f"    {market_id} (hash % 10 = {h}) -> {bucket.upper()}")
    
    print(f"\n🎉 Hash-based split completed successfully!")
    print(f"📁 Output files:")
    print(f"  • {train_file}")
    print(f"  • {test_file}")
    print(f"  • {val_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split races.jsonl.gz into train/test/validation sets using hash-based deterministic split")
    parser.add_argument("input_file", 
                        help="Path to the input races.jsonl.gz file")
    parser.add_argument("--output-prefix", default="races",
                        help="Prefix for output files (default: races)")
    
    args = parser.parse_args()
    
    split_races_streaming(args.input_file, args.output_prefix)
