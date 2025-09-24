#!/usr/bin/env python3
"""
Split races.jsonl.gz into training, test, and validation sets.

Training: 80% (races 1-8 of every 10)
Test: 10% (race 9 of every 10) 
Validation: 10% (race 10 of every 10)
"""

import argparse
import gzip
import json
import os
from pathlib import Path


def split_races(input_file: str, output_prefix: str = "races"):
    """
    Split races file into train/test/validation sets.
    
    Args:
        input_file: Path to races.jsonl.gz
        output_prefix: Prefix for output files (default: "races")
    """
    
    # Output file paths
    train_file = f"{output_prefix}_train.jsonl.gz"
    test_file = f"{output_prefix}_test.jsonl.gz"
    val_file = f"{output_prefix}_validation.jsonl.gz"
    
    print(f"🔧 SPLITTING PARAMETERS:")
    print(f"  • Input file: {input_file}")
    print(f"  • Training file: {train_file}")
    print(f"  • Test file: {test_file}")
    print(f"  • Validation file: {val_file}")
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
        
        print("📂 Processing races...")
        
        for line in infile:
            race_count += 1
            
            # Determine which set this race belongs to
            # Position within batch of 10 (1-indexed)
            batch_position = ((race_count - 1) % 10) + 1
            
            if batch_position == 9:
                # 9th race in batch goes to test
                test_out.write(line)
                test_count += 1
            elif batch_position == 10:
                # 10th race in batch goes to validation
                val_out.write(line)
                val_count += 1
            else:
                # Races 1-8 in batch go to training
                train_out.write(line)
                train_count += 1
            
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
    
    # Verify the split pattern with first few examples
    print(f"\n🔍 SPLIT PATTERN VERIFICATION:")
    print("  Race positions -> Dataset:")
    for i in range(1, 21):  # Show first 20 races
        batch_pos = ((i - 1) % 10) + 1
        if batch_pos == 9:
            dataset = "TEST"
        elif batch_pos == 10:
            dataset = "VALIDATION"
        else:
            dataset = "TRAINING"
        print(f"    Race {i:2d} (batch pos {batch_pos:2d}) -> {dataset}")
    
    print(f"\n🎉 Split completed successfully!")
    print(f"📁 Output files:")
    print(f"  • {train_file}")
    print(f"  • {test_file}")
    print(f"  • {val_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split races.jsonl.gz into train/test/validation sets")
    parser.add_argument("input_file", 
                        help="Path to the input races.jsonl.gz file")
    parser.add_argument("--output-prefix", default="races",
                        help="Prefix for output files (default: races)")
    
    args = parser.parse_args()
    
    split_races(args.input_file, args.output_prefix)
