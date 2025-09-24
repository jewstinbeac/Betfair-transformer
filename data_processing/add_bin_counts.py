#!/usr/bin/env python3
"""
Add n_bins field to each race, counting the number of time snapshots.
"""

import argparse
import gzip
import json
import os
from pathlib import Path


def add_bin_counts(input_file: str, output_file: str):
    """
    Add n_bins field to each race in the dataset.
    
    Args:
        input_file: Path to input races.jsonl.gz
        output_file: Path to output races.jsonl.gz with n_bins added
    """
    
    print(f"🔧 BIN COUNT PARAMETERS:")
    print(f"  • Input file: {input_file}")
    print(f"  • Output file: {output_file}")
    print("============================================================")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        return
    
    # Statistics
    total_races = 0
    total_bins = 0
    min_bins = float('inf')
    max_bins = 0
    bin_counts = {}  # For distribution analysis
    
    with gzip.open(input_file, 'rt') as infile, \
         gzip.open(output_file, 'wt') as outfile:
        
        print("📂 Processing races...")
        
        for line in infile:
            total_races += 1
            race = json.loads(line)
            
            # Count bins
            bins = race.get('bins', [])
            n_bins = len(bins)
            
            # Add n_bins field to race
            race['n_bins'] = n_bins
            
            # Update statistics
            total_bins += n_bins
            min_bins = min(min_bins, n_bins)
            max_bins = max(max_bins, n_bins)
            
            # Track distribution
            bin_counts[n_bins] = bin_counts.get(n_bins, 0) + 1
            
            # Write updated race
            outfile.write(json.dumps(race) + '\n')
            
            # Progress logging
            if total_races % 1000 == 0:
                avg_bins = total_bins / total_races
                print(f"  📊 Progress: {total_races:,} races processed | "
                      f"Avg bins: {avg_bins:.1f} | "
                      f"Range: {min_bins}-{max_bins}")
    
    # Final statistics
    avg_bins = total_bins / total_races if total_races > 0 else 0
    
    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"📈 BIN COUNT STATISTICS:")
    print(f"  • Total races processed: {total_races:,}")
    print(f"  • Total bins across all races: {total_bins:,}")
    print(f"  • Average bins per race: {avg_bins:.1f}")
    print(f"  • Minimum bins: {min_bins}")
    print(f"  • Maximum bins: {max_bins}")
    
    # Show distribution of bin counts
    print(f"\n📊 BIN COUNT DISTRIBUTION (top 10):")
    sorted_bins = sorted(bin_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for n_bins, count in sorted_bins:
        percentage = (count / total_races) * 100
        print(f"  • {n_bins:3d} bins: {count:4d} races ({percentage:5.1f}%)")
    
    # Show some percentiles
    all_bin_counts = []
    for n_bins, count in bin_counts.items():
        all_bin_counts.extend([n_bins] * count)
    all_bin_counts.sort()
    
    def percentile(data, p):
        index = int(len(data) * p / 100)
        return data[min(index, len(data) - 1)]
    
    print(f"\n📊 BIN COUNT PERCENTILES:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = percentile(all_bin_counts, p)
        print(f"  • {p:2d}th percentile: {val:3d} bins")
    
    # File size information
    def get_file_size_mb(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"\n📁 FILE SIZES:")
    print(f"  • Input: {get_file_size_mb(input_file):.2f} MB")
    print(f"  • Output: {get_file_size_mb(output_file):.2f} MB")
    
    print(f"\n🎉 Bin counting completed successfully!")
    print(f"📁 Output file: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add n_bins field to races dataset")
    parser.add_argument("input_file", 
                        help="Path to the input races.jsonl.gz file")
    parser.add_argument("output_file", 
                        help="Path to the output races.jsonl.gz file with n_bins added")
    
    args = parser.parse_args()
    
    add_bin_counts(args.input_file, args.output_file)
