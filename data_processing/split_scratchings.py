#!/usr/bin/env python3
"""
Split races with scratchings into separate segments to avoid model confusion.

For each race with scratchings:
1. Identify scratching times from marketDefinition_events
2. Remove 10-second buffer around each scratching
3. Split remaining bins into continuous segments  
4. Create separate race objects for each segment (e.g., 123456.1, 123456.2)
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class ScratchingEvent:
    """Represents a scratching event with timestamp."""
    pt_ms: int
    removed_runners: List[Dict]
    
    
@dataclass 
class RaceSegment:
    """Represents a continuous segment of a race."""
    start_ms: int
    end_ms: int
    bins: List[Dict]
    segment_id: int


def find_scratching_events(race: Dict) -> List[ScratchingEvent]:
    """Extract scratching events from marketDefinition_events."""
    scratchings = []
    
    for event in race.get('marketDefinition_events', []):
        # Look for events with removed runners, regardless of event_type
        if event.get('removed'):
            scratchings.append(ScratchingEvent(
                pt_ms=event['pt_ms'],
                removed_runners=event['removed']
            ))
    
    return scratchings


def create_exclusion_zones(scratchings: List[ScratchingEvent], buffer_ms: int = 10000) -> List[Tuple[int, int]]:
    """Create time ranges to exclude (scratching_time ± buffer_ms)."""
    exclusion_zones = []
    
    for scratching in scratchings:
        start_exclude = scratching.pt_ms - buffer_ms
        end_exclude = scratching.pt_ms + buffer_ms
        exclusion_zones.append((start_exclude, end_exclude))
    
    # Merge overlapping exclusion zones
    if not exclusion_zones:
        return []
    
    exclusion_zones.sort()
    merged = [exclusion_zones[0]]
    
    for start, end in exclusion_zones[1:]:
        if start <= merged[-1][1]:  # Overlapping
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    return merged


def is_bin_excluded(bin_time: int, exclusion_zones: List[Tuple[int, int]]) -> bool:
    """Check if a bin should be excluded due to scratching buffer."""
    for start, end in exclusion_zones:
        if start <= bin_time <= end:
            return True
    return False


def create_race_segments(race: Dict, exclusion_zones: List[Tuple[int, int]], min_segment_bins: int = 10) -> List[RaceSegment]:
    """Split race bins into continuous segments, avoiding exclusion zones."""
    bins = race.get('bins', [])
    if not bins:
        return []
    
    segments = []
    current_segment_bins = []
    segment_id = 1
    
    for bin_data in bins:
        bin_time = bin_data['t_ms']
        
        if is_bin_excluded(bin_time, exclusion_zones):
            # End current segment if we have enough bins
            if len(current_segment_bins) >= min_segment_bins:
                segments.append(RaceSegment(
                    start_ms=current_segment_bins[0]['t_ms'],
                    end_ms=current_segment_bins[-1]['t_ms'], 
                    bins=current_segment_bins.copy(),
                    segment_id=segment_id
                ))
                segment_id += 1
            
            # Reset for next segment
            current_segment_bins = []
        else:
            # Add bin to current segment
            current_segment_bins.append(bin_data)
    
    # Add final segment if it has enough bins
    if len(current_segment_bins) >= min_segment_bins:
        segments.append(RaceSegment(
            start_ms=current_segment_bins[0]['t_ms'],
            end_ms=current_segment_bins[-1]['t_ms'],
            bins=current_segment_bins.copy(), 
            segment_id=segment_id
        ))
    
    return segments


def filter_market_definition_events(race: Dict, segment: RaceSegment) -> List[Dict]:
    """Filter marketDefinition_events to only include those within the segment timeframe."""
    segment_events = []
    
    for event in race.get('marketDefinition_events', []):
        event_time = event['pt_ms']
        if segment.start_ms <= event_time <= segment.end_ms:
            segment_events.append(event)
    
    return segment_events


def create_segmented_race(race: Dict, segment: RaceSegment) -> Dict:
    """Create a new race object for a segment."""
    new_race = race.copy()
    
    # Update marketId to include segment suffix
    original_market_id = race['marketId']
    new_race['marketId'] = f"{original_market_id}.{segment.segment_id}"
    
    # Update bins to only include this segment
    new_race['bins'] = segment.bins
    
    # Filter marketDefinition_events to this segment's timeframe
    new_race['marketDefinition_events'] = filter_market_definition_events(race, segment)
    
    # Add metadata about the segmentation
    new_race['_segment_info'] = {
        'original_marketId': original_market_id,
        'segment_id': segment.segment_id,
        'segment_start_ms': segment.start_ms,
        'segment_end_ms': segment.end_ms,
        'total_bins': len(segment.bins),
        'has_scratchings': True
    }
    
    return new_race


def process_races(input_file: str, output_file: str, buffer_seconds: int = 10, min_segment_bins: int = 10):
    """Process races file and split races with scratchings."""
    
    buffer_ms = buffer_seconds * 1000
    
    print(f"🔧 SCRATCHING SPLIT PARAMETERS:")
    print(f"  • Input file: {input_file}")
    print(f"  • Output file: {output_file}")
    print(f"  • Buffer around scratchings: {buffer_seconds} seconds")
    print(f"  • Minimum bins per segment: {min_segment_bins}")
    print("============================================================")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        return
    
    # Statistics
    total_races = 0
    races_with_scratchings = 0
    races_split = 0
    total_segments_created = 0
    races_kept_intact = 0
    races_discarded = 0
    
    with gzip.open(input_file, 'rt') as infile, \
         gzip.open(output_file, 'wt') as outfile:
        
        print("📂 Processing races...")
        
        for line in infile:
            total_races += 1
            race = json.loads(line)
            
            # Find scratching events
            scratchings = find_scratching_events(race)
            
            if not scratchings:
                # No scratchings - keep race as-is but add metadata
                race['_segment_info'] = {
                    'original_marketId': race['marketId'],
                    'segment_id': 1,
                    'has_scratchings': False
                }
                outfile.write(json.dumps(race) + '\n')
                races_kept_intact += 1
            else:
                races_with_scratchings += 1
                
                # Create exclusion zones around scratchings
                exclusion_zones = create_exclusion_zones(scratchings, buffer_ms)
                
                # Split into segments
                segments = create_race_segments(race, exclusion_zones, min_segment_bins)
                
                if segments:
                    races_split += 1
                    total_segments_created += len(segments)
                    
                    # Create separate race objects for each segment
                    for segment in segments:
                        segmented_race = create_segmented_race(race, segment)
                        outfile.write(json.dumps(segmented_race) + '\n')
                else:
                    # No valid segments after splitting
                    races_discarded += 1
            
            # Progress logging with detailed stats
            if total_races % 500 == 0:
                print(f"  📊 Progress: {total_races:,} processed | "
                      f"Kept: {races_kept_intact:,} | "
                      f"Split: {races_split:,} | "
                      f"Segments: {total_segments_created:,} | "
                      f"Discarded: {races_discarded:,}")
    
    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"📈 FINAL STATISTICS:")
    print(f"  • Total input races: {total_races:,}")
    print(f"  • Races with scratchings: {races_with_scratchings:,} ({races_with_scratchings/total_races*100:.1f}%)")
    print(f"  • Races kept intact: {races_kept_intact:,}")
    print(f"  • Races successfully split: {races_split:,}")
    print(f"  • Total segments created: {total_segments_created:,}")
    print(f"  • Races discarded (too short): {races_discarded:,}")
    print(f"  • Final output races: {races_kept_intact + total_segments_created:,}")
    
    # File size information
    def get_file_size_mb(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"\n📁 FILE SIZES:")
    print(f"  • Input: {get_file_size_mb(input_file):.2f} MB")
    print(f"  • Output: {get_file_size_mb(output_file):.2f} MB")
    
    print(f"\n🎉 Scratching split completed successfully!")
    print(f"📁 Output file: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split races with scratchings into separate segments")
    parser.add_argument("input_file", 
                        help="Path to the input races.jsonl.gz file")
    parser.add_argument("output_file",
                        help="Path to the output races.jsonl.gz file")
    parser.add_argument("--buffer-seconds", type=int, default=10,
                        help="Buffer time around scratchings in seconds (default: 10)")
    parser.add_argument("--min-segment-bins", type=int, default=10,
                        help="Minimum bins required for a valid segment (default: 10)")
    
    args = parser.parse_args()
    
    process_races(args.input_file, args.output_file, args.buffer_seconds, args.min_segment_bins)
