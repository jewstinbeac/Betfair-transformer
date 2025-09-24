"""
Betfair data processing package.

This package contains utilities for processing Betfair stream data:
- build_race_json: Convert .bz2 stream files to race JSON format
- split_races_streaming: Hash-based deterministic train/test/validation split
- split_scratchings: Handle races with scratched horses
- add_bin_counts: Add bin count metadata to races
"""

__version__ = "1.0.0"
