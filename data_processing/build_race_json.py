#!/usr/bin/env python3
"""
Betfair WIN race builder - converts stream .bz2 files to compact races.jsonl.gz format.

Input: downloads/WIN_bz2/<month>/<day>/<meetnumber>/<marketId>.bz2 (raw Betfair stream JSON)
Output: races.jsonl.gz, one line per race (per marketId), compressed

Goal: Create memory-efficient, compact JSON format with 500ms bins containing
full order book state within the 30-minute pre-off window when liquidity >= 10k.
"""

import argparse
import bz2
import gc
import gzip
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import orjson
    JSON_LOADS = orjson.loads
    JSON_DUMPS = lambda x: orjson.dumps(x, option=orjson.OPT_SORT_KEYS).decode()
except ImportError:
    JSON_LOADS = json.loads
    JSON_DUMPS = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'))


@dataclass
class RunnerState:
    """State for a single runner/selection."""
    active: bool = True
    ltp: Optional[float] = None
    back: Dict[float, float] = field(default_factory=dict)  # price -> size
    lay: Dict[float, float] = field(default_factory=dict)   # price -> size


@dataclass
class RunnerMeta:
    """Static metadata for a runner."""
    selection_id: int
    name: str
    sort_priority: int


@dataclass
class MarketDefinitionEvent:
    """Market definition change event."""
    pt_ms: int
    version: int
    status: str
    in_play: bool
    number_active_runners: int
    event_type: str  # "UPDATE", "REMOVAL", etc.
    removed: List[Dict] = field(default_factory=list)


@dataclass
class EventData:
    """Single event/message data for replay."""
    pt_ms: int
    market_def: Optional[Dict] = None
    runner_changes: List[Dict] = field(default_factory=list)
    total_matched: Optional[float] = None
    img_flag: bool = False


@dataclass
class MarketState:
    """Complete state for a single market during processing."""
    # Static metadata (captured from first MD)
    market_id: Optional[str] = None
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    venue: Optional[str] = None
    country_code: Optional[str] = None
    timezone: Optional[str] = None
    market_type: Optional[str] = None
    name: Optional[str] = None
    market_time_ms: Optional[int] = None
    bsp_market: Optional[bool] = None
    number_of_winners: Optional[int] = None
    
    # Runner metadata
    runners_meta: Dict[int, RunnerMeta] = field(default_factory=dict)
    
    # Event log for replay
    events: List[EventData] = field(default_factory=list)
    md_events: List[MarketDefinitionEvent] = field(default_factory=list)
    last_md_version: int = 0
    
    # Track runner status changes for removal detection
    prev_runner_status: Dict[int, str] = field(default_factory=dict)


class BetfairRaceBuilder:
    """Main class for building race JSON from Betfair streams."""
    
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
    def find_bz2_files(self, root_path: str) -> List[str]:
        """Recursively find all .bz2 files."""
        bz2_files = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.bz2'):
                    bz2_files.append(os.path.join(root, file))
        return sorted(bz2_files)
        
    def parse_stream(self, file_path: str) -> MarketState:
        """Parse a single .bz2 stream file and collect events for replay."""
        state = MarketState()
        
        try:
            with bz2.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if not line.strip():
                            continue
                            
                        msg = JSON_LOADS(line.strip())
                        pt = msg.get('pt', 0)
                        
                        # Process market changes
                        for mc in msg.get('mc', []):
                            market_id = mc.get('id')
                            if market_id and not state.market_id:
                                state.market_id = market_id
                                
                            # Check for img flag at mc level
                            img_flag = mc.get('img', False)
                            
                            # Collect market definition for processing
                            market_def = mc.get('marketDefinition')
                            if market_def and not state.event_name:
                                self.capture_static_metadata(state, market_def)
                                
                            # Process market definition changes
                            if market_def:
                                self.process_market_definition(state, market_def, pt)
                                
                            # Collect runner changes
                            runner_changes = mc.get('rc', [])
                            
                            # Get total matched from various sources
                            total_matched = None
                            if 'tv' in mc:
                                total_matched = float(mc['tv'])
                            elif 'tv' in msg:
                                total_matched = float(msg['tv'])
                            else:
                                total_matched = None
                                        
                            # Create event for replay
                            event = EventData(
                                pt_ms=pt,
                                market_def=market_def,
                                runner_changes=runner_changes,
                                total_matched=total_matched,
                                img_flag=img_flag
                            )
                            state.events.append(event)
                                
                    except Exception as e:
                        self.logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
            
        return state
        
    def process_market_definition(self, state: MarketState, md: Dict, pt: int):
        """Process market definition and track changes."""
        if not state.event_name:
            self.capture_static_metadata(state, md)

        version = md.get('version', 0)
        status = md.get('status', 'UNKNOWN')
        in_play = md.get('inPlay', False)
        num_active = md.get('numberOfActiveRunners', 0)

        removed_runners = []
        if 'runners' in md:
            for r in md['runners']:
                rid = r.get('id')
                if rid is None:
                    continue
                new_status = r.get('status', 'ACTIVE')
                old_status = state.prev_runner_status.get(rid, 'ACTIVE')
                if old_status != 'REMOVED' and new_status == 'REMOVED':
                    removed_runners.append({
                        'selectionId': rid,
                        'name': state.runners_meta.get(rid, RunnerMeta(rid, '', 0)).name,
                        'removalDate_ms': self._to_ms(r.get('removalDate'), pt),
                        'adjustmentFactor': r.get('adjustmentFactor', 0.0),
                    })
                state.prev_runner_status[rid] = new_status

        event_type = "INITIAL" if state.last_md_version == 0 else ("REMOVAL" if removed_runners else "UPDATE")
        state.md_events.append(MarketDefinitionEvent(
            pt_ms=pt, version=version, status=status, in_play=in_play,
            number_active_runners=num_active, event_type=event_type, removed=removed_runners
        ))
        state.last_md_version = version
        
    def capture_static_metadata(self, state: MarketState, md: Dict):
        """Capture static metadata from first market definition."""
        # Market time
        if 'marketTime' in md:
            market_time_str = md['marketTime']
            market_time = datetime.fromisoformat(market_time_str.replace('Z', '+00:00'))
            state.market_time_ms = int(market_time.timestamp() * 1000)
            
        # Event information - try both nested event and direct fields
        if 'event' in md:
            event = md['event']
            state.event_id = event.get('id')
            state.event_name = event.get('name')
            state.venue = event.get('venue')
            state.country_code = event.get('countryCode')
            state.timezone = event.get('timezone')
        else:
            # Try direct fields on MD
            state.event_id = md.get('eventId')
            state.event_name = md.get('eventName')
            state.venue = md.get('venue')
            state.country_code = md.get('countryCode')
            state.timezone = md.get('timezone')
            
        # Market information
        state.market_type = md.get('marketType')
        state.name = md.get('name')
        state.bsp_market = md.get('bspMarket', False)
        state.number_of_winners = md.get('numberOfWinners', 1)
        
        # Runners
        if 'runners' in md:
            for runner in md['runners']:
                runner_id = runner.get('id')
                if runner_id:
                    meta = RunnerMeta(
                        selection_id=runner_id,
                        name=runner.get('name', runner.get('fullName', '')),
                        sort_priority=runner.get('sortPriority', 0)
                    )
                    state.runners_meta[runner_id] = meta
                    
    def _to_ms(self, v, fallback_pt):
        """Convert various date formats to milliseconds timestamp."""
        if v is None:
            return fallback_pt
        if isinstance(v, (int, float)):
            return int(v)
        try:
            return int(datetime.fromisoformat(str(v).replace('Z', '+00:00')).timestamp() * 1000)
        except Exception:
            return fallback_pt
    
    def calculate_stop_time(self, state: MarketState) -> int:
        """Calculate when to stop collecting data based on stop criterion."""
        if self.args.stop_criterion == 'inPlay':
            # Find first inPlay=True event
            for event in state.events:
                if event.market_def and event.market_def.get('inPlay'):
                    return event.pt_ms
            # Fallback to marketTime if never went inPlay
            return state.market_time_ms
        else:
            # Default: stop at marketTime
            return state.market_time_ms
            
    def create_bins(self, state: MarketState) -> List[Dict]:
        """Create time bins with order book snapshots by replaying events."""
        if not state.market_time_ms or not state.events:
            return []
            
        # Calculate time window
        window_ms = self.args.window_min * 60 * 1000
        start_ms = max(min(e.pt_ms for e in state.events), state.market_time_ms - window_ms)
        end_ms = self.calculate_stop_time(state)
        
        # Initialize runtime state for replay
        runtime_runners = {runner_id: RunnerState() for runner_id in state.runners_meta.keys()}
        runtime_in_play = False
        runtime_suspended = False
        runtime_num_active = len(state.runners_meta)
        runtime_tv = 0.0
        runtime_trd_seen = {}  # (selection_id, price) -> last cumulative matched size at that price
        last_removal_pt = None
        
        bins = []
        event_idx = 0
        
        # Sort events by time
        sorted_events = sorted(state.events, key=lambda e: e.pt_ms)
        
        # Generate bins
        for t_ms in range(start_ms, end_ms, self.args.bin_ms):
            # Apply all events up to this bin time
            while event_idx < len(sorted_events) and sorted_events[event_idx].pt_ms <= t_ms:
                event = sorted_events[event_idx]
                
                # Apply market definition changes
                if event.market_def:
                    runtime_in_play = event.market_def.get('inPlay', runtime_in_play)
                    runtime_suspended = (event.market_def.get('status') == 'SUSPENDED')
                    runtime_num_active = event.market_def.get('numberOfActiveRunners', runtime_num_active)
                    
                    # Check for runner removals
                    if 'runners' in event.market_def:
                        for md_runner in event.market_def['runners']:
                            runner_id = md_runner.get('id')
                            if runner_id in runtime_runners:
                                if md_runner.get('status') == 'REMOVED':
                                    runtime_runners[runner_id].active = False
                                    last_removal_pt = event.pt_ms
                
                # Apply runner changes
                if event.img_flag:
                    # Image: reset all order books and trade tracking
                    for runner in runtime_runners.values():
                        runner.back.clear()
                        runner.lay.clear()
                    runtime_trd_seen.clear()  # Clear trade tracking to bound memory
                        
                for rc in event.runner_changes:
                    selection_id = rc.get('id')
                    if selection_id not in runtime_runners:
                        continue
                        
                    runner = runtime_runners[selection_id]
                    
                    # Update back prices
                    if 'atb' in rc:
                        for price_size in rc['atb']:
                            if len(price_size) >= 2:
                                price, size = float(price_size[0]), float(price_size[1])
                                if size == 0:
                                    runner.back.pop(price, None)
                                else:
                                    runner.back[price] = size
                                    
                    # Update lay prices
                    if 'atl' in rc:
                        for price_size in rc['atl']:
                            if len(price_size) >= 2:
                                price, size = float(price_size[0]), float(price_size[1])
                                if size == 0:
                                    runner.lay.pop(price, None)
                                else:
                                    runner.lay[price] = size
                                    
                    # Update LTP
                    if 'ltp' in rc:
                        runner.ltp = float(rc['ltp'])
                    
                    # --- NEW: accumulate totalMatched from trd deltas ---
                    if 'trd' in rc and rc['trd']:
                        for price_size in rc['trd']:
                            if len(price_size) < 2:
                                continue
                            price = float(price_size[0])
                            cum = float(price_size[1])  # cumulative matched at this price
                            key = (selection_id, price)
                            prev = runtime_trd_seen.get(key, 0.0)
                            delta = cum - prev
                            if delta > 1e-12:            # guard against rounding noise
                                runtime_tv += delta
                                runtime_trd_seen[key] = cum
                            else:
                                # keep memory sane if a snapshot reports a smaller cum
                                runtime_trd_seen[key] = max(prev, cum)
                
                # Clamp up if the event carried an official totalMatched
                if event.total_matched is not None:
                    runtime_tv = max(runtime_tv, event.total_matched)
                    
                event_idx += 1
                
            # Skip if liquidity too low at this time
            if runtime_tv < self.args.liquidity_min:
                continue
                
            # Create bin snapshot
            bin_obj = {
                't_ms': t_ms,
                'inPlay': runtime_in_play,
                'suspended': runtime_suspended,
                'numberActiveRunners': runtime_num_active,
                'market_total_matched': runtime_tv,
                'removal_event': (last_removal_pt == t_ms),
                'runners': {}
            }
            
            # Add runner states
            for selection_id, runner in runtime_runners.items():
                runner_data = {
                    'active': runner.active,
                    'ltp': runner.ltp
                }
                
                # Sort and clean order books
                if runner.back:
                    back_sorted = sorted(
                        [(p, s) for p, s in runner.back.items() if s > 0],
                        key=lambda x: x[0],
                        reverse=True  # Descending by price
                    )
                    if self.args.flat_book:
                        runner_data['back_flat'] = [item for pair in back_sorted for item in pair]
                    else:
                        runner_data['back'] = back_sorted
                        
                if runner.lay:
                    lay_sorted = sorted(
                        [(p, s) for p, s in runner.lay.items() if s > 0],
                        key=lambda x: x[0]  # Ascending by price
                    )
                    if self.args.flat_book:
                        runner_data['lay_flat'] = [item for pair in lay_sorted for item in pair]
                    else:
                        runner_data['lay'] = lay_sorted
                        
                # Apply topk depth limit if specified
                if hasattr(self.args, 'topk') and self.args.topk > 0:
                    if 'back' in runner_data:
                        runner_data['back'] = runner_data['back'][:self.args.topk]
                    if 'lay' in runner_data:
                        runner_data['lay'] = runner_data['lay'][:self.args.topk]
                    if 'back_flat' in runner_data:
                        # For flat format, each price-size pair takes 2 elements
                        max_elements = self.args.topk * 2
                        runner_data['back_flat'] = runner_data['back_flat'][:max_elements]
                    if 'lay_flat' in runner_data:
                        max_elements = self.args.topk * 2
                        runner_data['lay_flat'] = runner_data['lay_flat'][:max_elements]
                        
                # Skip empty runners if requested
                if (self.args.omit_empty_runners and 
                    not runner.active and 
                    runner.ltp is None and 
                    not runner.back and 
                    not runner.lay):
                    continue
                    
                bin_obj['runners'][str(selection_id)] = runner_data
                
            bins.append(bin_obj)
            
        return bins
        
    def build_race_object(self, state: MarketState, bins: List[Dict]) -> Dict:
        """Build the final race JSON object."""
        # Convert runners metadata
        runners_list = []
        for meta in state.runners_meta.values():
            runners_list.append({
                'selectionId': meta.selection_id,
                'name': meta.name,
                'sortPriority': meta.sort_priority
            })
        runners_list.sort(key=lambda x: x['sortPriority'])
        
        # Convert MD events
        md_events_list = []
        for event in state.md_events:
            event_dict = {
                'pt_ms': event.pt_ms,
                'version': event.version,
                'status': event.status,
                'inPlay': event.in_play,
                'numberActiveRunners': event.number_active_runners,
                'type': event.event_type
            }
            if event.removed:
                event_dict['removed'] = event.removed
            md_events_list.append(event_dict)
            
        # Build race object
        race_obj = {
            'marketId': state.market_id,
            'eventId': state.event_id,
            'eventName': state.event_name,
            'venue': state.venue,
            'countryCode': state.country_code,
            'timezone': state.timezone,
            'marketType': state.market_type,
            'name': state.name,
            'marketTime_ms': state.market_time_ms,
            'bspMarket': state.bsp_market,
            'numberOfWinners': state.number_of_winners,
            'runners': runners_list,
            'marketDefinition_events': md_events_list,
            'bin_ms': self.args.bin_ms,
            'bins': bins
        }
        
        return race_obj
        
    def process_file(self, file_path: str) -> bool:
        """Process a single .bz2 file and return success status."""
        try:
            # Parse the stream
            state = self.parse_stream(file_path)
            
            if not state.market_id:
                self.logger.warning(f"Skipping {file_path}: missing market ID")
                return False
                
            if not state.market_time_ms:
                self.logger.warning(f"Skipping {state.market_id}: missing market time")
                return False
                
            # Create bins
            bins = self.create_bins(state)
            
            if not bins:
                self.logger.debug(f"SKIP {state.market_id}: no bins created")
                return False
                
            # Build race object
            race_obj = self.build_race_object(state, bins)
            
            self.logger.debug(f"KEEP {state.market_id}: {len(bins)} bins")
            return race_obj
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return False
            
    def run(self):
        """Main execution method."""
        self.logger.info(f"Starting Betfair race JSON builder...")
        self.logger.info(f"Root directory: {self.args.root}")
        self.logger.info(f"Output file: {self.args.out}")
        self.logger.info(f"Bin size: {self.args.bin_ms}ms")
        self.logger.info(f"Window: {self.args.window_min} minutes")
        self.logger.info(f"Liquidity threshold: ${self.args.liquidity_min:,.2f}")
        
        # Find all .bz2 files
        bz2_files = self.find_bz2_files(self.args.root)
        
        # Limit files for debugging if requested
        if self.args.max_files > 0:
            bz2_files = bz2_files[:self.args.max_files]
            self.logger.info(f"LIMITED to first {len(bz2_files)} .bz2 files for debugging")
        else:
            self.logger.info(f"Found {len(bz2_files)} .bz2 files")
        
        if not bz2_files:
            self.logger.warning("No .bz2 files found")
            return
            
        # Ensure output directory exists
        output_dir = os.path.dirname(self.args.out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Process files one by one
        total_processed = 0
        total_written = 0
        
        with gzip.open(self.args.out, 'wt', encoding='utf-8') as out_file:
            for i, file_path in enumerate(bz2_files, 1):
                total_processed += 1
                
                race_obj = self.process_file(file_path)
                
                if race_obj:
                    # Write to JSONL
                    json_line = JSON_DUMPS(race_obj)
                    out_file.write(json_line + '\n')
                    total_written += 1
                    
                # Log progress
                if i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(bz2_files)} processed, {total_written} written")
                    
                # Force garbage collection for memory management
                if i % 1000 == 0:
                    gc.collect()
                    
        # Final summary
        self.logger.info(f"=== SUMMARY ===")
        self.logger.info(f"Total files processed: {total_processed}")
        self.logger.info(f"Total races written: {total_written}")
        self.logger.info(f"Output file: {self.args.out}")
        
    def iter_races_from_paths(self, paths):
        """Generator that yields race objects from a list of .bz2 file paths."""
        for file_path in paths:
            race_obj = self.process_file(file_path)
            if race_obj:
                yield race_obj


def list_bz2_files(root, max_files=0):
    """Find all .bz2 files in root directory, optionally limited to max_files."""
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith('.bz2'):
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths[:max_files] if max_files and max_files > 0 else paths


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Betfair WIN race builder - converts stream .bz2 files to compact races.jsonl.gz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--root', required=True,
                       help='Root directory to scan for *.bz2 files')
    parser.add_argument('--out', required=True,
                       help='Output file path (races.jsonl.gz)')
    parser.add_argument('--bin-ms', type=int, default=500,
                       help='Bin size in milliseconds')
    parser.add_argument('--window-min', type=int, default=30,
                       help='Pre-off window in minutes')
    parser.add_argument('--liquidity-min', type=float, default=10000.0,
                       help='Minimum total matched threshold')
    parser.add_argument('--stop-criterion', choices=['marketTime', 'inPlay'], default='marketTime',
                       help='When to stop collecting data')
    parser.add_argument('--flat-book', action='store_true',
                       help='Use flat book format: [p1,s1,p2,s2,...] instead of [[p1,s1],[p2,s2],...]')
    parser.add_argument('--omit-empty-runners', action='store_true',
                       help='Skip inactive runners with no LTP/book data')
    parser.add_argument('--topk', type=int, default=0,
                       help='Keep top-k price levels per side (0=all)')
    parser.add_argument('--max-files', type=int, default=0,
                       help='Maximum files to process (0 = all files, useful for debugging)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isdir(args.root):
        print(f"Error: Root directory '{args.root}' does not exist")
        sys.exit(1)
        
    if args.bin_ms <= 0:
        print("Error: --bin-ms must be positive")
        sys.exit(1)
        
    if args.window_min <= 0:
        print("Error: --window-min must be positive")
        sys.exit(1)
        
    if args.liquidity_min < 0:
        print("Error: --liquidity-min must be non-negative")
        sys.exit(1)
        
    # Run the builder
    builder = BetfairRaceBuilder(args)
    builder.run()


if __name__ == '__main__':
    main()
