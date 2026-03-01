import os
import argparse
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

def parse_timestamp(time_str):
    """Parse timestamp in DD.MM.YYYY HH:MM:SS,mmm format"""
    try:
        time_str = time_str.strip().replace(',', '.')
        # Handle both with and without milliseconds
        if '.' in time_str.split()[-1]:
            return datetime.strptime(time_str, "%d.%m.%Y %H:%M:%S.%f")
        else:
            return datetime.strptime(time_str, "%d.%m.%Y %H:%M:%S")
    except:
        return None

def parse_spo2_file(filepath):
    """Parse SpO2 text file"""
    timestamps = []
    values = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Pattern: DD.MM.YYYY HH:MM:SS,mmm; VALUE
    pattern = r'(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2},\d{3});\s*(\d+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        ts = parse_timestamp(match[0])
        val = int(match[1])
        if ts:
            timestamps.append(ts)
            values.append(val)
    return timestamps, values

def parse_events_file(filepath):
    """Parse Flow Events text file"""
    events = []
    if not os.path.exists(filepath):
        return events
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Pattern: DD.MM.YYYY HH:MM:SS,mmm-HH:MM:SS,mmm; DURATION; TYPE; STAGE
    pattern = r'(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2},\d{3})-(\d{2}:\d{2}:\d{2},\d{3});\s*(\d+);\s*([^;]+);'
    matches = re.findall(pattern, content)
    
    for match in matches:
        date_part = match[0].split()[0]  # DD.MM.YYYY
        start_time_str = f"{date_part} {match[0].split()[1]}"
        end_time_str = f"{date_part} {match[1]}"
        
        start = parse_timestamp(start_time_str)
        end = parse_timestamp(end_time_str)
        event_type = match[3].strip()
        
        if start and end:
            events.append({'start': start, 'end': end, 'type': event_type})
    return events

def create_windows(timestamps, values, events, window_size=30):
    if not timestamps:
        return []
    
    start_time = min(timestamps)
    end_time = max(timestamps)
    windows = []
    current_time = start_time
    
    while current_time + timedelta(seconds=window_size) <= end_time:
        window_end = current_time + timedelta(seconds=window_size)
        
        # Get values in this window
        window_vals = [v for t, v in zip(timestamps, values) if current_time <= t < window_end]
        
        if not window_vals:
            current_time += timedelta(seconds=window_size/2)
            continue
            
        # Calculate features
        avg_spo2 = np.mean(window_vals)
        min_spo2 = np.min(window_vals)
        max_spo2 = np.max(window_vals)
        std_spo2 = np.std(window_vals)
        
        # Determine Label - check overlap with events
        label = "Normal"
        for ev in events:
            # Calculate overlap in seconds
            overlap_start = max(current_time, ev['start'])
            overlap_end = min(window_end, ev['end'])
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            
            # If overlap > 50% of window, assign event label
            if overlap_seconds > (window_size * 0.5):
                label = ev['type']
                break
        
        windows.append({
            'avg_spo2': avg_spo2,
            'min_spo2': min_spo2,
            'max_spo2': max_spo2,
            'std_spo2': std_spo2,
            'label': label,
            'start_time': current_time,
            'end_time': window_end
        })
        
        current_time += timedelta(seconds=window_size/2)  # 50% overlap
        
    return windows

def process_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_windows = []
    
    participants = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    for p in participants:
        p_folder = os.path.join(input_dir, p)
        print(f"Processing {p}...")
        
        spo2_f = None
        events_f = None
        for f in os.listdir(p_folder):
            if 'spo2' in f.lower(): 
                spo2_f = os.path.join(p_folder, f)
            if 'flow' in f.lower() or 'event' in f.lower(): 
                events_f = os.path.join(p_folder, f)
            
        if spo2_f:
            ts, vals = parse_spo2_file(spo2_f)
            evs = parse_events_file(events_f)
            print(f"  Found {len(ts)} SpO2 readings, {len(evs)} events")
            wins = create_windows(ts, vals, evs)
            for w in wins:
                w['participant'] = p
            all_windows.extend(wins)
            
    df = pd.DataFrame(all_windows)
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    out_path = os.path.join(output_dir, 'breathing_dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}")
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', type=str, required=True)
    parser.add_argument('-out_dir', type=str, required=True)
    args = parser.parse_args()
    process_all(args.in_dir, args.out_dir)
