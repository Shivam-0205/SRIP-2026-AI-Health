import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

def parse_timestamp(time_str):
    """Convert text timestamp to datetime object"""
    try:
        time_str = time_str.strip().replace(',', '.')
        return datetime.strptime(time_str, "%d.%m.%Y %H:%M:%S.%f")
    except:
        return None

def parse_spo2_file(filepath):
    """Parse SpO2 text file"""
    timestamps = []
    values = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
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
        lines = f.readlines()
    for line in lines:
        if ';' in line and '-' in line:
            parts = line.strip().split(';')
            if len(parts) >= 3:
                time_range = parts[0].strip()
                event_type = parts[2].strip()
                times = time_range.split('-')
                if len(times) == 2:
                    start = parse_timestamp(times[0])
                    end = parse_timestamp(times[1])
                    if start and end:
                        events.append({'start': start, 'end': end, 'type': event_type})
    return events

def create_visualization(participant_folder, output_dir="Visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    participant_name = os.path.basename(participant_folder)
    
    spo2_file = None
    events_file = None
    
    for f in os.listdir(participant_folder):
        if 'spo2' in f.lower():
            spo2_file = os.path.join(participant_folder, f)
        if 'flow' in f.lower() or 'event' in f.lower():
            events_file = os.path.join(participant_folder, f)
            
    if not spo2_file:
        print(f"SpO2 file not found in {participant_folder}")
        return

    print(f"Processing {participant_name}...")
    timestamps, values = parse_spo2_file(spo2_file)
    events = parse_events_file(events_file) if events_file else []
    
    if not timestamps:
        print(f"No data parsed for {participant_name}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.plot(timestamps, values, label='SpO2 (%)', color='blue', linewidth=0.5)
    
    for ev in events:
        ax.axvspan(ev['start'], ev['end'], color='red', alpha=0.3, label='Event' if ev==events[0] else "")
        
    ax.set_title(f'Sleep Monitoring - {participant_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('SpO2 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"{participant_name}_visualization.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, required=True)
    parser.add_argument('-out_dir', type=str, default='Visualizations')
    args = parser.parse_args()
    create_visualization(args.name, args.out_dir)
