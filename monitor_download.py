import os
import time
import sys

TARGET_DIR = "data/cpsc2018_raw/"

def get_dir_stats(path):
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            fp = os.path.join(root, file)
            total_size += os.path.getsize(fp)
            file_count += 1
    return file_count, total_size

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

print("="*40)
print("  CPSC 2018 DOWNLOAD MONITOR")
print("="*40)
print(f"Target: {TARGET_DIR}")

try:
    # Initial stats
    last_count, last_size = get_dir_stats(TARGET_DIR)
    start_time = time.time()
    
    while True:
        time.sleep(5)
        count, size = get_dir_stats(TARGET_DIR)
        
        diff_count = count - last_count
        diff_size = size - last_size
        
        # Calculate speed (bytes/sec)
        elapsed = 5
        speed = diff_size / elapsed
        
        # Clear line and print
        sys.stdout.write("\033[K") # Clear line
        print(f"\rFiles: {count} (+{diff_count}) | Size: {format_size(size)} (+{format_size(diff_size)}) | Speed: {format_size(speed)}/s")
        sys.stdout.flush()
        
        last_count = count
        last_size = size
        
except KeyboardInterrupt:
    print("\nMonitor stopped.")
