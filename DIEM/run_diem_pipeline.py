import subprocess
import sys
import csv
import os

# List the scripts you want to run
scripts = ["parse_gaze.py", "build_heatmaps.py", "extract_frames.py"]

for script in scripts:
    print(f"Starting {script}...")
    
    # sys.executable ensures it uses the same Python environment
    result = subprocess.run([sys.executable, script])
    
    # Check if the script executed successfully
    if result.returncode == 0:
        print(f"{script} finished successfully.\n")
    else:
        print(f"{script} failed with exit code {result.returncode}.\n")

# Parse calibration_report.csv to find videos that don't need review
csv_path = os.path.join("DIEM_Processed", "calibration_report.csv")
videos_no_review = []
if os.path.exists(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('needs_review') == 'no':
                videos_no_review.append(row['video'])

if videos_no_review:
    videos_arg = ",".join(videos_no_review)
    inspect_cmd = ["inspect_calibration.py", "--videos", videos_arg, "--frames", "5"]
else:
    inspect_cmd = ["inspect_calibration.py", "--frames", "5"]

# Run the remaining scripts
remaining_cmds = [
    inspect_cmd,
    ["fine_tune.py"]
]

for cmd in remaining_cmds:
    script_name = cmd[0]
    print(f"Starting {' '.join(cmd)}...")
    
    result = subprocess.run([sys.executable] + cmd)
    
    if result.returncode == 0:
        print(f"{script_name} finished successfully.\n")
    else:
        print(f"{script_name} failed with exit code {result.returncode}.\n")