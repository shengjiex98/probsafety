import os
import json
import pandas as pd
import re

# Path to your directory containing the 100 JSON files
data_dir = '/Users/jerry/Downloads/output-O2-samer'

# Initialize a list to store each file's data
data = []

# Regex pattern to capture the number immediately before the .json extension
pattern = re.compile(r'-(\d+)\.json$')

# Process each JSON file
for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(data_dir, filename)
        match = pattern.search(filename)
        if match:
            file_id = int(match.group(1))
            with open(filepath, 'r') as f:
                content = json.load(f)
                # Extract the compiler flags
                flags = content.get('opts', [])
                data.append({'file_id': file_id, 'flags': set(flags)})

# Gather all unique flags across all files
all_flags = set(flag for entry in data for flag in entry['flags'])

# Initialize a DataFrame with 'file_id' and each flag as a column
df = pd.DataFrame(columns=['file_id'] + sorted(all_flags))

# Prepare the list of dictionaries with 'file_id' and flags as keys
rows = []
for entry in data:
    row = {'file_id': entry['file_id']}
    # Set flag presence (1 or 0) for each flag
    for flag in all_flags:
        row[flag] = 1 if flag in entry['flags'] else 0
    rows.append(row)

# Create DataFrame from the list of dictionaries
df = pd.DataFrame(rows)

# Display the result
print(df)

# Save to the `compiler_flags` subdirectory
csv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'compiler_flags', 
    'compiler_flags_summary.csv'
)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path, index=False)
