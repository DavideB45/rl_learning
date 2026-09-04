import pandas as pd
import os

# Ask for inputs
csv_path = input("CSV path: ")
eval_id = int(input("Eval ID to remove: "))

# Read CSV
df = pd.read_csv(csv_path)

# Remove rows with the selected eval ID
filtered_df = df[df["eval"] != eval_id]

# Create output path
base, ext = os.path.splitext(csv_path)
output_path = f"{base}_without_{eval_id}{ext}"

# Save copy
filtered_df.to_csv(output_path, index=False)

print(f"Saved filtered CSV to: {output_path}")