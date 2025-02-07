import os
import pandas as pd
import re

# Define the folder path containing the files
caps_folder = "CAPS"

# List all files in the CAPS folder
files = os.listdir(caps_folder)

# Define a regex pattern to extract city, state, year, and plan type
pattern = re.compile(r"^(.*?),\s([A-Z]{2})\s(.{3,}?)\s(\d{4})\.pdf$")

# Extract information from file names
data = []
for file in files:
    match = pattern.match(file)
    if match:
        city, state, plan_type, year = match.groups()
        data.append([city.strip(), state, year, plan_type.strip()])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["City", "State", "Year", "Plan Type"])

# Save to CSV
df.to_csv("caps_plans.csv", index=False)

print(f"CSV file saved to: caps_plans.csv")
