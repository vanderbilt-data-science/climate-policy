import pandas as pd
import os

# Define file paths
input_txt_file = os.path.join(os.getcwd(), "city_county_mapping.txt")  # Load from current directory
output_csv_file = os.path.join(os.getcwd(), "city_county_mapping.csv")  # Save in current directory

# Read the text file
with open(input_txt_file, "r") as file:
    lines = file.readlines()

# Define headers
headers = ["State Name", "State Code", "City Code", "City Name", "County Code", "County Name"]

# Process the lines, removing repeated headers and handling inconsistent spacing
data = []
for line in lines:
    row = line.strip().split()
    if len(row) == len(headers) and row[0] == "State":  # Skip repeated headers
        continue
    if len(row) >= 6:  # Ensure proper alignment of columns
        state_name = " ".join(row[:-5])
        state_code, city_code, city_name = row[-5], row[-4], row[-3]
        county_code, county_name = row[-2], row[-1]
        data.append([state_name, state_code, city_code, city_name, county_code, county_name])

# Convert to DataFrame
df = pd.DataFrame(data, columns=headers)

# Save to CSV
df.to_csv(output_csv_file, index=False)

print(f"CSV file saved to: {output_csv_file}")
