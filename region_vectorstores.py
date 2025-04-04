import os
import pandas as pd
import shutil

# === Config ===
input_folder = "Individual_All_Vectorstores"
output_folder = "Combined_By_Region_Vectorstores"
epa_csv_path = "epa_regions.csv"

# === Step 1: Load EPA region mapping ===
epa_df = pd.read_csv(epa_csv_path, skipinitialspace=True)
state_abbr_to_region = {
    str(s).strip().strip('"'): region
    for s, region in zip(epa_df['States'], epa_df['Region'])
}
print("EPA Mapping keys:", list(state_abbr_to_region.keys()))

# === Step 2: Group summary vectorstores by EPA region ===
region_to_stores = {}

for fname in os.listdir(input_folder):
    # Process only files/folders that end with "_Summary_vectorstore"
    if not fname.endswith("_Summary_vectorstore"):
        continue

    try:
        # Expected filename format: "City, ST PlanName_Summary_vectorstore"
        parts = fname.split(", ")
        if len(parts) < 2:
            print(f"⚠️ Unexpected filename format: {fname}")
            continue

        # Extract the state abbreviation (e.g., "FL" from "Tampa, FL PlanName_Summary_vectorstore")
        state_rest = parts[1]
        state_abbr = state_rest.split(" ")[0].strip()

        region = state_abbr_to_region.get(state_abbr)
        if region is None:
            print(f"⚠️ State abbreviation '{state_abbr}' not found in EPA mapping for file: {fname}")
            continue

        full_path = os.path.join(input_folder, fname)
        region_to_stores.setdefault(region, []).append(full_path)

    except Exception as e:
        print(f"❌ Failed to parse filename: {fname}, error: {e}")

# === Step 3: Copy summary vectorstores per region ===
os.makedirs(output_folder, exist_ok=True)

for region, paths in region_to_stores.items():
    region_dir = os.path.join(output_folder, f"Region_{region}")
    os.makedirs(region_dir, exist_ok=True)
    print(f"🔄 Copying {len(paths)} summary vectorstores for EPA Region {region}")

    for store_path in paths:
        dest_path = os.path.join(region_dir, os.path.basename(store_path))
        try:
            # Copy the entire vectorstore directory.
            shutil.copytree(store_path, dest_path, dirs_exist_ok=True)
            print(f"✅ Copied {store_path} to {dest_path}")
        except Exception as e:
            print(f"❌ Failed to copy {store_path} to {dest_path}, error: {e}")
