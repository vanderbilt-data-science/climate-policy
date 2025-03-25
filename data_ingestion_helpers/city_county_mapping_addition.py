import argparse
import csv

import csv
import os

import argparse
import csv
import os

def append_to_csv(city_name, state_name, county_name, city_center_coordinates):
    file_path = './city_county_mapping.csv'
    
    # Split coordinates
    latitude, longitude = city_center_coordinates.split(',')

    # If file doesn't exist, create with headers
    if not os.path.exists(file_path):
        with open(file_path, "w", newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["City", "State", "County", "Latitude", "Longitude"])

    # Read existing data to check for duplicates
    existing_entries = set()
    with open(file_path, "r", newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)  # Skip header
        for row in csv_reader:
            if len(row) >= 3:
                key = (row[0].strip().lower(), row[1].strip().lower(), row[2].strip().lower())
                existing_entries.add(key)

    # Split counties and write one row per county (NEW BEHAVIOR)
    for county in county_name.split(','):
        county_clean = county.strip()
        entry_key = (city_name.strip().lower(), state_name.strip().lower(), county_clean.lower())

        if entry_key in existing_entries:
            print(f"Entry for {city_name}, {state_name}, {county_clean} already exists. Skipping append.")
        else:
            with open(file_path, "a", newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    city_name.strip(),
                    state_name.strip(),
                    county_clean,
                    latitude.strip(),
                    longitude.strip()
                ])
                print(f"Added {city_name}, {state_name}, {county_clean} to the CSV file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add city to city_county_mapping.csv")
    parser.add_argument("city", type=str, help="Name of the city")
    parser.add_argument("state", type=str, help="Name of the state")
    parser.add_argument("county", type=str, help="Name of the county (can be comma-separated)")
    parser.add_argument("city_center_coordinates", type=str, help="City center coordinates (latitude,longitude)")

    args = parser.parse_args()
    append_to_csv(args.city, args.state, args.county, args.city_center_coordinates)