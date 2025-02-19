import argparse
import csv

import csv
import os

def append_to_csv(city_name, state_name, county_name, city_center_coordinates):
    file_path = './city_county_mapping.csv'
    
    # Split the city center coordinates into latitude and longitude
    latitude, longitude = city_center_coordinates.split(',')
    
    # Check if file exists
    if not os.path.exists(file_path):
        # If file does not exist, create it with headers
        with open(file_path, "w", newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["City", "State", "County", "Latitude", "Longitude"])

    # Read existing data to check for duplicates
    with open(file_path, "r", newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)  # Skip header
        for row in csv_reader:
            if len(row) >= 2 and row[0].strip().lower() == city_name.lower() and row[1].strip().lower() == state_name.lower():
                print(f"Entry for {city_name}, {state_name} already exists. Skipping append.")
                return  # Exit function if duplicate found

    # Open the CSV file in append mode and add the new row
    with open(file_path, "a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([city_name, state_name, county_name, latitude.strip(), longitude.strip()])
        print(f"Added {city_name}, {state_name} to the CSV file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add city to city_county_mapping.csv")
    parser.add_argument("city", type=str, help="Name of the city")
    parser.add_argument("state", type=str, help="Name of the state")
    parser.add_argument("county", type=str, help="Name of the county")
    parser.add_argument("city_center_coordinates", type=str, help="City center coordinates")

    args = parser.parse_args()

    append_to_csv(args.city, args.state, args.county, args.city_center_coordinates)