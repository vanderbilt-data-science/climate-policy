import argparse
import csv

def append_to_csv(city_name, state_name, county_name, city_center_coordinates):
    # Split the city center coordinates into latitude and longitude
    latitude, longitude = city_center_coordinates.split(',')

    # Open the CSV file in append mode
    with open('./city_county_mapping.csv', "a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data as a new row
        csv_writer.writerow([city_name, state_name, county_name, latitude.strip(), longitude.strip()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add city to city_county_mapping.csv")
    parser.add_argument("city", type=str, help="Name of the city")
    parser.add_argument("state", type=str, help="Name of the state")
    parser.add_argument("county", type=str, help="Name of the county")
    parser.add_argument("city_center_coordinates", type=str, help="City center coordinates")

    args = parser.parse_args()

    append_to_csv(args.city, args.state, args.county, args.city_center_coordinates)