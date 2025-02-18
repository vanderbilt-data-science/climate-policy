import requests
import pandas as pd

API_KEY = "ea9637fd9f0c41f3e2e932faa99dfcd76f8041aa"

def fetch_county_data():
    """Fetch county-level Census data, return as DataFrame."""
    county_url = (
        f"https://api.census.gov/data/2019/pep/population"
        f"?get=NAME&for=county:*&key={API_KEY}"
    )
    r_counties = requests.get(county_url)
    county_data = r_counties.json()
    df = pd.DataFrame(county_data[1:], columns=county_data[0])
    # Split the NAME column into state and county
    df[['stateName', 'countyName']] = df['NAME'].str.split(',', expand=True)
    # Combine state and county to form FIPS
    df["FIPS"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df.to_csv("us_counties.csv", index=False)

fetch_county_data()