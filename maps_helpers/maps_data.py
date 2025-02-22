import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

API_KEY = "ea9637fd9f0c41f3e2e932faa99dfcd76f8041aa"

def fetch_state_data():
    """Fetch state-level Census data and return as a DataFrame."""
    state_url = (
        f"https://api.census.gov/data/2019/pep/population"
        f"?get=NAME,POP&for=state:*&key={API_KEY}"
    )
    r_states = requests.get(state_url)
    state_data = r_states.json()
    df = pd.DataFrame(state_data[1:], columns=state_data[0])
    df["POP"] = df["POP"].astype(int)
    return df

def fetch_county_data():
    """Fetch county-level Census data and return as a DataFrame."""
    county_url = (
        f"https://api.census.gov/data/2019/pep/population"
        f"?get=NAME,POP&for=county:*&key={API_KEY}"
    )
    r_counties = requests.get(county_url)
    county_data = r_counties.json()
    df = pd.DataFrame(county_data[1:], columns=county_data[0])
    df["POP"] = df["POP"].astype(int)
    # Split the NAME column into county and state names
    df[['countyName', 'stateName']] = df['NAME'].str.split(',', expand=True)
    df["FIPS"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    return df

def fetch_geojson(url: str):
    """Fetch the GeoJSON data from the provided URL."""
    return requests.get(url).json()

def build_states_gdf(state_df, state_abbrev_to_fips):
    """Build a GeoDataFrame for US states with Census data attached."""
    state_pop_dict = state_df.set_index("state")["POP"].to_dict()
    state_name_dict = state_df.set_index("state")["NAME"].to_dict()
    url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
    geo_data = fetch_geojson(url)
    rows = []
    for feat in geo_data["features"]:
        abbrev = feat["id"]
        geom = shape(feat["geometry"])
        fips = state_abbrev_to_fips.get(abbrev)
        if fips:
            pop_val = state_pop_dict.get(fips, "No data")
            name_val = state_name_dict.get(fips, "No data")
            rows.append({
                "geometry": geom,
                "STATE_FIPS": fips,
                "NAME": name_val,
                "POP": pop_val
            })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    # Precompute a formatted population column for tooltips (e.g., with commas)
    gdf["POP_TT"] = gdf["POP"].apply(lambda x: f"{int(x):,}" if isinstance(x, int) else "No data")
    return gdf

def build_counties_gdf(county_df):
    """Build a GeoDataFrame for US counties and simplify geometries for performance."""
    county_pop_dict = county_df.set_index("FIPS")["POP"].to_dict()
    county_name_dict = county_df.set_index("FIPS")["NAME"].to_dict()
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    geo_data = fetch_geojson(url)
    rows = []
    for feat in geo_data["features"]:
        fips = feat["id"]
        geom = shape(feat["geometry"])
        pop_val = county_pop_dict.get(fips, "No data")
        name_val = county_name_dict.get(fips, "No data")
        rows.append({
            "geometry": geom,
            "FIPS": fips,
            "NAME": name_val,
            "POP": pop_val
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    # Increase the simplification tolerance to reduce geometry complexity
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.05, preserve_topology=True)
    return gdf

state_abbrev_to_fips = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56"
}

def load_and_merge_caps(states_gdf):
    """Merge state-level climate action plan data with the states GeoDataFrame and precompute display columns."""
    caps_df = pd.read_csv("caps_plans.csv")
    caps_df["State"] = caps_df["State"].str.strip().str.upper()
    caps_df["STATE_FIPS"] = caps_df["State"].map(state_abbrev_to_fips)
    caps_df["plan_info"] = caps_df.apply(
        lambda row: f"{row['City']}, {row['Year']}, {row['Plan Type']}", axis=1
    )
    grouped = caps_df.groupby("STATE_FIPS").agg(
        n_caps=("Plan Type", "count"),
        plan_list=("plan_info", lambda x: list(x))
    ).reset_index()
    merged = states_gdf.merge(grouped, on="STATE_FIPS", how="left")
    merged["n_caps"] = merged["n_caps"].fillna(0).astype(int)
    merged["plan_list"] = merged["plan_list"].apply(lambda x: x if isinstance(x, list) else [])
    # Precompute a formatted population column (if not already computed)
    if "POP_TT" not in merged.columns:
        merged["POP_TT"] = merged["POP"].apply(lambda x: f"{int(x):,}" if isinstance(x, int) else "No data")
    return merged

def load_and_merge_caps_county(counties_gdf):
    """Merge county-level climate action plan data with the counties GeoDataFrame and precompute display columns."""
    caps_df = pd.read_csv("caps_plans.csv")
    mapping_df = pd.read_csv("city_county_mapping.csv")
    # Standardize text for matching
    caps_df["State"] = caps_df["State"].str.strip().str.upper()
    mapping_df["CountyKey"] = mapping_df["CountyName"].apply(
        lambda x: x.upper().split(',')[0].replace(" COUNTY", "").strip()
    )
    merged_caps = pd.merge(
        caps_df, mapping_df, 
        left_on=["City", "State"], 
        right_on=["CityName", "StateName"], 
        how="left"
    )
    merged_caps["plan_info"] = merged_caps.apply(
        lambda row: f"{row['City']}, {row['Year']}, {row['Plan Type']}", axis=1
    )
    merged_caps["CountyKey"] = merged_caps["CountyName"].apply(
        lambda x: x.upper().split(',')[0].replace(" COUNTY", "").strip() if pd.notnull(x) else None
    )
    grouped = merged_caps.groupby(["CountyKey", "StateName"]).agg(
        n_caps=("Plan Type", "count"),
        plan_list=("plan_info", lambda x: list(x))
    ).reset_index()
    fips_to_abbrev = {v: k for k, v in state_abbrev_to_fips.items()}
    counties_gdf["STATE"] = counties_gdf["FIPS"].str[:2].map(fips_to_abbrev)
    counties_gdf["CountyKey"] = counties_gdf["NAME"].apply(
        lambda x: x.upper().split(',')[0].replace(" COUNTY", "").strip()
    )
    merged_counties = counties_gdf.merge(
        grouped, 
        left_on=["CountyKey", "STATE"], 
        right_on=["CountyKey", "StateName"], 
        how="left"
    )
    merged_counties["n_caps"] = merged_counties["n_caps"].fillna(0).astype(int)
    merged_counties["plan_list"] = merged_counties["plan_list"].apply(lambda x: x if isinstance(x, list) else [])
    # Precompute display columns for county tooltips
    merged_counties["POP_TT"] = merged_counties["POP"].apply(
        lambda x: f"{int(x):,}" if pd.notnull(x) and isinstance(x, (int, float)) else "No data"
    )
    # Updated FIPS_TT lambda: if x is a string and is all digits, pad to 5 digits.
    merged_counties["FIPS_TT"] = merged_counties["FIPS"].apply(
        lambda x: x.zfill(5) if isinstance(x, str) and x.isdigit() else "No data"
    )
    return merged_counties

def load_city_mapping():
    """Load the city mapping CSV for marker locations."""
    df = pd.read_csv("city_county_mapping.csv")
    df["CityName"] = df["CityName"].str.strip().str.upper()
    df["StateName"] = df["StateName"].str.strip().str.upper()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df

def load_city_plans():
    """Load and group climate action plan data by city and state."""
    df = pd.read_csv("caps_plans.csv")
    df["City"] = df["City"].str.strip().str.upper()
    df["State"] = df["State"].str.strip().str.upper()
    df["plan_info"] = df.apply(lambda row: f"{row['Year']}, {row['Plan Type']}", axis=1)
    grouped = df.groupby(["City", "State"]).agg(plan_list=("plan_info", lambda x: list(x))).reset_index()
    return grouped

def merge_nri_data(states_gdf_caps, counties_gdf_caps):
    """Merge the NRI Future Risk Index data with the state and county GeoDataFrames."""
    nri_df = pd.read_excel("data/NRI Future Risk Index.xlsx")
    nri_df = nri_df[["STATEABBRV", "STATE", "COUNTY", "STCOFIPS", 
                     "CFLD_MID_HIGHER_PRISKS", "CFLD_LATE_HIGHER_PRISKS",
                     "CFLD_MID_HIGHER_HM", "CFLD_LATE_HIGHER_HM",
                     "WFIR_MID_HIGHER_PRISKS", "WFIR_LATE_HIGHER_PRISKS",
                     "WFIR_MID_HIGHER_HM", "WFIR_LATE_HIGHER_HM",
                     "DRGT_MID_HIGHER_PRISKS", "DRGT_LATE_HIGHER_PRISKS",
                     "DRGT_MID_HIGHER_HM", "DRGT_LATE_HIGHER_HM",
                     "HRCN_MID_HIGHER_PRISKS", "HRCN_LATE_HIGHER_PRISKS",
                     "HRCN_MID_HIGHER_HM", "HRCN_LATE_HIGHER_HM"]].round(2)
    
    grouped_states = nri_df.groupby("STATE").agg(
        CFLD_MID_HIGHER_PRISKS=("CFLD_MID_HIGHER_PRISKS", "mean"),
        CFLD_LATE_HIGHER_PRISKS=("CFLD_LATE_HIGHER_PRISKS", "mean"),
        CFLD_MID_HIGHER_HM=("CFLD_MID_HIGHER_HM", "mean"),
        CFLD_LATE_HIGHER_HM=("CFLD_LATE_HIGHER_HM", "mean"),
        WFIR_MID_HIGHER_PRISKS=("WFIR_MID_HIGHER_PRISKS", "mean"),
        WFIR_LATE_HIGHER_PRISKS=("WFIR_LATE_HIGHER_PRISKS", "mean"),
        WFIR_MID_HIGHER_HM=("WFIR_MID_HIGHER_HM", "mean"),
        WFIR_LATE_HIGHER_HM=("WFIR_LATE_HIGHER_HM", "mean"),
        DRGT_MID_HIGHER_PRISKS=("DRGT_MID_HIGHER_PRISKS", "mean"),
        DRGT_LATE_HIGHER_PRISKS=("DRGT_LATE_HIGHER_PRISKS", "mean"),
        DRGT_MID_HIGHER_HM=("DRGT_MID_HIGHER_HM", "mean"),
        DRGT_LATE_HIGHER_HM=("DRGT_LATE_HIGHER_HM", "mean"),
        HRCN_MID_HIGHER_PRISKS=("HRCN_MID_HIGHER_PRISKS", "mean"),
        HRCN_LATE_HIGHER_PRISKS=("HRCN_LATE_HIGHER_PRISKS", "mean"),
        HRCN_MID_HIGHER_HM=("HRCN_MID_HIGHER_HM", "mean"),
        HRCN_LATE_HIGHER_HM=("HRCN_LATE_HIGHER_HM", "mean"),
    ).round(2)
    
    counties_gdf_caps['FIPS'] = pd.to_numeric(counties_gdf_caps['FIPS'], errors='coerce').fillna(0).astype(int)
    nri_df['STCOFIPS'] = pd.to_numeric(nri_df['STCOFIPS'], errors='coerce').fillna(0).astype(int)

    merged_states_gdf = states_gdf_caps.merge(grouped_states, left_on="NAME", right_on="STATE", how="left")
    merged_counties_gdf = counties_gdf_caps.merge(nri_df, left_on="FIPS", right_on="STCOFIPS", how="left")
    return merged_states_gdf, merged_counties_gdf

if __name__ == "__main__":
    # Fetch and process Census data
    state_df = fetch_state_data()
    county_df = fetch_county_data()

    states_gdf = build_states_gdf(state_df, state_abbrev_to_fips)
    counties_gdf = build_counties_gdf(county_df)

    states_gdf_caps = load_and_merge_caps(states_gdf)
    counties_gdf_caps = load_and_merge_caps_county(counties_gdf)

    states_gdf_caps, counties_gdf_caps = merge_nri_data(states_gdf_caps, counties_gdf_caps)

    city_mapping_df = load_city_mapping()
    city_plans_df = load_city_plans()

    # Save all data as binary (pickle) files for fast loading later
    state_df.to_pickle("./maps_helpers/state_df.pkl")
    county_df.to_pickle("./maps_helpers/county_df.pkl")
    states_gdf_caps.to_pickle("./maps_helpers/states_gdf_caps.pkl")
    counties_gdf_caps.to_pickle("./maps_helpers/counties_gdf_caps.pkl")
    city_mapping_df.to_pickle("./maps_helpers/city_mapping_df.pkl")
    city_plans_df.to_pickle("./maps_helpers/city_plans_df.pkl")
