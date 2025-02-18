import streamlit as st
import requests
import geopandas as gpd
import pandas as pd
import folium
from folium import IFrame
from shapely.geometry import shape
from streamlit_folium import st_folium
import branca.colormap as cm
import math

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Climate Policy Maps", layout="wide")

# ------------------------------------------------------------------------------
# 1) FETCH AND PREPARE DATA (WITH CACHING)
# ------------------------------------------------------------------------------

API_KEY = "ea9637fd9f0c41f3e2e932faa99dfcd76f8041aa"  # Replace with your valid key

@st.cache_data
def fetch_state_data():
    """Fetch state-level Census data, return as DataFrame."""
    state_url = (
        f"https://api.census.gov/data/2019/pep/population"
        f"?get=NAME,POP&for=state:*&key={API_KEY}"
    )
    r_states = requests.get(state_url)
    state_data = r_states.json()
    df = pd.DataFrame(state_data[1:], columns=state_data[0])
    df["POP"] = df["POP"].astype(int)
    return df

@st.cache_data
def fetch_county_data():
    """Fetch county-level Census data, return as DataFrame."""
    county_url = (
        f"https://api.census.gov/data/2019/pep/population"
        f"?get=NAME,POP&for=county:*&key={API_KEY}"
    )
    r_counties = requests.get(county_url)
    county_data = r_counties.json()
    df = pd.DataFrame(county_data[1:], columns=county_data[0])
    df["POP"] = df["POP"].astype(int)
    # Split the NAME column into state and county
    df[['stateName', 'countyName']] = df['NAME'].str.split(',', expand=True)
    # Combine state and county to form FIPS
    df["FIPS"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    return df

@st.cache_data
def fetch_geojson(url: str):
    """Cache the geojson data from the provided URL."""
    return requests.get(url).json()

@st.cache_data
def build_states_gdf(state_df, state_abbrev_to_fips):
    """Build GeoDataFrame for US states."""
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
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")

@st.cache_data
def build_counties_gdf(county_df):
    """Build GeoDataFrame for US counties and simplify geometries for performance."""
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
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
    return gdf

# Helper dictionary: state abbreviation -> 2-digit FIPS
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

# ------------------------------------------------------------------------------
# Load census data and build GeoDataFrames
# ------------------------------------------------------------------------------
state_df = fetch_state_data()
county_df = fetch_county_data()

states_gdf = build_states_gdf(state_df, state_abbrev_to_fips)
counties_gdf = build_counties_gdf(county_df)

max_pop_state = state_df["POP"].max()
max_pop_county = county_df["POP"].max()

min_log_pop = math.log(county_df["POP"].min())
max_log_pop = math.log(max_pop_county)
blue_colors = ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"]
state_cm = cm.LinearColormap(blue_colors, vmin=0, vmax=max_pop_state, caption="Population")
county_cm = cm.LinearColormap(blue_colors, vmin=min_log_pop, vmax=max_log_pop, caption="Population (log scale)")

# ------------------------------------------------------------------------------
# 2a) LOAD AND MERGE caps_plans.csv DATA FOR STATES
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_merge_caps(_states_gdf):
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
    merged = _states_gdf.merge(grouped, on="STATE_FIPS", how="left")
    merged["n_caps"] = merged["n_caps"].fillna(0).astype(int)
    merged["plan_list"] = merged["plan_list"].apply(lambda x: x if isinstance(x, list) else [])
    return merged

states_gdf = load_and_merge_caps(states_gdf)

# ------------------------------------------------------------------------------
# 2b) LOAD AND MERGE CSV DATA FOR COUNTIES USING city_county_mapping.csv
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_merge_caps_county(_counties_gdf):
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
    _counties_gdf["STATE"] = _counties_gdf["FIPS"].str[:2].map(fips_to_abbrev)
    _counties_gdf["CountyKey"] = _counties_gdf["NAME"].apply(
        lambda x: x.upper().split(',')[0].replace(" COUNTY", "").strip()
    )
    merged_counties = _counties_gdf.merge(
        grouped, 
        left_on=["CountyKey", "STATE"], 
        right_on=["CountyKey", "StateName"], 
        how="left"
    )
    merged_counties["n_caps"] = merged_counties["n_caps"].fillna(0).astype(int)
    merged_counties["plan_list"] = merged_counties["plan_list"].apply(lambda x: x if isinstance(x, list) else [])
    return merged_counties

counties_gdf = load_and_merge_caps_county(counties_gdf)

# ------------------------------------------------------------------------------
# 2c) LOAD AND AGGREGATE CITY PLANS FOR OPTIONAL MARKERS
# ------------------------------------------------------------------------------
@st.cache_data
def load_city_mapping():
    """Load the city mapping CSV for marker locations."""
    df = pd.read_csv("city_county_mapping.csv")
    df["CityName"] = df["CityName"].str.strip().str.upper()
    df["StateName"] = df["StateName"].str.strip().str.upper()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df

@st.cache_data
def load_city_plans():
    """Load and group caps_plans data by city and state."""
    df = pd.read_csv("caps_plans.csv")
    df["City"] = df["City"].str.strip().str.upper()
    df["State"] = df["State"].str.strip().str.upper()
    df["plan_info"] = df.apply(lambda row: f"{row['Year']}, {row['Plan Type']}", axis=1)
    grouped = df.groupby(["City", "State"]).agg(plan_list=("plan_info", lambda x: list(x))).reset_index()
    return grouped

city_mapping_df = load_city_mapping()
city_plans_df = load_city_plans()

# ------------------------------------------------------------------------------
# 3) BUILD THE APP WITH TABS FOR STATE AND COUNTY MAPS
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["State Map", "County Map"])

# ----------
# Tab 1: State Map
# ----------
with tab1:
    st.subheader("State Map")
    # Create the map with no default tile layer and add OSM as base (control=False)
    m_state = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_state)
    
    # Add state boundaries in a FeatureGroup (always on)
    state_boundaries = folium.FeatureGroup(name="State Boundaries", control=False)
    tooltip_state = folium.GeoJsonTooltip(
        fields=["NAME", "POP"],
        aliases=["State:", "Population:"],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    folium.GeoJson(
        states_gdf,
        style_function=lambda x: {
            "fillColor": state_cm(x["properties"]["POP"])
            if x["properties"]["POP"] not in [None, "No data"]
            else "transparent",
            "color": "black",
            "fillOpacity": 0.4,
            "weight": 1
        },
        tooltip=tooltip_state,
        highlight_function=lambda x: {"weight": 2, "color": "blue"}
    ).add_to(state_boundaries)
    state_boundaries.add_to(m_state)
    state_cm.add_to(m_state)
    
    # ----- OPTIONAL: Add City Markers Layer to the State Map -----
    city_markers_fg = folium.FeatureGroup(name="City Markers", show=False)
    unique_cities = city_mapping_df[['CityName', 'StateName', 'Latitude', 'Longitude']].drop_duplicates()
    for idx, row in unique_cities.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]
        city = row["CityName"]
        state = row["StateName"]
        plans_row = city_plans_df[(city_plans_df["City"] == city) & (city_plans_df["State"] == state)]
        if not plans_row.empty:
            plan_list = plans_row.iloc[0]["plan_list"]
            plan_lines = "".join([f"<li>{plan}</li>" for plan in plan_list])
            popup_html = f"<b>{city}, {state}</b><br><ul>{plan_lines}</ul>"
        else:
            popup_html = f"<b>{city}, {state}</b><br>No plans found"
        popup = folium.Popup(popup_html, max_width=500)
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="darkgreen",
            fill=True,
            fill_color="darkgreen",
            fill_opacity=0.7,
            popup=popup,
            tooltip=f"{city}, {state}"
        ).add_to(city_markers_fg)
    m_state.add_child(city_markers_fg)
    
    # Add layer control (only optional overlays appear)
    folium.LayerControl(collapsed=False).add_to(m_state)
    
    cols = st.columns([1, 2])
    with cols[1]:
        st_data_state = st_folium(m_state, width=1000, height=800)
    with cols[0]:
        st.markdown("### Additional Information")
        if st_data_state.get("last_active_drawing"):
            props = st_data_state["last_active_drawing"].get("properties", {})
            state_name = props.get("NAME", "N/A")
            population = props.get("POP", "N/A")
            fips = props.get("STATE_FIPS", "N/A")
            n_caps = props.get("n_caps", 0)
            plan_list = props.get("plan_list", [])
            st.write("**State:**", state_name)
            st.write("**Population:**", population)
            st.write("**FIPS:**", fips)
            st.write("**Number of Climate Action Plans:**", n_caps)
            st.markdown("#### Cities with Climate Action Plans:")
            if plan_list:
                for plan in plan_list:
                    st.write(plan)
            else:
                st.write("None")
        else:
            st.info("Click on a state to view details.")
        user_input_state = st.text_input("Ask a Question about State:", key="state_question")
        if st.button("Submit State Query", key="state_submit"):
            st.write("This is some dummy response for your state input!")

# ----------
# Tab 2: County Map
# ----------
with tab2:
    st.subheader("County Map")
    m_county = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_county)
    
    # Add county boundaries in a FeatureGroup with control=False so they always display.
    county_boundaries = folium.FeatureGroup(name="County Boundaries", control=False)
    tooltip_county = folium.GeoJsonTooltip(
        fields=["NAME", "POP", "FIPS"],
        aliases=["County:", "Population:", "FIPS:"],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    folium.GeoJson(
        counties_gdf,
        style_function=lambda x: {
            "fillColor": county_cm(math.log(x["properties"]["POP"]))
            if x["properties"]["POP"] not in [None, "No data"]
            else "transparent",
            "color": "black",
            "fillOpacity": 0.4,
            "weight": 1
        },
        tooltip=tooltip_county,
        highlight_function=lambda x: {"weight": 2, "color": "blue"}
    ).add_to(county_boundaries)
    county_boundaries.add_to(m_county)
    
    county_cm.add_to(m_county)
    
    # ----- OPTIONAL: Add City Markers Layer to the County Map -----
    city_markers_fg_county = folium.FeatureGroup(name="City Markers", show=False)
    unique_cities = city_mapping_df[['CityName', 'StateName', 'Latitude', 'Longitude']].drop_duplicates()
    for idx, row in unique_cities.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]
        city = row["CityName"]
        state = row["StateName"]
        plans_row = city_plans_df[(city_plans_df["City"] == city) & (city_plans_df["State"] == state)]
        if not plans_row.empty:
            plan_list = plans_row.iloc[0]["plan_list"]
            plan_lines = "".join([f"<li>{plan}</li>" for plan in plan_list])
            popup_html = f"<b>{city}, {state}</b><br><ul>{plan_lines}</ul>"
        else:
            popup_html = f"<b>{city}, {state}</b><br>No plans found"
        popup = folium.Popup(popup_html, max_width=500)
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="darkgreen",
            fill=True,
            fill_color="darkgreen",
            fill_opacity=0.7,
            popup=popup,
            tooltip=f"{city}, {state}"
        ).add_to(city_markers_fg_county)
    m_county.add_child(city_markers_fg_county)
    
    # Add layer control (only optional overlays appear)
    folium.LayerControl(collapsed=False).add_to(m_county)
    
    cols = st.columns([1, 2])
    with cols[1]:
        st_data_county = st_folium(m_county, width=1000, height=800)
    with cols[0]:
        st.markdown("### Additional Information")
        if st_data_county.get("last_active_drawing"):
            props = st_data_county["last_active_drawing"].get("properties", {})
            county_name = props.get("NAME", "N/A")
            population = props.get("POP", "N/A")
            fips = props.get("FIPS", "N/A")
            st.write("**County:**", county_name)
            st.write("**Population:**", population)
            st.write("**FIPS:**", fips)
            n_caps = props.get("n_caps", 0)
            plan_list = props.get("plan_list", [])
            st.write("**Number of Climate Action Plans:**", n_caps)
            st.markdown("#### Cities with Climate Action Plans:")
            if plan_list:
                for plan in plan_list:
                    st.write(plan)
            else:
                st.write("None")
        else:
            st.info("Click on a county to view details.")
        user_input_county = st.text_input("**Ask a Question about County:**", key="county_question")
        if st.button("Submit County Query", key="county_submit"):
            st.write("This is some dummy response for your county input!")
