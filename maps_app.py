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
from shapely.wkt import loads
import ast

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
    df = pd.read_csv("./maps_helpers/state_df.csv")
    return df

@st.cache_data
def fetch_county_data():
    df = pd.read_csv("./maps_helpers/county_df.csv")
    return df

# ------------------------------------------------------------------------------
# Load census data and build GeoDataFrames
# ------------------------------------------------------------------------------
state_df = fetch_state_data()
county_df = fetch_county_data()

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
def load_and_merge_caps():
    merged = pd.read_csv("./maps_helpers/states_gdf_caps.csv")
    # Convert string representation of lists back to actual lists
    if "plan_list" in merged.columns:
        merged["plan_list"] = merged["plan_list"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return merged

states_gdf = load_and_merge_caps()
states_gdf['geometry'] = states_gdf['geometry'].apply(loads)
states_gdf = gpd.GeoDataFrame(states_gdf, geometry='geometry', crs="EPSG:4326")

# ------------------------------------------------------------------------------
# 2b) LOAD AND MERGE CSV DATA FOR COUNTIES USING city_county_mapping.csv
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_merge_caps_county():
    merged_counties = pd.read_csv("./maps_helpers/counties_gdf_caps.csv")
    # Convert string representation of lists back to actual lists
    if "plan_list" in merged_counties.columns:
        merged_counties["plan_list"] = merged_counties["plan_list"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return merged_counties

counties_gdf = load_and_merge_caps_county()
counties_gdf['geometry'] = counties_gdf['geometry'].apply(loads)
counties_gdf = gpd.GeoDataFrame(counties_gdf, geometry='geometry', crs="EPSG:4326")

# ------------------------------------------------------------------------------
# Add formatted columns for tooltip display in the county GeoDataFrame
# ------------------------------------------------------------------------------
# Format population with commas, but only if the value is numeric
counties_gdf["POP_TT"] = counties_gdf["POP"].apply(
    lambda x: f"{int(float(x)):,}" if pd.notnull(x) and str(x).strip().lower() != "no data" else "No data"
)
# Ensure FIPS is a five-digit string (preserving leading zeros) if numeric
counties_gdf["FIPS_TT"] = counties_gdf["FIPS"].apply(
    lambda x: f"{int(x):05d}" if pd.notnull(x) and str(x).strip().lower() != "no data" else "No data"
)

# ------------------------------------------------------------------------------
# 2c) LOAD AND AGGREGATE CITY PLANS FOR OPTIONAL MARKERS
# ------------------------------------------------------------------------------
@st.cache_data
def load_city_mapping():
    df = pd.read_csv("./maps_helpers/city_mapping_df.csv")
    return df

@st.cache_data
def load_city_plans():
    grouped = pd.read_csv("./maps_helpers/city_plans_df.csv")
    # Convert string representation of lists back to actual lists
    if "plan_list" in grouped.columns:
        grouped["plan_list"] = grouped["plan_list"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
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
            # Format population with comma separators if available
            if population != "N/A":
                pop_str = f"{int(population):,}"
            else:
                pop_str = population
            st.write("**Population:**", pop_str)
            st.write("**FIPS:**", f"{fips}")
            st.write("**Number of Climate Action Plans:**", f"{int(n_caps):,}")
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
    # Use the formatted fields for tooltip display
    tooltip_county = folium.GeoJsonTooltip(
        fields=["NAME", "POP_TT", "FIPS_TT"],
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
            "fillColor": county_cm(math.log(float(x["properties"]["POP"])))
            if x["properties"]["POP"] not in [None, "No data"] else "transparent",
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
            # Format population with comma separators if available
            if population != "N/A":
                try:
                    pop_str = f"{int(float(population)):,}"
                except Exception:
                    pop_str = population
            else:
                pop_str = population
            st.write("**Population:**", pop_str)
            # Ensure FIPS is displayed as a plain string without comma formatting
            st.write("**FIPS:**", str(fips))
            n_caps = props.get("n_caps", 0)
            st.write("**Number of Climate Action Plans:**", f"{int(n_caps):,}")
            st.markdown("#### Cities with Climate Action Plans:")
            plan_list = props.get("plan_list", [])
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
