import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import math

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Climate Policy Maps", layout="wide")

# ------------------------------------------------------------------------------
# 1) FETCH AND PREPARE DATA (WITH CACHING)
# ------------------------------------------------------------------------------
@st.cache_data
def fetch_state_data():
    """Load state-level data from the pickle file."""
    df = pd.read_pickle("./maps_helpers/state_df.pkl")
    return df

@st.cache_data
def fetch_county_data():
    """Load county-level data from the pickle file."""
    df = pd.read_pickle("./maps_helpers/county_df.pkl")
    return df

# ------------------------------------------------------------------------------
# Load Census Data and Setup Color Scales
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
# 2a) LOAD AND MERGE CAPS PLANS DATA FOR STATES
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_merge_caps():
    """Load state-level climate action plan data from pickle."""
    merged = pd.read_pickle("./maps_helpers/states_gdf_caps.pkl")
    return merged

states_gdf = load_and_merge_caps()
states_gdf = gpd.GeoDataFrame(states_gdf, geometry="geometry", crs="EPSG:4326")

# ------------------------------------------------------------------------------
# 2b) LOAD AND MERGE CAPS DATA FOR COUNTIES
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_merge_caps_county():
    """Load county-level climate action plan data from pickle."""
    merged_counties = pd.read_pickle("./maps_helpers/counties_gdf_caps.pkl")
    return merged_counties

counties_gdf = load_and_merge_caps_county()
counties_gdf = gpd.GeoDataFrame(counties_gdf, geometry="geometry", crs="EPSG:4326")

# Note: The precomputed display columns (e.g. POP_TT and FIPS_TT) are already in the pickled file.

# ------------------------------------------------------------------------------
# 2c) LOAD CITY MAPPING AND PLANS DATA
# ------------------------------------------------------------------------------
@st.cache_data
def load_city_mapping():
    """Load city mapping data from the pickle file."""
    df = pd.read_pickle("./maps_helpers/city_mapping_df.pkl")
    return df

@st.cache_data
def load_city_plans():
    """Load city-level climate action plan data from the pickle file."""
    grouped = pd.read_pickle("./maps_helpers/city_plans_df.pkl")
    return grouped

city_mapping_df = load_city_mapping()
city_plans_df = load_city_plans()

# ------------------------------------------------------------------------------
# 3) BUILD THE APP WITH TABS FOR STATE AND COUNTY MAPS
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["State Map", "County Map"])

# ================================
# Tab 1: State Map
# ================================
with tab1:
    st.subheader("State Map")
    m_state = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_state)
    
    state_boundaries = folium.FeatureGroup(name="State Boundaries", control=False)
    
    tooltip_state = folium.GeoJsonTooltip(
        fields=["NAME", "POP_TT"],
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
            "fillColor": state_cm(x["properties"]["POP"]) if x["properties"]["POP"] not in [None, "No data"] else "transparent",
            "color": "black",
            "fillOpacity": 0.4,
            "weight": 1
        },
        tooltip=tooltip_state,
        highlight_function=lambda x: {"weight": 2, "color": "blue"}
    ).add_to(state_boundaries)
    
    state_boundaries.add_to(m_state)
    state_cm.add_to(m_state)
    
    # Optional: Add city markers to the state map
    city_markers_fg = folium.FeatureGroup(name="City Markers", show=False)
    unique_cities = city_mapping_df[['CityName', 'StateName', 'Latitude', 'Longitude']].drop_duplicates()
    for _, row in unique_cities.iterrows():
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
    folium.LayerControl(collapsed=False).add_to(m_state)
    
    cols = st.columns([1, 2])
    with cols[1]:
        st_data_state = st_folium(m_state, width=1000, height=800)
    with cols[0]:
        st.markdown("### Additional Information")
        if st_data_state.get("last_active_drawing"):
            props = st_data_state["last_active_drawing"].get("properties", {})
            state_name = props.get("NAME", "N/A")
            population = props.get("POP_TT", "N/A")
            fips = props.get("STATE_FIPS", "N/A")
            n_caps = props.get("n_caps", 0)
            plan_list = props.get("plan_list", [])
            # Risk index details
            cfld_mid_higher_prisks = props.get("CFLD_MID_HIGHER_PRISKS", "N/A")
            cfld_late_higher_prisks = props.get("CFLD_LATE_HIGHER_PRISKS", "N/A")
            cfld_mid_higher_hm = props.get("CFLD_MID_HIGHER_HM", "N/A")
            cfld_late_higher_hm = props.get("CFLD_LATE_HIGHER_HM", "N/A")
            wfir_mid_higher_prisks = props.get("WFIR_MID_HIGHER_PRISKS", "N/A")
            wfir_late_higher_prisks = props.get("WFIR_LATE_HIGHER_PRISKS", "N/A")
            wfir_mid_higher_hm = props.get("WFIR_MID_HIGHER_HM", "N/A")
            wfir_late_higher_hm = props.get("WFIR_LATE_HIGHER_HM", "N/A")
            drgt_mid_higher_prisks = props.get("DRGT_MID_HIGHER_PRISKS", "N/A")
            drgt_late_higher_prisks = props.get("DRGT_LATE_HIGHER_PRISKS", "N/A")
            drgt_mid_higher_hm = props.get("DRGT_MID_HIGHER_HM", "N/A")
            drgt_late_higher_hm = props.get("DRGT_LATE_HIGHER_HM", "N/A")
            hrcn_mid_higher_prisks = props.get("HRCN_MID_HIGHER_PRISKS", "N/A")
            hrcn_late_higher_prisks = props.get("HRCN_LATE_HIGHER_PRISKS", "N/A")
            hrcn_mid_higher_hm = props.get("HRCN_MID_HIGHER_HM", "N/A")
            hrcn_late_higher_hm = props.get("HRCN_LATE_HIGHER_HM", "N/A")
            
            st.write("**State:**", state_name)
            st.write("**Population:**", population)
            st.write("**FIPS:**", f"{fips}")
            st.write("**Number of Climate Action Plans:**", f"{int(n_caps):,}")
            
            with st.expander("Cities with Climate Action Plans:"):
                if plan_list:
                    for plan in plan_list:
                        st.write(plan)
                else:
                    st.write("None")
            
            with st.expander("NRI Future Risk Index (Higher Warming Pathway):"):
                st.write("**Coastal Flooding Mid-Century Projected Risk:**", cfld_mid_higher_prisks)
                st.write("**Coastal Flooding Late-Century Projected Risk:**", cfld_late_higher_prisks)
                st.write("**Coastal Flooding Mid-Century Hazard Multiplier:**", cfld_mid_higher_hm)
                st.write("**Coastal Flooding Late-Century Hazard Multiplier:**", cfld_late_higher_hm)
                st.write("**Wildfire Mid-Century Projected Risk:**", wfir_mid_higher_prisks)
                st.write("**Wildfire Late-Century Projected Risk:**", wfir_late_higher_prisks)
                st.write("**Wildfire Mid-Century Hazard Multiplier:**", wfir_mid_higher_hm)
                st.write("**Wildfire Late-Century Hazard Multiplier:**", wfir_late_higher_hm)
                st.write("**Drought Mid-Century Projected Risk:**", drgt_mid_higher_prisks)
                st.write("**Drought Late-Century Projected Risk:**", drgt_late_higher_prisks)
                st.write("**Drought Mid-Century Hazard Multiplier:**", drgt_mid_higher_hm)
                st.write("**Drought Late-Century Hazard Multiplier:**", drgt_late_higher_hm)
                st.write("**Hurricane Mid-Century Projected Risk:**", hrcn_mid_higher_prisks)
                st.write("**Hurricane Late-Century Projected Risk:**", hrcn_late_higher_prisks)
                st.write("**Hurricane Mid-Century Hazard Multiplier:**", hrcn_mid_higher_hm)
                st.write("**Hurricane Late-Century Hazard Multiplier:**", hrcn_late_higher_hm)
        else:
            st.info("Click on a state to view details.")
        
        user_input_state = st.text_input("Ask a Question about State:", key="state_question")
        if st.button("Submit State Query", key="state_submit"):
            st.write("This is some dummy response for your state input!")
            
# ================================
# Tab 2: County Map
# ================================
with tab2:
    st.subheader("County Map")
    m_county = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_county)
    
    county_boundaries = folium.FeatureGroup(name="County Boundaries", control=False)
    
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
    
    city_markers_fg_county = folium.FeatureGroup(name="City Markers", show=False)
    unique_cities = city_mapping_df[['CityName', 'StateName', 'Latitude', 'Longitude']].drop_duplicates()
    for _, row in unique_cities.iterrows():
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
    folium.LayerControl(collapsed=False).add_to(m_county)
    
    cols = st.columns([1, 2])
    with cols[1]:
        st_data_county = st_folium(m_county, width=1000, height=800)
    with cols[0]:
        st.markdown("### Additional Information")
        if st_data_county.get("last_active_drawing"):
            props = st_data_county["last_active_drawing"].get("properties", {})
            county_name = props.get("NAME", "N/A")
            population = props.get("POP_TT", "N/A")
            fips = props.get("FIPS_TT", "N/A")
            n_caps = props.get("n_caps", 0)
            cfld_mid_higher_prisks = props.get("CFLD_MID_HIGHER_PRISKS", "N/A")
            cfld_late_higher_prisks = props.get("CFLD_LATE_HIGHER_PRISKS", "N/A")
            cfld_mid_higher_hm = props.get("CFLD_MID_HIGHER_HM", "N/A")
            cfld_late_higher_hm = props.get("CFLD_LATE_HIGHER_HM", "N/A")
            wfir_mid_higher_prisks = props.get("WFIR_MID_HIGHER_PRISKS", "N/A")
            wfir_late_higher_prisks = props.get("WFIR_LATE_HIGHER_PRISKS", "N/A")
            wfir_mid_higher_hm = props.get("WFIR_MID_HIGHER_HM", "N/A")
            wfir_late_higher_hm = props.get("WFIR_LATE_HIGHER_HM", "N/A")
            drgt_mid_higher_prisks = props.get("DRGT_MID_HIGHER_PRISKS", "N/A")
            drgt_late_higher_prisks = props.get("DRGT_LATE_HIGHER_PRISKS", "N/A")
            drgt_mid_higher_hm = props.get("DRGT_MID_HIGHER_HM", "N/A")
            drgt_late_higher_hm = props.get("DRGT_LATE_HIGHER_HM", "N/A")
            hrcn_mid_higher_prisks = props.get("HRCN_MID_HIGHER_PRISKS", "N/A")
            hrcn_late_higher_prisks = props.get("HRCN_LATE_HIGHER_PRISKS", "N/A")
            hrcn_mid_higher_hm = props.get("HRCN_MID_HIGHER_HM", "N/A")
            hrcn_late_higher_hm = props.get("HRCN_LATE_HIGHER_HM", "N/A")
            
            st.write("**County:**", county_name)
            st.write("**Population:**", population)
            st.write("**FIPS:**", fips)
            st.write("**Number of Climate Action Plans:**", f"{int(n_caps):,}")
            
            with st.expander("#### Cities with Climate Action Plans:"):
                plan_list = props.get("plan_list", [])
                if plan_list:
                    for plan in plan_list:
                        st.write(plan)
                else:
                    st.write("None")
            
            with st.expander("#### NRI Future Risk Index (Higher Warming Pathway):"):
                st.write("**Coastal Flooding Mid-Century Projected Risk:**", cfld_mid_higher_prisks)
                st.write("**Coastal Flooding Late-Century Projected Risk:**", cfld_late_higher_prisks)
                st.write("**Coastal Flooding Mid-Century Hazard Multiplier:**", cfld_mid_higher_hm)
                st.write("**Coastal Flooding Late-Century Hazard Multiplier:**", cfld_late_higher_hm)
                st.write("**Wildfire Mid-Century Projected Risk:**", wfir_mid_higher_prisks)
                st.write("**Wildfire Late-Century Projected Risk:**", wfir_late_higher_prisks)
                st.write("**Wildfire Mid-Century Hazard Multiplier:**", wfir_mid_higher_hm)
                st.write("**Wildfire Late-Century Hazard Multiplier:**", wfir_late_higher_hm)
                st.write("**Drought Mid-Century Projected Risk:**", drgt_mid_higher_prisks)
                st.write("**Drought Late-Century Projected Risk:**", drgt_late_higher_prisks)
                st.write("**Drought Mid-Century Hazard Multiplier:**", drgt_mid_higher_hm)
                st.write("**Drought Late-Century Hazard Multiplier:**", drgt_late_higher_hm)
                st.write("**Hurricane Mid-Century Projected Risk:**", hrcn_mid_higher_prisks)
                st.write("**Hurricane Late-Century Projected Risk:**", hrcn_late_higher_prisks)
                st.write("**Hurricane Mid-Century Hazard Multiplier:**", hrcn_mid_higher_hm)
                st.write("**Hurricane Late-Century Hazard Multiplier:**", hrcn_late_higher_hm)
        else:
            st.info("Click on a county to view details.")
        
        user_input_county = st.text_input("**Ask a Question about County:**", key="county_question")
        if st.button("Submit County Query", key="county_submit"):
            st.write("This is some dummy response for your county input!")
