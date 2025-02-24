import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
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

@st.cache_data
def load_states_gdf_caps():
    return pd.read_pickle("./maps_helpers/states_gdf_caps.pkl")

@st.cache_data
def load_counties_gdf_caps():
    return pd.read_pickle("./maps_helpers/counties_gdf_caps.pkl")

# ------------------------------------------------------------------------------
# Load EPA region color mapping (discrete colors for regions 1 to 10)
# ------------------------------------------------------------------------------
region_colors = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#984ea3",
    5: "#ff7f00",
    6: "#ffff33",
    7: "#a65628",
    8: "#f781bf",
    9: "#999999",
    10: "#66c2a5"
}

# ------------------------------------------------------------------------------
# Load other datasets for CAPS and city mapping
# ------------------------------------------------------------------------------
@st.cache_data
def load_city_mapping():
    df = pd.read_pickle("./maps_helpers/city_mapping_df.pkl")
    return df

@st.cache_data
def load_city_plans():
    df = pd.read_pickle("./maps_helpers/city_plans_df.pkl")
    return df

state_df = fetch_state_data()
county_df = fetch_county_data()
states_gdf = load_states_gdf_caps()
counties_gdf = load_counties_gdf_caps()
city_mapping_df = load_city_mapping()
city_plans_df = load_city_plans()

# ------------------------------------------------------------------------------
# Function to generate legend HTML
# ------------------------------------------------------------------------------
def generate_legend_html(region_colors):
    legend_html = """
    <div style="
         font-size:14px;
         opacity: 1;
         ">
         <b>EPA Regions</b><br>
    """
    for region, color in region_colors.items():
        legend_html += f'<i style="background:{color}; width:18px; height:18px; display:inline-block; margin-right:5px;"></i>Region {region}<br>'
    legend_html += "</div>"
    return legend_html

# ------------------------------------------------------------------------------
# 2) BUILD THE APP WITH TABS FOR STATE AND COUNTY MAPS
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["State Map", "County Map"])

# ================================
# Tab 1: State Map
# ================================
with tab1:
    st.subheader("State Map")
    # Create map with no default tiles
    m_state = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    # Add OSM tile layer with control disabled
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_state)
    
    state_boundaries = folium.FeatureGroup(name="State Boundaries", control=False)
    tooltip_state = folium.GeoJsonTooltip(
        fields=["NAME", "POP_TT", "EPA_REGION"],
        aliases=["State:", "Population:", "EPA Region:"],
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
            "fillColor": region_colors.get(x["properties"].get("EPA_REGION"), "transparent"),
            "color": "black",
            "fillOpacity": 0.4,
            "weight": 1
        },
        tooltip=tooltip_state,
        highlight_function=lambda x: {"weight": 2, "color": "blue"}
    ).add_to(state_boundaries)
    state_boundaries.add_to(m_state)
    
    # Add city markers to the state map
    city_markers_fg = folium.FeatureGroup(name="City Markers", show=False)
    for _, row in city_mapping_df[['CityName', 'StateName', 'Latitude', 'Longitude']].drop_duplicates().iterrows():
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
    
    # Create three columns: left (Additional Info), middle (Map), right (Legend - skinny)
    cols_state = st.columns([3, 6, 1])
    
    with cols_state[1]:
        st_data_state = st_folium(m_state, width=900, height=650)
    
    with cols_state[0]:
        st.markdown("### Additional Information")
        if st_data_state.get("last_active_drawing"):
            props = st_data_state["last_active_drawing"].get("properties", {})
            state_name = props.get("NAME", "N/A")
            population = props.get("POP_TT", "N/A")
            fips = props.get("STATE_FIPS", "N/A")
            n_caps = props.get("n_caps", 0)
            plan_list = props.get("plan_list", [])
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
                st.write("**Coastal Flooding Mid-Century Projected Risk:**", props.get("CFLD_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Coastal Flooding Late-Century Projected Risk:**", props.get("CFLD_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Coastal Flooding Mid-Century Hazard Multiplier:**", props.get("CFLD_MID_HIGHER_HM", "N/A"))
                st.write("**Coastal Flooding Late-Century Hazard Multiplier:**", props.get("CFLD_LATE_HIGHER_HM", "N/A"))
                st.write("**Wildfire Mid-Century Projected Risk:**", props.get("WFIR_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Wildfire Late-Century Projected Risk:**", props.get("WFIR_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Wildfire Mid-Century Hazard Multiplier:**", props.get("WFIR_MID_HIGHER_HM", "N/A"))
                st.write("**Wildfire Late-Century Hazard Multiplier:**", props.get("WFIR_LATE_HIGHER_HM", "N/A"))
                st.write("**Drought Mid-Century Projected Risk:**", props.get("DRGT_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Drought Late-Century Projected Risk:**", props.get("DRGT_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Drought Mid-Century Hazard Multiplier:**", props.get("DRGT_MID_HIGHER_HM", "N/A"))
                st.write("**Drought Late-Century Hazard Multiplier:**", props.get("DRGT_LATE_HIGHER_HM", "N/A"))
                st.write("**Hurricane Mid-Century Projected Risk:**", props.get("HRCN_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Hurricane Late-Century Projected Risk:**", props.get("HRCN_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Hurricane Mid-Century Hazard Multiplier:**", props.get("HRCN_MID_HIGHER_HM", "N/A"))
                st.write("**Hurricane Late-Century Hazard Multiplier:**", props.get("HRCN_LATE_HIGHER_HM", "N/A"))
        else:
            st.info("Click on a state to view details.")
        user_input_state = st.text_input("Ask a Question about State:", key="state_question")
        if st.button("Submit State Query", key="state_submit"):
            st.write("This is some dummy response for your state input!")
    
    with cols_state[2]:
        legend_html = generate_legend_html(region_colors)
        st.markdown(legend_html, unsafe_allow_html=True)

# ================================
# Tab 2: County Map
# ================================
with tab2:
    st.subheader("County Map")
    m_county = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_county)
    
    county_boundaries = folium.FeatureGroup(name="County Boundaries", control=False)
    tooltip_county = folium.GeoJsonTooltip(
        fields=["NAME", "POP_TT", "FIPS_TT", "EPA_REGION"],
        aliases=["County:", "Population:", "FIPS:", "EPA Region:"],
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
            "fillColor": region_colors.get(x["properties"].get("EPA_REGION"), "transparent"),
            "color": "black",
            "fillOpacity": 0.4,
            "weight": 1
        },
        tooltip=tooltip_county,
        highlight_function=lambda x: {"weight": 2, "color": "blue"}
    ).add_to(county_boundaries)
    county_boundaries.add_to(m_county)
    
    # Add city markers for counties
    city_markers_fg_county = folium.FeatureGroup(name="City Markers", show=False)
    for _, row in city_mapping_df[['CityName', 'StateName', 'Latitude', 'Longitude']].drop_duplicates().iterrows():
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
    
    # For the county tab, also use three columns: left (Additional Info), middle (Map), right (Legend)
    cols_county = st.columns([3, 6, 1])
    
    with cols_county[1]:
        st_data_county = st_folium(m_county, width=900, height=650)
    
    with cols_county[0]:
        st.markdown("### Additional Information")
        if st_data_county.get("last_active_drawing"):
            props = st_data_county["last_active_drawing"].get("properties", {})
            county_name = props.get("NAME", "N/A")
            population = props.get("POP_TT", "N/A")
            fips = props.get("FIPS_TT", "N/A")
            n_caps = props.get("n_caps", 0)
            st.write("**County:**", county_name)
            st.write("**Population:**", population)
            st.write("**FIPS:**", f"{fips}")
            st.write("**Number of Climate Action Plans:**", f"{int(n_caps):,}")
            with st.expander("#### Cities with Climate Action Plans:"):
                plan_list = props.get("plan_list", [])
                if plan_list:
                    for plan in plan_list:
                        st.write(plan)
                else:
                    st.write("None")
            with st.expander("#### NRI Future Risk Index (Higher Warming Pathway):"):
                st.write("**Coastal Flooding Mid-Century Projected Risk:**", props.get("CFLD_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Coastal Flooding Late-Century Projected Risk:**", props.get("CFLD_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Coastal Flooding Mid-Century Hazard Multiplier:**", props.get("CFLD_MID_HIGHER_HM", "N/A"))
                st.write("**Coastal Flooding Late-Century Hazard Multiplier:**", props.get("CFLD_LATE_HIGHER_HM", "N/A"))
                st.write("**Wildfire Mid-Century Projected Risk:**", props.get("WFIR_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Wildfire Late-Century Projected Risk:**", props.get("WFIR_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Wildfire Mid-Century Hazard Multiplier:**", props.get("WFIR_MID_HIGHER_HM", "N/A"))
                st.write("**Wildfire Late-Century Hazard Multiplier:**", props.get("WFIR_LATE_HIGHER_HM", "N/A"))
                st.write("**Drought Mid-Century Projected Risk:**", props.get("DRGT_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Drought Late-Century Projected Risk:**", props.get("DRGT_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Drought Mid-Century Hazard Multiplier:**", props.get("DRGT_MID_HIGHER_HM", "N/A"))
                st.write("**Drought Late-Century Hazard Multiplier:**", props.get("DRGT_LATE_HIGHER_HM", "N/A"))
                st.write("**Hurricane Mid-Century Projected Risk:**", props.get("HRCN_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Hurricane Late-Century Projected Risk:**", props.get("HRCN_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Hurricane Mid-Century Hazard Multiplier:**", props.get("HRCN_MID_HIGHER_HM", "N/A"))
                st.write("**Hurricane Late-Century Hazard Multiplier:**", props.get("HRCN_LATE_HIGHER_HM", "N/A"))
        else:
            st.info("Click on a county to view details.")
        user_input_county = st.text_input("**Ask a Question about County:**", key="county_question")
        if st.button("Submit County Query", key="county_submit"):
            st.write("This is some dummy response for your county input!")
    
    with cols_county[2]:
        legend_html = generate_legend_html(region_colors)
        st.markdown(legend_html, unsafe_allow_html=True)
