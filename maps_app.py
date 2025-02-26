import os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Climate Policy Maps", layout="wide")

# ------------------------------------------------------------------------------
# DATA LOADING UTILITIES
# ------------------------------------------------------------------------------
@st.cache_data
def load_pickle_data(file_path):
    """Load and cache a pickle file from the given path."""
    return pd.read_pickle(file_path)

# Load data files
state_df = load_pickle_data("./maps_helpers/state_df.pkl")
county_df = load_pickle_data("./maps_helpers/county_df.pkl")
states_gdf = load_pickle_data("./maps_helpers/states_gdf_caps.pkl")
counties_gdf = load_pickle_data("./maps_helpers/counties_gdf_caps.pkl")
city_mapping_df = load_pickle_data("./maps_helpers/city_mapping_df.pkl")
city_plans_df = load_pickle_data("./maps_helpers/city_plans_df.pkl")

# ------------------------------------------------------------------------------
# CONSTANTS & CONFIGURATIONS
# ------------------------------------------------------------------------------
REGION_COLORS = {
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
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def generate_legend_html(region_colors):
    """Generate HTML code for the EPA region legend."""
    legend_html = """
    <div style="font-size:14px; opacity: 1;">
         <b>EPA Regions</b><br>
    """
    for region, color in region_colors.items():
        legend_html += (
            f'<i style="background:{color}; width:18px; height:18px; '
            f'display:inline-block; margin-right:5px;"></i>Region {region}<br>'
        )
    legend_html += "</div>"
    return legend_html

def format_plan_name(plan, state_abbr):
    """
    Format a plan string (e.g. "Oakland, 2020, Mitigation Primary CAP")
    using the state abbreviation (e.g. "CA") to match the vector store naming convention.
    Expected output: "Oakland, CA Mitigation Primary CAP 2020"
    """
    parts = [p.strip() for p in plan.split(",")]
    if len(parts) == 3:
        city, year, title = parts
        return f"{city}, {state_abbr} {title} {year}"
    return plan

def add_city_markers(map_object):
    """
    Add city markers with plan information to the given Folium map object.
    Uses global data: city_mapping_df and city_plans_df.
    """
    city_markers = folium.FeatureGroup(name="City Markers", show=False)
    for _, row in city_mapping_df[['CityName', 'StateName', 'Latitude', 'Longitude']].drop_duplicates().iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        city, state = row["CityName"], row["StateName"]
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
        ).add_to(city_markers)
    map_object.add_child(city_markers)

def answer_question(api_key, user_input, extra_context, plan_list, state_abbr):
    """
    Answer a user's question by retrieving relevant document chunks from individual and combined vector stores.
    
    Parameters:
        api_key (str): OpenAI API key.
        user_input (str): The user question.
        extra_context (str): Extra context to include in the QA.
        plan_list (list): List of plan names.
        state_abbr (str): State abbreviation.
    
    Returns:
        str: The answer from the language model.
    """
    os.environ["OPENAI_API_KEY"] = api_key
    all_retrieved_chunks = []

    # Retrieve chunks from each individual plan vector store
    for plan in plan_list:
        formatted_plan = format_plan_name(plan, state_abbr)
        vectorstore_path = os.path.join("Individual_All_Vectorstores", formatted_plan + "_vectorstore")
        try:
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_store = FAISS.load_local(
                vectorstore_path, embedding_model, allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store for plan '{formatted_plan}': {e}")
            continue

        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        retrieved_chunks = retriever.invoke(user_input)
        all_retrieved_chunks.extend(retrieved_chunks)
    
    # Retrieve chunks from the combined vector store
    combined_vectorstore_path = "Combined_Summary_Vectorstore"
    try:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        combined_vector_store = FAISS.load_local(
            combined_vectorstore_path, embedding_model, allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading combined vector store: {e}")
    else:
        combined_retriever = combined_vector_store.as_retriever(search_kwargs={"k": 5})
        combined_retrieved_chunks = combined_retriever.invoke(user_input)
        all_retrieved_chunks.extend(combined_retrieved_chunks)

    # Append extra context as a Document
    all_retrieved_chunks.append(Document(page_content=extra_context))

    # Load system prompt for QA
    prompt_path = "Prompts/maps_qa.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")
    
    # Create prompt and chain for QA
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    result = question_answer_chain.invoke({"input": user_input, "context": all_retrieved_chunks})
    
    answer = result["answer"] if "answer" in result else result
    return answer

# ------------------------------------------------------------------------------
# APPLICATION LAYOUT: STATE & COUNTY MAPS
# ------------------------------------------------------------------------------
tab_state, tab_county = st.tabs(["State Map", "County Map"])

# ================================
# TAB 1: STATE MAP
# ================================
with tab_state:
    st.subheader("State Map")
    # Initialize state map with no default tiles and add OSM layer.
    m_state = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_state)

    # Add state boundaries with tooltips.
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
            "fillColor": REGION_COLORS.get(x["properties"].get("EPA_REGION"), "transparent"),
            "color": "black",
            "fillOpacity": 0.4,
            "weight": 1
        },
        tooltip=tooltip_state,
        highlight_function=lambda x: {"weight": 2, "color": "blue"}
    ).add_to(state_boundaries)
    state_boundaries.add_to(m_state)

    # Add city markers.
    add_city_markers(m_state)
    folium.LayerControl(collapsed=False).add_to(m_state)

    # Layout columns: left (info & QA), middle (map), right (legend)
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
            state_abbr = props.get("STATE_ABBR", "N/A")
            n_caps = props.get("n_caps", 0)
            plan_list = props.get("plan_list", [])
            
            st.write("**State:**", state_name)
            st.write("**Population:**", population)
            st.write("**FIPS:**", fips)
            st.write("**Number of Climate Action Plans:**", f"{int(n_caps):,}")
            
            with st.expander("Cities with Climate Action Plans:"):
                if plan_list:
                    for plan in plan_list:
                        st.write(plan)
                else:
                    st.write("None")
            
            with st.expander("NRI Future Risk Index (Higher Warming Pathway):"):
                st.write("**Mid-Century Coastal Flooding Risk (Percentile):**", props.get("CFLD_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Late-Century Coastal Flooding Risk (Percentile):**", props.get("CFLD_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Mid-Century Coastal Flooding Hazard Multiplier:**", props.get("CFLD_MID_HIGHER_HM", "N/A"))
                st.write("**Late-Century Coastal Flooding Hazard Multiplier:**", props.get("CFLD_LATE_HIGHER_HM", "N/A"))
                st.write("**Mid-Century Wildfire Risk (Percentile):**", props.get("WFIR_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Late-Century Wildfire Risk (Percentile):**", props.get("WFIR_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Mid-Century Wildfire Hazard Multiplier:**", props.get("WFIR_MID_HIGHER_HM", "N/A"))
                st.write("**Late-Century Wildfire Hazard Multiplier:**", props.get("WFIR_LATE_HIGHER_HM", "N/A"))
                st.write("**Mid-Century Drought Risk (Percentile):**", props.get("DRGT_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Late-Century Drought Risk (Percentile):**", props.get("DRGT_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Mid-Century Drought Hazard Multiplier:**", props.get("DRGT_MID_HIGHER_HM", "N/A"))
                st.write("**Late-Century Drought Hazard Multiplier:**", props.get("DRGT_LATE_HIGHER_HM", "N/A"))
                st.write("**Mid-Century Hurricane Risk (Percentile):**", props.get("HRCN_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Late-Century Hurricane Risk (Percentile):**", props.get("HRCN_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Mid-Century Hurricane Hazard Multiplier:**", props.get("HRCN_MID_HIGHER_HM", "N/A"))
                st.write("**Late-Century Hurricane Hazard Multiplier:**", props.get("HRCN_LATE_HIGHER_HM", "N/A"))
                st.write("**Mid-Century Extreme Heat Risk (Percentile):**", props.get("EXHT_L95_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Late-Century Extreme Heat Risk (Percentile):**", props.get("EXHT_L95_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Mid-Century Extreme Heat Hazard Multiplier:**", props.get("EXHT_L95_MID_HIGHER_HM", "N/A"))
                st.write("**Late-Century Extreme Heat Hazard Multiplier:**", props.get("EXHT_L95_LATE_HIGHER_HM", "N/A"))

            with st.expander("FEMA Risk Profile:"):
                st.write("**Disaster Risk (Percentile):**", props.get("RISK_SCORE", "N/A"))
                st.write("**Disaster Loss ($/year):**", props.get("EAL_VALT", "N/A"))
                st.write("**Social Vulnerability (Percentile):**", props.get("SOVI_SCORE", "N/A"))
                st.write("**Community Resilience (Percentile):**", props.get("RESL_SCORE", "N/A"))
                st.write("**Annual Avalanche Loss ($/year):**", props.get("AVLN_EALT", "N/A"))
                st.write("**Annual Avalanche Loss (Percentile):**", props.get("AVLN_EALS", "N/A"))
                st.write("**Annual Coastal Flooding Loss ($/year):**", props.get("CFLD_EALT", "N/A"))
                st.write("**Annual Coastal Flooding Loss (Percentile):**", props.get("CFLD_EALS", "N/A"))
                st.write("**Annual Cold Wave Loss ($/year):**", props.get("CWAV_EALT", "N/A"))
                st.write("**Annual Cold Wave Loss (Percentile):**", props.get("CWAV_EALS", "N/A"))
                st.write("**Annual Drought Loss ($/year):**", props.get("DRGT_EALT", "N/A"))
                st.write("**Annual Drought Loss (Percentile):**", props.get("DRGT_EALS", "N/A"))
                st.write("**Annual Hail Loss ($/year):**", props.get("HAIL_EALT", "N/A"))
                st.write("**Annual Hail Loss (Percentile):**", props.get("HAIL_EALS", "N/A"))
                st.write("**Annual Heat Wave Loss ($/year):**", props.get("HWAV_EALT", "N/A"))
                st.write("**Annual Heat Wave Loss (Percentile):**", props.get("HWAV_EALS", "N/A"))
                st.write("**Annual Hurricane Loss ($/year):**", props.get("HRCN_EALT", "N/A"))
                st.write("**Annual Hurricane Loss (Percentile):**", props.get("HRCN_EALS", "N/A"))
                st.write("**Annual Ice Storm Loss ($/year):**", props.get("ISTM_EALT", "N/A"))
                st.write("**Annual Ice Storm Loss (Percentile):**", props.get("ISTM_EALS", "N/A"))
                st.write("**Annual Landslide Loss ($/year):**", props.get("LNDS_EALT", "N/A"))
                st.write("**Annual Landslide Loss (Percentile):**", props.get("LNDS_EALS", "N/A"))
                st.write("**Annual River Flooding Loss ($/year):**", props.get("RFLD_EALT", "N/A"))
                st.write("**Annual River Flooding Loss (Percentile):**", props.get("RFLD_EALS", "N/A"))
                st.write("**Annual Wind Loss ($/year):**", props.get("SWND_EALT", "N/A"))
                st.write("**Annual Wind Loss (Percentile):**", props.get("SWND_EALS", "N/A"))
                st.write("**Annual Tornado Loss ($/year):**", props.get("TRND_EALT", "N/A"))
                st.write("**Annual Tornado Loss (Percentile):**", props.get("TRND_EALS", "N/A"))
                st.write("**Annual Winter Weather Loss ($/year):**", props.get("WNTW_EALT", "N/A"))
                st.write("**Annual Winter Weather Loss (Percentile):**", props.get("WNTW_EALS", "N/A"))

            with st.expander("CEJST Data:"):
                st.write("**Share of properties at risk of flood in 30 years (percentile):**", props.get("Share of properties at risk of flood in 30 years (percentile)", "N/A"))
                st.write("**Share of properties at risk of flood in 30 years:**", props.get("Share of properties at risk of flood in 30 years", "N/A"))
                st.write("**Share of properties at risk of fire in 30 years (percentile):**", props.get("Share of properties at risk of fire in 30 years (percentile)", "N/A"))
                st.write("**Share of properties at risk of fire in 30 years:**", props.get("Share of properties at risk of fire in 30 years", "N/A"))
                st.write("**Energy burden (percentile):**", props.get("Energy burden (percentile)", "N/A"))
                st.write("**PM2.5 (percentile):**", props.get("PM2.5 in the air (percentile)", "N/A"))
                st.write("**PM2.5 (Volume):**", props.get("PM2.5 in the air", "N/A"))
                st.write("**Impervious surface or cropland:**", props.get("Share of the tract's land area that is covered by impervious surface or cropland as a percent", "N/A"))
                st.write("**Asthma Prevalence (Percentile):**", props.get("Current asthma among adults aged greater than or equal to 18 years", "N/A"))
                       
            # Build extra context for the QA chain
            extra_context = (
                f"State: {state_name}\n"
                f"Population: {population}\n"
                f"FIPS: {fips}\n"
                f"Climate Action Plans: {', '.join(plan_list) if plan_list else 'No climate action plans'}\n"
                f"NRI Future Risk Index (Higher Warming Pathway):\n"
                f"Mid-Century Coastal Flooding Risk (Percentile): {props.get('CFLD_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Late-Century Coastal Flooding Risk (Percentile): {props.get('CFLD_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Mid-Century Coastal Flooding Hazard Multiplier: {props.get('CFLD_MID_HIGHER_HM', 'N/A')}\n"
                f"Late-Century Coastal Flooding Hazard Multiplier: {props.get('CFLD_LATE_HIGHER_HM', 'N/A')}\n"
                f"Mid-Century Wildfire Risk (Percentile): {props.get('WFIR_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Late-Century Wildfire Risk (Percentile): {props.get('WFIR_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Mid-Century Wildfire Hazard Multiplier: {props.get('WFIR_MID_HIGHER_HM', 'N/A')}\n"
                f"Late-Century Wildfire Hazard Multiplier: {props.get('WFIR_LATE_HIGHER_HM', 'N/A')}\n"
                f"Mid-Century Drought Risk (Percentile): {props.get('DRGT_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Late-Century Drought Risk (Percentile): {props.get('DRGT_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Mid-Century Drought Hazard Multiplier: {props.get('DRGT_MID_HIGHER_HM', 'N/A')}\n"
                f"Late-Century Drought Hazard Multiplier: {props.get('DRGT_LATE_HIGHER_HM', 'N/A')}\n"
                f"Mid-Century Hurricane Risk (Percentile): {props.get('HRCN_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Late-Century Hurricane Risk (Percentile): {props.get('HRCN_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Mid-Century Hurricane Hazard Multiplier: {props.get('HRCN_MID_HIGHER_HM', 'N/A')}\n"
                f"Late-Century Hurricane Hazard Multiplier: {props.get('HRCN_LATE_HIGHER_HM', 'N/A')}\n"
                f"Mid-Century Extreme Heat Risk (Percentile): {props.get('EXHT_L95_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Late-Century Extreme Heat Risk (Percentile): {props.get('EXHT_L95_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Mid-Century Extreme Heat Hazard Multiplier: {props.get('EXHT_L95_MID_HIGHER_HM', 'N/A')}\n"
                f"Late-Century Extreme Heat Hazard Multiplier: {props.get('EXHT_L95_LATE_HIGHER_HM', 'N/A')}\n"
                f"FEMA Risk Profile:\n"
                f"Disaster Risk (Percentile): {props.get('RISK_SCORE', 'N/A')}\n"
                f"Disaster Loss ($/year): {props.get('EAL_VALT', 'N/A')}\n"
                f"Social Vulnerability (Percentile): {props.get('SOVI_SCORE', 'N/A')}\n"
                f"Community Resilience (Percentile): {props.get('RESL_SCORE', 'N/A')}\n"
                f"Annual Avalanche Loss ($/year): {props.get('AVLN_EALT', 'N/A')}\n"
                f"Annual Avalanche Loss (Percentile): {props.get('AVLN_EALS', 'N/A')}\n"
                f"Annual Coastal Flooding Loss ($/year): {props.get('CFLD_EALT', 'N/A')}\n"
                f"Annual Coastal Flooding Loss (Percentile): {props.get('CFLD_EALS', 'N/A')}\n"
                f"Annual Cold Wave Loss ($/year): {props.get('CWAV_EALT', 'N/A')}\n"
                f"Annual Cold Wave Loss (Percentile): {props.get('CWAV_EALS', 'N/A')}\n"
                f"Annual Drought Loss ($/year): {props.get('DRGT_EALT', 'N/A')}\n"
                f"Annual Drought Loss (Percentile): {props.get('DRGT_EALS', 'N/A')}\n"
                f"Annual Hail Loss ($/year): {props.get('HAIL_EALT', 'N/A')}\n"
                f"Annual Hail Loss (Percentile): {props.get('HAIL_EALS', 'N/A')}\n"
                f"Annual Heat Wave Loss ($/year): {props.get('HWAV_EALT', 'N/A')}\n"
                f"Annual Heat Wave Loss (Percentile): {props.get('HWAV_EALS', 'N/A')}\n"
                f"Annual Hurricane Loss ($/year): {props.get('HRCN_EALT', 'N/A')}\n"
                f"Annual Hurricane Loss (Percentile): {props.get('HRCN_EALS', 'N/A')}\n"
                f"Annual Ice Storm Loss ($/year): {props.get('ISTM_EALT', 'N/A')}\n"
                f"Annual Ice Storm Loss (Percentile): {props.get('ISTM_EALS', 'N/A')}\n"
                f"Annual Landslide Loss ($/year): {props.get('LNDS_EALT', 'N/A')}\n"
                f"Annual Landslide Loss (Percentile): {props.get('LNDS_EALS', 'N/A')}\n"
                f"Annual River Flooding Loss ($/year): {props.get('RFLD_EALT', 'N/A')}\n"
                f"Annual River Flooding Loss (Percentile): {props.get('RFLD_EALS', 'N/A')}\n"
                f"Annual Wind Loss ($/year): {props.get('SWND_EALT', 'N/A')}\n"
                f"Annual Wind Loss (Percentile): {props.get('SWND_EALS', 'N/A')}\n"
                f"Annual Tornado Loss ($/year): {props.get('TRND_EALT', 'N/A')}\n"
                f"Annual Tornado Loss (Percentile): {props.get('TRND_EALS', 'N/A')}\n"
                f"Annual Winter Weather Loss ($/year): {props.get('WNTW_EALT', 'N/A')}\n"
                f"Annual Winter Weather Loss (Percentile): {props.get('WNTW_EALS', 'N/A')}\n"
                f"CEJST Data:\n"
                f"Share of properties at risk of flood in 30 years (percentile): {props.get('Share of properties at risk of flood in 30 years (percentile)', 'N/A')}\n"
                f"Share of properties at risk of flood in 30 years: {props.get('Share of properties at risk of flood in 30 years', 'N/A')}\n"
                f"Share of properties at risk of fire in 30 years (percentile): {props.get('Share of properties at risk of fire in 30 years (percentile)', 'N/A')}\n"
                f"Share of properties at risk of fire in 30 years: {props.get('Share of properties at risk of fire in 30 years', 'N/A')}\n"
                f"Energy burden (percentile): {props.get('Energy burden (percentile)', 'N/A')}\n"
                f"PM2.5 (percentile): {props.get('PM2.5 in the air (percentile)', 'N/A')}\n"
                f"PM2.5 (Volume): {props.get('PM2.5 in the air', 'N/A')}\n"
                f"Impervious surface or cropland: {props.get('Share of the tract\'s land area that is covered by impervious surface or cropland as a percent', 'N/A')}\n"
                f"Asthma Prevalence (Percentile): {props.get('Current asthma among adults aged greater than or equal to 18 years', 'N/A')}\n"
            )

            api_key_input = st.text_input("Enter your OpenAI API key:", type="password")
            user_question = st.text_input("Ask a Question about the selected State:", key="state_question")
            if st.button("Submit State Query", key="state_submit"):
                if api_key_input and user_question:
                    result = answer_question(api_key_input, user_question, extra_context, plan_list, state_abbr)
                    st.write(result)
                else:
                    st.write("Please provide both an API key and a question.")
        else:
            st.info("Click on a state to view details.")
    
    with cols_state[2]:
        legend_html = generate_legend_html(REGION_COLORS)
        st.markdown(legend_html, unsafe_allow_html=True)

# ================================
# TAB 2: COUNTY MAP
# ================================
with tab_county:
    st.subheader("County Map")
    # Initialize county map with no default tiles and add OSM layer.
    m_county = folium.Map(location=[35.3, -97.6], zoom_start=4, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m_county)

    # Add county boundaries with tooltips.
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
            "fillColor": REGION_COLORS.get(x["properties"].get("EPA_REGION"), "transparent"),
            "color": "black",
            "fillOpacity": 0.4,
            "weight": 1
        },
        tooltip=tooltip_county,
        highlight_function=lambda x: {"weight": 2, "color": "blue"}
    ).add_to(county_boundaries)
    county_boundaries.add_to(m_county)

    # Add city markers.
    add_city_markers(m_county)
    folium.LayerControl(collapsed=False).add_to(m_county)

    # Layout columns: left (info & QA), middle (map), right (legend)
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
            state_abbr = props.get("STATE_ABBR", "N/A")
            plan_list = props.get("plan_list", [])
            
            st.write("**County:**", county_name)
            st.write("**Population:**", population)
            st.write("**FIPS:**", fips)
            st.write("**Number of Climate Action Plans:**", f"{int(n_caps):,}")
            
            with st.expander("#### Cities with Climate Action Plans:"):
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
                st.write("**Extreme Heat Mid-Century Projected Risk:**", props.get("EXHT_L95_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Extreme Heat Late-Century Projected Risk:**", props.get("EXHT_L95_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Extreme Heat Mid-Century Hazard Multiplier:**", props.get("EXHT_L95_MID_HIGHER_HM", "N/A"))
                st.write("**Extreme Heat Late-Century Hazard Multiplier:**", props.get("EXHT_L95_LATE_HIGHER_HM", "N/A"))
            
            with st.expander("#### FEMA Risk Profile:"):
                st.write("**Disaster Risk (Percentile):**", props.get("RISK_SCORE", "N/A"))
                st.write("**Disaster Risk (Percentile, relative to state):**", props.get("RISK_SPCTL", "N/A"))
                st.write("**Disaster Loss ($/year):**", props.get("EAL_VALT", "N/A"))
                st.write("**Social Vulnerability (Percentile):**", props.get("SOVI_SCORE", "N/A"))
                st.write("**Community Resilience (Percentile):**", props.get("RESL_SCORE", "N/A"))
                st.write("**Annual Avalanche Loss ($/year):**", props.get("AVLN_EALT", "N/A"))
                st.write("**Annual Avalanche Loss (Percentile):**", props.get("AVLN_EALS", "N/A"))
                st.write("**Annual Coastal Flooding Loss ($/year):**", props.get("CFLD_EALT", "N/A"))
                st.write("**Annual Coastal Flooding Loss (Percentile):**", props.get("CFLD_EALS", "N/A"))
                st.write("**Annual Cold Wave Loss ($/year):**", props.get("CWAV_EALT", "N/A"))
                st.write("**Annual Cold Wave Loss (Percentile):**", props.get("CWAV_EALS", "N/A"))
                st.write("**Annual Drought Loss ($/year):**", props.get("DRGT_EALT", "N/A"))
                st.write("**Annual Drought Loss (Percentile):**", props.get("DRGT_EALS", "N/A"))
                st.write("**Annual Hail Loss ($/year):**", props.get("HAIL_EALT", "N/A"))
                st.write("**Annual Hail Loss (Percentile):**", props.get("HAIL_EALS", "N/A"))
                st.write("**Annual Heat Wave Loss ($/year):**", props.get("HWAV_EALT", "N/A"))
                st.write("**Annual Heat Wave Loss (Percentile):**", props.get("HWAV_EALS", "N/A"))
                st.write("**Annual Hurricane Loss ($/year):**", props.get("HRCN_EALT", "N/A"))
                st.write("**Annual Hurricane Loss (Percentile):**", props.get("HRCN_EALS", "N/A"))
                st.write("**Annual Ice Storm Loss ($/year):**", props.get("ISTM_EALT", "N/A"))
                st.write("**Annual Ice Storm Loss (Percentile):**", props.get("ISTM_EALS", "N/A"))
                st.write("**Annual Landslide Loss ($/year):**", props.get("LNDS_EALT", "N/A"))
                st.write("**Annual Landslide Loss (Percentile):**", props.get("LNDS_EALS", "N/A"))
                st.write("**Annual River Flooding Loss ($/year):**", props.get("RFLD_EALT", "N/A"))
                st.write("**Annual River Flooding Loss (Percentile):**", props.get("RFLD_EALS", "N/A"))
                st.write("**Annual Wind Loss ($/year):**", props.get("SWND_EALT", "N/A"))
                st.write("**Annual Wind Loss (Percentile):**", props.get("SWND_EALS", "N/A"))
                st.write("**Annual Tornado Loss ($/year):**", props.get("TRND_EALT", "N/A"))
                st.write("**Annual Tornado Loss (Percentile):**", props.get("TRND_EALS", "N/A"))
                st.write("**Annual Winter Weather Loss ($/year):**", props.get("WNTW_EALT", "N/A"))
                st.write("**Annual Winter Weather Loss (Percentile):**", props.get("WNTW_EALS", "N/A"))

            with st.expander("#### CEJST Data:"):
                st.write("**Share of properties at risk of flood in 30 years (percentile):**", props.get("Share of properties at risk of flood in 30 years (percentile)", "N/A"))
                st.write("**Share of properties at risk of flood in 30 years:**", props.get("Share of properties at risk of flood in 30 years", "N/A"))
                st.write("**Share of properties at risk of fire in 30 years (percentile):**", props.get("Share of properties at risk of fire in 30 years (percentile)", "N/A"))
                st.write("**Share of properties at risk of fire in 30 years:**", props.get("Share of properties at risk of fire in 30 years", "N/A"))
                st.write("**Energy burden (percentile):**", props.get("Energy burden (percentile)", "N/A"))
                st.write("**PM2.5 (percentile):**", props.get("PM2.5 in the air (percentile)", "N/A"))
                st.write("**PM2.5 (Volume):**", props.get("PM2.5 in the air", "N/A"))
                st.write("**Impervious surface or cropland:**", props.get("Share of the tract's land area that is covered by impervious surface or cropland as a percent", "N/A"))
                st.write("**Asthma Prevalence (Percentile):**", props.get("Current asthma among adults aged greater than or equal to 18 years", "N/A"))
                        
            # Build extra context for the QA chain
            extra_context = (
                f"County: {county_name}\n"
                f"Population: {population}\n"
                f"FIPS: {fips}\n"
                f"Climate Action Plans: {', '.join(plan_list) if plan_list else 'No climate action plans'}\n"
                f"NRI Future Risk Index (Higher Warming Pathway):\n"
                f"Coastal Flooding Mid-Century Projected Risk: {props.get('CFLD_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Coastal Flooding Late-Century Projected Risk: {props.get('CFLD_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Coastal Flooding Mid-Century Hazard Multiplier: {props.get('CFLD_MID_HIGHER_HM', 'N/A')}\n"
                f"Coastal Flooding Late-Century Hazard Multiplier: {props.get('CFLD_LATE_HIGHER_HM', 'N/A')}\n"
                f"Wildfire Mid-Century Projected Risk: {props.get('WFIR_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Wildfire Late-Century Projected Risk: {props.get('WFIR_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Wildfire Mid-Century Hazard Multiplier: {props.get('WFIR_MID_HIGHER_HM', 'N/A')}\n"
                f"Wildfire Late-Century Hazard Multiplier: {props.get('WFIR_LATE_HIGHER_HM', 'N/A')}\n"
                f"Drought Mid-Century Projected Risk: {props.get('DRGT_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Drought Late-Century Projected Risk: {props.get('DRGT_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Drought Mid-Century Hazard Multiplier: {props.get('DRGT_MID_HIGHER_HM', 'N/A')}\n"
                f"Drought Late-Century Hazard Multiplier: {props.get('DRGT_LATE_HIGHER_HM', 'N/A')}\n"
                f"Hurricane Mid-Century Projected Risk: {props.get('HRCN_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Hurricane Late-Century Projected Risk: {props.get('HRCN_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Hurricane Mid-Century Hazard Multiplier: {props.get('HRCN_MID_HIGHER_HM', 'N/A')}\n"
                f"Hurricane Late-Century Hazard Multiplier: {props.get('HRCN_LATE_HIGHER_HM', 'N/A')}\n"
                f"Extreme Heat Mid-Century Projected Risk: {props.get('EXHT_L95_MID_HIGHER_PRISKS', 'N/A')}\n"
                f"Extreme Heat Late-Century Projected Risk: {props.get('EXHT_L95_LATE_HIGHER_PRISKS', 'N/A')}\n"
                f"Extreme Heat Mid-Century Hazard Multiplier: {props.get('EXHT_L95_MID_HIGHER_HM', 'N/A')}\n"
                f"Extreme Heat Late-Century Hazard Multiplier: {props.get('EXHT_L95_LATE_HIGHER_HM', 'N/A')}\n"
                f"FEMA Risk Profile:\n"
                f"Disaster Risk (Percentile): {props.get('RISK_SCORE', 'N/A')}\n"
                f"Disaster Risk (Percentile, relative to state): {props.get('RISK_SPCTL', 'N/A')}\n"
                f"Disaster Loss ($/year): {props.get('EAL_VALT', 'N/A')}\n"
                f"Social Vulnerability (Percentile): {props.get('SOVI_SCORE', 'N/A')}\n"
                f"Community Resilience (Percentile): {props.get('RESL_SCORE', 'N/A')}\n"
                f"Annual Avalanche Loss ($/year): {props.get('AVLN_EALT', 'N/A')}\n"
                f"Annual Avalanche Loss (Percentile): {props.get('AVLN_EALS', 'N/A')}\n"
                f"Annual Coastal Flooding Loss ($/year): {props.get('CFLD_EALT', 'N/A')}\n"
                f"Annual Coastal Flooding Loss (Percentile): {props.get('CFLD_EALS', 'N/A')}\n"
                f"Annual Cold Wave Loss ($/year): {props.get('CWAV_EALT', 'N/A')}\n"
                f"Annual Cold Wave Loss (Percentile): {props.get('CWAV_EALS', 'N/A')}\n"
                f"Annual Drought Loss ($/year): {props.get('DRGT_EALT', 'N/A')}\n"
                f"Annual Drought Loss (Percentile): {props.get('DRGT_EALS', 'N/A')}\n"
                f"Annual Hail Loss ($/year): {props.get('HAIL_EALT', 'N/A')}\n"
                f"Annual Hail Loss (Percentile): {props.get('HAIL_EALS', 'N/A')}\n"
                f"Annual Heat Wave Loss ($/year): {props.get('HWAV_EALT', 'N/A')}\n"
                f"Annual Heat Wave Loss (Percentile): {props.get('HWAV_EALS', 'N/A')}\n"
                f"Annual Hurricane Loss ($/year): {props.get('HRCN_EALT', 'N/A')}\n"
                f"Annual Hurricane Loss (Percentile): {props.get('HRCN_EALS', 'N/A')}\n"
                f"Annual Ice Storm Loss ($/year): {props.get('ISTM_EALT', 'N/A')}\n"
                f"Annual Ice Storm Loss (Percentile): {props.get('ISTM_EALS', 'N/A')}\n"
                f"Annual Landslide Loss ($/year): {props.get('LNDS_EALT', 'N/A')}\n"
                f"Annual Landslide Loss (Percentile): {props.get('LNDS_EALS', 'N/A')}\n"
                f"Annual River Flooding Loss ($/year): {props.get('RFLD_EALT', 'N/A')}\n"
                f"Annual River Flooding Loss (Percentile): {props.get('RFLD_EALS', 'N/A')}\n"   
                f"Annual Wind Loss ($/year): {props.get('SWND_EALT', 'N/A')}\n"
                f"Annual Wind Loss (Percentile): {props.get('SWND_EALS', 'N/A')}\n"
                f"Annual Tornado Loss ($/year): {props.get('TRND_EALT', 'N/A')}\n"
                f"Annual Tornado Loss (Percentile): {props.get('TRND_EALS', 'N/A')}\n"
                f"Annual Winter Weather Loss ($/year): {props.get('WNTW_EALT', 'N/A')}\n"
                f"Annual Winter Weather Loss (Percentile): {props.get('WNTW_EALS', 'N/A')}\n"
                f"CEJST Data:\n"
                f"Share of properties at risk of flood in 30 years (percentile): {props.get('Share of properties at risk of flood in 30 years (percentile)', 'N/A')}\n"
                f"Share of properties at risk of flood in 30 years: {props.get('Share of properties at risk of flood in 30 years', 'N/A')}\n"
                f"Share of properties at risk of fire in 30 years (percentile): {props.get('Share of properties at risk of fire in 30 years (percentile)', 'N/A')}\n"
                f"Share of properties at risk of fire in 30 years: {props.get('Share of properties at risk of fire in 30 years', 'N/A')}\n"
                f"Energy burden (percentile): {props.get('Energy burden (percentile)', 'N/A')}\n"
                f"PM2.5 (percentile): {props.get('PM2.5 in the air (percentile)', 'N/A')}\n"
                f"PM2.5 (Volume): {props.get('PM2.5 in the air', 'N/A')}\n"
                f"Impervious surface or cropland: {props.get('Share of the tract\'s land area that is covered by impervious surface or cropland as a percent', 'N/A')}\n"
                f"Asthma Prevalence (Percentile): {props.get('Current asthma among adults aged greater than or equal to 18 years', 'N/A')}\n"
            )

            api_key_input = st.text_input("Enter your OpenAI API key:", type="password", key="county_api_key")
            user_question = st.text_input("Ask a Question about the selected County:", key="county_question")
            if st.button("Submit County Query", key="county_submit"):
                if api_key_input and user_question:
                    result = answer_question(api_key_input, user_question, extra_context, plan_list, state_abbr)
                    st.write(result)
                else:
                    st.write("Please provide both an API key and a question.")
        else:
            st.info("Click on a county to view details.")
    
    with cols_county[2]:
        legend_html = generate_legend_html(REGION_COLORS)
        st.markdown(legend_html, unsafe_allow_html=True)
