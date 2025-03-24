import os
import re
from tempfile import NamedTemporaryFile

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import anthropic

# Import all helper functions from app_helpers.py
from app_helpers import *

# Import necessary modules from LangChain and its community extensions
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Climate Policy Tracker", layout="wide")

# ------------------------------------------------------------------------------
# LOAD DATA FILES
# ------------------------------------------------------------------------------
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

st.title("Climate Policy Tracker")

# ------------------------------------------------------------------------------
# API KEYS INPUT
# ------------------------------------------------------------------------------
openai_api_key = st.text_input("OpenAI API Key", type="password")
anthropic_api_key = st.text_input("Anthropic API Key", type="password")

# ------------------------------------------------------------------------------
# TABS SETUP
# ------------------------------------------------------------------------------
(summary_tab, multi_plan_qa_tab, document_qa_tab, plan_comparison_tab, 
 state_tab, county_tab) = st.tabs([
    "Summary Generation",
    "Multi-Plan Q&A",
    "Document Q&A Tool",
    "Plan Comparison Tool",
    "State-Level Policy Tracker",
    "County-Level Policy Tracker",
])

# ------------------------------------------------------------------------------
# TAB 1: SUMMARY GENERATION
# ------------------------------------------------------------------------------
with summary_tab:
    st.header("Summary Generation")
    uploaded_file = st.file_uploader(
        "Upload a Climate Action Plan in PDF format",
        type="pdf",
        key="upload_file"
    )

    # Set file paths for prompt and questions
    prompt_file_path = "Prompts/summary_tool_system_prompt.md"
    questions_file_path = "Prompts/summary_tool_questions.md"

    if st.button("Generate", key="generate_button"):
        if not openai_api_key:
            st.warning("Please provide your OpenAI API key.")
        elif not uploaded_file:
            st.warning("Please upload a PDF file.")
        else:
            display_placeholder = st.empty()
            with st.spinner("Processing..."):
                try:
                    # Call the new summary_generation function
                    results = summary_generation(
                        openai_api_key,
                        uploaded_file,
                        questions_file_path,
                        prompt_file_path,
                        display_placeholder
                    )
                    markdown_text = "\n".join(results)

                    # Use the uploaded file's base name for the download file
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    download_file_name = f"{base_name}_Summary.md"

                    st.download_button(
                        label="Download Results as Markdown",
                        data=markdown_text,
                        file_name=download_file_name,
                        mime="text/markdown",
                        key="download_button"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ------------------------------------------------------------------------------
# TAB 2: MULTI-PLAN Q&A
# ------------------------------------------------------------------------------
with multi_plan_qa_tab:
    st.header("Multi-Plan Q&A")
    st.markdown(
        "This tool answers questions using information from all plans in the database. "
        "Use it to answer general questions about the plans and their strategies. "
        "For questions about a specific plan, use the Document Q&A Tool."
    )
    input_text = st.text_input("Ask a question:", key="multi_plan_input")
    st.markdown("### Search Method")
    st.markdown(
        "The **Efficient** method uses a single shared vector store across all plans. "
        "The **Greedy** method uses multiple vector stores to retrieve the most relevant chunks for each plan."
    )
    search_method = st.radio("Select a search method: ", ["Efficient", "Greedy"])
    if st.button("Ask", key="multi_plan_qa_button"):
        if not openai_api_key:
            st.warning("Please provide your OpenAI API key.")
        elif not input_text:
            st.warning("Please enter a question.")
        else:
            display_placeholder2 = st.empty()
            with st.spinner("Processing..."):
                try:
                    if search_method == "Efficient":
                        # Call multi_plan_qa for the efficient (single vector store) method
                        multi_plan_qa(
                            openai_api_key,
                            input_text,
                            display_placeholder2
                        )
                    elif search_method == "Greedy":
                        # Call multi_plan_qa_multi_vectorstore for the greedy (multiple vector stores) method
                        multi_plan_qa_multi_vectorstore(
                            openai_api_key,
                            input_text,
                            display_placeholder2
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ------------------------------------------------------------------------------
# TAB 3: DOCUMENT Q&A TOOL
# ------------------------------------------------------------------------------
with document_qa_tab:
    st.header("Document Q&A Tool")

    # Get list of existing vector store documents
    vectorstore_documents = list_vector_store_documents()

    # Option to upload a new plan or select from existing vector stores
    focus_option = st.radio(
        "Choose a focus plan:",
        ("Select from existing vector stores", "Upload a new plan"),
        key="focus_option_qa"
    )

    if focus_option == "Upload a new plan":
        focus_uploaded_file = st.file_uploader(
            "Upload a Climate Action Plan",
            type="pdf",
            key="focus_upload_qa"
        )
        focus_input = focus_uploaded_file if focus_uploaded_file else None
    else:
        selected_focus_plan = st.selectbox(
            "Select a focus plan:",
            vectorstore_documents,
            key="select_focus_plan_qa"
        )
        focus_input = os.path.join(
            "Individual_All_Vectorstores",
            f"{selected_focus_plan.replace(' Summary', '_Summary')}_vectorstore"
        )

    # Display previous conversation messages
    if "chat_history" in st.session_state:
        for message in st.session_state.chat_history:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            st.chat_message(role).markdown(message.content)

    user_input = st.chat_input("Ask a question")
    if user_input:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if openai_api_key and focus_input:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.chat_message("user").markdown(user_input)
            with st.spinner("Processing..."):
                # Call the new document_qa function
                answer = document_qa(openai_api_key, focus_input, user_input)
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.chat_message("assistant").markdown(answer)
        else:
            st.warning("Please provide your OpenAI API key and select a focus plan.")

# ------------------------------------------------------------------------------
# TAB 4: PLAN COMPARISON TOOL
# ------------------------------------------------------------------------------
with plan_comparison_tab:
    st.header("Plan Comparison Tool")

    # Get list of existing vector store documents for plans
    vectorstore_documents = list_vector_store_documents()

    # Option to upload a new plan or select from existing vector stores for focus
    focus_option = st.radio(
        "Choose a focus plan:",
        ("Select from existing vector stores", "Upload a new plan"),
        key="focus_option"
    )
    if focus_option == "Upload a new plan":
        focus_uploaded_file = st.file_uploader(
            "Upload a Climate Action Plan to compare",
            type="pdf",
            key="focus_upload"
        )
        focus_input = focus_uploaded_file if focus_uploaded_file is not None else None
    else:
        selected_focus_plan = st.selectbox(
            "Select a focus plan:",
            vectorstore_documents,
            key="select_focus_plan"
        )
        focus_input = os.path.join(
            "Individual_All_Vectorstores",
            f"{selected_focus_plan.replace(' Summary', '_Summary')}_vectorstore"
        )

    # Option to upload comparison documents or select from existing vector stores
    comparison_option = st.radio(
        "Choose comparison documents:",
        ("Select from existing vector stores", "Upload new documents"),
        key="comparison_option"
    )
    if comparison_option == "Upload new documents":
        comparison_files = st.file_uploader(
            "Upload comparison documents",
            type="pdf",
            accept_multiple_files=True,
            key="comparison_files"
        )
        comparison_inputs = comparison_files
    else:
        selected_comparison_plans = st.multiselect(
            "Select comparison documents:",
            vectorstore_documents,
            key="select_comparison_plans"
        )
        comparison_inputs = [
            os.path.join(
                "Individual_All_Vectorstores",
                f"{doc.replace(' Summary', '_Summary')}_vectorstore"
            ) for doc in selected_comparison_plans
        ]

    input_text = st.text_input("Ask a comparison question:", key="comparison_input")

    st.markdown("### Model")
    st.markdown(
        "The **Standard (OpenAI)** model uses GPT-4o with RAG to answer questions. "
        "The **Long Context Model (Anthropic)** uses Claude for answering questions without RAG."
    )
    search_method = st.radio("Select an approach: ", ["Standard (OpenAI)", "Long Context Model (Anthropic)"])
    if st.button("Compare", key="compare_button"):
        if not openai_api_key:
            st.warning("Please provide your OpenAI API key.")
        elif not input_text:
            st.warning("Please enter a comparison question.")
        elif not focus_input:
            st.warning("Please provide a focus plan.")
        elif not comparison_inputs:
            st.warning("Please provide comparison documents.")
        else:
            display_placeholder3 = st.empty()
            with st.spinner("Processing..."):
                try:
                    if search_method == "Standard (OpenAI)":
                        # Call the new comparison_qa function (formerly process_one_to_many_query)
                        comparison_qa(
                            openai_api_key,
                            focus_input,
                            comparison_inputs,
                            input_text,
                            display_placeholder3
                        )
                    elif search_method == "Long Context Model (Anthropic)":
                        # For long-context, pass the focus plan and comparison inputs as selected summaries
                        comparison_qa_long_context(
                            openai_api_key,
                            anthropic_api_key,
                            input_text,
                            focus_input,
                            comparison_inputs,
                            display_placeholder3
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ------------------------------------------------------------------------------
# TAB 5: STATE-LEVEL POLICY TRACKER
# ------------------------------------------------------------------------------
with state_tab:
    st.subheader("State Map")
    # Initialize state map with no default tiles; add an OpenStreetMap layer.
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

    # Add city markers to the map.
    add_city_markers(m_state)
    folium.LayerControl(collapsed=False).add_to(m_state)

    # Define a three-column layout for additional info, the map, and the legend.
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
            
            # (Additional risk index and FEMA risk info displayed in expanders)
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
                # (Additional FEMA risk metrics can be added similarly)
            
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
            
            # Build extra context for the maps QA chain
            extra_context = (
                f"State: {state_name}\n"
                f"Population: {population}\n"
                f"FIPS: {fips}\n"
                f"Climate Action Plans: {', '.join(plan_list) if plan_list else 'No climate action plans'}\n"
                # Additional risk details can be appended here...
            )

            api_key_input = st.text_input("Enter your OpenAI API key:", type="password")
            user_question = st.text_input("Ask a Question about the selected State:", key="state_question")
            if st.button("Submit State Query", key="state_submit"):
                if api_key_input and user_question:
                    # Call the new maps_qa function (formerly answer_question)
                    result = maps_qa(api_key_input, user_question, extra_context, plan_list, state_abbr)
                    st.write(result)
                else:
                    st.write("Please provide both an API key and a question.")
        else:
            st.info("Click on a state to view details.")
    
    with cols_state[2]:
        legend_html = generate_legend_html(REGION_COLORS)
        st.markdown(legend_html, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 6: COUNTY-LEVEL POLICY TRACKER
# ------------------------------------------------------------------------------
with county_tab:
    st.subheader("County Map")
    # Initialize county map with no default tiles; add an OpenStreetMap layer.
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

    # Define a three-column layout for county info, map, and legend.
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
            
            with st.expander("Cities with Climate Action Plans:"):
                if plan_list:
                    for plan in plan_list:
                        st.write(plan)
                else:
                    st.write("None")
            
            # Display additional risk information in expanders (NRI, FEMA, CEJST, etc.)
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
                st.write("**Extreme Heat Mid-Century Projected Risk:**", props.get("EXHT_L95_MID_HIGHER_PRISKS", "N/A"))
                st.write("**Extreme Heat Late-Century Projected Risk:**", props.get("EXHT_L95_LATE_HIGHER_PRISKS", "N/A"))
                st.write("**Extreme Heat Mid-Century Hazard Multiplier:**", props.get("EXHT_L95_MID_HIGHER_HM", "N/A"))
                st.write("**Extreme Heat Late-Century Hazard Multiplier:**", props.get("EXHT_L95_LATE_HIGHER_HM", "N/A"))
            
            with st.expander("FEMA Risk Profile:"):
                st.write("**Disaster Risk (Percentile):**", props.get("RISK_SCORE", "N/A"))
                st.write("**Disaster Risk (Percentile, relative to state):**", props.get("RISK_SPCTL", "N/A"))
                st.write("**Disaster Loss ($/year):**", props.get("EAL_VALT", "N/A"))
                st.write("**Social Vulnerability (Percentile):**", props.get("SOVI_SCORE", "N/A"))
                st.write("**Community Resilience (Percentile):**", props.get("RESL_SCORE", "N/A"))
                # (Additional FEMA metrics as needed)
            
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
            
            # Build extra context for the maps QA chain
            extra_context = (
                f"County: {county_name}\n"
                f"Population: {population}\n"
                f"FIPS: {fips}\n"
                f"Climate Action Plans: {', '.join(plan_list) if plan_list else 'No climate action plans'}\n"
                # Additional details can be appended here...
            )

            api_key_input = st.text_input("Enter your OpenAI API key:", type="password", key="county_api_key")
            user_question = st.text_input("Ask a Question about the selected County:", key="county_question")
            if st.button("Submit County Query", key="county_submit"):
                if api_key_input and user_question:
                    # Call the new maps_qa function for counties
                    result = maps_qa(api_key_input, user_question, extra_context, plan_list, state_abbr)
                    st.write(result)
                else:
                    st.write("Please provide both an API key and a question.")
        else:
            st.info("Click on a county to view details.")
    
    with cols_county[2]:
        legend_html = generate_legend_html(REGION_COLORS)
        st.markdown(legend_html, unsafe_allow_html=True)