import streamlit as st
import re
import subprocess
import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

def get_coordinates(city, state, timeout=10):
    geolocator = Nominatim(user_agent="geo_locator")
    try:
        location = geolocator.geocode(f"{city}, {state}, USA", timeout=timeout)
        if location:
            return f"{location.latitude}, {location.longitude}"
        else:
            return ""
    except (GeocoderUnavailable, GeocoderTimedOut) as e:
        print(f"Geocoding error: {e}")
        return ""

@st.cache_data
def load_county_data():
    df = pd.read_csv("us_counties.csv")
    df["stateName"] = df["stateName"].str.strip()
    df["countyName"] = df["countyName"].str.strip()
    return df

county_data = load_county_data()

# Mapping of full state names to abbreviations (including District of Columbia)
state_abbr_map = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC"
}
# Inverse mapping: abbreviation to full state name
abbr_to_full = {abbr: name for name, abbr in state_abbr_map.items()}

st.title("Batch Data Ingestion Portal")
st.write("Upload multiple PDF files of climate action plans. Files should be named as follows:")
st.write("**City, State Plan Type Year.pdf** (e.g., *Carson, CA Mitigation Only CAP 2017.pdf* or *Washington, District of Columbia Green Plan 2019.pdf*)")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
api_key = st.text_input("OpenAI API Key", type="password")

file_info = {}

if uploaded_files:
    with st.form("metadata_form"):
        st.write("### File Details and County Selection")
        for uploaded_file in uploaded_files:
            with st.expander(f"File: {uploaded_file.name}", expanded=True):
                base_name = os.path.splitext(uploaded_file.name)[0]
                # Regex with alternation:
                # - Either exactly two letters as state_abbr (if followed by whitespace)
                # - Or a full state name (one or more words)
                pattern = r"^(?P<city>.+?),\s*((?P<state_abbr>[A-Za-z]{2})(?=\s)|(?P<state_full>[A-Za-z\.]+(?:\s+[A-Za-z\.]+)*?))\s+(?P<plan_type>.+?)\s+(?P<year>\d{4})$"
                match = re.match(pattern, base_name)
                if not match:
                    st.error("Filename format is incorrect. Please ensure it follows 'City, State Plan Type Year.pdf'")
                    continue
                
                city = match.group("city").strip()
                # Determine if the state was captured as an abbreviation or full name.
                if match.group("state_abbr"):
                    state_abbrev = match.group("state_abbr").upper()
                    full_state = abbr_to_full.get(state_abbrev)
                    if not full_state:
                        st.error(f"State abbreviation {state_abbrev} not recognized.")
                        continue
                else:
                    full_state = match.group("state_full").strip()
                    # Normalize common variations for District of Columbia.
                    if full_state.lower() in ["district", "d.c.", "dc"]:
                        full_state = "District of Columbia"
                    if full_state in state_abbr_map:
                        state_abbrev = state_abbr_map[full_state]
                    else:
                        st.error(f"State name {full_state} not recognized.")
                        continue
                
                plan_type = match.group("plan_type").strip()
                year = match.group("year").strip()
                
                st.write(f"**City:** {city}")
                st.write(f"**State:** {full_state} ({state_abbrev})")
                st.write(f"**Plan Type:** {plan_type}")
                st.write(f"**Year:** {year}")
                
                county_options = county_data[county_data["stateName"] == full_state]["countyName"].tolist()
                selected_counties = st.multiselect("Select County(ies) for this plan", county_options, key=f"counties_{uploaded_file.name}")
                
                default_coords = get_coordinates(city, state_abbrev)
                coords = st.text_input("City Center Coordinates (latitude, longitude)", value=default_coords, key=f"coords_{uploaded_file.name}")
                
                file_info[uploaded_file.name] = {
                    "uploaded_file": uploaded_file,
                    "city": city,
                    "state": state_abbrev,
                    "plan_type": plan_type,
                    "year": year,
                    "counties": selected_counties,
                    "coords": coords
                }
        form_submitted = st.form_submit_button("Process All Files")

    if form_submitted:
        if not api_key:
            st.error("Please provide the OpenAI API Key.")
        else:
            with st.spinner("Processing files..."):
                for file_name, info in file_info.items():
                    if (not info["city"] or not info["state"] or not info["plan_type"] or 
                        not info["year"] or not api_key or not info["counties"] or not info["coords"]):
                        st.error(f"Missing required fields for file {file_name}. Please fill in all fields.")
                        continue
                    
                    county_str = ", ".join(info["counties"])
                    city = info["city"]
                    state_abbrev = info["state"]
                    plan_type = info["plan_type"]
                    year = info["year"]
                    coords = info["coords"]
                    uploaded_file = info["uploaded_file"]
                    
                    out_file_name = f"{city}, {state_abbrev} {plan_type} {year}.pdf"
                    summary_file_name = f"{city}, {state_abbrev} {plan_type} {year}_Summary.md"
                    file_path = os.path.join("CAPS", out_file_name)
                    
                    if os.path.exists(file_path):
                        st.error(f"File for {out_file_name} already exists. Skipping this file.")
                        continue
                    
                    os.makedirs("CAPS", exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.write(f"Saved {out_file_name} to CAPS folder.")
                    
                    subprocess.run(["python", "data_ingestion_helpers/city_county_mapping_addition.py", city, state_abbrev, county_str, coords])
                    st.write(f"City, State, County(s), and Coordinates added for {out_file_name}.")
                    
                    subprocess.run(["python", "data_ingestion_helpers/summary_generation.py", api_key, file_path])
                    st.write(f"Summary generated for {out_file_name}.")
                    
                    subprocess.run(["python", "data_ingestion_helpers/data_ingestion_vectorstores.py", api_key, out_file_name, summary_file_name])
                    st.write(f"Vector store created for {out_file_name}.")
                    
                    subprocess.run(["python", "data_ingestion_helpers/dataset_addition.py", api_key, file_path])
                    st.write(f"Data added to dataset for {out_file_name}.")
                
                # Run final batch scripts once after all files are processed.
                subprocess.run(["python", "batch_scripts/caps_directory_reader.py"])
                st.write("CAPS directory reader executed.")
                
                subprocess.run(["python", "maps_helpers/maps_data.py"])
                st.write("Maps data re-created.")
            
            st.success("All files processed successfully!")