import streamlit as st
import subprocess
import os
import pandas as pd
from geopy.geocoders import Nominatim

# Function to get coordinates from city and state
def get_coordinates(city, state):
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.geocode(f"{city}, {state}, USA")
    if location:
        return f"{location.latitude}, {location.longitude}"
    else:
        return None  # Return None instead of a string to handle better logic

# Load county data from CSV
@st.cache_data
def load_county_data():
    df = pd.read_csv("us_counties.csv")  # Replace with actual file path
    df["stateName"] = df["stateName"].str.strip()
    df["countyName"] = df["countyName"].str.strip()
    return df

county_data = load_county_data()

# State name to abbreviation mapping
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
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

st.title("Data Ingestion Portal")
st.write("Upload a climate action plan and provide relevant details.")

uploaded_file = st.file_uploader("Upload a PDF of a climate action plan", type="pdf")

if uploaded_file is not None:
    st.write("File uploaded successfully!")

city = st.text_input("City")

# Select state by full name
state_full_name = st.selectbox("State", sorted(state_abbr_map.keys()))
state = state_abbr_map[state_full_name]  # Convert to abbreviation

# Dynamically update county options based on selected state
filtered_counties = county_data[county_data["stateName"] == state_full_name]["countyName"].tolist()
selected_counties = st.multiselect("Select County(ies)", filtered_counties)

# Convert selected counties into a comma-separated string
county_str = ", ".join(selected_counties)

# Automatically get coordinates when city and state are entered
city_center_coordinates = get_coordinates(city, state)

if city_center_coordinates:
    st.text(f"City Coordinates: {city_center_coordinates}")
else:
    st.error("Coordinates not found. Please enter manually:")
    city_center_coordinates = st.text_input("City Center Coordinates (latitude, longitude)")

year = st.text_input("Year")

plan_type = st.selectbox("Type of Plan", [
    "Mitigation Only CAP",
    "Mitigation Primary CAP",
    "Equal Adaptation-Mitigation CAP",
    "Green Plan",
    "Resiliency Plan"
])

api_key = st.text_input("OpenAI API Key", type="password")

# File path for the uploaded document
file_name = f"{city}, {state} {plan_type} {year}.pdf"
summary_file_name = f"{city}, {state} {plan_type} {year}_Summary.md"
file_path = os.path.join("CAPS", file_name)

# Check if the file already exists
file_exists = os.path.exists(file_path)

# Button Logic
if st.button("Submit"):
    if not city_center_coordinates or city_center_coordinates == "Coordinates not found":
        st.error("Please provide valid city and state coordinates")
    elif uploaded_file is None:
        st.error("Please upload a PDF file.")
    elif not city or not state or not year or not plan_type or not api_key or not selected_counties:
        st.error("Please fill in all required fields.")
    elif file_exists:
        st.error("File for this plan already exists. Please provide a different plan.")
    else:
        os.makedirs("CAPS", exist_ok=True)

        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write(f"File saved as {file_name} in CAPS folder.")

        # Run subprocess scripts with selected county and manually entered coordinates
        subprocess.run(["python", "data_ingestion_helpers/city_county_mapping_addition.py", city, state, county_str, city_center_coordinates])
        st.write("City, State, County(s), and City Center Coordinates added to city_county_mapping.csv")

        subprocess.run(["python", "data_ingestion_helpers/summary_generation.py", api_key, file_path])
        st.write("Summary generated successfully!")

        subprocess.run(["python", "data_ingestion_helpers/data_ingestion_vectorstores.py", api_key, file_name, summary_file_name])
        st.write("Vector store created successfully")

        subprocess.run(["python", "data_ingestion_helpers/dataset_addition.py", api_key, file_path])
        st.write("Data added to dataset successfully")

        subprocess.run(["python", "batch_scripts/caps_directory_reader.py"])
        st.write("CAPS directory reader executed successfully")

        st.success("All scripts executed successfully!")
