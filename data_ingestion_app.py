import streamlit as st
import subprocess
import os

st.title("Data Ingestion Portal")

st.write("This portal allows you to contribute to the dataset of Climate Action Plans (CAPs) in the United States. Please upload a PDF of a climate action plan and provide the city, state, and year and type of the plan.")

uploaded_file = st.file_uploader("Upload a PDF of a climate action plan", type="pdf")

if uploaded_file is not None:
    st.write("File uploaded successfully!")

city = st.text_input("City")
state = st.selectbox("State", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "WA", "WV", "WI", "WY"])
st.markdown("Please provide the county name as per the US Census Bureau: e.g. 'Forsyth County, North Carolina', 'Anchorage Municipality, Alaska', 'Orleans Parish, Louisiana' or 'District of Columbia, District of Columbia'")
county = st.text_input("County")
year = st.text_input("Year")
st.markdown("Please provide the city center coordinates e.g. '35.994033, -83.929849' for mapping purposes")
city_center_coordinates = st.text_input("City Center Coordinates (latitude, longitude)")

# Use a select box for the type of plan
plan_type = st.selectbox("Type of Plan", [
    "Mitigation Only CAP",
    "Mitigation Primary CAP",
    "Equal Adaptation-Mitigation CAP",
    "Green Plan",
    "Resiliency Plan"
])

# Add a password input for the OpenAI API key
api_key = st.text_input("OpenAI API Key", type="password")

if st.button("Submit"):
    if uploaded_file is not None and city and state and year and plan_type and api_key:
        # Create the CAPS directory if it doesn't exist
        os.makedirs("CAPS", exist_ok=True)
        
        # Define the file path
        file_name = f"{city}, {state} {plan_type} {year}.pdf"
        summary_file_name = f"{city}, {state} {plan_type} {year}_Summary.md"
        file_path = os.path.join("CAPS", file_name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write(f"File saved as {file_name} in CAPS folder.")
        
        # Run the external Python scripts with the required parameters
        subprocess.run(["python", "data_ingestion_helpers/city_county_mapping_addition.py", city, state, county, city_center_coordinates])
        st.write("City, State, County, and City Center Coordinates added to city_county_mapping.csv")
        subprocess.run(["python", "data_ingestion_helpers/summary_generation.py", api_key, file_path])
        st.write("Summary generated successfully!")

        subprocess.run(["python", "data_ingestion_helpers/data_ingestion_vectorstores.py", api_key, file_name, summary_file_name])
        st.write("Vector store created successfully")

        subprocess.run(["python", "data_ingestion_helpers/dataset_addition.py", api_key, file_path])
        st.write("Data added to dataset successfully")

        subprocess.run(["python", "caps_directory_reader.py"])
        st.write("CAPs directory reader executed successfully")
        
        st.write("All scripts executed successfully")
    else:
        st.error("Please fill in all fields, upload a file, and provide your OpenAI API key.")


