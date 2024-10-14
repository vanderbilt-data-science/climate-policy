import streamlit as st
import openai
import PyPDF2
import textwrap

st.set_page_config(page_title="PDF to Markdown Summary")

st.title("PDF to Markdown Summary Generator")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into smaller chunks (set chunk_size to smaller number to avoid token overload)
def split_text_into_chunks(text, chunk_size=1000):  # Set smaller chunk size (e.g., 1000)
    return textwrap.wrap(text, width=chunk_size)

# Function to generate a markdown summary using OpenAI
def generate_summary(api_key, document_text, questions):
    openai.api_key = api_key

    # Split document into smaller chunks
    chunks = split_text_into_chunks(document_text, chunk_size=1000)  # Reduce chunk size
    full_summary = ""

    for i, chunk in enumerate(chunks):
        prompt = f"Here is part {i+1} of the document:\n{chunk}\n\nSummarize it based on the following questions:\n{questions}"

        try:
            # Call the OpenAI API for each chunk, with max_tokens limited to 500 to avoid overload
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500  # Further reduce max tokens per chunk
            )
            summary = response['choices'][0]['message']['content'].strip()
            full_summary += f"### Part {i+1} Summary:\n{summary}\n\n"

        except Exception as e:
            return f"Error: {str(e)}"

    return full_summary

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Input for OpenAI API key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Predefined questions for the summary
questions = st.text_area("Enter the predefined questions for the summary", 
                         "1. What is the main topic?\n2. What are the key takeaways?\n3. What are the next steps?")

# Button to generate the markdown summary
if st.button("Generate Summary"):
    if uploaded_file is not None and api_key:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Generate the summary
        with st.spinner("Generating summary..."):
            summary = generate_summary(api_key, pdf_text, questions)
        
        # Display the markdown summary
        st.markdown("### Generated Markdown Summary")
        st.markdown(summary)

        # Provide a download option
        st.download_button(
            label="Download Markdown File",
            data=summary,
            file_name="summary.md",
            mime="text/markdown"
        )
    else:
        st.error("Please upload a PDF and enter a valid API key.")
