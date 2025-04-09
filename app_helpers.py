import os
import re
from tempfile import NamedTemporaryFile

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import anthropic

# Import necessary modules from LangChain and community extensions
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================
@st.cache_data
def load_pickle_data(file_path):
    """Load and cache a pickle file from the given path."""
    return pd.read_pickle(file_path)

city_mapping_df = load_pickle_data("./maps_helpers/city_mapping_df.pkl")
city_plans_df = load_pickle_data("./maps_helpers/city_plans_df.pkl")
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def remove_code_blocks(text):
    """
    Remove code block markers (triple backticks) from the text.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text without code block markers.
    """
    code_block_pattern = r"^```(?:\w+)?\n(.*?)\n```$"
    match = re.match(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def load_documents_from_pdf(file):
    """
    Load documents from an uploaded PDF file.

    Args:
        file: Uploaded PDF file.

    Returns:
        list: List of documents extracted from the PDF.
    """
    if not file.name.endswith('.pdf'):
        raise ValueError("The uploaded file is not a PDF. Please upload a PDF file.")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file.read())
        temp_pdf_path = temp_pdf.name

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    os.remove(temp_pdf_path)
    return docs


def load_vector_store_from_path(path):
    """
    Load a FAISS vector store from a given directory path.

    Args:
        path (str): Path to the vector store directory.

    Returns:
        FAISS: Loaded vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def list_vector_store_documents():
    """
    List available vector store documents from the 'Individual_All_Vectorstores' directory.

    Returns:
        list: List of document names.
    """
    directory_path = "Individual_All_Vectorstores"
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"The directory '{directory_path}' does not exist. "
            "Run `create_and_save_individual_vector_stores()` to create it."
        )
    documents = [
        f.replace("_vectorstore", "").replace("_", " ")
        for f in os.listdir(directory_path)
        if f.endswith("_vectorstore")
    ]
    return documents


# =============================================================================
# SUMMARY AND QA FUNCTIONS
# =============================================================================
def summary_generation(api_key, uploaded_file, questions_path, prompt_path, display_placeholder):
    """
    Generate a summary from a PDF file by processing its content and performing Q&A.

    Steps:
      - Save the uploaded PDF temporarily.
      - Load and split the PDF into document chunks.
      - Create a FAISS vector store and set up a retriever.
      - Load a system prompt and initialize the language model.
      - For each question in the questions file, run retrieval and QA.
      - Update the display placeholder with results.

    Args:
        api_key (str): OpenAI API key.
        uploaded_file: Uploaded PDF file.
        questions_path (str): Path to the text file with questions.
        prompt_path (str): Path to the system prompt file.
        display_placeholder: Streamlit placeholder for displaying results.

    Returns:
        list: List of formatted Q&A results.
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # Save the uploaded PDF to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    # Load and split the PDF into documents
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    # Create a FAISS vector store for the document splits
    vectorstore = FAISS.from_documents(splits, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Load the system prompt for Q&A
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    # Create a prompt template with the system and human messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Initialize the language model (using GPT-4o in this case)
    llm = ChatOpenAI(model="gpt-4o")

    # Create the document chain and retrieval chain for Q&A
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Load questions from the provided questions file
    if os.path.exists(questions_path):
        with open(questions_path, "r") as file:
            questions = [line.strip() for line in file.readlines() if line.strip()]
    else:
        raise FileNotFoundError(f"The specified file was not found: {questions_path}")

    qa_results = []
    for question in questions:
        result = rag_chain.invoke({"input": question})
        answer = remove_code_blocks(result["answer"])
        qa_text = f"### Question: {question}\n**Answer:**\n{answer}\n"
        qa_results.append(qa_text)
        display_placeholder.markdown("\n".join(qa_results), unsafe_allow_html=True)

    # Remove the temporary PDF file
    os.remove(temp_pdf_path)

    return qa_results


def multi_plan_qa(api_key, input_text, display_placeholder):
    """
    Perform multi-plan Q&A using an existing combined vector store.

    Args:
        api_key (str): OpenAI API key.
        input_text (str): The question to ask.
        display_placeholder: Streamlit placeholder for displaying results.
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # Load the combined vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.load_local("Combined_Summary_Vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 50})

    # Load the multi-document Q&A system prompt
    prompt_path = "Prompts/multi_document_qa_system_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = rag_chain.invoke({"input": input_text})
    answer = result["answer"]
    display_placeholder.markdown(f"**Answer:**\n{answer}")


def multi_plan_qa_multi_vectorstore(api_key, input_text, display_placeholder):
    """
    Perform multi-plan Q&A using multiple individual vector stores.

    Args:
        api_key (str): OpenAI API key.
        input_text (str): The question to ask.
        display_placeholder: Streamlit placeholder for displaying results.
    """
    os.environ["OPENAI_API_KEY"] = api_key

    vectorstore_directory = "Individual_Summary_Vectorstores"
    vectorstore_names = [
        d for d in os.listdir(vectorstore_directory)
        if os.path.isdir(os.path.join(vectorstore_directory, d))
    ]

    all_retrieved_chunks = []
    for vectorstore_name in vectorstore_names:
        vectorstore_path = os.path.join(vectorstore_directory, vectorstore_name)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        retrieved_chunks = retriever.invoke(input_text)
        all_retrieved_chunks.extend(retrieved_chunks)

    # Load system prompt for multi-document QA
    prompt_path = "Prompts/multi_document_qa_system_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")

    result = question_answer_chain.invoke({
        "input": input_text,
        "context": all_retrieved_chunks
    })

    answer = result["answer"] if "answer" in result else result
    display_placeholder.markdown(f"**Answer:**\n{answer}")


def comparison_qa(api_key, focus_input, comparison_inputs, input_text, display_placeholder):
    """
    Compare a focus document against multiple comparison documents using a one-to-many query.

    Args:
        api_key (str): OpenAI API key.
        focus_input: Focus document (uploaded PDF or vector store path).
        comparison_inputs: List of comparison documents (uploaded PDFs or vector store paths).
        input_text (str): The comparison question to ask.
        display_placeholder: Streamlit placeholder for displaying results.
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # Load focus document or vector store and set up retriever
    if isinstance(focus_input, st.runtime.uploaded_file_manager.UploadedFile):
        focus_docs = load_documents_from_pdf(focus_input)
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        focus_splits = splitter.split_documents(focus_docs)
        focus_vector_store = FAISS.from_documents(focus_splits, OpenAIEmbeddings(model="text-embedding-3-large"))
    elif isinstance(focus_input, str) and os.path.isdir(focus_input):
        focus_vector_store = load_vector_store_from_path(focus_input)
    else:
        raise ValueError("Invalid focus input type. Must be a PDF file or a path to a vector store.")

    focus_retriever = focus_vector_store.as_retriever(search_kwargs={"k": 5})
    focus_docs = focus_retriever.invoke(input_text)

    # Process each comparison input
    comparison_chunks = []
    for comparison_input in comparison_inputs:
        if isinstance(comparison_input, st.runtime.uploaded_file_manager.UploadedFile):
            comparison_docs = load_documents_from_pdf(comparison_input)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
            comparison_splits = splitter.split_documents(comparison_docs)
            comparison_vector_store = FAISS.from_documents(comparison_splits, OpenAIEmbeddings(model="text-embedding-3-large"))
        elif isinstance(comparison_input, str) and os.path.isdir(comparison_input):
            comparison_vector_store = load_vector_store_from_path(comparison_input)
        else:
            raise ValueError("Invalid comparison input type. Must be a PDF file or a path to a vector store.")
        comparison_retriever = comparison_vector_store.as_retriever(search_kwargs={"k": 5})
        comparison_docs = comparison_retriever.invoke(input_text)
        comparison_chunks.extend(comparison_docs)

    combined_context = focus_docs + comparison_chunks

    # Load system prompt for comparison QA
    prompt_path = "Prompts/comparison_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    result = question_answer_chain.invoke({
        "context": combined_context,
        "input": input_text
    })

    answer = result["answer"] if "answer" in result else result
    display_placeholder.markdown(f"**Answer:**\n{answer}")


def document_qa(api_key, focus_input, user_input):
    """
    Query a single document (PDF or vector store) to answer a question.

    Args:
        api_key (str): OpenAI API key.
        focus_input: Focus document (uploaded PDF or path to vector store).
        user_input (str): The question to ask.

    Returns:
        str: The model's answer.
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # Create a vector store from the focus input
    if isinstance(focus_input, st.runtime.uploaded_file_manager.UploadedFile):
        docs = load_documents_from_pdf(focus_input)
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        splits = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-large"))
    else:
        vector_store = load_vector_store_from_path(focus_input)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieved_chunks = retriever.invoke(user_input)

    # Load the multi-document QA system prompt
    prompt_path = "Prompts/multi_document_qa_system_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    # Create a conversation RAG chain that incorporates chat history
    llm = ChatOpenAI(model="gpt-4o")
    history_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    conversation_rag_chain = create_retrieval_chain(history_retriever_chain, document_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.get("chat_history", []),
        "input": user_input,
        "context": retrieved_chunks
    })

    return response["answer"]


def comparison_qa_long_context(api_key, anthropic_api_key, input_text, focus_plan_path, selected_summaries, display_placeholder):
    """
    Compare plans using a long-context language model (e.g., Anthropic Claude).

    Args:
        api_key (str): OpenAI API key.
        anthropic_api_key (str): Anthropic API key.
        input_text (str): The comparison question to ask.
        focus_plan_path: Path to the focus plan (PDF file or file path).
        selected_summaries (list): List of selected summary filenames.
        display_placeholder: Streamlit placeholder for displaying results.
    """
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load focus document from PDF or file path
    if isinstance(focus_plan_path, st.runtime.uploaded_file_manager.UploadedFile):
        focus_docs = load_documents_from_pdf(focus_plan_path)
    elif isinstance(focus_plan_path, str):
        loader = PyPDFLoader(focus_plan_path)
        focus_docs = loader.load()
    else:
        raise ValueError("Invalid focus plan input type. Must be an uploaded file or a file path.")

    # Concatenate summaries from selected summary files
    summaries_directory = "CAPS_Summaries"
    summaries_content = ""
    for filename in selected_summaries:
        summary_filename = f"{filename.replace(' Summary', '_Summary')}.md"
        with open(os.path.join(summaries_directory, summary_filename), 'r') as file:
            summaries_content += file.read() + "\n\n"

    # Prepare focus context from the loaded documents
    focus_context = "\n\n".join([doc.page_content for doc in focus_docs])

    # Create Anthropic client and send the prompt for comparison
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    response = client.completions.create(
        model="claude-2",
        max_tokens_to_sample=1024,
        prompt=f"{input_text}\n\nFocus Document:\n{focus_context}\n\nSummaries:\n{summaries_content}"
    )

    answer = response.completion
    display_placeholder.markdown(f"**Answer:**\n{answer}", unsafe_allow_html=True)


# =============================================================================
# MAP AND PLAN UTILITIES
# =============================================================================
def generate_legend_html(region_colors):
    """
    Generate HTML code for an EPA region legend.

    Args:
        region_colors (dict): Mapping of region to color.

    Returns:
        str: HTML string for the legend.
    """
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
    Format a plan string using a state abbreviation to match vector store naming.

    Example:
      Input: "Oakland, 2020, Mitigation Primary CAP" and state_abbr "CA"
      Output: "Oakland, CA Mitigation Primary CAP 2020"

    Args:
        plan (str): Plan description.
        state_abbr (str): State abbreviation.

    Returns:
        str: Formatted plan name.
    """
    parts = [p.strip() for p in plan.split(",")]
    if len(parts) == 3:
        city, year, title = parts
        return f"{city}, {state_abbr} {title} {year}"
    return plan


def add_city_markers(map_object):
    """
    Add city markers with plan information to a Folium map.
    
    This function assumes the existence of two global DataFrames:
      - city_mapping_df: Contains 'CityName', 'StateName', 'Latitude', 'Longitude'
      - city_plans_df: Contains plan information per city and state.
    
    Args:
        map_object (folium.Map): The Folium map to add markers to.
    """
    city_markers = folium.FeatureGroup(name="City Markers", show=False)
    for _, row in city_mapping_df[['City', 'State', 'Latitude', 'Longitude']].drop_duplicates().iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        city, state = row["City"], row["State"]
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
            radius=3,
            color="darkgreen",
            fill=True,
            fill_color="darkgreen",
            fill_opacity=0.7,
            popup=popup,
            tooltip=f"{city}, {state}"
        ).add_to(city_markers)
    map_object.add_child(city_markers)


def maps_qa(api_key, user_input, extra_context, plan_list, state_abbr, epa_region):
    """
    Answer a user's question about maps and plans by retrieving document chunks from
    individual, combined, and EPA region vector stores, plus extra context and region cities.
    
    Now, the EPA region vector store is a single combined store, and an additional
    CSV file listing the cities in the region is also added as a document.
    """
    # Set API key
    os.environ["OPENAI_API_KEY"] = api_key
    all_documents = []

    # 1. Retrieve and aggregate individual plan vector store results
    individual_texts = []
    for plan in plan_list:
        formatted_plan = format_plan_name(plan, state_abbr)
        vectorstore_path = os.path.join("Individual_All_Vectorstores", formatted_plan + "_vectorstore")
        try:
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_store = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading vector store for plan '{formatted_plan}': {e}")
            continue

        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        retrieved_chunks = retriever.invoke(user_input)
        if retrieved_chunks:
            # Combine all retrieved chunks for this plan into one text
            individual_texts.append(" ".join([chunk.page_content for chunk in retrieved_chunks]))
    
    aggregated_individual_text = "\n".join(individual_texts) if individual_texts else ""
    individual_doc = Document(page_content=aggregated_individual_text)
    all_documents.append(individual_doc)

    # 2. Retrieve from combined summary vector store (using k=5 and taking the first result)
    combined_vectorstore_path = "Combined_Summary_Vectorstore"
    try:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        combined_vector_store = FAISS.load_local(combined_vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading combined vector store: {e}")
        combined_doc = Document(page_content="")
    else:
        combined_retriever = combined_vector_store.as_retriever(search_kwargs={"k": 5})
        combined_retrieved_chunks = combined_retriever.invoke(user_input)
        combined_doc = combined_retrieved_chunks[0] if combined_retrieved_chunks else Document(page_content="")
    all_documents.append(combined_doc)

    # 3. Retrieve from combined EPA region vector store
    # Note: We now expect one vector store per region.
    region_vectorstore_path = os.path.join("Combined_By_Region_Vectorstores", f"Region_{epa_region}", f"Region_{epa_region}_vectorstore")
    try:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        region_vector_store = FAISS.load_local(region_vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
        region_retriever = region_vector_store.as_retriever(search_kwargs={"k": 1})
        retrieved_chunks = region_retriever.invoke(user_input)
        aggregated_region_text = " ".join([chunk.page_content for chunk in retrieved_chunks]) if retrieved_chunks else ""
    except Exception as e:
        st.error(f"Error loading region vector store for EPA region '{epa_region}': {e}")
        aggregated_region_text = ""
    region_doc = Document(page_content=aggregated_region_text)
    all_documents.append(region_doc)

    # 4. Append extra context as its own document
    extra_context_doc = Document(page_content=extra_context)
    all_documents.append(extra_context_doc)

    # 5. Add the cities document (read CSV listing cities in the region)
    cities_csv_path = os.path.join("Combined_By_Region_Vectorstores", f"Region_{epa_region}", f"Region_{epa_region}_cities.csv")
    if os.path.exists(cities_csv_path):
        with open(cities_csv_path, "r") as f:
            cities_info = f.read()
    else:
        cities_info = f"No city information available for EPA region {epa_region}."
    cities_doc = Document(page_content=f"Cities in EPA Region {epa_region}:\n{cities_info}")
    all_documents.append(cities_doc)

    # Load the system prompt for maps QA
    prompt_path = "Prompts/maps_qa.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")
    
    # Update prompt to accept both "context" and "input"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{context}\n\nQuestion: {input}")
    ])    
    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")

    result = question_answer_chain.invoke({"input": user_input, "context": all_documents})
    
    return result["answer"] if "answer" in result else result