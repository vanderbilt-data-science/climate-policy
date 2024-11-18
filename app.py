import os
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import anthropic

# Function to remove code block markers from the answer
def remove_code_blocks(text):
    code_block_pattern = r"^```(?:\w+)?\n(.*?)\n```$"
    match = re.match(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text

# Function to process PDF, run Q&A, and return results
def process_pdf(api_key, uploaded_file, questions_path, prompt_path, display_placeholder):
    os.environ["OPENAI_API_KEY"] = api_key

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-large")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    if os.path.exists(questions_path):
        with open(questions_path, "r") as file:
            questions = [line.strip() for line in file.readlines() if line.strip()]
    else:
        raise FileNotFoundError(f"The specified file was not found: {questions_path}")

    qa_results = []
    for question in questions:
        result = rag_chain.invoke({"input": question})
        answer = result["answer"]

        answer = remove_code_blocks(answer)

        qa_text = f"### Question: {question}\n**Answer:**\n{answer}\n"
        qa_results.append(qa_text)
        display_placeholder.markdown("\n".join(qa_results), unsafe_allow_html=True)

    os.remove(temp_pdf_path)

    return qa_results

# New function to process multi-plan QA using an existing vector store
def process_multi_plan_qa(api_key, input_text, display_placeholder):
    os.environ["OPENAI_API_KEY"] = api_key

    # Load the existing vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.load_local("Combined_Summary_Vectorstore", embeddings, allow_dangerous_deserialization=True)

    # Convert the vector store to a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 50})

    # Read the system prompt for multi-document QA
    prompt_path = "Prompts/multi_document_qa_system_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the question-answering chain
    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Process the input text
    result = rag_chain.invoke({"input": input_text})
    answer = result["answer"]

    # Display the answer
    display_placeholder.markdown(f"**Answer:**\n{answer}")

def multi_plan_qa_multi_vectorstore(api_key, input_text, display_placeholder):
    os.environ["OPENAI_API_KEY"] = api_key

    # Directory containing individual vector stores
    vectorstore_directory = "Individual_Summary_Vectorstores"

    # List all vector store directories
    vectorstore_names = [d for d in os.listdir(vectorstore_directory) if os.path.isdir(os.path.join(vectorstore_directory, d))]

    # Initialize a list to collect all retrieved chunks
    all_retrieved_chunks = []

    # Process each vector store
    for vectorstore_name in vectorstore_names:
        vectorstore_path = os.path.join(vectorstore_directory, vectorstore_name)

        # Load the vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        # Convert the vector store to a retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        # Retrieve relevant chunks for the input text
        retrieved_chunks = retriever.invoke("input_text")
        all_retrieved_chunks.extend(retrieved_chunks)

    # Read the system prompt for multi-document QA
    prompt_path = "Prompts/multi_document_qa_system_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the question-answering chain
    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")

    # Process the combined context
    result = question_answer_chain.invoke({"input": input_text, "context": all_retrieved_chunks})

    # Display the answer
    display_placeholder.markdown(f"**Answer:**\n{result}")


# Function to compare document via one-to-many query approach
def process_one_to_many_query(api_key, focus_input, comparison_inputs, input_text, display_placeholder):
    os.environ["OPENAI_API_KEY"] = api_key

    def load_documents_from_pdf(file):
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file.read())
            temp_pdf_path = temp_pdf.name

        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        os.remove(temp_pdf_path)
        return docs

    def load_vector_store_from_path(path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    # Load focus documents or vector store
    if isinstance(focus_input, st.runtime.uploaded_file_manager.UploadedFile):
        focus_docs = load_documents_from_pdf(focus_input)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        focus_splits = text_splitter.split_documents(focus_docs)
        focus_vector_store = FAISS.from_documents(focus_splits, OpenAIEmbeddings(model="text-embedding-3-large"))
        focus_retriever = focus_vector_store.as_retriever(search_kwargs={"k": 5})
    elif isinstance(focus_input, str) and os.path.isdir(focus_input):
        focus_vector_store = load_vector_store_from_path(focus_input)
        focus_retriever = focus_vector_store.as_retriever(search_kwargs={"k": 5})
    else:
        raise ValueError("Invalid focus input type. Must be a PDF file or a path to a vector store.")

    focus_docs = focus_retriever.invoke(input_text)

    comparison_chunks = []
    for comparison_input in comparison_inputs:
        if isinstance(comparison_input, st.runtime.uploaded_file_manager.UploadedFile):
            comparison_docs = load_documents_from_pdf(comparison_input)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
            comparison_splits = text_splitter.split_documents(comparison_docs)
            comparison_vector_store = FAISS.from_documents(comparison_splits, OpenAIEmbeddings(model="text-embedding-3-large"))
            comparison_retriever = comparison_vector_store.as_retriever(search_kwargs={"k": 5})
        elif isinstance(comparison_input, str) and os.path.isdir(comparison_input):
            comparison_vector_store = load_vector_store_from_path(comparison_input)
            comparison_retriever = comparison_vector_store.as_retriever(search_kwargs={"k": 5})
        else:
            raise ValueError("Invalid comparison input type. Must be a PDF file or a path to a vector store.")

        comparison_docs = comparison_retriever.invoke(input_text)
        comparison_chunks.extend(comparison_docs)

    # Construct the combined context
    combined_context = (
        focus_docs +
        comparison_chunks
    )

    # Read the system prompt
    prompt_path = "Prompts/comparison_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # Create the question-answering chain
    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(
        llm, 
        prompt,
        document_variable_name="context"
    )

    # Process the combined context
    result = question_answer_chain.invoke({
        "context": combined_context,
        "input": input_text
    })

    # Display the answer
    display_placeholder.markdown(f"**Answer:**\n{result}")

# Function to list vector store documents
def list_vector_store_documents():
    # Assuming documents are stored in the "Individual_All_Vectorstores" directory
    directory_path = "Individual_All_Vectorstores"
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist. Run `create_and_save_individual_vector_stores()` to create it.")
    # List all available vector stores by document name
    documents = [f.replace("_vectorstore", "").replace("_", " ") for f in os.listdir(directory_path) if f.endswith("_vectorstore")]
    return documents

def compare_with_long_context(api_key, anthropic_api_key, input_text, focus_plan_path, focus_city_name, selected_summaries, display_placeholder):
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load the focus plan
    focus_docs = []
    if focus_plan_path.endswith('.pdf'):
        focus_loader = PyPDFLoader(focus_plan_path)
        focus_docs = focus_loader.load()
    elif focus_plan_path.endswith('.md'):
        focus_loader = TextLoader(focus_plan_path)
        focus_docs = focus_loader.load()
    else:
        raise ValueError("Unsupported file format for focus plan.")

    # Concatenate selected summary documents
    summaries_directory = "CAPS_Summaries"
    summaries_content = ""
    for filename in selected_summaries:
        with open(os.path.join(summaries_directory, filename), 'r') as file:
            summaries_content += file.read() + "\n\n"

    # Prepare the context
    focus_context = "\n\n".join([doc.page_content for doc in focus_docs])

    # Create the client and message
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{input_text}\n\nFocus Document:\n{focus_context}\n\nSummaries:\n{summaries_content}"}
        ]
    )

    # Display the answer
    display_placeholder.markdown(f"**Answer:**\n{message.content}", unsafe_allow_html=True)


# Streamlit app layout with tabs
st.title("Climate Policy Analysis Tool")

# API Key Input
api_key = st.text_input("Enter your OpenAI API key:", type="password", key="openai_key")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary Generation", "Multi-Plan QA (Shared Vectorstore)", "Multi-Plan QA (Multi-Vectorstore)", "Plan Comparison Tool", "Plan Comparison with Long Context Model"])

# First tab: Summary Generation
with tab1:
    uploaded_file = st.file_uploader("Upload a Climate Action Plan in PDF format", type="pdf", key="upload_file")

    prompt_file_path = "Prompts/summary_tool_system_prompt.md"
    questions_file_path = "Prompts/summary_tool_questions.md"

    if st.button("Generate", key="generate_button") and api_key and uploaded_file:
        display_placeholder = st.empty()

        with st.spinner("Processing..."):
            try:
                results = process_pdf(api_key, uploaded_file, questions_file_path, prompt_file_path, display_placeholder)
                
                markdown_text = "\n".join(results)
                
                # Use the uploaded file's name for the download file
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

# Second tab: Multi-Plan QA
with tab2:
    input_text = st.text_input("Ask a question:", key="multi_plan_input")
    if input_text and api_key:
        display_placeholder2 = st.empty()
        process_multi_plan_qa(api_key, input_text, display_placeholder2)

with tab3:
    user_input = st.text_input("Ask a Question", key="multi_vectorstore_input")
    if user_input and api_key:
        display_placeholder3 = st.empty()
        multi_plan_qa_multi_vectorstore(api_key, user_input, display_placeholder3)

# Fourth tab: Plan Comparison Tool
with tab4:
    st.header("Plan Comparison Tool")

    # List of documents from vector stores
    vectorstore_documents = list_vector_store_documents()

    # Option to upload a new plan or select from existing vector stores
    focus_option = st.radio("Choose a focus plan:", ("Select from existing vector stores", "Upload a new plan"), key="focus_option")

    if focus_option == "Upload a new plan":
        focus_uploaded_file = st.file_uploader("Upload a Climate Action Plan to compare", type="pdf", key="focus_upload")
        focus_city_name = st.text_input("Enter the city name for the uploaded plan:", key="focus_city_name")
        if focus_uploaded_file is not None and focus_city_name:
            # Directly use the uploaded file
            focus_input = focus_uploaded_file
        else:
            focus_input = None
    else:
        # Select a focus plan from existing vector stores
        selected_focus_plan = st.selectbox("Select a focus plan:", vectorstore_documents, key="select_focus_plan")
        focus_input = os.path.join("Individual_All_Vectorstores", f"{selected_focus_plan}_vectorstore")
        focus_city_name = selected_focus_plan.replace("_", " ")

    # Option to upload comparison documents or select from existing vector stores
    comparison_option = st.radio("Choose comparison documents:", ("Select from existing vector stores", "Upload new documents"), key="comparison_option")

    if comparison_option == "Upload new documents":
        comparison_files = st.file_uploader("Upload comparison documents", type="pdf", accept_multiple_files=True, key="comparison_files")
        comparison_inputs = comparison_files
    else:
        # Select comparison documents from existing vector stores
        selected_comparison_plans = st.multiselect("Select comparison documents:", vectorstore_documents, key="select_comparison_plans")
        comparison_inputs = [os.path.join("Individual_All_Vectorstores", f"{doc}_vectorstore") for doc in selected_comparison_plans]

    input_text = st.text_input("Ask a comparison question:", key="comparison_input")

    if st.button("Compare", key="compare_button") and api_key and input_text and focus_input and comparison_inputs:
        display_placeholder4 = st.empty()
        with st.spinner("Processing..."):
            try:
                # Call the process_one_to_many_query function
                process_one_to_many_query(api_key, focus_input, comparison_inputs, input_text, display_placeholder4)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Fifth tab: Plan Comparison with Long Context Model
with tab5:
    st.header("Plan Comparison with Long Context Model")

    # Anthropics API Key Input
    anthropic_api_key = st.text_input("Enter your Anthropic API key:", type="password", key="anthropic_key")

    # Option to upload a new plan or select from a list
    upload_option = st.radio("Choose a focus plan:", ("Select from existing plans", "Upload a new plan"), key="upload_option_long_context")

    if upload_option == "Upload a new plan":
        focus_uploaded_file = st.file_uploader("Upload a Climate Action Plan to compare", type="pdf", key="focus_upload_long_context")
        focus_city_name = st.text_input("Enter the city name for the uploaded plan:", key="focus_city_name_long_context")
        if focus_uploaded_file is not None and focus_city_name:
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(focus_uploaded_file.read())
                focus_plan_path = temp_pdf.name
        else:
            focus_plan_path = None
    else:
        # List of existing plans in CAPS
        plan_list = [f.replace(".pdf", "") for f in os.listdir("CAPS") if f.endswith('.pdf')]
        selected_plan = st.selectbox("Select a plan:", plan_list, key="selected_plan_long_context")
        focus_plan_path = os.path.join("CAPS", selected_plan)
        # Extract city name from the file name
        focus_city_name = os.path.splitext(selected_plan)[0].replace("_", " ")

    # List available summary documents for selection
    summaries_directory = "CAPS_Summaries"
    summary_files = [f.replace(".md", "").replace("_", " ") for f in os.listdir(summaries_directory) if f.endswith('.md')]
    selected_summaries = st.multiselect("Select summary documents for comparison:", summary_files, key="selected_summaries")

    input_text = st.text_input("Ask a comparison question:", key="comparison_input_long_context")

    if st.button("Compare with Long Context", key="compare_button_long_context") and api_key and anthropic_api_key and input_text and focus_plan_path and focus_city_name:
        display_placeholder = st.empty()
        with st.spinner("Processing..."):
            try:
                compare_with_long_context(api_key, anthropic_api_key, input_text, focus_plan_path, focus_city_name, selected_summaries, display_placeholder)
            except Exception as e:
                st.error(f"An error occurred: {e}")