import os
import re
import streamlit as st
from tempfile import NamedTemporaryFile


# Import necessary modules from LangChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# Function to remove code block markers from the answer
def remove_code_blocks(text):
    """
    Removes code block markers from the answer text.

    Args:
        text (str): The text from which code block markers should be removed.

    Returns:
        str: The text without code block markers.
    """
    code_block_pattern = r"^```(?:\w+)?\n(.*?)\n```$"
    match = re.match(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text

# Function to process PDF, run Q&A, and return results
def process_pdf(api_key, uploaded_file, questions_path, prompt_path, display_placeholder):
    """
    Processes a PDF file, runs Q&A, and returns the results.

    Args:
        api_key (str): OpenAI API key.
        uploaded_file: Uploaded PDF file.
        questions_path (str): Path to the questions file.
        prompt_path (str): Path to the system prompt file.
        display_placeholder: Streamlit placeholder for displaying results.

    Returns:
        list: List of QA results.
    """
    # Set the OpenAI API key
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

    # Create a vector store from the documents
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Load the system prompt
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

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o")

    # Create the question-answering chain
    question_answer_chain = create_stuff_documents_chain(
        llm, prompt, document_variable_name="context"
    )
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Load the questions
    if os.path.exists(questions_path):
        with open(questions_path, "r") as file:
            questions = [line.strip() for line in file.readlines() if line.strip()]
    else:
        raise FileNotFoundError(f"The specified file was not found: {questions_path}")

    # Process each question
    qa_results = []
    for question in questions:
        result = rag_chain.invoke({"input": question})
        answer = result["answer"]

        # Remove code block markers
        answer = remove_code_blocks(answer)

        qa_text = f"### Question: {question}\n**Answer:**\n{answer}\n"
        qa_results.append(qa_text)
        display_placeholder.markdown("\n".join(qa_results), unsafe_allow_html=True)

    # Clean up temporary PDF file
    os.remove(temp_pdf_path)

    return qa_results



# Function to perform multi-plan QA using an existing vector store
def process_multi_plan_qa(api_key, input_text, display_placeholder):
    """
    Performs multi-plan QA using an existing shared vector store.

    Args:
        api_key (str): OpenAI API key.
        input_text (str): The question to ask.
        display_placeholder: Streamlit placeholder for displaying results.
    """
    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key

    # Load the existing vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.load_local(
        "Combined_Summary_Vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

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
    question_answer_chain = create_stuff_documents_chain(
        llm, prompt, document_variable_name="context"
    )
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Process the input text
    result = rag_chain.invoke({"input": input_text})
    answer = result["answer"]

    # Display the answer
    display_placeholder.markdown(f"**Answer:**\n{answer}")

# Function to perform multi-plan QA using multiple individual vector stores
def process_multi_plan_qa_multi_vectorstore(api_key, input_text, display_placeholder):
    """
    Performs multi-plan QA using multiple individual vector stores.

    Args:
        api_key (str): OpenAI API key.
        input_text (str): The question to ask.
        display_placeholder: Streamlit placeholder for displaying results.
    """
    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key

    # Directory containing individual vector stores
    vectorstore_directory = "Individual_Summary_Vectorstores"

    # List all vector store directories
    vectorstore_names = [
        d for d in os.listdir(vectorstore_directory)
        if os.path.isdir(os.path.join(vectorstore_directory, d))
    ]

    # Initialize a list to collect all retrieved chunks
    all_retrieved_chunks = []

    # Process each vector store
    for vectorstore_name in vectorstore_names:
        vectorstore_path = os.path.join(vectorstore_directory, vectorstore_name)

        # Load the vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Convert the vector store to a retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        # Retrieve relevant chunks for the input text
        retrieved_chunks = retriever.invoke(input_text)
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
    question_answer_chain = create_stuff_documents_chain(
        llm, prompt, document_variable_name="context"
    )

    # Process the combined context
    result = question_answer_chain.invoke({
        "input": input_text,
        "context": all_retrieved_chunks
    })

    # Display the answer
    answer = result["answer"] if "answer" in result else result
    display_placeholder.markdown(f"**Answer:**\n{answer}")

def load_documents_from_pdf(file):
    """
    Loads documents from a PDF file.

    Args:
        file: Uploaded PDF file.

    Returns:
        list: List of documents.
    """
    # Check if the file is a PDF
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
    Loads a vector store from a given path.

    Args:
        path (str): Path to the vector store.

    Returns:
        FAISS: Loaded vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

# Function to compare documents via one-to-many query approach
def process_one_to_many_query(api_key, focus_input, comparison_inputs, input_text, display_placeholder):
    """
    Compares a focus document against multiple comparison documents using a one-to-many query approach.

    Args:
        api_key (str): OpenAI API key.
        focus_input: Focus document (uploaded file or path to vector store).
        comparison_inputs: List of comparison documents (uploaded files or paths to vector stores).
        input_text (str): The comparison question to ask.
        display_placeholder: Streamlit placeholder for displaying results.
    """
    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key
    print(comparison_inputs)
    # Load focus documents or vector store
    if isinstance(focus_input, st.runtime.uploaded_file_manager.UploadedFile):
        # If focus_input is an uploaded PDF file
        focus_docs = load_documents_from_pdf(focus_input)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        focus_splits = text_splitter.split_documents(focus_docs)
        focus_vector_store = FAISS.from_documents(
            focus_splits,
            OpenAIEmbeddings(model="text-embedding-3-large")
        )
        focus_retriever = focus_vector_store.as_retriever(search_kwargs={"k": 5})
    elif isinstance(focus_input, str) and os.path.isdir(focus_input):
        # If focus_input is a path to a vector store
        focus_vector_store = load_vector_store_from_path(focus_input)
        focus_retriever = focus_vector_store.as_retriever(search_kwargs={"k": 5})
    else:
        raise ValueError("Invalid focus input type. Must be a PDF file or a path to a vector store.")


    # Retrieve relevant chunks from the focus document
    focus_docs = focus_retriever.invoke(input_text)

    # Initialize list to collect comparison chunks
    comparison_chunks = []
    for comparison_input in comparison_inputs:
        if isinstance(comparison_input, st.runtime.uploaded_file_manager.UploadedFile):
            # If comparison_input is an uploaded PDF file
            comparison_docs = load_documents_from_pdf(comparison_input)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
            comparison_splits = text_splitter.split_documents(comparison_docs)
            comparison_vector_store = FAISS.from_documents(
                comparison_splits,
                OpenAIEmbeddings(model="text-embedding-3-large")
            )
            comparison_retriever = comparison_vector_store.as_retriever(search_kwargs={"k": 5})
        elif isinstance(comparison_input, str) and os.path.isdir(comparison_input):
            # If comparison_input is a path to a vector store
            comparison_vector_store = load_vector_store_from_path(comparison_input)
            comparison_retriever = comparison_vector_store.as_retriever(search_kwargs={"k": 5})
        else:
            raise ValueError("Invalid comparison input type. Must be a PDF file or a path to a vector store.")

        # Retrieve relevant chunks from the comparison document
        comparison_docs = comparison_retriever.invoke(input_text)
        comparison_chunks.extend(comparison_docs)

    # Construct the combined context
    combined_context = focus_docs + comparison_chunks

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
    answer = result["answer"] if "answer" in result else result
    display_placeholder.markdown(f"**Answer:**\n{answer}")

# Function to list vector store documents
def list_vector_store_documents():
    """
    Lists available vector store documents.

    Returns:
        list: List of document names.
    """
    # Assuming documents are stored in the "Individual_All_Vectorstores" directory
    directory_path = "Individual_All_Vectorstores"
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"The directory '{directory_path}' does not exist. "
            "Run `create_and_save_individual_vector_stores()` to create it."
        )
    # List all available vector stores by document name
    documents = [
        f.replace("_vectorstore", "").replace("_", " ")
        for f in os.listdir(directory_path)
        if f.endswith("_vectorstore")
    ]
    return documents


# Streamlit app layout with tabs
st.title("Climate Policy Analysis Tool")

# API Key Input
api_key = st.text_input("Enter your OpenAI API key:", type="password", key="openai_key")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Summary Generation",
    "Multi-Plan QA (Shared Vectorstore)",
    "Multi-Plan QA (Multi-Vectorstore)",
    "Plan Comparison Tool",
    "Plan Analysis with Conversation History"
])

# First tab: Summary Generation
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a Climate Action Plan in PDF format",
        type="pdf",
        key="upload_file"
    )

    prompt_file_path = "Prompts/summary_tool_system_prompt.md"
    questions_file_path = "Prompts/summary_tool_questions.md"

    if st.button("Generate", key="generate_button"):
        if not api_key:
            st.warning("Please provide your OpenAI API key.")
        elif not uploaded_file:
            st.warning("Please upload a PDF file.")
        else:
            display_placeholder = st.empty()
            with st.spinner("Processing..."):
                try:
                    results = process_pdf(
                        api_key,
                        uploaded_file,
                        questions_file_path,
                        prompt_file_path,
                        display_placeholder
                    )
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

# Second tab: Multi-Plan QA (Shared Vectorstore)
with tab2:
    input_text = st.text_input("Ask a question:", key="multi_plan_input")
    if st.button("Ask", key="multi_plan_qa_button"):
        if not api_key:
            st.warning("Please provide your OpenAI API key.")
        elif not input_text:
            st.warning("Please enter a question.")
        else:
            display_placeholder2 = st.empty()
            with st.spinner("Processing..."):
                try:
                    process_multi_plan_qa(
                        api_key,
                        input_text,
                        display_placeholder2
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Third tab: Multi-Plan QA (Multi-Vectorstore)
with tab3:
    user_input = st.text_input("Ask a question:", key="multi_vectorstore_input")
    if st.button("Ask", key="multi_vectorstore_qa_button"):
        if not api_key:
            st.warning("Please provide your OpenAI API key.")
        elif not user_input:
            st.warning("Please enter a question.")
        else:
            display_placeholder3 = st.empty()
            with st.spinner("Processing..."):
                try:
                    process_multi_plan_qa_multi_vectorstore(
                        api_key,
                        user_input,
                        display_placeholder3
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Fourth tab: Plan Comparison Tool
with tab4:
    st.header("Plan Comparison Tool")

    # List of documents from vector stores
    vectorstore_documents = list_vector_store_documents()

    # Option to upload a new plan or select from existing vector stores
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
        if focus_uploaded_file is not None:
            # Directly use the uploaded file
            focus_input = focus_uploaded_file
        else:
            focus_input = None
    else:
        # Select a focus plan from existing vector stores
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
        # Select comparison documents from existing vector stores
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

    input_text = st.text_input(
        "Ask a comparison question:",
        key="comparison_input"
    )

    if st.button("Compare", key="compare_button"):
        if not api_key:
            st.warning("Please provide your OpenAI API key.")
        elif not input_text:
            st.warning("Please enter a comparison question.")
        elif not focus_input:
            st.warning("Please provide a focus plan.")
        elif not comparison_inputs:
            st.warning("Please provide comparison documents.")
        else:
            display_placeholder4 = st.empty()
            with st.spinner("Processing..."):
                try:
                    # Call the process_one_to_many_query function
                    process_one_to_many_query(
                        api_key,
                        focus_input,
                        comparison_inputs,
                        input_text,
                        display_placeholder4
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")











# Fifth tab: Plan Analysis with Conversation History

# Function to perform question and answering with a document
def question_answer(api_key, focus_input, user_input):
    """
    Queries a single document (uploaded file or vector store) to answer a question.

    Args:
        api_key (str): OpenAI API key.
        focus_input: Focus document (uploaded file or path to vector store).
        user_input (str): The question to ask.

    Returns:
        str: The model's answer to the input question.
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # Create retriever
    if isinstance(focus_input, st.runtime.uploaded_file_manager.UploadedFile):
        docs = load_documents_from_pdf(focus_input)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        splits = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-large"))
    else:
        vector_store = load_vector_store_from_path(focus_input)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        # Retrieve relevant chunks for the input text
        retrieved_chunks = retriever.invoke(input_text)    

    # Load system prompt
    prompt_path = "Prompts/multi_document_qa_system_prompt.md"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            system_prompt = file.read()
    else:
        raise FileNotFoundError(f"The specified file was not found: {prompt_path}")

    # Create conversation RAG chain
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

    # Invoke the model
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.get("chat_history", []),
        "input": user_input,
        "context": retrieved_chunks
    })

    return response["answer"]

# Function to list vector store documents
def list_vector_store_documents():
    """
    Lists available vector store documents.

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

# Fifth Tab: Plan Analysis with Conversation History
with tab5:
    st.header("Document Q&A Tool")

    # List of documents from vector stores
    vectorstore_documents = list_vector_store_documents()

    # Option to upload a new plan or select from existing vector stores
    focus_option = st.radio(
        "Choose a focus plan:",
        ("Select from existing vector stores", "Upload a new plan"),
        key="focus_option_qa"
    )

    if focus_option == "Upload a new plan":
        focus_uploaded_file = st.file_uploader(
            "Upload a Climate Action Plan to compare",
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


    
    #Display previous questions and inputs
    if "chat_history" in st.session_state:
        for message in st.session_state.chat_history:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            st.chat_message(role).markdown(message.content)

    user_input = st.chat_input("Ask a question") 
    
    # Chat input for new question
    if user_input:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        if api_key and focus_input:
            st.session_state.chat_history.append(HumanMessage(content=user_input))  # User input
            st.chat_message("user").markdown(user_input) #display User input first
            
            with st.spinner("Processing..."):
                #generating response with AI
                answer = question_answer(api_key, focus_input, user_input)

                # Display the new AI response
                st.session_state.chat_history.append(AIMessage(content=answer))
                # Append AI response to chat history
                st.chat_message("assistant").markdown(answer)
     
        else:
            st.warning("Please provide your OpenAI API key and select a focus plan.")


    
    
           



    
                    
                                                      



