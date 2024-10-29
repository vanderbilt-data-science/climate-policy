import os
import streamlit as st
from io import BytesIO
from tempfile import NamedTemporaryFile
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

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
    vector_store = FAISS.load_local("multi_plan_vectorstore", embeddings, allow_dangerous_deserialization=True)

    # Convert the vector store to a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Read the system prompt for multi-document QA
    prompt_path = "multi_document_qa_system_prompt.md"
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
    vectorstore_directory = "Individual_Vectorstores"

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
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Retrieve relevant chunks for the input text
        retrieved_chunks = retriever.invoke("input_text")
        print(retrieved_chunks)
        all_retrieved_chunks.extend(retrieved_chunks)

    # Read the system prompt for multi-document QA
    prompt_path = "multi_document_qa_system_prompt.md"
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


# Streamlit app layout with tabs
st.title("Climate Policy Analysis Tool")

# API Key Input
api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Summary Generation", "Multi-Plan QA (Shared Vectorstore)", "Multi-Plan QA (Multi-Vectorstore)"])

# First tab: Summary Generation
with tab1:
    uploaded_file = st.file_uploader("Upload a Climate Action Plan in PDF format", type="pdf")

    prompt_file_path = "summary_tool_system_prompt.md"
    questions_file_path = "summary_tool_questions.md"

    if st.button("Generate") and api_key and uploaded_file:
        display_placeholder = st.empty()

        with st.spinner("Processing..."):
            try:
                results = process_pdf(api_key, uploaded_file, questions_file_path, prompt_file_path, display_placeholder)
                
                markdown_text = "\n".join(results)
                
                # Use the uploaded file's name for the download file
                base_name = os.path.splitext(uploaded_file.name)[0]
                download_file_name = f"{base_name}_summary.md"
                
                st.download_button(
                    label="Download Results as Markdown",
                    data=markdown_text,
                    file_name=download_file_name,
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Second tab: Multi-Plan QA
with tab2:
    input_text = st.text_input("Ask a question:")
    if input_text and api_key:
        display_placeholder = st.empty()
        process_multi_plan_qa(api_key, input_text, display_placeholder)

with tab3:
    user_input = st.text_input("Ask a Question")
    if user_input and api_key:
        display_placeholder2 = st.empty()
        multi_plan_qa_multi_vectorstore(api_key, user_input, display_placeholder2)
