import os
import shutil
import argparse
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Function to create and save a combined vector store from all summary documents
def create_combined_summary_vector_store(api_key):
    # Directory containing the Markdown summaries (input directory)
    directory_path = "./CAPS_Summaries"
    os.environ["OPENAI_API_KEY"] = api_key

    # Check if the input directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Input directory '{directory_path}' did not exist and has been created. Please add your summary files.")
        return

    # List all Markdown files in the directory
    md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]

    # Load the Markdown documents
    documents = []
    for file_name in md_files:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Wrap the content in a Document object
            documents.append(Document(page_content=content))

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Define and create the output directory for the combined vector store
    combined_vector_store_dir = "Combined_Summary_Vectorstore"
    os.makedirs(combined_vector_store_dir, exist_ok=True)

    # Save the vector store locally
    vector_store.save_local(combined_vector_store_dir)
    print(f"Combined summary vector store creation complete and saved as '{combined_vector_store_dir}'.")


# Function to create and save individual vector store for a summary document
def create_individual_summary_vector_stores(api_key, summary_file_name):
    # Directory containing the Markdown summaries (input directory)
    directory_path = "./CAPS_Summaries"
    os.environ["OPENAI_API_KEY"] = api_key

    # Directory to save individual vector stores
    save_directory = "./Individual_Summary_Vectorstores"
    os.makedirs(save_directory, exist_ok=True)

    file_path = os.path.join(directory_path, summary_file_name)
    if not os.path.exists(file_path):
        print(f"Summary file {summary_file_name} not found in {directory_path}.")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Wrap the content in a Document object
        document = Document(page_content=content)

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        splits = text_splitter.split_documents([document])

        # Create embeddings and vector store for the document
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

        # Save the vector store locally with a unique name in the specified directory
        vector_store_name = os.path.join(save_directory, f"{os.path.splitext(summary_file_name)[0]}_vectorstore")
        os.makedirs(vector_store_name, exist_ok=True)  # Create destination directory if it doesn't exist
        vector_store.save_local(vector_store_name)
        print(f"Vector store for {summary_file_name} created and saved as '{vector_store_name}'.")


# Function to create and save individual vector stores for all documents in CAPS_Summaries and CAPS
def create_individual_vector_stores_for_all_documents(api_key, file_name, summary_file_name):
    # Directories containing the documents
    caps_directory = "./CAPS"
    os.environ["OPENAI_API_KEY"] = api_key

    # Directory to save individual vector stores
    save_directory = "./Individual_All_Vectorstores"
    os.makedirs(save_directory, exist_ok=True)

    # Source vector store path in Individual_Summary_Vectorstores
    source_vector_store_name = os.path.join("./Individual_Summary_Vectorstores", f"{os.path.splitext(summary_file_name)[0]}_vectorstore")
    # Destination vector store path in Individual_All_Vectorstores
    destination_vector_store_name = os.path.join(save_directory, f"{os.path.splitext(summary_file_name)[0]}_vectorstore")
    # Copy the vector store (this will create the destination directory if needed)
    shutil.copytree(source_vector_store_name, destination_vector_store_name, dirs_exist_ok=True)
    print(f"Copied vector store for {file_name} to '{destination_vector_store_name}'.")

    file_path = os.path.join(caps_directory, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_name} not found in {caps_directory}.")
        return

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Successfully loaded {file_name} from {caps_directory}.")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store for the document
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Save the vector store locally with a unique name in the specified directory
    vector_store_name = os.path.join(save_directory, f"{os.path.splitext(file_name)[0]}_vectorstore")
    os.makedirs(vector_store_name, exist_ok=True)  # Create destination directory if it doesn't exist
    vector_store.save_local(vector_store_name)
    print(f"Vector store for {file_name} created and saved as '{vector_store_name}'.")


# Run the functions to create and save the vector stores
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process vector store creation.")
    parser.add_argument("api_key", type=str, help="OpenAI API Key")
    parser.add_argument("file_name", type=str, help="Name of the file")
    parser.add_argument("summary_file_name", type=str, help="Name of the summary file")

    args = parser.parse_args()

    create_combined_summary_vector_store(args.api_key)
    create_individual_summary_vector_stores(args.api_key, args.summary_file_name)
    create_individual_vector_stores_for_all_documents(args.api_key, args.file_name, args.summary_file_name)
