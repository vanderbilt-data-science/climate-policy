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
    # Directory containing the Markdown summaries
    directory_path = "./CAPS_Summaries"
    os.environ["OPENAI_API_KEY"] = api_key
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
        #print(f"Successfully added {file_name} to the combined vector store.")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Save the vector store locally
    vector_store.save_local("Combined_Summary_Vectorstore")
    print("Combined summary vector store creation complete and saved as 'Combined_Summary_Vectorstore'.")

# Function to create and save individual vector store for summary document
def create_individual_summary_vector_stores(api_key, summary_file_name):
    # Directory containing the Markdown summaries
    directory_path = "./CAPS_Summaries"
    os.environ["OPENAI_API_KEY"] = api_key
    # Directory to save individual vector stores
    save_directory = "./Individual_Summary_Vectorstores"

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    # Process each file individually

    file_path = os.path.join(directory_path, summary_file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Wrap the content in a Document object
        document = Document(page_content=content)

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        splits = text_splitter.split_documents([document])

        # Create embeddings and vector store for each document
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

        # Save the vector store locally with a unique name in the specified directory
        vector_store_name = os.path.join(save_directory, f"{os.path.splitext(summary_file_name)[0]}_vectorstore")
        vector_store.save_local(vector_store_name)
        #print(f"Vector store for {summary_file_name} created and saved as '{vector_store_name}'.")
    #print(f"All Individual Summary Vectorstores created.")

# Function to create and save individual vector stores for all documents in CAPS_Summaries and CAPS
def create_individual_vector_stores_for_all_documents(api_key, file_name, summary_file_name):
    # Directories containing the documents
    caps_directory = "./CAPS"
    os.environ["OPENAI_API_KEY"] = api_key
    # Directory to save individual vector stores
    save_directory = "./Individual_All_Vectorstores"

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Source vector store path in Individual_Summary_Vectorstores
    source_vector_store_name = os.path.join("./Individual_Summary_Vectorstores", f"{os.path.splitext(summary_file_name)[0]}_vectorstore")
    # Destination vector store path in Individual_All_Vectorstores
    destination_vector_store_name = os.path.join(save_directory, f"{os.path.splitext(summary_file_name)[0]}_vectorstore")
    # Copy the vector store
    shutil.copytree(source_vector_store_name, destination_vector_store_name, dirs_exist_ok=True)
    #print(f"Copied vector store for {file_name} to '{destination_vector_store_name}'.")

    file_path = os.path.join(caps_directory, file_name)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    #print(f"Successfully loaded {file_name} from CAPS.")

        # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

        # Create embeddings and vector store for each document
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

        # Save the vector store locally with a unique name in the specified directory
    vector_store_name = os.path.join(save_directory, f"{os.path.splitext(file_name)[0]}_vectorstore")
    vector_store.save_local(vector_store_name)
    #print(f"Vector store for {file_name} created and saved as '{vector_store_name}'.")
    #print(f"All Individual Vectorstores for complete and summary plans created.")

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