import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # Import the Document class
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("Provide OpenAI API Key:")

# Load and process the documents
def create_and_save_vector_store():
    # Directory containing the Markdown summaries
    directory_path = "CAPS_Summaries"

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
        print(f"Successfully added {file_name} to the vector store.")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Save the vector store locally
    vector_store.save_local("multi_plan_vectorstore")
    print("Vector store creation complete and saved as 'multi_plan_vectorstore'.")


def create_and_save_individual_vector_stores():
    # Directory containing the Markdown summaries
    directory_path = "CAPS_Summaries"
    # Directory to save individual vector stores
    save_directory = "Individual_Vectorstores"

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # List all Markdown files in the directory
    md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]

    # Process each file individually
    for file_name in md_files:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Wrap the content in a Document object
            document = Document(page_content=content)
            print(f"Successfully loaded {file_name}.")

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        splits = text_splitter.split_documents([document])

        # Create embeddings and vector store for each document
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

        # Save the vector store locally with a unique name in the specified directory
        vector_store_name = os.path.join(save_directory, f"{os.path.splitext(file_name)[0]}_vectorstore")
        vector_store.save_local(vector_store_name)
        print(f"Vector store for {file_name} created and saved as '{vector_store_name}'.")

# Run the function to create and save the vector stores
if __name__ == "__main__":
    create_and_save_vector_store()
    create_and_save_individual_vector_stores()

# create_vector_store.py
def list_vector_store_documents():
    # Assuming documents are stored in the "Individual_Vectorstores" directory
    directory_path = "Individual_Vectorstores"
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist. Run `create_and_save_individual_vector_stores()` to create it.")
    # List all available vector stores by document name
    documents = [f.replace("_vectorstore", "") for f in os.listdir(directory_path) if f.endswith("_vectorstore")]
    return documents