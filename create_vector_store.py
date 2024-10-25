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

# Run the function to create and save the vector store
if __name__ == "__main__":
    create_and_save_vector_store()