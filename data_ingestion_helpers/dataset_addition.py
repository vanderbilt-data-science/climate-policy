import os
import re
import csv
import argparse
from tempfile import NamedTemporaryFile
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf_to_csv(api_key, pdf_path, questions, city, state, year, plan_type):
    os.environ["OPENAI_API_KEY"] = api_key
    prompt_path = "./Prompts/dataset_tool_system_prompt.md"
    csv_file_path = "./climate_action_plans_dataset.csv"

    with open(pdf_path, "rb") as file:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file.read())
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-4o")
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    answers = [city, state, year, plan_type]
    for question in questions:
        result = rag_chain.invoke({"input": question})
        answer = result["answer"]
        answers.append(answer)

    with open(csv_file_path, "a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(answers)

    os.remove(temp_pdf_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add data to an existing CSV from a PDF.")
    parser.add_argument("api_key", type=str, help="OpenAI API Key")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")

    args = parser.parse_args()

    pattern = re.compile(r"^(.*?),\s([A-Z]{2})\s(.{3,}?)\s(\d{4})\.pdf$")
    match = pattern.match(os.path.basename(args.pdf_path))

    if match:
        city, state, plan_type, year = match.groups()
        city = city.strip()
        state = state.strip()
        year = year.strip()
        plan_type = plan_type.strip()

        questions = [
            "List 5 threats identified and discussed most often in the plan.",
            "List every single adaptation measure in the plan.",
            "List every single mitigation measure in the plan.",
            "List every single resilience measure in the plan.",
        ]

        try:
            process_pdf_to_csv(args.api_key, args.pdf_path, questions, city, state, year, plan_type)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Filename format does not match the expected pattern.")
