import os
import re
import csv
import pandas as pd
from tempfile import NamedTemporaryFile
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf_to_csv(api_key, pdf_path, questions, prompt_path, csv_writer, city, state, year, plan_type):
    os.environ["OPENAI_API_KEY"] = api_key

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

    csv_writer.writerow(answers)
    os.remove(temp_pdf_path)


def main():
    directory_path = input("Enter the path to the folder containing the PDF plans: ").strip()
    api_key = input("Enter your OpenAI API key: ").strip()
    prompt_file_path = "Prompts/dataset_tool_system_prompt.md"

    questions = [
        "List 5 threats identified and discussed most often in the plan.",
        "List every single adaptation measure in the plan.",
        "List every single mitigation measure in the plan.",
        "List every single resilience measure in the plan.",
    ]

    output_file_path = "climate_action_plans_dataset.csv"
    with open(output_file_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["City Name", "State Name", "Year", "Plan Type", "Threats", "Adaptation Measures", "Mitigation Measures", "Resilience Measures"])

        pattern = re.compile(r"^(.*?),\s([A-Z]{2})\s(.{3,}?)\s(\d{4})\.pdf$")

        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                match = pattern.match(filename)
                if match:
                    city, state, plan_type, year = match.groups()
                    pdf_path = os.path.join(directory_path, filename)
                    print(f"Processing {filename}...")

                    try:
                        process_pdf_to_csv(api_key, pdf_path, questions, prompt_file_path, csv_writer, city.strip(), state, year, plan_type.strip())
                        print(f"Data for {filename} added to dataset.")
                    except Exception as e:
                        print(f"An error occurred while processing {filename}: {e}")


if __name__ == "__main__":
    main()
