import os
import pandas as pd
import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_region_vectorstores(api_key):
    # Set the OpenAI API key in the environment
    os.environ["OPENAI_API_KEY"] = api_key

    # === Configuration ===
    # Folder containing individual vector stores
    input_folder = "Individual_All_Vectorstores"
    # Folder to output the combined regional vector stores
    output_folder = "Combined_By_Region_Vectorstores"
    # CSV file with EPA region mapping
    epa_csv_path = "epa_regions.csv"

    # === Step 1: Load EPA Region Mapping ===
    epa_df = pd.read_csv(epa_csv_path, skipinitialspace=True)
    state_abbr_to_region = {
        str(s).strip().strip('"'): region for s, region in zip(epa_df['States'], epa_df['Region'])
    }
    print("EPA Mapping keys:", list(state_abbr_to_region.keys()))

    # === Step 2: Group individual vector store directories by EPA region ===
    # Process only directories ending with "_vectorstore" but skip those that are summary stores.
    region_to_store_paths = {}
    cities_by_region = {}

    for fname in os.listdir(input_folder):
        # Skip if the folder does not end with '_vectorstore'
        if not fname.endswith("_vectorstore"):
            continue

        # Explicitly skip any summary vector stores
        if fname.endswith("_Summary_vectorstore"):
            continue

        try:
            # Expected filename format: "City, ST PlanName_vectorstore"
            parts = fname.split(", ")
            if len(parts) < 2:
                print(f"⚠️ Unexpected filename format: {fname}")
                continue

            city = parts[0].strip()
            state_rest = parts[1]
            state_abbr = state_rest.split(" ")[0].strip()

            # Determine the EPA region from the state abbreviation
            region = state_abbr_to_region.get(state_abbr)
            if region is None:
                print(f"⚠️ State abbreviation '{state_abbr}' not found in EPA mapping for file: {fname}")
                continue

            full_path = os.path.join(input_folder, fname)
            region_to_store_paths.setdefault(region, []).append(full_path)
            cities_by_region.setdefault(region, set()).add(city)

        except Exception as e:
            print(f"❌ Failed to parse filename: {fname}, error: {e}")

    # === Step 3: Create combined vector store and cities CSV for each region ===
    os.makedirs(output_folder, exist_ok=True)

    for region, paths in region_to_store_paths.items():
        region_dir = os.path.join(output_folder, f"Region_{region}")
        os.makedirs(region_dir, exist_ok=True)
        print(f"🔄 Combining {len(paths)} vector stores for EPA Region {region}")

        all_documents = []
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        for store_path in paths:
            try:
                vector_store = FAISS.load_local(store_path, embedding_model, allow_dangerous_deserialization=True)
                # Extract stored documents using the underlying InMemoryDocstore's internal dictionary.
                docs = list(vector_store.docstore._dict.values())
                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ Failed to load or extract documents from '{store_path}' for region {region}: {e}")

        if all_documents:
            combined_vector_store = FAISS.from_documents(all_documents, embedding_model)
            combined_store_path = os.path.join(region_dir, f"Region_{region}_vectorstore")
            combined_vector_store.save_local(combined_store_path)
            print(f"✅ Created combined vector store for Region {region} at {combined_store_path}")
        else:
            print(f"⚠️ No documents found for Region {region}, skipping vector store creation.")

        # Create a CSV file listing the cities in this region
        cities = sorted(cities_by_region.get(region, []))
        cities_df = pd.DataFrame(cities, columns=["City"])
        cities_csv_path = os.path.join(region_dir, f"Region_{region}_cities.csv")
        cities_df.to_csv(cities_csv_path, index=False)
        print(f"✅ Created cities CSV file for Region {region} at {cities_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create combined EPA Region vector stores")
    parser.add_argument("api_key", type=str, help="OpenAI API Key")
    args = parser.parse_args()

    create_region_vectorstores(args.api_key)
