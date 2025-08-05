#!/usr/bin/env python3
import json
import os
import logging
from ollama import chat
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer  # New import

# ------------------------------------------------------------------------------
# 2. Logger, Pydantic Models, and Prompt File Loading (unchanged)
# ------------------------------------------------------------------------------
logger = logging.getLogger("DualLogger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("log.txt", mode="a")  # Use "w" to overwrite each run

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ------------------------------------------------------------------------------
# 1. Initialize the Embedding Model from SentenceTransformers
# ------------------------------------------------------------------------------
# This model is popular for its robustness. It will use a GPU if available.
embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

def get_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for a given text using SentenceTransformers.
    The model will leverage GPU acceleration if available.
    """
    return embedding_model.encode(text, convert_to_numpy=True)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_top_k(query: str, index: list, k: int = 1) -> list:
    # Filter index for URS entries only (i.e., doc_id ending with "_URS")
    urs_entries = [item for item in index if item["doc_id"].endswith("_URS")]
    query_embedding = get_embedding(query)
    scored_items = []
    for item in urs_entries:
        sim = cosine_similarity(query_embedding, item["embedding"])
        scored_items.append((sim, item))
    scored_items.sort(key=lambda x: x[0], reverse=True)
    top_urs = scored_items[:k]
    results = []
    for sim, item in top_urs:
        formatted = f"User Requirement: {item['text']}\n"
        for idx, srs_title in enumerate(item.get("srs_titles", []), start=1):
            formatted += f"     System Requirement {idx}: {srs_title}\n"
        results.append(formatted)
    return results

def build_embedding_index(data: dict) -> list:
    index = []
    for req_id, req_info in data.items():
        if "URS_Description" in req_info:
            text = req_info["URS_Description"]
            srs_titles = []
            srs_data = req_info.get("SRS_Data", {})
            for srs_id, srs_info in srs_data.items():
                if "SRS_Title" in srs_info:
                    srs_titles.append(srs_info["SRS_Title"])
            index.append({
                "doc_id": req_id + "_URS",
                "text": text,
                "embedding": get_embedding(text),
                "srs_titles": srs_titles
            })
    return index



# Define the structured output schema using Pydantic.
class GeneratedSystemRequirement(BaseModel):
    GeneratedSystemRequirementId: str
    GeneratedSystemRequirement: str

class UserSystemRequirement(BaseModel):
    UserRequirementId: str
    GeneratedSystemRequirements: list[GeneratedSystemRequirement]

class OutputFormat(BaseModel):
    AllRequirements: list[UserSystemRequirement]

# Load initial explanation files.
script_dir = os.path.dirname(os.path.abspath(__file__))
explanation_files = [
    os.path.join(script_dir, "Prompts", "User-System", "Prompt-4.txt")
]
initial_explanations = []
for file in explanation_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            initial_explanations.append(f.read())
    else:
        logger.warning(f"Explanation file not found: {file}")

# ------------------------------------------------------------------------------
# 3. Load JSON Input and Build the Embedding Index
# ------------------------------------------------------------------------------
# Read input JSON file containing requirements.
input_file = os.path.join(script_dir, "Extracted Data", "srs_data_v3.json")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create the embedding index for retrieval.
# Load the JSON file for building the embedding index (embeddingExample.json)
embedding_file = os.path.join(script_dir, "Extracted Data", "Investigator_Requirements.json")
with open(embedding_file, "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# Build the embedding index from the embedding data.
embedding_index = build_embedding_index(embedding_data)
logger.info("Built embedding index with {} documents.".format(len(embedding_index)))

# Also, separate requirements based on IDs for later batch processing.
neuro_requirements_list = []
mri_requirements_list = []
for req_id, req_data in data.items():
    description = req_data.get("URS_Description", "")
    formatted_requirement = (
        f"User Requirement Id: {req_id}:\n"
        f"     Description: {description}\n"
    )
    if "cNeuro" in req_id:
        neuro_requirements_list.append(formatted_requirement)
    elif "cMRI" in req_id:
        mri_requirements_list.append(formatted_requirement)

# ------------------------------------------------------------------------------
# 4. Main Loop: Integrate RAG into Batch Processing
# ------------------------------------------------------------------------------
# Define the models to test.
model_paths = {
    "Llama 3.2 3B": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:F16",
    "Mistral 24B": "hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:F16",
    "Llama 3.3 70B": "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q5_K_S",
    "Qwen 14B": "hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:F16",
    "R1 Distill 32B": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q8_0",
}

for model_name, model_path in model_paths.items():
    logger.info(f"\n=== Processing Model: {model_name} ===\n")
    
    # Create output directory for this model.
    output_dir = os.path.join("JSON Outputs New", model_name, "User-System")
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, explanation in enumerate(initial_explanations, start=1):
        errorText = ""
        if not explanation.strip():
            logger.info(f"Skipping empty explanation #{idx} for model {model_name}.\n")
            continue

        logger.info(f"Processing explanation #{idx} for model {model_name}...")
        
        # Initialize conversation history with the initial explanation.
        conversation_history = [{"role": "user", "content": explanation}]
        
        # Initial call to set context (its response is not aggregated).
        try:
            response = chat(
                messages=conversation_history,
                model=model_path
            )
            initial_response = response.message.content
            conversation_history.append({"role": "assistant", "content": initial_response})
        except Exception as e:
            logger.error(f"Error processing initial explanation for explanation #{idx} on model {model_name}: {e}")
            continue

        # Prepare aggregator for the LLM structured JSON outputs.
        aggregated_results = {"AllRequirements": {}, "Hallucinations": {}}
        
        # Process each requirement individually (e.g., for cNeuro and cMRI).
        for batch_label, requirements_list in [("cNeuro", neuro_requirements_list), ("cMRI", mri_requirements_list)]:
            for req in requirements_list:
                logger.info(f"Model: {model_name} | Explanation #{idx} | {batch_label} requirement processing.")
                # Retrieve relevant context from the embedding index using RAG for a single user requirement.
                relevant_contexts = retrieve_top_k(req, embedding_index, k=2)
                context_text = "\n\n".join(relevant_contexts)
                
                # Combine the retrieved context with the individual user requirement.
                combined_prompt = f"Relevant Context:\n{context_text}\n\nGiven the relevant examples, break down this higher-level user requirement into actionable system requirements with similar detail and structure:\n{req}"
                conversation_history.append({"role": "user", "content": combined_prompt})
                
                try:
                    response = chat(
                        messages=conversation_history,
                        model=model_path,
                        format=OutputFormat.model_json_schema()
                    )
                    llm_response_str = response.message.content
                    # Parse the structured JSON output from the LLM.
                    try:
                        structured_output = json.loads(llm_response_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON for {batch_label} requirement in explanation #{idx} for model {model_name}: {e}")
                        errorText += llm_response_str + "\n\n\n"
                        continue
                    
                    # Aggregate results from this requirement.
                    for req_item in structured_output.get("AllRequirements", []):
                        user_req_id = req_item.get("UserRequirementId")
                        gen_system_reqs = req_item.get("GeneratedSystemRequirements", [])
                        if user_req_id not in aggregated_results["AllRequirements"]:
                            aggregated_results["AllRequirements"][user_req_id] = gen_system_reqs
                        else:
                            if user_req_id not in aggregated_results["Hallucinations"]:
                                aggregated_results["Hallucinations"][user_req_id] = []
                            aggregated_results["Hallucinations"][user_req_id].extend(gen_system_reqs)
                    
                    # Optionally add LLM response to the conversation history.
                    conversation_history.append({"role": "assistant", "content": llm_response_str})
                    logger.info(f"--> {batch_label} requirement processed.")
                except Exception as e:
                    logger.error(f"Error processing {batch_label} requirement for explanation #{idx} on model {model_name}: {e}")


        # Convert the aggregated dictionary into the final structured JSON.
        final_aggregated = {
            "AllRequirements": [
                {"UserRequirementId": req_id, "GeneratedSystemRequirements": gen_sys_reqs}
                for req_id, gen_sys_reqs in aggregated_results["AllRequirements"].items()
            ],
            "Hallucinations": [
                {"UserRequirementId": req_id, "GeneratedSystemRequirements": gen_sys_reqs}
                for req_id, gen_sys_reqs in aggregated_results["Hallucinations"].items()
            ]
        }
        
        # Save the aggregated JSON, error log, and conversation history.
        output_file = os.path.join(output_dir, f"explanation_4_aggregated.json")
        error_file = os.path.join(output_dir, f"explanation_4_error.txt")
        conversation_file = os.path.join(output_dir, f"explanation_4_history.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_aggregated, f, indent=4)
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(errorText)
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, indent=4)
            logger.info(f"Saved aggregated JSON output for explanation #{idx} to: {output_file}\n")
        except Exception as e:
            logger.error(f"Error saving aggregated JSON for explanation #{idx} on model {model_name}: {e}")
