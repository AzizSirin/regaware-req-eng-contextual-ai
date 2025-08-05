#!/usr/bin/env python3
import json
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer  # Added for embedding
from ollama import chat
from pydantic import BaseModel

# ----------------------------------------------------------------------------------
# 1. Configure logger to output to both the console and log.txt
# ----------------------------------------------------------------------------------
logger = logging.getLogger("DualLogger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("log.txt", mode="a")  # Use "w" to overwrite each run

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------------------------------------------------------------------------------
# 2. Define the structured output schema using Pydantic models
# ----------------------------------------------------------------------------------
class SoftwareRequirement(BaseModel):
    GeneratedRequirementId: str
    GeneratedRequirement: str

class Requirement(BaseModel):
    SystemRequirementId: str
    SoftwareRequirements: list[SoftwareRequirement]

class OutputFormat(BaseModel):
    AllRequirements: list[Requirement]

# ----------------------------------------------------------------------------------
# 3. Initialize the Embedding Model from SentenceTransformers
# ----------------------------------------------------------------------------------
# This model will use a GPU if available.
embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

def get_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for a given text using SentenceTransformers.
    """
    return embedding_model.encode(text, convert_to_numpy=True)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ----------------------------------------------------------------------------------
# 4. Build the RAG Embedding Index from the RAG file (RAGfile.json)
# ----------------------------------------------------------------------------------
def build_rag_index(data: dict) -> list:
    """
    Build an embedding index from the RAG file.
    For each requirement in RAGfile.json, combine the SRS titles and IRS data as context.
    """
    index = []
    for req_id, req_info in data.items():
        srs_data = req_info.get("SRS_Data", {})
        combined_text = ""
        for srs_req_id, srs_info in srs_data.items():
            title = srs_info.get("SRS_Title", "")
            # Combine any IRS details if available
            irs_data = srs_info.get("IRS_Data", {})
            irs_text = " ".join(irs_data.values())
            combined_text += f"SRS_Title: {title}\nIRS_Data: {irs_text}\n"
        index.append({
            "doc_id": req_id,  # using the key from RAGfile.json as document ID
            "text": combined_text,
            "embedding": get_embedding(combined_text)
        })
    return index

def retrieve_top_k_rag(query: str, index: list, k: int = 2) -> list:
    """
    Retrieve the top k documents from the RAG index based on cosine similarity.
    Formats the retrieved context in a structured manner.
    """
    query_embedding = get_embedding(query)
    scored_items = []
    for item in index:
        sim = cosine_similarity(query_embedding, item["embedding"])
        scored_items.append((sim, item))
    scored_items.sort(key=lambda x: x[0], reverse=True)
    top_items = scored_items[:k]
    results = []
    for sim, item in top_items:
        formatted = f"System Requirement Example: {item['doc_id']}:\n{item['text']}"
        results.append(formatted)
    return results

# ----------------------------------------------------------------------------------
# 5. Load Initial Explanation Files (unchanged logic)
# ----------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
explanation_files = [
    #os.path.join(script_dir, "Prompts", "System-Software", "Prompt-1.txt"),
    #os.path.join(script_dir, "Prompts", "System-Software", "Prompt-2.txt"),
    #os.path.join(script_dir, "Prompts", "System-Software", "Prompt-3.txt"),
    os.path.join(script_dir, "Prompts", "System-Software", "Prompt-4.txt")
]
initial_explanations = []
for file in explanation_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            initial_explanations.append(f.read())
    else:
        logger.warning(f"Explanation file not found: {file}")

# ----------------------------------------------------------------------------------
# 6. Read input JSON file with system requirements (srs_data_v3 remains unchanged)
# ----------------------------------------------------------------------------------
input_file = os.path.join(script_dir, "Extracted Data", "srs_data_v3.json")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

srs_requirements_list = []
for top_key, requirement in data.items():
    srs_data = requirement.get("SRS_Data", {})
    for srs_id, srs_item in srs_data.items():
        srs_title = srs_item.get("SRS_Title", "")
        srs_description = srs_item.get("SRS_Description", "")
        formatted_requirement = (
            f"System Requirement Id: {srs_id}:\n"
            f"     Description: {srs_description}\n"
            f"     Title: {srs_title}"
        )
        srs_requirements_list.append(formatted_requirement)

# ----------------------------------------------------------------------------------
# 7. Load the RAGfile.json for retrieval context
# ----------------------------------------------------------------------------------
rag_file = os.path.join(script_dir, "Extracted Data", "Investigator_Requirements.json")
if os.path.exists(rag_file):
    with open(rag_file, "r", encoding="utf-8") as f:
        rag_data = json.load(f)
    rag_index = build_rag_index(rag_data)
    logger.info(f"Built RAG index with {len(rag_index)} documents.")
else:
    logger.error(f"RAG file not found at {rag_file}. RAG retrieval will not be performed.")
    rag_index = []  # Fallback: empty index

# ----------------------------------------------------------------------------------
# 8. Define the models you want to test.
# ----------------------------------------------------------------------------------
model_paths = {
    "Llama 3.2 3B": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:F16",
    "Mistral 24B": "hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:F16",
    "Llama 3.3 70B": "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q5_K_S",
    "Qwen 14B": "hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:F16",
    "R1 Distill 32B": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q8_0"
}

# ----------------------------------------------------------------------------------
# 9. Main Loop: Process each model and each explanation, handling system requirements one-by-one
# ----------------------------------------------------------------------------------
for model_name, model_path in model_paths.items():
    logger.info(f"\n=== Processing Model: {model_name} ===\n")
    
    # Create output directory for this model.
    output_dir = os.path.join("JSON Outputs New", model_name, "System-Software")
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, explanation in enumerate(initial_explanations, start=1):
        errorText = ""
        if not explanation.strip():
            logger.info(f"Skipping empty explanation #{idx} for model {model_name}.\n")
            continue

        logger.info(f"Processing explanation #{idx} for model {model_name}...")

        # Initialize conversation history with the initial explanation.
        conversation_history = [{"role": "user", "content": explanation}]
        
        # --------------------------------------------------------------------------
        # Initial explanation call (context-setting). Its response is not aggregated.
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Prepare aggregator for structured JSON outputs.
        # ----------------------------------------------------------------------------------
        aggregated_results = {"AllRequirements": {}, "Hallucinations": {}}
        
        # ----------------------------------------------------------------------------------
        # Process each system requirement individually (instead of in batches of 10)
        # ----------------------------------------------------------------------------------
        req_number = 1
        for req in srs_requirements_list:
            logger.info(f"Model: {model_name} | Explanation #{idx} | Processing system requirement {req_number}")
            
            # Retrieve relevant context via RAG from RAGfile.json – compare against the current system requirement.
            if rag_index:
                relevant_contexts = retrieve_top_k_rag(req, rag_index, k=2)
                context_text = "\n\n".join(relevant_contexts)
            else:
                context_text = "No relevant context available."

            prompt = (
                f"Relevant Context:\n{context_text}\n\n"
                f"Given the relevant examples, break down this higher-level system requirement into actionable software requirements with similar detail and structure:\n{req}"
            )
            conversation_history.append({"role": "user", "content": prompt})
            
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
                    logger.error(f"Error parsing JSON for system requirement {req_number} in explanation #{idx} for model {model_name}: {e}")
                    errorText += llm_response_str + "\n\n\n"
                    req_number += 1
                    continue

                # Aggregate the output from this requirement.
                for requirement_item in structured_output.get("AllRequirements", []):
                    sys_req_id = requirement_item.get("SystemRequirementId")
                    software_reqs = requirement_item.get("SoftwareRequirements", [])
                    if sys_req_id not in aggregated_results["AllRequirements"]:
                        aggregated_results["AllRequirements"][sys_req_id] = software_reqs
                    else:
                        if sys_req_id not in aggregated_results["Hallucinations"]:
                            aggregated_results["Hallucinations"][sys_req_id] = []
                        aggregated_results["Hallucinations"][sys_req_id].extend(software_reqs)
                
                # Optionally, add the LLM response to conversation history (if context persistence is desired).
                conversation_history.append({"role": "assistant", "content": llm_response_str})
                logger.info(f"--> System requirement {req_number} processed.")
            except Exception as e:
                logger.error(f"Error processing system requirement {req_number} for explanation #{idx} on model {model_name}: {e}")
            
            req_number += 1

        # ------------------------------------------------------------------------------
        # Convert the aggregated results (dict form) into final JSON structure.
        # ------------------------------------------------------------------------------
        final_aggregated = {
            "AllRequirements": [
                {"SystemRequirementId": req_id, "SoftwareRequirements": software_reqs}
                for req_id, software_reqs in aggregated_results["AllRequirements"].items()
            ],
            "Hallucinations": [
                {"SystemRequirementId": req_id, "SoftwareRequirements": software_reqs}
                for req_id, software_reqs in aggregated_results["Hallucinations"].items()
            ]
        }
        
        # Save the aggregated JSON output, error log, and conversation history.
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
