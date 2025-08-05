#!/usr/bin/env python3
import json
import os
import logging
from ollama import chat
from pydantic import BaseModel

# ----------------------------------------------------------------------------------
# Configure logger to output to both the console and log.txt
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
# Define the structured output schema using Pydantic models
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
# 1. Load initial explanation files.
# ----------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
explanation_files = [
    os.path.join(script_dir, "Prompts", "System-Software", "Prompt-1.txt"),
    os.path.join(script_dir, "Prompts", "System-Software", "Prompt-2.txt"),
    os.path.join(script_dir, "Prompts", "System-Software", "Prompt-3.txt"),
    # os.path.join(script_dir, "Prompts", "System-Software", "Prompt-4.txt")
]
initial_explanations = []
for file in explanation_files:
    with open(file, "r", encoding="utf-8") as f:
        initial_explanations.append(f.read())

# ----------------------------------------------------------------------------------
# 2. Define the models you want to test.
# ----------------------------------------------------------------------------------
model_paths = {
    "Llama 3.2 3B": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:F16",
    "Mistral 24B": "hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:F16",
    "Llama 3.3 70B": "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q5_K_S",
    "Qwen 14B": "hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:F16",
    "R1 Distill 32B": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q8_0"
}

# ----------------------------------------------------------------------------------
# 3. Read input JSON file with system requirements.
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
# 4. Main loop: For each model and for each explanation, run the conversation in batches.
# ----------------------------------------------------------------------------------
for model_name, model_path in model_paths.items():
    logger.info(f"\n=== Processing Model: {model_name} ===\n")
    
    # Create the output directory for this model.
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
        # Initial explanation call (context-setting). Its structured response is 
        # not used for aggregation.
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
        # We use dictionaries keyed by SystemRequirementId to detect duplicates.
        # --------------------------------------------------------------------------
        aggregated_results = {"AllRequirements": {}, "Hallucinations": {}}
        
        batch_number = 1
        # Process system requirements in batches of 10.
        for i in range(0, len(srs_requirements_list), 10):
            logger.info(f"Model: {model_name} | Explanation #{idx} | Batch {batch_number}")
            batch_items = srs_requirements_list[i:i+10]
            batch_prompt = "\n\n".join(batch_items)
            conversation_history.append({"role": "user", "content": batch_prompt})
            
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
                    print("Error parsing JSON. Response message:", llm_response_str)
                    logger.error(f"Error parsing JSON for batch {batch_number} in explanation #{idx} for model {model_name}: {e}")
                    errorText += llm_response_str + "\n\n\n"
                    batch_number += 1
                    continue

                # Aggregate the output from this batch.
                for req in structured_output.get("AllRequirements", []):
                    req_id = req.get("SystemRequirementId")
                    software_reqs = req.get("SoftwareRequirements", [])
                    if req_id not in aggregated_results["AllRequirements"]:
                        aggregated_results["AllRequirements"][req_id] = software_reqs
                    else:
                        if req_id not in aggregated_results["Hallucinations"]:
                            aggregated_results["Hallucinations"][req_id] = []
                        aggregated_results["Hallucinations"][req_id].extend(software_reqs)

                # Optionally, add the LLM's response to conversation history (if needed for context).
                conversation_history.append({"role": "assistant", "content": llm_response_str})
                logger.info(f"--> Batch {batch_number} processed.")
            except Exception as e:
                logger.error(f"Error processing batch {batch_number} for explanation #{idx} on model {model_name}: {e}")
            
            batch_number += 1

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
        
        # Save the aggregated JSON output for this explanation run.
        output_file = os.path.join(output_dir, f"explanation_{idx}_aggregated.json")
        error_file = os.path.join(output_dir, f"explanation_{idx}_error.txt")
        conversation_file = os.path.join(output_dir, f"explanation_{idx}_history.json")
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
            
