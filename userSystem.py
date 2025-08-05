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
# Each output object now contains a UserRequirementId and a list of generated system 
# requirements, each with its GeneratedSystemRequirementId and GeneratedSystemRequirement.
# ----------------------------------------------------------------------------------
class GeneratedSystemRequirement(BaseModel):
    GeneratedSystemRequirementId: str
    GeneratedSystemRequirement: str

class UserSystemRequirement(BaseModel):
    UserRequirementId: str
    GeneratedSystemRequirements: list[GeneratedSystemRequirement]

class OutputFormat(BaseModel):
    AllRequirements: list[UserSystemRequirement]

# ----------------------------------------------------------------------------------
# 1. Load initial explanation files for User-System generation.
# ----------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
explanation_files = [
    os.path.join(script_dir, "Prompts", "User-System", "Prompt-1.txt"),
    #os.path.join(script_dir, "Prompts", "User-System", "Prompt-2.txt"),
    #os.path.join(script_dir, "Prompts", "User-System", "Prompt-3.txt")
]
initial_explanations = []
for file in explanation_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            initial_explanations.append(f.read())
    else:
        logger.warning(f"Explanation file not found: {file}")

# ----------------------------------------------------------------------------------
# 2. Define the models you want to test.
# ----------------------------------------------------------------------------------
model_paths = {
    "R1 Distill 32B": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q8_0",
    "Llama 3.2 3B": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:F16",
    "Mistral 24B": "hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:F16",
    "Llama 3.3 70B": "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q5_K_S",
    "Qwen 14B": "hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:F16"
}

# ----------------------------------------------------------------------------------
# 3. Read input JSON file with user requirements.
# ----------------------------------------------------------------------------------
input_file = os.path.join(script_dir, "Extracted Data", "srs_data_v3.json")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Separate requirements into two batches based on the Id prefix.
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

# ----------------------------------------------------------------------------------
# 4. Main loop: For each model and for each explanation prompt, run conversations in batches.
# ----------------------------------------------------------------------------------
for model_name, model_path in model_paths.items():
    logger.info(f"\n=== Processing Model: {model_name} ===\n")
    
    # Create the output directory for this model.
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
        # Aggregated results is a dict mapping UserRequirementId to a list of generated system requirement objects.
        # --------------------------------------------------------------------------
        aggregated_results = {"AllRequirements": {}, "Hallucinations": {}}
        
        # Process each batch separately: one for cNeuro and one for cMRI.
        for batch_label, requirements_list in [("cNeuro", neuro_requirements_list), ("cMRI", mri_requirements_list)]:
            batch_number = 1
            # Process requirements in batches of 10.
            for i in range(0, len(requirements_list), 10):
                logger.info(f"Model: {model_name} | Explanation #{idx} | Batch {batch_label} Batch {batch_number}")
                batch_items = requirements_list[i:i+10]
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
                        logger.error(f"Error parsing JSON for batch {batch_label} {batch_number} in explanation #{idx} for model {model_name}: {e}")
                        errorText += llm_response_str + "\n\n\n"
                        batch_number += 1
                        continue

                    # Aggregate the output from this batch.
                    for req in structured_output.get("AllRequirements", []):
                        user_req_id = req.get("UserRequirementId")
                        gen_system_reqs = req.get("GeneratedSystemRequirements", [])
                        if user_req_id not in aggregated_results["AllRequirements"]:
                            aggregated_results["AllRequirements"][user_req_id] = gen_system_reqs
                        else:
                            if user_req_id not in aggregated_results["Hallucinations"]:
                                aggregated_results["Hallucinations"][user_req_id] = []
                            aggregated_results["Hallucinations"][user_req_id].extend(gen_system_reqs)

                    # Optionally, add the LLM's response to conversation history for context.
                    conversation_history.append({"role": "assistant", "content": llm_response_str})
                    logger.info(f"--> Batch {batch_label} {batch_number} processed.")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_label} {batch_number} for explanation #{idx} on model {model_name}: {e}")
                
                batch_number += 1

        # ------------------------------------------------------------------------------
        # Convert the aggregated results (dict form) into final JSON structure.
        # ------------------------------------------------------------------------------
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
