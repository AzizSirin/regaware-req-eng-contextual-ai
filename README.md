# Enhancing Regulation-Adherent Requirement Engineering with Contextual AI

This repository contains scripts used in the thesis titled:  
**"Enhancing Regulation-Adherent Requirement Engineering with Contextual AI – An Empirical Study"**  
and the accompanying paper:  
**"Enhancing Regulation-Adherent Requirement Engineering with Contextual AI: An Industrial Study"**

> ⚠️ **Note**: This is an experimental pipeline. It is **not** production-ready and requires further adaptation for real-world application.

---

## 📌 Purpose

The scripts implement a pipeline to explore the generation of **lower-level requirements** (system and software) from **higher-level user requirements** in the **medical device domain**, using various **on-premise LLMs** and **prompting strategies**.  
Additionally, a **Retrieval-Augmented Generation (RAG)** mechanism is integrated to enrich the input context using past requirement relationships.

---

## 📂 Project Structure

### 1. `userSystem.py`
> Generates **system-level requirements** from **user-level requirements** using a selection of LLMs.

- Loads input user requirements from JSON.
- Uses one or more prompt templates.
- Sends batch prompts to each model via `Ollama`.
- Stores outputs as structured JSON.
- No contextual retrieval (i.e., RAG) used in this script.

### 2. `userSystemRAG.py`
> Same goal as `userSystem.py` but integrates **RAG** to retrieve relevant user → system mappings.

- Builds an embedding index with **SentenceTransformers**.
- For each user requirement, retrieves similar user requirements from 3rd product and their mapped system requirements.
- Combines retrieved context with the input to the LLM.
- Outputs structured JSON + logs.

### 3. `systemSoftware.py`
> Generates **software-level requirements** from **system-level requirements** using prompting only.

- Loads input system requirements from JSON.
- Batches them and sends to selected LLMs via `Ollama`.
- Stores outputs as structured JSON.

### 4. `systemSoftwareRAG.py`
> Like `systemSoftware.py` but adds **RAG** to guide generation with prior system → software mappings.

- Builds embeddings for previously generated system-software mappings.
- Retrieves top-k most similar examples from 3rd product for each system requirement.
- Feeds retrieved context and current requirement into the LLM.
- Outputs structured JSON + logs.
  
---

## 🧠 LLMs Used

These models were tested in the pipeline using Ollama:
- Llama 3.2 3B
- Llama 3.3 70B
- Mistral 24B
- Qwen 14B
- R1 Distill 32B

> All models were tested with structured prompt-response conversations.

---

## 📦 Dependencies

- `ollama`
- `pydantic`
- `numpy`
- `sentence-transformers`
- `json`, `os`, `logging` (standard libraries)
