# LLM4PharmaQA

MCQ DATASET: For each question in mcq dataset
PROMPT ENGINEERING (not really engineering at this stage): Generate a prompt with the question
LLMs: Give that prompt to LLMs
Fine-Tuned, Reasoning, Foundational
    Rag-Based Pipeline -  PrimeKG knowledge graph data -> GraphRAG -> LLM API
    Store the generated content in csv or json (?)
EVALUATION METHODS: Script for each evaluation
Run each generated output through each evaluation method and store the scores in csv/json
Statistical analysis on results