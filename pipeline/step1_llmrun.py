#!/Users/keshigagopalarajah/Restart/venv/bin/python3
# Models and API keys
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Import LLM classes from llms.py
sys.path.insert(0, os.path.dirname(__file__))
from llms import (
    Llama3_8B_Model,
    Llama3_70B_Model,
    OpenBio8BModel,
    OpenBio70BModel,
    Gemini25FlashNonThinking,
    Gemini25FlashThinking,
    Med42_8B_Model,
    Med42_70B_Model,
    DeepSeekR1_1_5_Model,
)

# Load environment variables from .env file
load_dotenv()

# Initialize LLM instances
all_llms = [
        Llama3_8B_Model(),
        Llama3_70B_Model(),
        Med42_8B_Model(),
        Med42_70B_Model(),
        DeepSeekR1_1_5_Model(),
]

print(f"Initialized {len(all_llms)} LLM models")


def basic_mcq_prompt(subject_name, subject_name_adjective, question, opa, opb, opc, opd):
    return f"""
You are an assistant specializing in {subject_name}. Please select the correct option from the given choices for the following multiple-choice question related to {subject_name}.
Each question will have four options (A, B, C, or D). After selecting the correct option, provide a detailed explanation that includes the reasoning behind your choice, supported by relevant {subject_name_adjective} knowledge and information.

Format your response in JSON in the structure shown below:
{{
    "Correct answer": "A",
    "Reasoning": "[Provide reasoning that clearly explains why option A is the correct choice using a maximum of 100 words, including relevant {subject_name_adjective} details and knowledge.]"
}}

Question:
{question}

Options:
A) {opa}
B) {opb}
C) {opc}
D) {opd}
"""

def get_llm_response(prompt, llm_model):
    """Get response from the LLM for a given prompt using the LLM interface."""
    try:
        return llm_model.run(prompt)
    except Exception as e:
        print(f"Error getting LLM response from {llm_model._model_name}: {e}")
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Run LLM models on a CSV dataset")
    parser.add_argument('dataset_path', type=str, help='Path to the input CSV file')
    parser.add_argument('-o', '--output', type=str, help='Path to the combined output CSV file (default: input filename with _with_llm_responses suffix inside output directory)')
    
    args = parser.parse_args()
    
    # Load the dataset
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    
    dataset = pd.read_csv(args.dataset_path)
    base_dataset = dataset.copy()
    
    print(f"Loaded dataset with {len(dataset)} rows from {args.dataset_path}")
    print(f"Processing questions with {len(all_llms)} LLM models...")
    
    # Prepare output directory
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_basename = os.path.basename(args.dataset_path)
    output_root = os.path.dirname(args.dataset_path) or '.'
    per_model_output_dir = os.path.join(output_root, f"output-{date_str}")
    os.makedirs(per_model_output_dir, exist_ok=True)
    print(f"Per-model outputs will be written to {per_model_output_dir}")

    combined_dataset = base_dataset.copy()
    
    # Process each LLM model
    for llm in all_llms:
        print(f"\nProcessing with {llm._model_name}...")
        llm_responses = []
        
        for idx, row in dataset.iterrows():
            print(f"  Question {idx + 1}/{len(dataset)}...")
            
            # Format the question as a string
            # Todo: Subject name adjective - do a mapping of some kind  
            formatted_question = basic_mcq_prompt(
                row['subject_name'], 
                row['subject_name'], 
                row['question'], 
                row['opa'], 
                row['opb'], 
                row['opc'], 
                row['opd']
            )
            
            # Get LLM response (this call waits for the response before continuing)
            response = get_llm_response(formatted_question, llm)
            llm_responses.append(response)
        
        # Add LLM responses as a new column
        column_name = f"{llm._model_name}_response"
        combined_dataset[column_name] = llm_responses
        
        # Create per-model dataset and save
        model_df = base_dataset.copy()
        model_df[column_name] = llm_responses
        model_filename = f"{llm._model_name}-{date_str}-{dataset_basename}"
        model_path = os.path.join(per_model_output_dir, model_filename)
        model_df.to_csv(model_path, index=False)
        print(f"  Saved per-model output to {model_path}")
        print(f"  Completed {llm._model_name}")
    
    # Generate output filename
    if args.output:
        output_filename = args.output
        if not os.path.isabs(output_filename):
            output_filename = os.path.join(per_model_output_dir, output_filename)
    else:
        # Generate output filename based on input filename
        base_name = os.path.splitext(dataset_basename)[0]
        output_filename = os.path.join(per_model_output_dir, f"{base_name}_with_llm_responses.csv")
    
    # Save the updated dataset
    combined_dataset.to_csv(output_filename, index=False)
    
    print(f"\nCompleted! Updated dataset saved to: {output_filename}")
    print(f"Total questions processed: {len(dataset)}")
    print(f"Models used: {[llm._model_name for llm in all_llms]}")

if __name__ == "__main__":
    main()
