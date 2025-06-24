#!/usr/bin/env python3

from dotenv import load_dotenv
import pandas as pd
import llm_interface

def get_medmcqa_questions(subject_name: str, num_questions: int, filename: str) -> pd.DataFrame:
    """
    Extracts a specified number of questions from a DataFrame based on subject_name.
    
    Parameters:
    - df (pd.DataFrame): The full dataset.
    - subject_name (str): The subject to filter by (e.g., 'Pathology').
    - num_questions (int): Number of questions to extract.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered questions.
    """

    splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
    }
    df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["train"])

    # Filter by subject name
    filtered_df = df[df['subject_name'] == subject_name]
    
    subset = filtered_df.head(num_questions)

    subset.to_csv(filename, index=False)

    return subset

def add_structured_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'prompt' column using a JSON-style pharmacology prompt for each question.

    Parameters:
        df (pd.DataFrame): A DataFrame with question, opa, opb, opc, opd.

    Returns:
        pd.DataFrame: Original DataFrame with a new 'prompt' column.
    """
    subject_name = "Pharmacology"
    subject_name_adjective = "pharmacological"

    def build_prompt(row):
        return f"""
            You are an assistant specializing in {subject_name}. Please select the correct option from the given choices for the following multiple-choice question related to {subject_name}. 
            Each question will have four options (A, B, C, or D). After selecting the correct option, provide a detailed explanation that includes the reasoning behind your choice, supported by relevant {subject_name_adjective} knowledge and information.

            Please format your response in JSON as shown below:
            {{
                "Correct answer": "A",
                "Reasoning": "[Provide reasoning that clearly explains why option A is the correct choice, including relevant {subject_name_adjective} details and knowledge.]"
            }}

            Question:
            {row['question']}

            Options:
            A) {row['opa']}
            B) {row['opb']}
            C) {row['opc']}
            D) {row['opd']}
            """.strip()

    df["prompt"] = df.apply(build_prompt, axis=1)
    return df



def main():

    load_dotenv

    print("Getting medmcqa questions")
    medmcqa_questions = get_medmcqa_questions("Pharmacology", 100, "medmcqa_train.csv")
    prompts = add_structured_prompts(medmcqa_questions)
    prompts.to_csv("medmcqa_questions.csv", index=False)



if __name__ == "__main__":
    main()