#!/usr/bin/env python3
"""
Simplify step3: Evaluate answers by comparing them to correct answers.

Reads an input CSV file with 'answer' and 'response' columns,
compares answers to the 'cop' (correct option) column,
and calculates accuracy metrics.

Usage:
    python step3_eval.py <input_folder>
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
import bert_score
import evaluate
import torch

# Add BARTScore directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BARTScore'))
from bart_score import BARTScorer

# Add AlignScore directory to path
from alignscore import AlignScore 

# Map numeric correct answer (0-3) to letter (A-D)
LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def evaluate_csv(input_path: Path, output_path: Path):
    """Evaluate a single CSV file and save results."""
    print(f"\nProcessing {input_path.name}...")
    
    # Read the input CSV
    df = pd.read_csv(input_path)

    # Check required columns
    if 'cop' not in df.columns:
        print(f"  Warning: 'cop' column (correct option) not found in {input_path.name}")
        return
    
    # Convert numeric correct answer (0-3) to letter (A-D)
    df['correct_answer'] = df['cop'].map(LETTER_MAP)

    # Find all answer columns (could be 'answer' or '{model}_answer')
    answer_columns = []
    if 'answer' in df.columns:
        answer_columns.append('answer')
    # Also look for model-specific answer columns
    answer_columns.extend([col for col in df.columns if col.endswith('_answer') and col != 'answer'])

    if not answer_columns:
        print(f"  Warning: No 'answer' column found in {input_path.name}")
        return

    # Process each answer column
    for answer_col in answer_columns:
        print(f"  Evaluating {answer_col}...")
        
        # Compare predicted answer to correct answer
        # Normalize answers to uppercase and strip whitespace
        predicted_option = df[answer_col].astype(str).str.strip().str.upper()
        correct_option = df['correct_answer'].astype(str)

        # Calculate accuracy (1 if correct, 0 if incorrect)
        accuracy_col = f"{answer_col.replace('_answer', '')}_accuracy" if answer_col != 'answer' else 'accuracy'
        df[accuracy_col] = (predicted_option == correct_option).astype(int)

        # Find corresponding response column
        response_col = None
        if answer_col == 'answer':
            # Look for 'response' or 'response_parsed'
            if 'response' in df.columns:
                response_col = 'response'
            elif 'response_parsed' in df.columns:
                response_col = 'response_parsed'
        else:
            # Look for model-specific response column
            base_name = answer_col.replace('_answer', '')
            if f"{base_name}_response_parsed" in df.columns:
                response_col = f"{base_name}_response_parsed"
            elif f"{base_name}_response" in df.columns:
                response_col = f"{base_name}_response"

        # Evaluate explanations if response column exists
        if response_col and response_col in df.columns:
            predicted_explanation = df[response_col].astype(str)
            
            # Check if explanation column exists
            if 'exp' in df.columns:
                correct_explanation = df['exp'].astype(str)
            else:
                print(f"    Warning: No correct explanation column found ('exp') for {answer_col}")
                continue
            
            # Initialize scorers (only once, reuse for efficiency)
            # Initialize ROUGE scorer (only ROUGE-L)
            my_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            # Initialize BERT scorer
            bert_scorer = bert_score.BERTScorer(lang='en', rescale_with_baseline=True)
            
            # Initialize BART scorer
            print(f"    Initializing BART scorer...")
            bart_scorer = BARTScorer(device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint='facebook/bart-large-cnn')
            # Load custom checkpoint if available
            bart_checkpoint_path = os.path.join(os.path.dirname(__file__), 'BARTScore', 'bart_score.pth')
            if os.path.exists(bart_checkpoint_path):
                bart_scorer.model.load_state_dict(torch.load(bart_checkpoint_path, map_location=bart_scorer.device))
                bart_scorer.model.eval()
            
            # Initialize AlignScore scorer
            alignscore_checkpoint_path = os.path.join(os.path.dirname(__file__), 'AlignScore', 'src', 'alignscore', 'AlignScore-large.ckpt')
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            alignscore_scorer = AlignScore(model='roberta-large', batch_size=32, device=device_str, ckpt_path=alignscore_checkpoint_path, evaluation_mode='nli_sp')
            
            # Convert to lists for scoring
            pred_list = predicted_explanation.tolist()
            ref_list = correct_explanation.tolist()
            
            # Calculate BERT scores
            print(f"    Calculating BERT scores...")
            P, R, F1 = bert_scorer.score(pred_list, ref_list)
            bert_col = f"{answer_col.replace('_answer', '')}_bert_scores" if answer_col != 'answer' else 'bert_scores'
            df[bert_col] = F1.tolist()
            
            # Calculate BART scores
            print(f"    Calculating BART scores...")
            bart_scores = bart_scorer.score(ref_list, pred_list)  # Note: BARTScore expects (src, tgt) = (reference, prediction)
            bart_col = f"{answer_col.replace('_answer', '')}_bart_scores" if answer_col != 'answer' else 'bart_scores'
            df[bart_col] = bart_scores
            
            # Calculate AlignScore
            print(f"    Calculating AlignScore...")
            align_scores = alignscore_scorer.score(contexts=ref_list, claims=pred_list)
            align_col = f"{answer_col.replace('_answer', '')}_alignscore" if answer_col != 'answer' else 'alignscore'
            df[align_col] = align_scores
            
            # Calculate ROUGE-L and METEOR scores
            print(f"    Calculating ROUGE-L and METEOR scores...")
            rougeL_scores = []
            meteor = evaluate.load("meteor")
            meteor_scores = []
            
            for idx in df.index:
                pred_text = str(predicted_explanation.loc[idx])
                ref_text = str(correct_explanation.loc[idx])
                scores = my_rouge_scorer.score(ref_text, pred_text)
                meteor_score = meteor.compute(predictions=[pred_text], references=[ref_text])
                meteor_scores.append(meteor_score['meteor'])
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            rouge_col = f"{answer_col.replace('_answer', '')}_rougeL" if answer_col != 'answer' else 'rougeL'
            meteor_col = f"{answer_col.replace('_answer', '')}_meteor" if answer_col != 'answer' else 'meteor'
            df[rouge_col] = rougeL_scores
            df[meteor_col] = meteor_scores

    # Save to output file
    df.to_csv(output_path, index=False)
    print(f"  Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM responses in CSV files")
    parser.add_argument('input_folder', type=str, help='Path to the folder containing CSV files')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    
    if not input_folder.exists():
        print(f"Error: Folder not found: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a directory")
        sys.exit(1)
    
    # Find all CSV files in the folder
    csv_files = list(input_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {input_folder}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s) in {input_folder}")
    
    # Create output directory with timestamp (in the same directory as input folder)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = input_folder.parent / f"step3-{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each CSV file
    for csv_file in csv_files:
        # Generate output filename: step3_original_filename.csv
        original_name = csv_file.stem  # filename without extension
        output_filename = f"step3_{original_name}.csv"
        output_path = output_dir / output_filename
        
        evaluate_csv(csv_file, output_path)
    
    print(f"\nCompleted! Processed {len(csv_files)} file(s).")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
