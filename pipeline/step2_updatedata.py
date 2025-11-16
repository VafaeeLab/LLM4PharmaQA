#!/usr/bin/env python3
"""
Simplify step2: Read input CSV, parse llm_response column into answer and response,
and output a new CSV with these columns.

Usage:
    python step2_updatedata.py <input_folder>
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_llm_response(raw: str) -> tuple[str, str]:
    """
    Parse llm_response and extract answer and response (reasoning).
    Returns (answer, response) tuple.
    """
    if not raw or raw.strip() == "":
        return "ERROR", "Empty response"
    
    # Handle error responses
    if raw.startswith("ERROR:"):
        return "ERROR", raw
    
    try:
        # Try to extract JSON from markdown code blocks first
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, flags=re.DOTALL)
        if code_block_match:
            json_text = code_block_match.group(1)
        else:
            # Try to find JSON object in the text
            json_match = re.search(r'\{.*?\}', raw, flags=re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found")
            json_text = json_match.group(0)
        
        # Clean up the JSON text
        json_text = json_text.replace('""', '"')  # normalize doubled quotes
        
        # Parse JSON
        data = json.loads(json_text)
        
        # Extract answer and reasoning
        answer = str(data.get("Correct answer", "")).strip()
        response = str(data.get("Reasoning", "")).strip()

        # Extract only the MCQ letter (A-D) at the start
        match = re.match(r'^["\'\(\[]*([A-Da-d])[^A-Da-d]?.*', answer)
        if match:
            answer = match.group(1).upper()
        else:
            answer = answer[:1].upper() if answer and answer[0].upper() in "ABCD" else answer

        return answer, response
        
    except Exception as e:
        # Fallback to regex parsing
        try:
            answer_pattern = re.compile(
                r'"?Correct answer"?\s*[:=]\s*"?(?P<ans>[A-Da-d])[^"]*"?',
                flags=re.IGNORECASE | re.MULTILINE,
            )
            reason_pattern = re.compile(
                r'"?Reasoning"?\s*[:=]\s*"?(?P<reason>.+?)"?\s*[}\n]',
                flags=re.IGNORECASE | re.DOTALL,
            )
            
            ans_match = answer_pattern.search(raw)
            reason_match = reason_pattern.search(raw)
            
            if ans_match:
                answer = ans_match.group("ans").upper()
            else:
                answer = "ERROR"
                
            if reason_match:
                response = reason_match.group("reason").strip()
            else:
                response = raw  # Fallback to full response
            
            return answer, response
        except Exception:
            return "ERROR", f"Could not parse response: {str(e)}"


def process_csv_file(input_path: Path, output_dir: Path):
    """Process a single CSV file and save the updated version."""
    print(f"\nProcessing {input_path.name}...")
    
    # Read the input CSV
    df = pd.read_csv(input_path)
    
    # Find all columns ending with '_response'
    response_columns = [col for col in df.columns if col.endswith('_response')]
    
    if not response_columns:
        print(f"  Warning: No columns ending with '_response' found in {input_path.name}")
        return
    
    # Process each response column
    for response_col in response_columns:
        print(f"  Parsing {response_col}...")
        parsed_results = df[response_col].apply(parse_llm_response)
        
        # Create answer and response column names based on the response column name
        base_name = response_col.replace('_response', '')
        answer_col = f"{base_name}_answer"
        response_col_new = f"{base_name}_response_parsed"
        
        df[answer_col] = [result[0] for result in parsed_results]
        df[response_col_new] = [result[1] for result in parsed_results]
    
    # Generate output filename: step2_original_filename.csv
    original_name = input_path.stem  # filename without extension
    output_filename = f"step2_{original_name}.csv"
    output_path = output_dir / output_filename
    
    # Save to output file
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process CSV files in a folder and parse LLM responses")
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
    output_dir = input_folder.parent / f"step2-{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, output_dir)
    
    print(f"\nCompleted! Processed {len(csv_files)} file(s).")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
