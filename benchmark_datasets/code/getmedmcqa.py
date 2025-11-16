#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
from datetime import datetime
import os
from typing import Optional, List
import sys 

# Configure logging
def setup_logging(quiet_mode=False):
    """Setup logging configuration based on quiet mode."""
    if quiet_mode:
        # Disable all logging
        logging.getLogger().setLevel(logging.CRITICAL)
        # Remove all handlers
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

logger = logging.getLogger(__name__)

def get_medmcqa_questions(subject_name: Optional[str] = None, 
                         max_questions: Optional[int] = None, 
                         dataset: Optional[str] = None,
                         choice_type: Optional[str] = None,
                         exp: Optional[str] = None,
                         min_exp_words: Optional[int] = None) -> pd.DataFrame:
    """
    Extracts questions from MedMCQA dataset based on specified filters.
    
    Parameters:
    - subject_name (Optional[str]): The subject to filter by. If None, gets all subjects.
    - max_questions (Optional[int]): Maximum number of questions to extract. If None, gets all.
    - dataset (Optional[str]): Dataset split to use ('train', 'test', 'validation'). If None, uses all.
    - choice_type (Optional[str]): Filter by choice_type column ('single', 'multiple', or None for all)
    - exp (Optional[str]): Filter by explanation presence ('with', 'without', or None for all)
    - min_exp_words (Optional[int]): Minimum number of words in explanation. If None, no word count filtering.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered questions.
    """
    
    splits = {
        'train': 'data/train-00000-of-00001.parquet',
        'test': 'data/test-00000-of-00001.parquet',
        'validation': 'data/validation-00000-of-00001.parquet'
    }
    
    if dataset:
        if dataset not in splits:
            raise ValueError(f"Invalid dataset: {dataset}. Must be one of {list(splits.keys())}")
        datasets_to_process = [dataset]
    else:
        datasets_to_process = list(splits.keys())
    
    all_data = []
    
    for split in datasets_to_process:
        logger.info(f"Processing {split} dataset...")
        file_path = f"hf://datasets/openlifescienceai/medmcqa/{splits[split]}"
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} questions from {split} dataset")
            
            # Add dataset column for tracking
            df['dataset_split'] = split
            
            # Debug: Show available subject names if filtering
            if subject_name:
                available_subjects = df['subject_name'].unique()
                logger.info(f"Available subjects in {split} dataset: {sorted(available_subjects)}")
                logger.info(f"Looking for subject: '{subject_name}'")
                
                # Try case-insensitive matching
                filtered_df = df[df['subject_name'].str.lower() == subject_name.lower()]
                if len(filtered_df) == 0:
                    # Try partial matching
                    filtered_df = df[df['subject_name'].str.lower().str.contains(subject_name.lower())]
                    if len(filtered_df) > 0:
                        logger.info(f"Found {len(filtered_df)} questions using partial matching")
                    else:
                        # Subject not found in this dataset - exit immediately
                        error_msg = f"Subject '{subject_name}' not found in dataset '{split}'.\n\nAvailable subjects in this dataset:\n"
                        error_msg += "\n".join(sorted(available_subjects))
                        logger.error(f"Subject not found: {error_msg}")
                        print(f"\n{error_msg}")
                        sys.exit(1)
                else:
                    logger.info(f"Found {len(filtered_df)} questions using exact matching")
            else:
                filtered_df = df
                logger.info(f"Using all subjects: {len(filtered_df)} questions")

            # Only filter and limit if filtered_df is a DataFrame (avoid linter errors)
            if isinstance(filtered_df, pd.DataFrame):
                # Filter by choice_type if specified
                if choice_type and choice_type != "all":
                    if "choice_type" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["choice_type"] == choice_type]
                        if not isinstance(filtered_df, pd.DataFrame):
                            filtered_df = pd.DataFrame(columns=df.columns)
                        logger.info(f"Filtered by choice_type='{choice_type}': {len(filtered_df)} questions")

                # Filter by explanation presence if specified
                if exp and exp != "all":
                    if "exp" in filtered_df.columns:
                        if exp == "with":
                            filtered_df = filtered_df[filtered_df["exp"].notnull() & (filtered_df["exp"].astype(str).str.strip() != "")]
                            if not isinstance(filtered_df, pd.DataFrame):
                                filtered_df = pd.DataFrame(columns=df.columns)
                            logger.info(f"Filtered to questions WITH explanation: {len(filtered_df)} questions")
                        elif exp == "without":
                            filtered_df = filtered_df[filtered_df["exp"].isnull() | (filtered_df["exp"].astype(str).str.strip() == "")]
                            if not isinstance(filtered_df, pd.DataFrame):
                                filtered_df = pd.DataFrame(columns=df.columns)
                            logger.info(f"Filtered to questions WITHOUT explanation: {len(filtered_df)} questions")

                # Filter by minimum explanation word count if specified
                if min_exp_words and min_exp_words > 0:
                    if "exp" in filtered_df.columns:
                        # Count words in explanations (handle null values)
                        def count_words(exp_text):
                            if pd.isna(exp_text) or str(exp_text).strip() == "":
                                return 0
                            return len(str(exp_text).split())
                        
                        # Apply word count filter
                        word_counts = filtered_df["exp"].apply(count_words)
                        filtered_df = filtered_df[word_counts >= min_exp_words]
                        if not isinstance(filtered_df, pd.DataFrame):
                            filtered_df = pd.DataFrame(columns=df.columns)
                        logger.info(f"Filtered to questions with >= {min_exp_words} words in explanation: {len(filtered_df)} questions")

                # Limit number of questions if specified
                if max_questions:
                    filtered_df = filtered_df.head(max_questions)
                    if not isinstance(filtered_df, pd.DataFrame):
                        filtered_df = pd.DataFrame(columns=df.columns)
                    logger.info(f"Limited to {len(filtered_df)} questions")
            else:
                # If not a DataFrame, skip filtering/limiting and just append as is
                logger.warning("filtered_df is not a DataFrame; skipping filtering and limiting.")

            all_data.append(filtered_df)
            
        except Exception as e:
            logger.error(f"Error processing {split} dataset: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data could be loaded from any dataset")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset contains {len(combined_df)} questions")
    
    return combined_df

def get_available_subjects() -> dict:
    """
    Get all available subjects from all datasets.
    
    Returns:
    - dict: Dictionary with dataset names as keys and sets of subjects as values
    """
    splits = {
        'train': 'data/train-00000-of-00001.parquet',
        'test': 'data/test-00000-of-00001.parquet',
        'validation': 'data/validation-00000-of-00001.parquet'
    }
    
    available_subjects = {}
    
    for split, file_path in splits.items():
        try:
            logger.info(f"Loading subjects from {split} dataset...")
            full_path = f"hf://datasets/openlifescienceai/medmcqa/{file_path}"
            df = pd.read_parquet(full_path)
            subjects = set(df['subject_name'].unique())
            available_subjects[split] = subjects
            logger.info(f"Found {len(subjects)} subjects in {split} dataset")
        except Exception as e:
            logger.error(f"Error loading {split} dataset: {e}")
            available_subjects[split] = set()
    
    return available_subjects

def get_available_datasets() -> list:
    """
    Get all available datasets.
    
    Returns:
    - list: List of available dataset names
    """
    return ['train', 'test', 'validation']

def generate_filename(subject_name: Optional[str], 
                    max_questions: Optional[int], 
                    dataset: Optional[str], 
                    combine: bool,
                    destination_dir: str) -> str:
    """
    Generate filename based on parameters.
    
    Parameters:
    - subject_name (Optional[str]): Subject name or 'all'
    - max_questions (Optional[int]): Max questions or 'all'
    - dataset (Optional[str]): Dataset name or 'all'
    - combine (bool): Whether combining datasets
    - destination_dir (str): Destination directory
    
    Returns:
    - str: Generated filename with path
    """
    
    # Set default values
    subject_str = subject_name if subject_name else "all"
    questions_str = str(max_questions) if max_questions else "all"
    
    if combine and not dataset:
        dataset_str = "all"
    elif dataset:
        dataset_str = dataset
    else:
        dataset_str = "all"
    
    # Generate datetime string
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{subject_str}-{questions_str}-{dataset_str}-{datetime_str}.csv"
    
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Return full path
    return os.path.join(destination_dir, filename)

def main():
    
    parser = argparse.ArgumentParser(
        description="Extract MedMCQA questions with various filtering options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract questions
  python getmedmcqa.py -d /path/to/output
  python getmedmcqa.py -s "Pharmacology" -n 100 -d /path/to/output
  python getmedmcqa.py -s "Ophthalmology" -t validation -d /path/to/output
  python getmedmcqa.py -c -d /path/to/output
  # Filter by choice type
  python getmedmcqa.py --choice-type single -d /path/to/output
  # Filter by explanation presence (default: only questions with explanations)
  python getmedmcqa.py -d /path/to/output
  python getmedmcqa.py --exp with -d /path/to/output
  python getmedmcqa.py --exp all -d /path/to/output  # include questions without explanations
  # Filter by explanation word count
  python getmedmcqa.py --min-exp-words 10 -d /path/to/output
  # Get information
  python getmedmcqa.py --list-subjects
  python getmedmcqa.py --list-datasets
  python getmedmcqa.py --info
  # Quiet mode (no logging)
  python getmedmcqa.py --quiet --list-subjects
  python getmedmcqa.py --quiet -s "Ophthalmology" -d /path/to/output
        """
    )
    
    # Add arguments
    parser.add_argument('-s', '--subject', 
                       help='Subject name to filter by (default: all subjects)')
    parser.add_argument('-n', '--num-questions', type=int,
                       help='Maximum number of questions to extract (default: all questions)')
    parser.add_argument('-t', '--dataset', choices=['train', 'test', 'validation'],
                       help='Dataset split to use (default: all datasets)')
    parser.add_argument('-c', '--combine', action='store_true',
                       help='Combine all datasets into one file (only when no dataset specified)')
    parser.add_argument('-d', '--destination',
                       help='Destination directory path for output files (required for extraction)')
    
    # Info options
    parser.add_argument('--list-subjects', action='store_true',
                       help='List all available subjects from all datasets')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all available datasets')
    parser.add_argument('--info', action='store_true',
                       help='Show both available subjects and datasets')
    parser.add_argument('--quiet', action='store_true',
                       help='Turn off logging (quiet mode)')
    
    # New choice type filter
    parser.add_argument('--choice-type', choices=['single', 'multiple', 'all'], default='all',
                       help="Filter by choice_type column: 'single', 'multiple', or 'all' (default: all)")
    parser.add_argument('--exp', choices=['with', 'without', 'all'], default='with',
                       help="Filter by explanation presence: 'with', 'without', or 'all' (default: with)")
    parser.add_argument('--min-exp-words', type=int, metavar='N',
                       help="Filter to questions with explanations containing at least N words (default: no minimum)")
    
    args = parser.parse_args()
    
    # Setup logging based on quiet mode
    setup_logging(args.quiet)
    
    # Handle info options first (these take precedence)
    if args.list_subjects or args.info:
        logger.info("Retrieving available subjects...")
        available_subjects = get_available_subjects()
        
        print("\n=== AVAILABLE SUBJECTS ===")
        for dataset, subjects in available_subjects.items():
            print(f"\n{dataset.upper()} dataset ({len(subjects)} subjects):")
            for subject in sorted(subjects):
                print(f"  • {subject}")
        
        # Show combined unique subjects
        all_subjects = set()
        for subjects in available_subjects.values():
            all_subjects.update(subjects)
        print(f"\nTotal unique subjects across all datasets: {len(all_subjects)}")
        print("Combined list:")
        for subject in sorted(all_subjects):
            print(f"  • {subject}")
        print("\n=== CHOICE TYPES ===\n  • single\n  • multiple\n  • all (default)")
        print("\n=== EXPLANATION PRESENCE (exp) ===\n  • with (has explanation) - default\n  • without (no explanation)\n  • all")
        print("\n=== EXPLANATION WORD COUNT ===\n  • --min-exp-words N (filter to explanations with >= N words)")
        
        # Also show choice_type distribution
        splits = {
            'train': 'data/train-00000-of-00001.parquet',
            'test': 'data/test-00000-of-00001.parquet',
            'validation': 'data/validation-00000-of-00001.parquet'
        }

        if args.subject:
            # If a subject was provided, show per-subject choice_type counts (by split and combined)
            print(f"\n=== CHOICE TYPE COUNTS FOR SUBJECT: {args.subject} ===")
            combined_counts = {}
            total_matched = 0
            for split, rel_path in splits.items():
                try:
                    df = pd.read_parquet(f"hf://datasets/openlifescienceai/medmcqa/{rel_path}")
                    if 'subject_name' not in df.columns:
                        print(f"\n{split.upper()} dataset: 'subject_name' column not found")
                        continue
                    # Case-insensitive exact match, then fallback to contains
                    mask = df['subject_name'].str.lower() == args.subject.lower()
                    filtered = df[mask]
                    if len(filtered) == 0:
                        mask = df['subject_name'].str.lower().str.contains(args.subject.lower(), na=False)
                        filtered = df[mask]
                    if len(filtered) == 0:
                        print(f"\n{split.upper()} dataset: no questions matched subject '{args.subject}'")
                        continue
                    total_matched += len(filtered)
                    if 'choice_type' in filtered.columns:
                        counts = filtered['choice_type'].fillna('unknown').value_counts()
                        print(f"\n{split.upper()} dataset ({len(filtered)} questions) choice_type counts for subject '{args.subject}':")
                        for ct, cnt in counts.items():
                            print(f"  • {ct}: {cnt}")
                            combined_counts[ct] = combined_counts.get(ct, 0) + cnt
                    else:
                        print(f"\n{split.upper()} dataset: 'choice_type' column not found for subject '{args.subject}'")
                except Exception as e:
                    print(f"\n{split.upper()} dataset: error computing subject choice_type counts ({e})")
            # Combined totals
            if total_matched > 0:
                print(f"\nCOMBINED across datasets for subject '{args.subject}' ({total_matched} questions):")
                for ct, cnt in sorted(combined_counts.items(), key=lambda x: (-x[1], x[0])):
                    print(f"  • {ct}: {cnt}")
        else:
            # No subject provided: show overall choice_type distribution per dataset
            print("\n=== CHOICE TYPE COUNTS BY DATASET ===")
            for split, rel_path in splits.items():
                try:
                    df = pd.read_parquet(f"hf://datasets/openlifescienceai/medmcqa/{rel_path}")
                    if 'choice_type' in df.columns:
                        counts = df['choice_type'].fillna('unknown').value_counts()
                        print(f"\n{split.upper()} dataset choice_type counts:")
                        for ct, cnt in counts.items():
                            print(f"  • {ct}: {cnt}")
                    else:
                        print(f"\n{split.upper()} dataset: 'choice_type' column not found")
                except Exception as e:
                    print(f"\n{split.upper()} dataset: error reading choice_type counts ({e})")

        # Explanations statistics (count with exp and word-count stats)
        def exp_mask(s: pd.Series) -> pd.Series:
            return s.notnull() & (s.astype(str).str.strip() != "")
        def exp_word_counts(s: pd.Series) -> pd.Series:
            return s.astype(str).str.strip().str.split().apply(len)

        if args.subject:
            print(f"\n=== EXPLANATION STATS FOR SUBJECT: {args.subject} ===")
            combined_wc = []
            total_with_exp = 0
            total_rows = 0
            for split, rel_path in splits.items():
                try:
                    df = pd.read_parquet(f"hf://datasets/openlifescienceai/medmcqa/{rel_path}")
                    total_rows += len(df)
                    if 'subject_name' not in df.columns or 'exp' not in df.columns:
                        print(f"\n{split.upper()} dataset: missing 'subject_name' or 'exp' column")
                        continue
                    mask = df['subject_name'].str.lower() == args.subject.lower()
                    filtered = df[mask]
                    if len(filtered) == 0:
                        mask = df['subject_name'].str.lower().str.contains(args.subject.lower(), na=False)
                        filtered = df[mask]
                    if len(filtered) == 0:
                        print(f"\n{split.upper()} dataset: no questions matched subject '{args.subject}'")
                        continue
                    m = exp_mask(filtered['exp'])
                    wc = exp_word_counts(filtered.loc[m, 'exp'])
                    total_with_exp += m.sum()
                    combined_wc.append(wc)
                    print(f"\n{split.upper()} dataset: {m.sum()} with explanations (of {len(filtered)})")
                    if len(wc) > 0:
                        print(f"  • mean: {wc.mean():.2f}")
                        print(f"  • median: {wc.median():.2f}")
                        try:
                            mode_val = wc.mode().iloc[0]
                            print(f"  • mode: {int(mode_val)}")
                        except Exception:
                            print("  • mode: n/a")
                    else:
                        print("  • mean/median/mode: n/a")
                except Exception as e:
                    print(f"\n{split.upper()} dataset: error computing explanation stats ({e})")
            if len(combined_wc) > 0:
                all_wc = pd.concat(combined_wc, ignore_index=True)
                print(f"\nCOMBINED across datasets for subject '{args.subject}': {total_with_exp} with explanations")
                if len(all_wc) > 0:
                    print(f"  • mean: {all_wc.mean():.2f}")
                    print(f"  • median: {all_wc.median():.2f}")
                    try:
                        mode_val = all_wc.mode().iloc[0]
                        print(f"  • mode: {int(mode_val)}")
                    except Exception:
                        print("  • mode: n/a")
            else:
                print(f"\nNo explanation stats available for subject '{args.subject}'.")
        else:
            print("\n=== EXPLANATION STATS BY DATASET (ALL SUBJECTS) ===")
            combined_wc = []
            total_with_exp = 0
            for split, rel_path in splits.items():
                try:
                    df = pd.read_parquet(f"hf://datasets/openlifescienceai/medmcqa/{rel_path}")
                    if 'exp' not in df.columns:
                        print(f"\n{split.upper()} dataset: 'exp' column not found")
                        continue
                    m = exp_mask(df['exp'])
                    wc = exp_word_counts(df.loc[m, 'exp'])
                    total_with_exp += m.sum()
                    combined_wc.append(wc)
                    print(f"\n{split.upper()} dataset: {m.sum()} with explanations (of {len(df)})")
                    if len(wc) > 0:
                        print(f"  • mean: {wc.mean():.2f}")
                        print(f"  • median: {wc.median():.2f}")
                        try:
                            mode_val = wc.mode().iloc[0]
                            print(f"  • mode: {int(mode_val)}")
                        except Exception:
                            print("  • mode: n/a")
                    else:
                        print("  • mean/median/mode: n/a")
                except Exception as e:
                    print(f"\n{split.upper()} dataset: error computing explanation stats ({e})")
            if len(combined_wc) > 0:
                all_wc = pd.concat(combined_wc, ignore_index=True)
                print(f"\nCOMBINED across datasets: {total_with_exp} with explanations")
                if len(all_wc) > 0:
                    print(f"  • mean: {all_wc.mean():.2f}")
                    print(f"  • median: {all_wc.median():.2f}")
                    try:
                        mode_val = all_wc.mode().iloc[0]
                        print(f"  • mode: {int(mode_val)}")
                    except Exception:
                        print("  • mode: n/a")
    
    if args.list_datasets or args.info:
        logger.info("Retrieving available datasets...")
        available_datasets = get_available_datasets()
        
        print("\n=== AVAILABLE DATASETS ===")
        for dataset in available_datasets:
            print(f"  • {dataset}")
    
    # If info options were used, exit without extraction
    if args.list_subjects or args.list_datasets or args.info:
        return
    
    # Validate arguments for extraction
    if not args.destination:
        logger.error("Destination directory (-d/--destination) is required for extraction operations.")
        return
    
    if args.combine and args.dataset:
        logger.error("Cannot use --combine with --dataset. Use --combine only when no dataset is specified.")
        return
    
    logger.info("Starting MedMCQA question extraction...")
    logger.info(f"Arguments: subject={args.subject}, num_questions={args.num_questions}, "
                f"dataset={args.dataset}, combine={args.combine}, destination={args.destination}")
    
    try:
        # Get questions based on arguments
        questions_df = get_medmcqa_questions(
            subject_name=args.subject,
            max_questions=args.num_questions,
            dataset=args.dataset,
            choice_type=args.choice_type,
            exp=args.exp,
            min_exp_words=args.min_exp_words
        )
        
        # Generate filename and save
        filename = generate_filename(
            subject_name=args.subject,
            max_questions=args.num_questions,
            dataset=args.dataset,
            combine=args.combine,
            destination_dir=args.destination
        )
        
        questions_df.to_csv(filename, index=False)
        logger.info(f"Successfully saved {len(questions_df)} questions to {filename}")
        
        # Log summary statistics
        if 'subject_name' in questions_df.columns:
            subject_counts = questions_df['subject_name'].value_counts()
            logger.info("Question distribution by subject:")
            for subject, count in subject_counts.items():
                logger.info(f"  {subject}: {count} questions")
        
        if 'dataset_split' in questions_df.columns:
            dataset_counts = questions_df['dataset_split'].value_counts()
            logger.info("Question distribution by dataset:")
            for dataset, count in dataset_counts.items():
                logger.info(f"  {dataset}: {count} questions")
        
        # Log choice_type distribution
        if 'choice_type' in questions_df.columns:
            ct_counts = questions_df['choice_type'].fillna('unknown').value_counts()
            logger.info("Question distribution by choice_type:")
            for ct, count in ct_counts.items():
                logger.info(f"  {ct}: {count} questions")
                
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise

if __name__ == "__main__":
    main() 