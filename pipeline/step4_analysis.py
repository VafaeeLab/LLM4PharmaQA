#!/usr/bin/env python3

import numpy as np
import pandas as pd
import glob
import os
import argparse 
from sklearn.metrics import f1_score, mean_gamma_deviance
from scipy import stats
import pprint
from datetime import datetime

from itertools import combinations
from scipy import stats
import numpy as np

# =========================================
# Helper function: Calculate t-confidence interval
# =========================================
def calculate_t_confidence_interval(data, confidence=0.95):
    """
    Calculates the confidence interval for the mean of a sample using the t-distribution.

    Args:
        data (array-like): The sample data.
        confidence (float): The desired confidence level (e.g., 0.95 for 95%).

    Returns:
        tuple: A tuple containing (sample_mean, lower_bound, upper_bound, standard_error, n).
    """
    sample_array = np.array(data)
    n = len(sample_array) # Sample size must be greater than 2
    df = n - 1 # Degrees of freedom

    sample_mean = np.mean(sample_array)
    standard_error = np.std(sample_array, ddof=1) / np.sqrt(n)  # Standard error of the mean (using sample std)
    t_value = stats.t.ppf((1 - confidence) / 2, n - 1)
    margin_of_error = t_value * standard_error

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return sample_mean, lower_bound, upper_bound, standard_error, n

# =========================================
# Helper function: Collect metric values from models
# =========================================
def collect_metric_values(models, df_key, metrics_list):
    """
    Collects all metric values from all models' dataframes.
    
    Args:
        models (dict): Dictionary of model information containing dataframes
        df_key (str): Key to access the dataframe in each model's info dict (e.g., "df" or "df_no_error")
        metrics_list (list): List of metric names to collect
    
    Returns:
        dict: Dictionary mapping metric names to lists of all collected values
    """
    metric_values = {metric: [] for metric in metrics_list}
    for model_name, info in models.items():
        df = info[df_key]
        for metric in metrics_list:
            col = model_name + "_" + metric
            if col and col in df.columns:
                values = df[col].dropna().tolist()
                metric_values[metric].extend(values)
    return metric_values

# =========================================
# Helper function: Calculate metric ranges
# =========================================
def calculate_metric_ranges(metric_values, metrics_list):
    """
    Calculates min/max ranges for each metric from collected values.
    
    Args:
        metric_values (dict): Dictionary mapping metric names to lists of values
        metrics_list (list): List of metric names to process
    
    Returns:
        dict: Dictionary mapping metric names to {"min": min_value, "max": max_value}
    """
    ranges = {}
    for metric in metrics_list:
        ranges[metric] = {
            "min": min(metric_values[metric]),
            "max": max(metric_values[metric])
        }
    return ranges

# =========================================
# Helper function: Normalize dataframe metrics
# =========================================
def normalize_dataframe(df, model_name, metrics_list, metric_ranges):
    """
    Normalizes metric columns in a dataframe using the provided ranges.
    
    Args:
        df (pd.DataFrame): Dataframe to normalize
        model_name (str): Name of the model (used to construct column names)
        metrics_list (list): List of metric names to normalize
        metric_ranges (dict): Dictionary mapping metric names to {"min": min_value, "max": max_value}
    
    Returns:
        pd.DataFrame: Normalized dataframe
    """
    df_normalized = df.copy()
    for metric in metrics_list:
        col = model_name + "_" + metric
        if col and col in df_normalized.columns:
            m_range = metric_ranges[metric]
            if m_range["max"] != m_range["min"]:
                df_normalized[col] = (df_normalized[col] - m_range["min"]) / (m_range["max"] - m_range["min"])
            else:
                # All same value -> set to 0.5
                df_normalized[col] = 0.5
    return df_normalized

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to the folder containing CSV files')
args = parser.parse_args()
path = args.path

# Create output folder with date timestamp
output_folder = os.path.join(path, f"step4-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder created: {output_folder}")

# =========================================
# 1. Load all files and identify models
# =========================================
files = [f for f in glob.glob(os.path.join(path, "*.csv"))]

models = {}
metrics_list = ["accuracy", "bert_scores", "bart_scores", "alignscore", "rougeL", "meteor"]

for f in files:
    df = pd.read_csv(f)
    
    # Find the column that ends in "_accuracy" (excluding 'correct_answer' or similar)
    llm_response_cols = [c for c in df.columns if c.endswith("_response")]
    model_name = llm_response_cols[0][:-len("_response")]
    
    # Remove rows where accuracy column says "Error"
    answer_col = model_name + "_answer"
    df_no_error = df[df[answer_col] != "ERROR"]

    
    models[model_name] = {
        "df": df,
        "df_no_error": df_no_error,
        "file": f
    }

# =========================================
# 2. Collect all metric values for global normalization
# =========================================
all_metric_values = collect_metric_values(models, "df", metrics_list)
no_error_metric_values = collect_metric_values(models, "df_no_error", metrics_list)   

# Calculate global min/max for normalization
metric_ranges = calculate_metric_ranges(all_metric_values, metrics_list)
no_error_metric_ranges = calculate_metric_ranges(no_error_metric_values, metrics_list)

print(metric_ranges)
print(no_error_metric_ranges)
print(len(models["deepseek-r1-1.5"]["df_no_error"]["deepseek-r1-1.5_bert_scores"]))
print(len(models["deepseek-r1-1.5"]["df"]["deepseek-r1-1.5_bert_scores"]))

# Normalise data and save normalised data to a file - 2 versions, one with errors, one without
for model_name, info in models.items():
    df = normalize_dataframe(info["df"], model_name, metrics_list, metric_ranges)
    df_no_error = normalize_dataframe(info["df_no_error"], model_name, metrics_list, no_error_metric_ranges)
    
    # Add weighted average score column (mean of all normalized metrics excluding accuracy)
    other_metrics = [m for m in metrics_list if m != "accuracy"]
    metric_cols = [model_name + "_" + metric for metric in other_metrics]
    available_cols_df = [col for col in metric_cols if col in df.columns]
    available_cols_df_no_error = [col for col in metric_cols if col in df_no_error.columns]
    
    if available_cols_df:
        df[f"{model_name}_weighted_normalized_score"] = df[available_cols_df].mean(axis=1)
    if available_cols_df_no_error:
        df_no_error[f"{model_name}_weighted_normalized_score"] = df_no_error[available_cols_df_no_error].mean(axis=1)
    
    df.to_csv(os.path.join(output_folder, f"{model_name}_norm.csv"), index=False)
    df_no_error.to_csv(os.path.join(output_folder, f"{model_name}_norm_no_error.csv"), index=False)
    print(f"Saved normalised data: {os.path.join(output_folder, f"{model_name}_norm.csv")}")
    print(f"Saved normalised data without errors: {os.path.join(output_folder, f"{model_name}_norm_no_error.csv")}")

    models[model_name]["df_norm"] = df
    models[model_name]["df_no_error_norm"] = df_no_error

# See if there is a statistical significant difference in performance between models - for accuracy, f1 score, each metric and weighted normalized scores    

# Now are data is done YAY

# =========================================
# Functions to calculate metrics for a single model
# =========================================

def calculate_model_accuracy(model_name, model_info, use_no_error=True):
    """
    Calculate accuracy for a model (num correct / num questions) with confidence intervals.
    
    Args:
        model_name (str): Name of the model
        model_info (dict): Dictionary containing model dataframes
        use_no_error (bool): Whether to use df_no_error_norm (True) or df_norm (False)
    
    Returns:
        dict: Dictionary with mean, ci_lower, ci_upper, standard_error, n for accuracy
    """
    df_key = "df_no_error_norm" if use_no_error else "df_norm"
    df = model_info[df_key]
    
    accuracy_col = model_name + "_accuracy"
    if accuracy_col not in df.columns:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 0}
    
    accuracy_values = df[accuracy_col].dropna().values.astype(float)
    
    if len(accuracy_values) >= 2:
        mean, lower, upper, standard_error, n = calculate_t_confidence_interval(accuracy_values, confidence=0.95)
        return {"mean": mean, "ci_lower": lower, "ci_upper": upper, "standard_error": standard_error, "n": n}
    elif len(accuracy_values) == 1:
        return {"mean": float(accuracy_values[0]), "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 1}
    else:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 0}



def calculate_model_metric_means(model_name, model_info, metrics_list, use_no_error=True):
    """
    Calculate mean of normalized metrics for a model with confidence intervals.
    
    Args:
        model_name (str): Name of the model
        model_info (dict): Dictionary containing model dataframes
        metrics_list (list): List of metric names (excluding "accuracy")
        use_no_error (bool): Whether to use df_no_error_norm (True) or df_norm (False)
    
    Returns:
        dict: Dictionary mapping metric names to {"mean": mean, "ci_lower": lower, "ci_upper": upper, "standard_error": se, "n": n}
    """
    df_key = "df_no_error_norm" if use_no_error else "df_norm"
    df = model_info[df_key]
    
    results = {}
    for metric in metrics_list:
        col = model_name + "_" + metric
        if col and col in df.columns:
            values = df[col].dropna().values.astype(float)
            
            if len(values) >= 2:
                mean, lower, upper, standard_error, n = calculate_t_confidence_interval(values, confidence=0.95)
                results[metric] = {"mean": mean, "ci_lower": lower, "ci_upper": upper, "standard_error": standard_error, "n": n}
            elif len(values) == 1:
                results[metric] = {"mean": float(values[0]), "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 1}
            else:
                results[metric] = {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 0}
        else:
            results[metric] = {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 0}
    
    return results


def calculate_model_weighted_normalized_score(model_name, model_info, use_no_error=True):
    """
    Calculate weighted normalized score for a model using the pre-calculated column in the dataframe.
    The weighted score column contains the mean of all normalized metrics (excluding accuracy) for each row.
    
    Args:
        model_name (str): Name of the model
        model_info (dict): Dictionary containing model dataframes
        use_no_error (bool): Whether to use df_no_error_norm (True) or df_norm (False)
    
    Returns:
        dict: Dictionary with mean, ci_lower, ci_upper, standard_error, n for weighted normalized score
    """
    df_key = "df_no_error_norm" if use_no_error else "df_norm"
    df = model_info[df_key]
    
    # Use the pre-calculated weighted normalized score column
    weighted_score_col = f"{model_name}_weighted_normalized_score"
    
    if weighted_score_col not in df.columns:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 0}
    
    weighted_scores = df[weighted_score_col].dropna().values.astype(float)
    
    if len(weighted_scores) >= 2:
        mean, lower, upper, standard_error, n = calculate_t_confidence_interval(weighted_scores, confidence=0.95)
        return {"mean": mean, "ci_lower": lower, "ci_upper": upper, "standard_error": standard_error, "n": n}
    elif len(weighted_scores) == 1:
        return {"mean": float(weighted_scores[0]), "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 1}
    else:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "standard_error": np.nan, "n": 0} 

def calculate_all_model_metrics(model_name, model_info, metrics_list, use_no_error=True):
    """
    Calculate all metrics for a single model.
    
    Args:
        model_name (str): Name of the model
        model_info (dict): Dictionary containing model dataframes
        metrics_list (list): List of metric names
        use_no_error (bool): Whether to use df_no_error_norm (True) or df_norm (False)
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Exclude accuracy from metrics_list for metric means calculation
    other_metrics = [m for m in metrics_list if m != "accuracy"]
    
    results = {
        "model_name": model_name,
        "accuracy": calculate_model_accuracy(model_name, model_info, use_no_error),
        "metric_means": calculate_model_metric_means(model_name, model_info, other_metrics, use_no_error),
        "weighted_normalized_score": calculate_model_weighted_normalized_score(model_name, model_info, use_no_error)
    }
    
    return results



# =========================================
# Apply calculations to all models
# =========================================
print("\n=== Calculating metrics for all models ===")
all_model_results = []

for model_name, model_info in models.items():
    results = calculate_all_model_metrics(model_name, model_info, metrics_list, use_no_error=True)
    all_model_results.append(results)
    print(f"Completed calculations for {model_name}")


pprint.pprint(all_model_results)


def run_wilcoxon_signed_for_metric(models, metric_name, use_no_error=True, alpha=0.05):
    """
    Run pairwise 2-tailed Wilcoxon signed-rank tests across all models
    for a given metric, with Bonferroni correction.

    This assumes each model is evaluated on the SAME item set (paired design).

    Args:
        models (dict): your models dict with "df_norm" / "df_no_error_norm"
        metric_name (str): suffix of the metric column, e.g.
                           "weighted_normalized_score", "rougeL", "bert_scores"
                           (full column is f"{model_name}_{metric_name}")
        use_no_error (bool): if True, use "df_no_error_norm", else "df_norm"
        alpha (float): nominal significance level (for reference; we just report p-values)

    Returns:
        list of dict: one entry per pairwise comparison with raw and Bonferroni-corrected p-values
    """
    model_names = list(models.keys())
    df_key = "df_no_error_norm" if use_no_error else "df_norm"

    pairs = list(combinations(model_names, 2))
    m = len(pairs)  # number of comparisons for Bonferroni

    results = []

    print(f"\n=== Wilcoxon signed-rank tests for metric '{metric_name}' "
          f"using {df_key} (2-tailed, Bonferroni corrected) ===")

    for m1, m2 in pairs:
        df1 = models[m1][df_key]
        df2 = models[m2][df_key]

        col1 = f"{m1}_{metric_name}"
        col2 = f"{m2}_{metric_name}"

        if col1 not in df1.columns or col2 not in df2.columns:
            print(f"{m1} vs {m2}: metric column missing ({col1} or {col2})")
            continue

        # Align on index and keep only rows where BOTH are non-NaN
        paired = pd.concat(
            [df1[col1], df2[col2]],
            axis=1,
            keys=["x", "y"]
        ).dropna()

        x = paired["x"].astype(float).values
        y = paired["y"].astype(float).values

        n = len(paired)

        if n < 2:
            print(f"{m1} vs {m2}: not enough paired data (n = {n})")
            continue

        # Wilcoxon signed-rank test (paired, 2-sided)
        # Note: wilcoxon drops zero-differences internally.
        w_stat, p_raw = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")

        p_bonf = min(p_raw * m, 1.0)

        result = {
            "metric": metric_name,
            "model_1": m1,
            "model_2": m2,
            "w_stat": w_stat,
            "p_raw": p_raw,
            "p_bonf": p_bonf,
            "n_paired": n,
            "alpha": alpha
        }
        results.append(result)

        print(
            f"{m1} vs {m2}: W = {w_stat:.4f}, "
            f"p_raw = {p_raw:.4g}, p_bonf = {p_bonf:.4g} "
            f"(n_paired = {n})"
        )

    return results


def run_wilcoxon_for_metric(models, metric_name, use_no_error=True, alpha=0.05):
    """
    Run pairwise 2-tailed Wilcoxon rank-sum (Mann–Whitney U) tests
    across all models for a given metric, with Bonferroni correction.

    Args:
        models (dict): your models dict with "df_norm" / "df_no_error_norm"
        metric_name (str): suffix of the metric column, e.g.
                           "weighted_normalized_score", "rougeL", "bert_scores"
                           (full column is assumed to be f"{model_name}_{metric_name}")
        use_no_error (bool): if True, use "df_no_error_norm", else "df_norm"
        alpha (float): nominal significance level (for reference; we just report p-values)

    Returns:
        list of dict: one entry per pairwise comparison with raw and Bonferroni-corrected p-values
    """
    model_names = list(models.keys())

    def get_metric_values(model_name):
        df_key = "df_no_error_norm" if use_no_error else "df_norm"
        df = models[model_name][df_key]

        col = f"{model_name}_{metric_name}"
        if col not in df.columns:
            return np.array([])

        return df[col].dropna().astype(float).values

    pairs = list(combinations(model_names, 2))
    m = len(pairs)  # number of comparisons for Bonferroni

    results = []

    dataset_label = "df_no_error_norm" if use_no_error else "df_norm"
    print(f"\n=== Wilcoxon rank-sum tests for metric '{metric_name}' "
          f"using {dataset_label} (2-tailed, Bonferroni corrected) ===")

    for m1, m2 in pairs:
        x = get_metric_values(m1)
        y = get_metric_values(m2)

        if len(x) >= 2 and len(y) >= 2:
            u_stat, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_bonf = min(p_raw * m, 1.0)  # Bonferroni correction

            result = {
                "metric": metric_name,
                "model_1": m1,
                "model_2": m2,
                "u_stat": u_stat,
                "p_raw": p_raw,
                "p_bonf": p_bonf,
                "n1": len(x),
                "n2": len(y),
                "alpha": alpha
            }
            results.append(result)

            print(
                f"{m1} vs {m2}: U = {u_stat:.4f}, "
                f"p_raw = {p_raw:.4g}, p_bonf = {p_bonf:.4g} "
                f"(n1 = {len(x)}, n2 = {len(y)})"
            )
        else:
            print(
                f"{m1} vs {m2}: not enough data "
                f"(n1 = {len(x)}, n2 = {len(y)})"
            )

    return results


# =========================================
# Category-based analysis functions
# =========================================

def get_model_category(model_name):
    """
    Extract category from model name.
    
    Args:
        model_name (str): Model name (e.g., "deepseek-r1-1.5", "llama3-8b", "med42-70b")
    
    Returns:
        str: Category name (e.g., "deepseek", "llama", "med42")
    """
    model_lower = model_name.lower()
    
    if "deepseek" in model_lower:
        return "deepseek"
    elif "llama" in model_lower:
        return "llama"
    elif "med42" in model_lower or "medv2" in model_lower:
        return "med42"
    else:
        # Default: use first part of name before first dash or underscore
        parts = model_name.replace("_", "-").split("-")
        return parts[0].lower()


def group_models_by_category(models):
    """
    Group models by their category.
    
    Args:
        models (dict): Dictionary of model information
    
    Returns:
        dict: Dictionary mapping category names to lists of model names
    """
    categories = {}
    for model_name in models.keys():
        category = get_model_category(model_name)
        if category not in categories:
            categories[category] = []
        categories[category].append(model_name)
    return categories


def get_category_metric_values(models, category, metric_name, use_no_error=True):
    """
    Aggregate all metric values from all models in a category.
    
    Args:
        models (dict): Dictionary of model information
        category (str): Category name
        metric_name (str): Metric name (e.g., "accuracy", "weighted_normalized_score")
        use_no_error (bool): Whether to use df_no_error_norm (True) or df_norm (False)
    
    Returns:
        np.array: Array of all metric values from all models in the category
    """
    df_key = "df_no_error_norm" if use_no_error else "df_norm"
    all_values = []
    
    category_models = [m for m in models.keys() if get_model_category(m) == category]
    
    for model_name in category_models:
        df = models[model_name][df_key]
        col = f"{model_name}_{metric_name}"
        if col in df.columns:
            values = df[col].dropna().astype(float).values
            all_values.extend(values)
    
    return np.array(all_values)


def run_wilcoxon_for_category_metric(models, metric_name, use_no_error=True, alpha=0.05):
    """
    Run pairwise 2-tailed Wilcoxon rank-sum (Mann–Whitney U) tests
    across model categories for a given metric, with Bonferroni correction.
    
    Args:
        models (dict): Dictionary of model information
        metric_name (str): Metric name (e.g., "accuracy", "weighted_normalized_score")
        use_no_error (bool): Whether to use df_no_error_norm (True) or df_norm (False)
        alpha (float): Nominal significance level
    
    Returns:
        list of dict: One entry per pairwise category comparison with raw and Bonferroni-corrected p-values
    """
    categories_dict = group_models_by_category(models)
    category_names = list(categories_dict.keys())
    
    pairs = list(combinations(category_names, 2))
    m = len(pairs)  # number of comparisons for Bonferroni
    
    results = []
    
    dataset_label = "df_no_error_norm" if use_no_error else "df_norm"
    print(f"\n=== Wilcoxon rank-sum tests for metric '{metric_name}' by CATEGORY "
          f"using {dataset_label} (2-tailed, Bonferroni corrected) ===")
    print(f"Categories: {category_names}")
    print(f"Category groupings: {categories_dict}")
    
    for cat1, cat2 in pairs:
        x = get_category_metric_values(models, cat1, metric_name, use_no_error)
        y = get_category_metric_values(models, cat2, metric_name, use_no_error)
        
        if len(x) >= 2 and len(y) >= 2:
            u_stat, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_bonf = min(p_raw * m, 1.0)  # Bonferroni correction
            
            result = {
                "metric": metric_name,
                "category_1": cat1,
                "category_2": cat2,
                "u_stat": u_stat,
                "p_raw": p_raw,
                "p_bonf": p_bonf,
                "n1": len(x),
                "n2": len(y),
                "alpha": alpha
            }
            results.append(result)
            
            print(
                f"{cat1} vs {cat2}: U = {u_stat:.4f}, "
                f"p_raw = {p_raw:.4g}, p_bonf = {p_bonf:.4g} "
                f"(n1 = {len(x)}, n2 = {len(y)})"
            )
        else:
            print(
                f"{cat1} vs {cat2}: not enough data "
                f"(n1 = {len(x)}, n2 = {len(y)})"
            )
    
    return results


def run_wilcoxon_signed_for_category_metric(models, metric_name, use_no_error=True, alpha=0.05):
    """
    Run pairwise 2-tailed Wilcoxon signed-rank tests across model categories
    for a given metric, with Bonferroni correction.
    
    This assumes models within each category are evaluated on the SAME item set (paired design).
    We pair corresponding items across categories (e.g., item 1 from all llama models vs item 1 from all deepseek models).
    
    Args:
        models (dict): Dictionary of model information
        metric_name (str): Metric name (e.g., "accuracy", "weighted_normalized_score")
        use_no_error (bool): Whether to use df_no_error_norm (True) or df_norm (False)
        alpha (float): Nominal significance level
    
    Returns:
        list of dict: One entry per pairwise category comparison with raw and Bonferroni-corrected p-values
    """
    categories_dict = group_models_by_category(models)
    category_names = list(categories_dict.keys())
    
    pairs = list(combinations(category_names, 2))
    m = len(pairs)  # number of comparisons for Bonferroni
    
    results = []
    
    df_key = "df_no_error_norm" if use_no_error else "df_norm"
    print(f"\n=== Wilcoxon signed-rank tests for metric '{metric_name}' by CATEGORY "
          f"using {df_key} (2-tailed, Bonferroni corrected) ===")
    print(f"Categories: {category_names}")
    print(f"Category groupings: {categories_dict}")
    
    for cat1, cat2 in pairs:
        # Get all models in each category
        cat1_models = categories_dict[cat1]
        cat2_models = categories_dict[cat2]
        
        # Collect paired values (same row index across models in each category)
        paired_values = []
        
        # Use the first model's dataframe to get row indices
        if cat1_models and cat2_models:
            df1_ref = models[cat1_models[0]][df_key]
            df2_ref = models[cat2_models[0]][df_key]
            
            # Find common indices
            common_indices = df1_ref.index.intersection(df2_ref.index)
            
            for idx in common_indices:
                # Get mean value across all models in category 1 for this row
                cat1_values = []
                for model_name in cat1_models:
                    df = models[model_name][df_key]
                    col = f"{model_name}_{metric_name}"
                    if col in df.columns and idx in df.index:
                        val = df.loc[idx, col]
                        if pd.notna(val):
                            cat1_values.append(float(val))
                
                # Get mean value across all models in category 2 for this row
                cat2_values = []
                for model_name in cat2_models:
                    df = models[model_name][df_key]
                    col = f"{model_name}_{metric_name}"
                    if col in df.columns and idx in df.index:
                        val = df.loc[idx, col]
                        if pd.notna(val):
                            cat2_values.append(float(val))
                
                # If both categories have at least one valid value, use the mean
                if cat1_values and cat2_values:
                    paired_values.append((np.mean(cat1_values), np.mean(cat2_values)))
        
        if len(paired_values) >= 2:
            x = np.array([p[0] for p in paired_values])
            y = np.array([p[1] for p in paired_values])
            
            w_stat, p_raw = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            p_bonf = min(p_raw * m, 1.0)
            
            result = {
                "metric": metric_name,
                "category_1": cat1,
                "category_2": cat2,
                "w_stat": w_stat,
                "p_raw": p_raw,
                "p_bonf": p_bonf,
                "n_paired": len(paired_values),
                "alpha": alpha
            }
            results.append(result)
            
            print(
                f"{cat1} vs {cat2}: W = {w_stat:.4f}, "
                f"p_raw = {p_raw:.4g}, p_bonf = {p_bonf:.4g} "
                f"(n_paired = {len(paired_values)})"
            )
        else:
            print(
                f"{cat1} vs {cat2}: not enough paired data "
                f"(n_paired = {len(paired_values)})"
            )
    
    return results


# Individual model comparisons - run Wilcoxon tests for all metrics
print("\n" + "="*80)
print("INDIVIDUAL MODEL COMPARISONS - WILCOXON RANK-SUM TESTS")
print("="*80)
for metric in metrics_list:
    print(f"\n--- Metric: {metric} ---")
    results = run_wilcoxon_for_metric(models, metric)
    pprint.pprint(results)

print("\n" + "="*80)
print("INDIVIDUAL MODEL COMPARISONS - WILCOXON SIGNED-RANK TESTS")
print("="*80)
for metric in metrics_list:
    print(f"\n--- Metric: {metric} ---")
    results = run_wilcoxon_signed_for_metric(models, metric)
    pprint.pprint(results)

# Weighted normalized score (separate since it's not in metrics_list)
print("\n" + "="*80)
print("INDIVIDUAL MODEL COMPARISONS - WEIGHTED NORMALIZED SCORE")
print("="*80)
print("--- Wilcoxon Rank-Sum Test ---")
pprint.pprint(run_wilcoxon_for_metric(models, "weighted_normalized_score"))
print("--- Wilcoxon Signed-Rank Test ---")
pprint.pprint(run_wilcoxon_signed_for_metric(models, "weighted_normalized_score"))

# Category-based analysis - run Wilcoxon tests for all metrics
print("\n" + "="*80)
print("CATEGORY-BASED ANALYSIS - WILCOXON RANK-SUM TESTS")
print("="*80)
for metric in metrics_list:
    print(f"\n--- Metric: {metric} ---")
    results = run_wilcoxon_for_category_metric(models, metric)
    pprint.pprint(results)

print("\n" + "="*80)
print("CATEGORY-BASED ANALYSIS - WILCOXON SIGNED-RANK TESTS")
print("="*80)
for metric in metrics_list:
    print(f"\n--- Metric: {metric} ---")
    results = run_wilcoxon_signed_for_category_metric(models, metric)
    pprint.pprint(results)

# Weighted normalized score for categories
print("\n" + "="*80)
print("CATEGORY-BASED ANALYSIS - WEIGHTED NORMALIZED SCORE")
print("="*80)
print("--- Wilcoxon Rank-Sum Test ---")
pprint.pprint(run_wilcoxon_for_category_metric(models, "weighted_normalized_score"))
print("--- Wilcoxon Signed-Rank Test ---")
pprint.pprint(run_wilcoxon_signed_for_category_metric(models, "weighted_normalized_score"))

# Calculate if there is a significant difference in performance between models - for accuracy, f1 score, each metric and weighted normalized scores    
# two sample t test with 0.05 significance level and bonferroni correction
