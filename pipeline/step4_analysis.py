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
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
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
# print("\n=== Calculating metrics for all models ===")
# all_model_results = []

# for model_name, model_info in models.items():
#     results = calculate_all_model_metrics(model_name, model_info, metrics_list, use_no_error=True)
#     all_model_results.append(results)
#     print(f"Completed calculations for {model_name}")

# pprint.pprint(all_model_results)

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    boot_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return lower, upper


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

        diffs = x - y # mean difference
    # Mean difference
        mean_diff = np.mean(diffs)

        # # Bootstrap CI
        # ci_lower, ci_upper = bootstrap_ci(diffs)      
        # if n < 2:
        #     print(f"{m1} vs {m2}: not enough paired data (n = {n})")
        #     continue

        # USING T TEST FOR CONFIDENCE INTERVAL
        mean_diff, lower, upper, standard_error, n = calculate_t_confidence_interval(diffs, confidence=0.95)

        # Wilcoxon signed-rank test (paired, 2-sided)
        # Note: wilcoxon drops zero-differences internally.
        t_stat, p_raw = stats.ttest_rel(x, y, nan_policy="omit")    

        p_bonf = min(p_raw * m, 1.0)

        result = {
            "mean_diff": mean_diff,
            "ci_lower": lower,
            "ci_upper": upper,
            "standard_error": standard_error,
            "n": n,
            "metric": metric_name,
            "model_1": m1,
            "model_2": m2,
            "t_stat": t_stat,
            "p_raw": p_raw,
            "p_bonf": p_bonf,
            "n_paired": n,
            "alpha": alpha,
            "m": m,
        }
        results.append(result)

        print(
            f"{m1} vs {m2}: W = {t_stat:.4f}, "
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
            
            # Calculate mean difference (unpaired: mean(x) - mean(y))
            mean_diff = np.mean(x) - np.mean(y)

            result = {
                "mean_diff": mean_diff,
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





# Weighted normalized score (separate since it's not in metrics_list)
print("\n" + "="*80)
print("INDIVIDUAL MODEL COMPARISONS - WEIGHTED NORMALIZED SCORE")
print("="*80)
print("--- Wilcoxon Signed-Rank Test ---")
pprint.pprint(run_wilcoxon_signed_for_metric(models, "weighted_normalized_score"))






def paired_t_test(metric_name, models, use_no_error=True, alpha=0.05):
    """
    Calculate the mean difference in accuracy between models.
    """
    model_names = list(models.keys())
    df_key = "df_no_error_norm" if use_no_error else "df_norm"

    pairs = list(combinations(model_names, 2))
    m = len(pairs)  # number of comparisons for Bonferroni

    results = []

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

        diffs = x - y # mean difference
        mean_diff, lower, upper, standard_error, n = calculate_t_confidence_interval(diffs, confidence=0.95)
        
        if n < 2:
            print(f"{m1} vs {m2}: not enough paired data (n = {n})")
            continue

        # Wilcoxon signed-rank test (paired, 2-sided)
        # Note: wilcoxon drops zero-differences internally.
        w_stat, p_raw = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")

        p_bonf = min(p_raw * m, 1.0)

        result = {
            "mean_diff": mean_diff,
            "ci_lower": lower,
            "ci_upper": upper,
            "standard_error": standard_error,
            "n": n,
            "metric": metric_name,
            "model_1": m1,
            "model_2": m2,
            "w_stat": w_stat,
            "p_raw": p_raw,
            "p_bonf": p_bonf,
            "n_paired": n,
            "alpha": alpha,
            "m": m,
        }
        results.append(result)

        print(
            f"{m1} vs {m2}: W = {w_stat:.4f}, "
            f"p_raw = {p_raw:.4g}, p_bonf = {p_bonf:.4g} "
            f"(n_paired = {n})"
        )

    return results



# Accuracy (separate since it's handled differently)
print("\n" + "="*80)
print("INDIVIDUAL MODEL COMPARISONS - ACCURACY")
print("="*80)
print("--- Paired T Test ---")
pprint.pprint(paired_t_test("accuracy", models))


# Calculate if there is a significant difference in performance between models - for accuracy, f1 score, each metric and weighted normalized scores    
# two sample t test with 0.05 significance level and bonferroni correction
