import pandas as pd
import numpy as np
import os
import re
from scipy.stats import ttest_rel

# -------- CONFIG --------
input_folder = "all_data" # put all data files into one folder for analysis
output_file = "big5_confidence_results.csv"

PAIR_DEFS = {
    "makeup": ("have_makeup", "no_makeup"),
    "hair": ("long_hair", "short_hair"),
    "jewelry": ("have_jewerly", "no_jewerly")
}

# -------- LOAD FUNCTION --------
def load_excel(filepath):
    xls = pd.ExcelFile(filepath)
    rows = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)

        for _, row in df.iterrows():
            if pd.notna(row["Positive Trait"]):
                rows.append({
                    "Category": sheet,
                    "Trait": row["Positive Trait"],
                    "Valence": "Positive",
                    "Score": row["Positive Probability"]
                })
            if pd.notna(row["Negative Trait"]):
                rows.append({
                    "Category": sheet,
                    "Trait": row["Negative Trait"],
                    "Valence": "Negative",
                    "Score": row["Negative Probability"]
                })

    return pd.DataFrame(rows)

# -------- PARSE FILENAME --------
def parse_filename(filename):
    name = filename.replace("clip_big5_", "").replace(".xlsx", "")

    # example: male1_have_makeup
    parts = name.split("_")

    identity = parts[0]   # male1 / female1
    condition = "_".join(parts[1:])

    return identity, condition

# -------- LOAD ALL DATA --------
data = {}

for file in os.listdir(input_folder):
    if file.endswith(".xlsx"):
        identity, condition = parse_filename(file)
        key = (identity, condition)

        filepath = os.path.join(input_folder, file)
        data[key] = load_excel(filepath)

# -------- ANALYSIS --------
results = []

for condition_name, (with_tag, without_tag) in PAIR_DEFS.items():

    identities = set([k[0] for k in data.keys()])

    for identity in identities:

        key_with = (identity, with_tag)
        key_without = (identity, without_tag)

        if key_with not in data or key_without not in data:
            continue

        df_with = data[key_with]
        df_without = data[key_without]

        merged = pd.merge(
            df_with,
            df_without,
            on=["Category", "Trait", "Valence"],
            suffixes=("_with", "_without")
        )

        merged["Diff"] = merged["Score_with"] - merged["Score_without"]

        # -------- TRAIT LEVEL --------
        for trait in merged["Trait"].unique():
            sub = merged[merged["Trait"] == trait]

            results.append({
                "Identity": identity,
                "Condition": condition_name,
                "Category": sub["Category"].iloc[0],
                "Trait": trait,
                "Difference": sub["Diff"].values[0]
            })

# Convert to dataframe
df = pd.DataFrame(results)

# -------- STATISTICS --------
final_results = []

group_cols = ["Condition", "Category", "Trait"]

for name, group in df.groupby(group_cols):

    diffs = group["Difference"].values

    if len(diffs) < 2:
        continue  # not enough data

    mean = np.mean(diffs)
    std = np.std(diffs, ddof=1)

    # paired t-test vs zero
    t_stat, p_val = ttest_rel(diffs, np.zeros_like(diffs))

    # effect size
    d = mean / std if std != 0 else 0

    # confidence interval
    n = len(diffs)
    ci_low = mean - 1.96 * (std / np.sqrt(n))
    ci_high = mean + 1.96 * (std / np.sqrt(n))

    final_results.append({
        "Condition": name[0],
        "Category": name[1],
        "Trait": name[2],
        "Mean_Diff": mean,
        "P_Value": p_val,
        "Cohens_d": d,
        "CI_Low": ci_low,
        "CI_High": ci_high,
        "N": n
    })

# -------- CATEGORY-LEVEL (STRONGER SIGNAL) --------
for name, group in df.groupby(["Condition", "Category"]):

    diffs = group["Difference"].values

    if len(diffs) < 2:
        continue

    mean = np.mean(diffs)
    std = np.std(diffs, ddof=1)

    t_stat, p_val = ttest_rel(diffs, np.zeros_like(diffs))

    d = mean / std if std != 0 else 0

    n = len(diffs)
    ci_low = mean - 1.96 * (std / np.sqrt(n))
    ci_high = mean + 1.96 * (std / np.sqrt(n))

    final_results.append({
        "Condition": name[0],
        "Category": name[1],
        "Trait": "ALL",
        "Mean_Diff": mean,
        "P_Value": p_val,
        "Cohens_d": d,
        "CI_Low": ci_low,
        "CI_High": ci_high,
        "N": n
    })

# -------- SAVE --------
final_df = pd.DataFrame(final_results)
final_df.sort_values(by="P_Value", inplace=True)

final_df.to_csv(output_file, index=False)

print(f"Saved results to {output_file}")