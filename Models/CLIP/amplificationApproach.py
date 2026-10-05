import pandas as pd
import numpy as np
import os
import re

from scipy.stats import (
    ttest_ind,
    ttest_rel,
    sem,
    norm
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "all_data"

BASELINE_CONDITION = "baseline"

# Human reference data.
#
# Expected columns:
#   Category, Human_Cohens_d
#
# Human_Cohens_d MUST represent:
#       Female - Male
#
HUMAN_REFERENCE_FILE = "human_big5_effects.csv"

# Output files
GENDER_GAP_OUTPUT = "clip_gender_gaps.csv"
APPEARANCE_OUTPUT = "appearance_effects.csv"
AMPLIFICATION_OUTPUT = "amplification_results.csv"

# Appearance pair definitions.
#
# The spelling "jewerly" is intentional because that is how
# the existing filenames are named.
PAIR_DEFS = {
    "makeup": ("have_makeup", "no_makeup"),
    "hair": ("long_hair", "short_hair"),
    "jewelry": ("have_jewerly", "no_jewerly")
}

# ============================================================
# LOAD HUMAN REFERENCE DATA
# ============================================================

if os.path.exists(HUMAN_REFERENCE_FILE):

    human_df = pd.read_csv(HUMAN_REFERENCE_FILE)

    required_columns = {
        "Category",
        "Human_Cohens_d"
    }

    missing = required_columns - set(human_df.columns)

    if missing:
        raise ValueError(
            f"Human reference file is missing columns: {missing}"
        )

    human_df["Category"] = human_df["Category"].astype(str).str.strip()

    human_reference = dict(
        zip(
            human_df["Category"],
            human_df["Human_Cohens_d"]
        )
    )

else:

    print(
        f"WARNING: {HUMAN_REFERENCE_FILE} was not found.\n"
        "Amplification relative to human data will not be calculated."
    )

    human_reference = {}


# ============================================================
# LOAD ONE EXCEL FILE
# ============================================================

def load_excel(filepath):

    xls = pd.ExcelFile(filepath)

    rows = []

    for sheet in xls.sheet_names:

        df = pd.read_excel(
            xls,
            sheet_name=sheet
        )

        # ----------------------------------------------------
        # Positive traits
        # ----------------------------------------------------

        if (
            "Positive Trait" in df.columns
            and
            "Positive Probability" in df.columns
        ):

            for _, row in df.iterrows():

                if pd.notna(row["Positive Trait"]):

                    rows.append({
                        "Category": sheet,
                        "Trait": str(row["Positive Trait"]).strip(),
                        "Valence": "Positive",
                        "Score": float(row["Positive Probability"])
                    })

        # ----------------------------------------------------
        # Negative traits
        # ----------------------------------------------------

        if (
            "Negative Trait" in df.columns
            and
            "Negative Probability" in df.columns
        ):

            for _, row in df.iterrows():

                if pd.notna(row["Negative Trait"]):

                    rows.append({
                        "Category": sheet,
                        "Trait": str(row["Negative Trait"]).strip(),
                        "Valence": "Negative",
                        "Score": float(row["Negative Probability"])
                    })

    return pd.DataFrame(rows)


# ============================================================
# PARSE FILENAME
# ============================================================

def parse_filename(filename):
    """
    Parse the naming conventions used in the dataset.

    Examples:

        clip_big5_female1_have_jewerly.xlsx
            -> identity='female1'
            -> condition='have_jewerly'

        clip_big5_female1_no_jewerly.xlsx
            -> identity='female1'
            -> condition='no_jewerly'

        clip_big5_female1_long_hair.xlsx
            -> identity='female1'
            -> condition='long_hair'

        clip_big5_woman1_cropped.xlsx
            -> identity='female1'
            -> condition='baseline'

        clip_big5_male3_short_hair.xlsx
            -> identity='male3'
            -> condition='short_hair'

        clip_big5_man3_cropped.xlsx
            -> identity='male3'
            -> condition='baseline'
    """

    # Remove extension
    name = os.path.splitext(filename)[0]

    # Remove CLIP prefix
    prefix = "clip_big5_"

    if not name.startswith(prefix):
        raise ValueError(
            f"Unexpected filename format: {filename}"
        )

    name = name[len(prefix):]

    # --------------------------------------------------------
    # CROPPED / BASELINE IMAGES
    # --------------------------------------------------------
    #
    # woman1_cropped -> female1, baseline
    # man1_cropped   -> male1, baseline
    #
    # These are explicitly mapped to the corresponding
    # female/male identity.
    # --------------------------------------------------------

    cropped_match = re.fullmatch(
        r"(woman|man)(\d+)_cropped",
        name,
        flags=re.IGNORECASE
    )

    if cropped_match:

        gender_word = cropped_match.group(1).lower()
        number = cropped_match.group(2)

        if gender_word == "woman":
            identity = f"female{number}"

        elif gender_word == "man":
            identity = f"male{number}"

        else:
            raise ValueError(
                f"Unknown gender in filename: {filename}"
            )

        return identity, "baseline"

    # --------------------------------------------------------
    # NORMAL EXPERIMENTAL CONDITIONS
    # --------------------------------------------------------
    #
    # female1_have_jewerly
    # female1_no_jewerly
    # female1_have_makeup
    # female1_no_makeup
    # female1_long_hair
    # female1_short_hair
    #
    # male1_...
    #
    # --------------------------------------------------------

    normal_match = re.fullmatch(
        r"(female|male)(\d+)_(.+)",
        name,
        flags=re.IGNORECASE
    )

    if normal_match:

        gender_word = normal_match.group(1).lower()
        number = normal_match.group(2)
        condition = normal_match.group(3).lower()

        identity = f"{gender_word}{number}"

        return identity, condition

    # --------------------------------------------------------
    # Anything that doesn't match the expected conventions
    # should produce an explicit error rather than silently
    # entering the analysis incorrectly.
    # --------------------------------------------------------

    raise ValueError(
        f"Could not parse filename: {filename}\n"
        f"Expected formats such as:\n"
        f"  clip_big5_female1_long_hair.xlsx\n"
        f"  clip_big5_male1_short_hair.xlsx\n"
        f"  clip_big5_woman1_cropped.xlsx\n"
        f"  clip_big5_man1_cropped.xlsx"
    )


# ============================================================
# LOAD ALL DATA RECURSIVELY
# ============================================================

data = {}

for root, dirs, files in os.walk(INPUT_FOLDER):

    for file in files:

        if not file.lower().endswith(".xlsx"):
            continue

        filepath = os.path.join(root, file)

        identity, condition = parse_filename(file)

        key = (identity, condition)

        print(
            f"Loading: {filepath}"
        )

        # Prevent accidental duplicate conditions
        if key in data:

            print(
                f"WARNING: Duplicate data for "
                f"{identity} / {condition}"
            )

            print(
                f"Existing file will be overwritten by: "
                f"{filepath}"
            )

        data[key] = load_excel(filepath)


print()
print(
    f"Loaded {len(data)} image conditions."
)


# ============================================================
# PRINT DATASET STRUCTURE
# ============================================================
#
# This makes it easy to verify that the naming conventions
# were interpreted correctly.
# ============================================================

print()
print("=" * 70)
print("RECOGNIZED DATASET STRUCTURE")
print("=" * 70)

identities = sorted(
    set(identity for identity, condition in data.keys())
)

for identity in identities:

    conditions = sorted(
        condition
        for stored_identity, condition in data.keys()
        if stored_identity == identity
    )

    print(
        f"{identity}: {', '.join(conditions)}"
    )

print()


# ============================================================
# LOAD ALL DATA RECURSIVELY
# ============================================================

data = {}

for root, dirs, files in os.walk(INPUT_FOLDER):

    for file in files:

        if not file.lower().endswith(".xlsx"):
            continue

        filepath = os.path.join(root, file)

        identity, condition = parse_filename(file)

        key = (identity, condition)

        print(f"Loading: {filepath}")

        data[key] = load_excel(filepath)


print()
print(f"Loaded {len(data)} image conditions.")


# ============================================================
# ADD GENDER INFORMATION
# ============================================================

def get_gender(identity):

    identity_lower = identity.lower()

    if identity_lower.startswith("female"):
        return "Female"

    if identity_lower.startswith("male"):
        return "Male"

    return "Unknown"


# ============================================================
# TRAIT-LEVEL GENDER DATA
# ============================================================
#
# For every condition, calculate:
#
#   Female mean
#   Male mean
#   Female - Male
#   Cohen's d
#   p-value
#
# This tells us whether CLIP associates the trait differently
# with female and male images.
#
# ============================================================

gender_rows = []

all_conditions = sorted(
    set(condition for identity, condition in data.keys())
)

all_categories = sorted(
    set(
        category
        for df in data.values()
        for category in df["Category"].unique()
    )
)


for condition in all_conditions:

    for category in all_categories:

        # ----------------------------------------------------
        # Collect every trait appearing in this category
        # ----------------------------------------------------

        traits = set()

        for (identity, cond), df in data.items():

            if cond != condition:
                continue

            category_df = df[df["Category"] == category]

            traits.update(
                category_df["Trait"].unique()
            )

        # ----------------------------------------------------
        # Analyze each trait
        # ----------------------------------------------------

        for trait in traits:

            female_scores = []
            male_scores = []

            for (identity, cond), df in data.items():

                if cond != condition:
                    continue

                gender = get_gender(identity)

                subset = df[
                    (df["Category"] == category)
                    &
                    (df["Trait"] == trait)
                ]

                if len(subset) == 0:
                    continue

                score = subset["Score"].iloc[0]

                if gender == "Female":
                    female_scores.append(score)

                elif gender == "Male":
                    male_scores.append(score)

            # Need at least two observations in each group
            # for a meaningful group comparison.
            if len(female_scores) < 2 or len(male_scores) < 2:
                continue

            female_scores = np.array(female_scores)
            male_scores = np.array(male_scores)

            female_mean = np.mean(female_scores)
            male_mean = np.mean(male_scores)

            gender_gap = female_mean - male_mean

            # ------------------------------------------------
            # Welch independent-samples t-test
            # ------------------------------------------------

            t_stat, p_value = ttest_ind(
                female_scores,
                male_scores,
                equal_var=False
            )

            # ------------------------------------------------
            # Cohen's d
            # ------------------------------------------------

            n_female = len(female_scores)
            n_male = len(male_scores)

            pooled_sd = np.sqrt(
                (
                    (n_female - 1) * np.var(
                        female_scores,
                        ddof=1
                    )
                    +
                    (n_male - 1) * np.var(
                        male_scores,
                        ddof=1
                    )
                )
                /
                (
                    n_female + n_male - 2
                )
            )

            if pooled_sd != 0:

                cohens_d = (
                    female_mean - male_mean
                ) / pooled_sd

            else:

                cohens_d = np.nan

            # ------------------------------------------------
            # 95% CI for raw mean difference
            # ------------------------------------------------

            se = np.sqrt(
                np.var(female_scores, ddof=1) / n_female
                +
                np.var(male_scores, ddof=1) / n_male
            )

            ci_low = gender_gap - 1.96 * se
            ci_high = gender_gap + 1.96 * se

            gender_rows.append({

                "Condition": condition,
                "Category": category,
                "Trait": trait,

                "Female_Mean": female_mean,
                "Male_Mean": male_mean,

                # Positive means Female > Male
                "Gender_Gap_Female_Minus_Male": gender_gap,

                "Cohens_d": cohens_d,

                "P_Value": p_value,

                "CI_Low": ci_low,
                "CI_High": ci_high,

                "N_Female": n_female,
                "N_Male": n_male

            })


gender_df = pd.DataFrame(gender_rows)


# ============================================================
# CATEGORY-LEVEL GENDER ANALYSIS
# ============================================================
#
# Instead of treating every individual descriptor as independent,
# calculate a category-level score for each image.
#
# Positive probability contributes positively.
# Negative probability contributes negatively.
#
# Category score =
#
#       mean(positive probabilities)
#       -
#       mean(negative probabilities)
#
# This produces one Big Five score per image.
#
# ============================================================

category_image_scores = []


for (identity, condition), df in data.items():

    gender = get_gender(identity)

    for category in df["Category"].unique():

        category_df = df[
            df["Category"] == category
        ]

        positive = category_df[
            category_df["Valence"] == "Positive"
        ]["Score"]

        negative = category_df[
            category_df["Valence"] == "Negative"
        ]["Score"]

        positive_mean = (
            positive.mean()
            if len(positive) > 0
            else 0
        )

        negative_mean = (
            negative.mean()
            if len(negative) > 0
            else 0
        )

        # Signed category score
        category_score = (
            positive_mean - negative_mean
        )

        category_image_scores.append({

            "Identity": identity,
            "Gender": gender,
            "Condition": condition,
            "Category": category,

            "Positive_Mean": positive_mean,
            "Negative_Mean": negative_mean,

            "Category_Score": category_score
        })


category_df = pd.DataFrame(
    category_image_scores
)


# ============================================================
# CATEGORY-LEVEL GENDER GAP
# ============================================================

category_gender_results = []


for (condition, category), group in category_df.groupby(
    ["Condition", "Category"]
):

    female = group[
        group["Gender"] == "Female"
    ]["Category_Score"].dropna().values

    male = group[
        group["Gender"] == "Male"
    ]["Category_Score"].dropna().values

    if len(female) < 2 or len(male) < 2:
        continue

    female_mean = np.mean(female)
    male_mean = np.mean(male)

    gap = female_mean - male_mean

    t_stat, p_value = ttest_ind(
        female,
        male,
        equal_var=False
    )

    n_female = len(female)
    n_male = len(male)

    pooled_sd = np.sqrt(
        (
            (n_female - 1) * np.var(
                female,
                ddof=1
            )
            +
            (n_male - 1) * np.var(
                male,
                ddof=1
            )
        )
        /
        (
            n_female + n_male - 2
        )
    )

    if pooled_sd != 0:
        cohens_d = gap / pooled_sd
    else:
        cohens_d = np.nan

    se = np.sqrt(
        np.var(female, ddof=1) / n_female
        +
        np.var(male, ddof=1) / n_male
    )

    ci_low = gap - 1.96 * se
    ci_high = gap + 1.96 * se

    category_gender_results.append({

        "Condition": condition,
        "Category": category,

        "Female_Mean": female_mean,
        "Male_Mean": male_mean,

        "Gender_Gap": gap,

        "CLIP_Cohens_d": cohens_d,

        "P_Value": p_value,

        "CI_Low": ci_low,
        "CI_High": ci_high,

        "N_Female": n_female,
        "N_Male": n_male

    })


category_gender_df = pd.DataFrame(
    category_gender_results
)


# ============================================================
# APPEARANCE EFFECT
# ============================================================
#
# This reproduces the useful part of your original analysis:
#
#   modified - baseline
#
# But now it is explicitly calculated separately for female
# and male images.
#
# Example:
#
#   Female long hair - Female short hair
#   Male long hair   - Male short hair
#
# ============================================================

appearance_results = []


for appearance, (modified_tag, baseline_tag) in PAIR_DEFS.items():

    identities = set(
        identity
        for identity, condition in data.keys()
    )

    for identity in identities:

        modified_key = (
            identity,
            modified_tag
        )

        baseline_key = (
            identity,
            baseline_tag
        )

        if (
            modified_key not in data
            or
            baseline_key not in data
        ):
            continue

        modified_df = data[modified_key]
        baseline_df = data[baseline_key]

        merged = pd.merge(
            modified_df,
            baseline_df,

            on=[
                "Category",
                "Trait",
                "Valence"
            ],

            suffixes=(
                "_modified",
                "_baseline"
            )
        )

        merged["Difference"] = (
            merged["Score_modified"]
            -
            merged["Score_baseline"]
        )

        gender = get_gender(identity)

        for _, row in merged.iterrows():

            appearance_results.append({

                "Identity": identity,
                "Gender": gender,

                "Appearance": appearance,

                "Category": row["Category"],
                "Trait": row["Trait"],
                "Valence": row["Valence"],

                "Modified_Score": row["Score_modified"],
                "Baseline_Score": row["Score_baseline"],

                "Appearance_Effect": row["Difference"]

            })


appearance_df = pd.DataFrame(
    appearance_results
)


# ============================================================
# APPEARANCE EFFECT STATISTICS
# ============================================================

appearance_stats = []


if len(appearance_df) > 0:

    for (
        appearance,
        gender,
        category,
        trait
    ), group in appearance_df.groupby(
        [
            "Appearance",
            "Gender",
            "Category",
            "Trait"
        ]
    ):

        differences = (
            group["Appearance_Effect"]
            .dropna()
            .values
        )

        if len(differences) < 2:
            continue

        mean_diff = np.mean(differences)

        std_diff = np.std(
            differences,
            ddof=1
        )

        t_stat, p_value = ttest_rel(
            group["Modified_Score"],
            group["Baseline_Score"]
        )

        if std_diff != 0:
            cohens_d = (
                mean_diff / std_diff
            )
        else:
            cohens_d = np.nan

        n = len(differences)

        ci_low = (
            mean_diff
            -
            1.96 * std_diff / np.sqrt(n)
        )

        ci_high = (
            mean_diff
            +
            1.96 * std_diff / np.sqrt(n)
        )

        appearance_stats.append({

            "Appearance": appearance,
            "Gender": gender,

            "Category": category,
            "Trait": trait,

            "Mean_Appearance_Effect": mean_diff,

            "Cohens_d": cohens_d,

            "P_Value": p_value,

            "CI_Low": ci_low,
            "CI_High": ci_high,

            "N": n

        })


appearance_stats_df = pd.DataFrame(
    appearance_stats
)


# ============================================================
# AMPLIFICATION ANALYSIS
# ============================================================
#
# This is the key part.
#
# For each Big Five category:
#
#     Human effect = documented female-male Cohen's d
#
#     CLIP effect = CLIP female-male Cohen's d
#
#     Amplification =
#
#             CLIP d - Human d
#
# Positive:
#     CLIP's gender difference is larger in the
#     Female-Male direction than the human reference.
#
# Negative:
#     CLIP's difference is smaller than the human reference.
#
# IMPORTANT:
# This comparison is meaningful only when the human effect
# and CLIP score represent the same Big Five construct and
# have the same Female-Male direction.
#
# ============================================================

amplification_results = []


for _, row in category_gender_df.iterrows():

    category = row["Category"]

    if category not in human_reference:
        continue

    human_d = human_reference[category]

    clip_d = row["CLIP_Cohens_d"]

    if pd.isna(clip_d) or pd.isna(human_d):
        continue

    amplification = (
        clip_d - human_d
    )

    # Ratio can become unstable when human_d is near zero,
    # so we do NOT use it as the primary measure.

    amplification_results.append({

        "Condition": row["Condition"],
        "Category": category,

        "Human_Cohens_d": human_d,

        "CLIP_Cohens_d": clip_d,

        "Amplification_d": amplification,

        "CLIP_Gender_Gap": row["Gender_Gap"],

        "CLIP_P_Value": row["P_Value"],

        "CLIP_CI_Low": row["CI_Low"],
        "CLIP_CI_High": row["CI_High"],

        "N_Female": row["N_Female"],
        "N_Male": row["N_Male"]

    })


amplification_df = pd.DataFrame(
    amplification_results
)


# ============================================================
# SAVE RESULTS
# ============================================================

if len(gender_df) > 0:

    gender_df.sort_values(
        by="P_Value",
        inplace=True
    )

    gender_df.to_csv(
        GENDER_GAP_OUTPUT,
        index=False
    )


if len(appearance_stats_df) > 0:

    appearance_stats_df.sort_values(
        by="P_Value",
        inplace=True
    )

    appearance_stats_df.to_csv(
        APPEARANCE_OUTPUT,
        index=False
    )


if len(amplification_df) > 0:

    amplification_df.sort_values(
        by="Amplification_d",
        ascending=False,
        inplace=True
    )

    amplification_df.to_csv(
        AMPLIFICATION_OUTPUT,
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()
print(f"Loaded conditions: {len(all_conditions)}")
print(f"Loaded identities: {len(set(identity for identity, _ in data.keys()))}")

print()
print("Output files:")

print(f"  1. {GENDER_GAP_OUTPUT}")
print(f"  2. {APPEARANCE_OUTPUT}")
print(f"  3. {AMPLIFICATION_OUTPUT}")

print()

if len(amplification_df) == 0:

    print(
        "No amplification results were calculated."
    )

    print(
        f"Make sure {HUMAN_REFERENCE_FILE} exists "
        "and contains matching Big Five categories."
    )

else:

    print(
        "Amplification results calculated:"
    )

    print()

    display_columns = [
        "Condition",
        "Category",
        "Human_Cohens_d",
        "CLIP_Cohens_d",
        "Amplification_d"
    ]

    print(
        amplification_df[
            display_columns
        ].to_string(index=False)
    )