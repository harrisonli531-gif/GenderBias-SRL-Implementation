import os
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Output of dfn.py:
#   data/<gender>/<subject>/<image name>/dfn_big5_<image name>_{pos,neg}.xlsx
INPUT_ROOT = "data"

OUTPUT_FILE = "dfn_big5_scores.xlsx"

CATEGORIES = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness"
]

# Image names look like:
#   female10_original                 (female original)
#   female10_original_long_hair       (female edited)
#   male10_original                   (male original)
#   male10_long_hair                  (male edited, no "_original_")
IMAGE_NAME_PATTERN = re.compile(
    r"^((?:fe)?male)(\d+)(?:_original)?(?:_(.*))?$",
    re.IGNORECASE
)

TRAIT_MAPPING = {
    "have_jewelry": "with jewelry",
    "no_jewelry": "no jewelry",
    "have_jewerly": "with jewelry",
    "no_jewerly": "no jewelry",
    "have_makeup": "with makeup",
    "no_makeup": "no makeup",
    "long_hair": "long hair",
    "short_hair": "short hair"
}


# ============================================================
# PARSE IMAGE NAME
# ============================================================

def parse_image_name(image_name):

    """
    Split an image folder name into gender, number and physical trait.

    Examples:
        female10_original            -> ("female", 10, "original")
        female10_original_have_makeup -> ("female", 10, "with makeup")
        male10_original              -> ("male", 10, "original")
        male10_no_jewerly            -> ("male", 10, "no jewelry")
    """

    match = IMAGE_NAME_PATTERN.match(image_name)

    if not match:
        return None, None, None

    gender = match.group(1).lower()
    number = int(match.group(2))
    trait = match.group(3)

    if not trait:
        return gender, number, "original"

    physical_trait = TRAIT_MAPPING.get(
        trait.lower(),
        trait.replace("_", " ")
    )

    return gender, number, physical_trait


# ============================================================
# READ EXCEL WORKBOOK AND CALCULATE CATEGORY AVERAGES
# ============================================================

def calculate_category_averages(excel_file):

    """
    Read a workbook with one sheet per Big Five category
    (columns: Trait, Similarity) and return the average
    similarity for each category.
    """

    averages = {}

    with pd.ExcelFile(excel_file) as excel:

        for category in CATEGORIES:

            if category not in excel.sheet_names:
                print(
                    f"WARNING: Missing sheet {category} "
                    f"in {excel_file}"
                )
                averages[category] = None
                continue

            df = pd.read_excel(excel, sheet_name=category)

            if "Similarity" not in df.columns:
                raise ValueError(
                    f"'Similarity' column missing in "
                    f"{excel_file}, sheet {category}"
                )

            averages[category] = pd.to_numeric(
                df["Similarity"],
                errors="coerce"
            ).mean()

    return averages


# ============================================================
# PROCESS ONE IMAGE FOLDER
# ============================================================

def process_image_folder(image_folder):

    image_name = os.path.basename(image_folder)

    gender, image_number, physical_trait = parse_image_name(image_name)

    if gender is None:
        print(f"WARNING: Could not parse image name: {image_name}")
        return []

    pos_file = os.path.join(
        image_folder,
        f"dfn_big5_{image_name}_pos.xlsx"
    )

    neg_file = os.path.join(
        image_folder,
        f"dfn_big5_{image_name}_neg.xlsx"
    )

    if not os.path.isfile(pos_file):
        print(f"WARNING: Missing positive file: {pos_file}")
        return []

    if not os.path.isfile(neg_file):
        print(f"WARNING: Missing negative file: {neg_file}")
        return []

    pos_averages = calculate_category_averages(pos_file)
    neg_averages = calculate_category_averages(neg_file)

    results = []

    for category in CATEGORIES:

        avg_pos = pos_averages[category]
        avg_neg = neg_averages[category]

        if pd.isna(avg_pos) or pd.isna(avg_neg):
            print(
                f"WARNING: Missing similarity data for "
                f"{image_name}, {category}"
            )
            continue

        results.append({
            "gender": gender,
            "number": image_number,
            "physical trait": physical_trait,
            "category": category.lower(),
            "average pos": avg_pos,
            "average neg": avg_neg,
            "score": avg_pos - avg_neg
        })

    print(
        f"Processed {image_name}: "
        f"{len(results)} category scores"
    )

    return results


# ============================================================
# HELPER: LIST SUBFOLDERS
# ============================================================

def list_subfolders(folder):

    return sorted([
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ])


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("CALCULATING BIG FIVE SCORES")
    print("=" * 60)

    if not os.path.isdir(INPUT_ROOT):
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_ROOT}"
        )

    all_results = []

    # data/<gender>/<subject>/<image>/
    for gender_folder in list_subfolders(INPUT_ROOT):

        subject_folders = list_subfolders(gender_folder)

        print(
            f"\nFound {len(subject_folders)} subject folders "
            f"in {os.path.basename(gender_folder)}."
        )

        for subject_folder in subject_folders:

            print(f"\nProcessing {os.path.basename(subject_folder)}...")

            for image_folder in list_subfolders(subject_folder):
                all_results.extend(process_image_folder(image_folder))

    # ========================================================
    # CREATE FINAL DATAFRAME
    # ========================================================

    columns = [
        "gender",
        "number",
        "physical trait",
        "category",
        "average pos",
        "average neg",
        "score"
    ]

    results_df = pd.DataFrame(all_results, columns=columns)

    results_df = results_df.sort_values(
        by=["gender", "number", "physical trait", "category"]
    ).reset_index(drop=True)

    # ========================================================
    # EXPORT TO EXCEL
    # ========================================================

    results_df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED!")
    print("=" * 60)

    print(f"Total images processed: {len(results_df) // len(CATEGORIES)}")
    print(f"Total category scores: {len(results_df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFirst 10 rows:")
    print(results_df.head(10).to_string(index=False))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
