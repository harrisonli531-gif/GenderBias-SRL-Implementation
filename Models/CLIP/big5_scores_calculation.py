
import os
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = "female_edited6_20_data"

OUTPUT_FILE = "female_edited6_20_big5_scores.xlsx"

CATEGORIES = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness"
]


# ============================================================
# EXTRACT IMAGE NUMBER
# ============================================================

def get_image_number(image_name):

    """
    Extract the number from an image folder name.

    Examples:
        female6_original          -> 6
        female10_original         -> 10
        female6_original_no_makeup -> 6
    """

    match = re.search(r"(?:female|male)(\d+)", image_name)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# EXTRACT PHYSICAL TRAIT
# ============================================================

def get_physical_trait(image_name):

    """
    Extract the physical characteristic from the image name.

    Examples:
        female6_original                    -> original
        female6_original_have_jewelry       -> with jewelry
        female6_original_no_jewelry         -> no jewelry
        female6_original_have_makeup        -> with makeup
        female6_original_no_makeup          -> no makeup
        female6_original_long_hair          -> long hair
        female6_original_short_hair         -> short hair
    """

    if "_original_" not in image_name:
        return "original"

    trait = image_name.split("_original_", 1)[1]

    trait_mapping = {
        "have_jewelry": "with jewelry",
        "no_jewelry": "no jewelry",
        "have_jewerly": "with jewelry",
        "no_jewerly": "no jewelry",
        "have_makeup": "with makeup",
        "no_makeup": "no makeup",
        "long_hair": "long hair",
        "short_hair": "short hair"
    }

    return trait_mapping.get(
        trait.lower(),
        trait.replace("_", " ")
    )


# ============================================================
# READ EXCEL WORKBOOK AND CALCULATE CATEGORY AVERAGES
# ============================================================

def calculate_category_averages(excel_file):

    """
    Read an Excel workbook containing five Big Five sheets.

    Each sheet contains:
        Trait
        Similarity

    Calculate the average similarity for each category.
    """

    averages = {}

    excel = pd.ExcelFile(excel_file)

    for category in CATEGORIES:

        if category not in excel.sheet_names:
            print(
                f"WARNING: Missing sheet {category} "
                f"in {excel_file}"
            )
            averages[category] = None
            continue

        df = pd.read_excel(
            excel,
            sheet_name=category
        )

        if "Similarity" not in df.columns:
            raise ValueError(
                f"'Similarity' column missing in "
                f"{excel_file}, sheet {category}"
            )

        # Calculate average similarity
        average = pd.to_numeric(
            df["Similarity"],
            errors="coerce"
        ).mean()

        averages[category] = average

    return averages


# ============================================================
# PROCESS ONE IMAGE FOLDER
# ============================================================

def process_image_folder(image_folder, subject_name):

    image_name = os.path.basename(image_folder)

    # Extract image number and physical trait
    image_number = get_image_number(image_name)
    physical_trait = get_physical_trait(image_name)

    # Expected Excel filenames
    pos_file = os.path.join(
        image_folder,
        f"clip_big5_{image_name}_pos.xlsx"
    )

    neg_file = os.path.join(
        image_folder,
        f"clip_big5_{image_name}_neg.xlsx"
    )

    # Check that both files exist
    if not os.path.isfile(pos_file):
        print(f"WARNING: Missing positive file: {pos_file}")
        return []

    if not os.path.isfile(neg_file):
        print(f"WARNING: Missing negative file: {neg_file}")
        return []

    # Calculate averages for positive and negative traits
    pos_averages = calculate_category_averages(pos_file)

    neg_averages = calculate_category_averages(neg_file)

    results = []

    # Calculate one score for each Big Five category
    for category in CATEGORIES:

        avg_pos = pos_averages[category]
        avg_neg = neg_averages[category]

        # Skip categories with missing data
        if pd.isna(avg_pos) or pd.isna(avg_neg):
            print(
                f"WARNING: Missing similarity data for "
                f"{image_name}, {category}"
            )
            continue

        # Positive minus negative
        score = avg_pos - avg_neg

        results.append({
            "number": image_number,
            "physical trait": physical_trait,
            "category": category.lower(),
            "average pos": avg_pos,
            "average neg": avg_neg,
            "score": score
        })

    print(
        f"Processed {image_name}: "
        f"{len(results)} category scores"
    )

    return results


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

    # Find subject folders (female6, female7, etc.)
    subject_folders = sorted([
        os.path.join(INPUT_ROOT, folder)
        for folder in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, folder))
    ])

    print(f"Found {len(subject_folders)} subject folders.")

    # Process every subject
    for subject_folder in subject_folders:

        subject_name = os.path.basename(subject_folder)

        print(f"\nProcessing {subject_name}...")

        # Find individual image folders
        image_folders = sorted([
            os.path.join(subject_folder, folder)
            for folder in os.listdir(subject_folder)
            if os.path.isdir(os.path.join(subject_folder, folder))
        ])

        print(f"Found {len(image_folders)} image folders.")

        # Process each image
        for image_folder in image_folders:

            results = process_image_folder(
                image_folder,
                subject_name
            )

            all_results.extend(results)

    # ========================================================
    # CREATE FINAL DATAFRAME
    # ========================================================

    columns = [
        "number",
        "physical trait",
        "category",
        "average pos",
        "average neg",
        "score"
    ]

    results_df = pd.DataFrame(
        all_results,
        columns=columns
    )

    # Sort results by image number, physical trait, category
    results_df = results_df.sort_values(
        by=["number", "physical trait", "category"]
    ).reset_index(drop=True)

    # ========================================================
    # EXPORT TO EXCEL
    # ========================================================

    results_df.to_excel(
        OUTPUT_FILE,
        index=False,
        engine="xlsxwriter"
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED!")
    print("=" * 60)

    print(f"Total images processed: {len(results_df) // 5}")
    print(f"Total category scores: {len(results_df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFirst 10 rows:")
    print(results_df.head(10).to_string(index=False))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
