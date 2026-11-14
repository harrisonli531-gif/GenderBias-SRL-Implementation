
import re
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Automatically locate the directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# Input folder (located in the same directory as this script)
INPUT_ROOT = BASE_DIR / "male_data6_53"

# Output Excel file (saved in the same directory as this script)
OUTPUT_FILE = BASE_DIR / "male6_53_big5_scores.xlsx"

# Big Five categories
CATEGORIES = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness"
]


# ============================================================
# FUNCTION: EXTRACT IMAGE NUMBER
# ============================================================

def extract_image_number(image_name):

    match = re.search(
        r"(?:female|male)(\d+)",
        image_name,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    print(f"Warning: Could not extract image number from {image_name}")
    return None


# ============================================================
# FUNCTION: CALCULATE AVERAGE SIMILARITY
# ============================================================

def calculate_average(file_path, category):

    df = pd.read_excel(
        file_path,
        sheet_name=category
    )

    if "Similarity" not in df.columns:
        raise ValueError(
            f"Missing Similarity column in {file_path.name}"
        )

    similarities = pd.to_numeric(
        df["Similarity"],
        errors="coerce"
    ).dropna()

    if similarities.empty:
        raise ValueError(
            f"No valid similarity scores in {file_path.name}, "
            f"category: {category}"
        )

    return similarities.mean()


# ============================================================
# FUNCTION: PROCESS ONE IMAGE
# ============================================================

def process_image(image_folder, physical_trait):

    image_name = image_folder.name

    print(f"\nProcessing: {image_name}")

    image_number = extract_image_number(image_name)

    if image_number is None:
        return []

    # Locate positive and negative Excel files
    pos_file = (
        image_folder /
        f"clip_big5_{image_name}_pos.xlsx"
    )

    neg_file = (
        image_folder /
        f"clip_big5_{image_name}_neg.xlsx"
    )

    # Check both files
    if not pos_file.is_file():
        print(f"WARNING: Missing positive file: {pos_file}")
        return []

    if not neg_file.is_file():
        print(f"WARNING: Missing negative file: {neg_file}")
        return []

    results = []

    # Calculate scores for all five categories
    for category in CATEGORIES:

        try:

            average_pos = calculate_average(
                pos_file,
                category
            )

            average_neg = calculate_average(
                neg_file,
                category
            )

            score = average_pos - average_neg

            results.append({
                "number": image_number,
                "physical trait": physical_trait,
                "category": category,
                "average pos": average_pos,
                "average neg": average_neg,
                "score": score
            })

        except Exception as e:

            print(
                f"  ERROR: {image_name} | "
                f"{category}: {e}"
            )

    return results


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("BIG FIVE SCORE CALCULATION")
    print("=" * 60)

    print(f"Script directory: {BASE_DIR}")
    print(f"Input folder:     {INPUT_ROOT}")
    print(f"Output file:      {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Validate input folder
    # --------------------------------------------------------

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            f"\nInput folder does not exist:\n{INPUT_ROOT}\n"
            "Make sure female_data6_50 is in the same "
            "directory as this Python script."
        )

    if not INPUT_ROOT.is_dir():

        raise NotADirectoryError(
            f"Input path is not a directory: {INPUT_ROOT}"
        )

    # --------------------------------------------------------
    # Find physical-trait folders
    # --------------------------------------------------------

    physical_trait_folders = sorted([
        folder
        for folder in INPUT_ROOT.iterdir()
        if folder.is_dir()
    ])

    if not physical_trait_folders:

        print("No physical-trait folders found.")
        return

    print(
        f"\nFound {len(physical_trait_folders)} "
        "physical-trait folders."
    )

    all_results = []

    total_images = 0

    # --------------------------------------------------------
    # Process each physical-trait folder
    # --------------------------------------------------------

    for trait_folder in physical_trait_folders:

        physical_trait = trait_folder.name

        print("\n" + "=" * 60)
        print(f"PHYSICAL TRAIT: {physical_trait}")
        print("=" * 60)

        image_folders = sorted([
            folder
            for folder in trait_folder.iterdir()
            if folder.is_dir()
        ])

        print(f"Found {len(image_folders)} image folders.")

        # Process each image
        for image_folder in image_folders:

            total_images += 1

            image_results = process_image(
                image_folder,
                physical_trait
            )

            all_results.extend(image_results)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not all_results:

        print("\nNo results generated. Nothing to save.")
        return

    result_df = pd.DataFrame(all_results)

    result_df = result_df.sort_values(
        by=["number", "physical trait", "category"]
    ).reset_index(drop=True)

    # Save Excel workbook
    result_df.to_excel(
        OUTPUT_FILE,
        sheet_name="Big Five Scores",
        index=False,
        engine="openpyxl"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED!")
    print("=" * 60)

    print(f"Physical-trait folders: {len(physical_trait_folders)}")
    print(f"Image folders examined: {total_images}")
    print(f"Total score records:    {len(result_df)}")
    print(f"Output file:            {OUTPUT_FILE}")

# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
