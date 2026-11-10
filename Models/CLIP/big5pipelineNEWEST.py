
import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

# Input folder containing female6, female7, ..., female20
INPUT_ROOT = "male_edited6_20"
#INPUT_ROOT = "test"

# Output folder
OUTPUT_ROOT = "male_edited6_20_data"
#OUTPUT_ROOT = "testoutput"

# Excel file containing the Big Five traits
TRAITS_FILE = "The_Big_Five_traits.xlsx"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD CLIP MODEL AND PROCESSOR ONCE
# ============================================================

print("Loading CLIP model...")

model = AutoModel.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print(f"CLIP loaded on {DEVICE}")


# ============================================================
# LOAD AND PREPARE BIG FIVE TRAIT DATASET
# ============================================================

print("Loading Big Five traits...")

df_master = pd.read_excel(TRAITS_FILE)

# Clean columns
df_master["Trait_clean"] = (
    df_master["Trait"].astype(str).str.lower().str.strip()
)

df_master["Category"] = (
    df_master["Category"].astype(str).str.strip()
)

df_master["Valence"] = (
    df_master["Valence"].astype(str).str.strip().str.capitalize()
)

# Use the exact trait descriptions as CLIP text prompts
df_master["Prompt"] = df_master["Trait_clean"]

categories = df_master["Category"].unique().tolist()

print(f"Categories: {categories}")
print(f"Total traits: {len(df_master)}")


# ============================================================
# FUNCTION: CALCULATE CLIP PROBABILITIES
# ============================================================

def calculate_similarities(image, trait_df):

    prompts = trait_df["Prompt"].tolist()

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        # Recover cosine similarities instead of softmax probabilities
        similarities = (
            outputs.logits_per_image / model.logit_scale.exp()
        )

    return similarities.squeeze(0).cpu().tolist()


# ============================================================
# FUNCTION: PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, image_output_dir):

    print(f"\nProcessing image: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Extract filename without extension
    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # Create an output folder for this specific image
    os.makedirs(image_output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Create two Excel workbooks:
    # 1. Positive traits
    # 2. Negative traits
    # --------------------------------------------------------

    pos_file = os.path.join(
        image_output_dir,
        f"clip_big5_{base_name}_pos.xlsx"
    )

    neg_file = os.path.join(
        image_output_dir,
        f"clip_big5_{base_name}_neg.xlsx"
    )

    # Process positive and negative valence separately
    for valence, output_file in [
        ("Positive", pos_file),
        ("Negative", neg_file)
    ]:

        print(f"  Processing {valence} traits...")

        # Create one Excel workbook for this valence
        with pd.ExcelWriter(
            output_file,
            engine="xlsxwriter"
        ) as writer:

            # Process all five Big Five categories
            for category in categories:

                print(f"    Category: {category}")

                # Select traits belonging to this category
                # AND this valence
                trait_df = df_master[
                    (df_master["Category"] == category) &
                    (df_master["Valence"] == valence)
                ].copy()

                # Skip if no traits exist
                if trait_df.empty:
                    print(
                        f"    Warning: No {valence} traits "
                        f"found for {category}"
                    )
                    continue

                # ------------------------------------------------
                # Calculate CLIP probabilities
                # ------------------------------------------------

                similarities = calculate_similarities(
                    image,
                    trait_df
                )

                trait_df["Similarity"] = similarities

                # Sort traits from highest to lowest probability
                trait_df = trait_df.sort_values(
                    by="Similarity",
                    ascending=False
                ).reset_index(drop=True)

                # ------------------------------------------------
                # Prepare Excel worksheet
                # ------------------------------------------------

                result_df = pd.DataFrame({
                    "Trait": trait_df["Trait"].values,
                    "Similarity": trait_df["Similarity"].values
                })

                # One worksheet per Big Five category
                result_df.to_excel(
                    writer,
                    sheet_name=category,
                    index=False
                )

        print(f"  Saved: {output_file}")

    image.close()

    print(f"Finished processing: {base_name}")


# ============================================================
# FUNCTION: PROCESS ALL IMAGES IN ONE SUBJECT FOLDER
# ============================================================

def process_subject_folder(subject_folder):

    subject_name = os.path.basename(subject_folder)

    print("\n" + "=" * 60)
    print(f"PROCESSING SUBJECT: {subject_name}")
    print("=" * 60)

    # Corresponding output subject folder
    subject_output_dir = os.path.join(
        OUTPUT_ROOT,
        subject_name
    )

    os.makedirs(subject_output_dir, exist_ok=True)

    # Find all images directly inside the subject folder
    image_files = sorted([
        filename
        for filename in os.listdir(subject_folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not image_files:
        print(f"No images found in {subject_folder}")
        return

    print(f"Found {len(image_files)} images.")

    # Process each image independently
    for filename in image_files:

        image_path = os.path.join(
            subject_folder,
            filename
        )

        # Create a separate output folder for each image
        image_name = os.path.splitext(filename)[0]

        image_output_dir = os.path.join(
            subject_output_dir,
            image_name
        )

        process_image(
            image_path,
            image_output_dir
        )


# ============================================================
# MAIN PIPELINE: PROCESS ALL SUBJECT FOLDERS
# ============================================================

def main():

    print("\nStarting Big Five CLIP Pipeline")
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")

    if not os.path.isdir(INPUT_ROOT):
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_ROOT}"
        )

    if not os.path.isfile(TRAITS_FILE):
        raise FileNotFoundError(
            f"Trait Excel file not found: {TRAITS_FILE}"
        )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Find all subject folders (female6, female7, etc.)
    subject_folders = sorted([
        os.path.join(INPUT_ROOT, folder)
        for folder in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, folder))
    ])

    print(f"\nFound {len(subject_folders)} subject folders.")

    # Process every subject
    for subject_folder in subject_folders:

        process_subject_folder(subject_folder)

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED!")
    print(f"Results saved in: {OUTPUT_ROOT}")
    print("=" * 60)


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":
    main()
