import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import open_clip
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"

# Input folder: New Pictures/{female,male}/<condition>/<images>
# (relative to Models/DFNCLIP, where this script is run from)
INPUT_ROOT = os.path.join("..", "..", "New Pictures")
#INPUT_ROOT = "test"

# Output folder, grouped by subject (same per-image layout as CLIP):
#   <OUTPUT_ROOT>/<gender>/<subject>/<image name>/dfn_big5_<image name>_{pos,neg}.xlsx
#   e.g. data/female/female10/female10_original_long_hair/
#            dfn_big5_female10_original_long_hair_pos.xlsx
OUTPUT_ROOT = "data"

# Image names start with the subject ID, e.g. "female10_..." or "male10_..."
SUBJECT_PATTERN = re.compile(r"^((?:fe)?male)(\d+)", re.IGNORECASE)
#OUTPUT_ROOT = "testoutput"

# Excel file containing the Big Five traits
TRAITS_FILE = "The_Big_Five_traits.xlsx"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD DFN MODEL, PREPROCESSOR AND TOKENIZER ONCE
# ============================================================

print("Loading DFN model...")

# DFN is loaded through the open_clip library (downloads from HF on first run)
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model = model.to(DEVICE).eval()

print(f"DFN loaded on {DEVICE}")


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

# Use the exact trait descriptions as text prompts
df_master["Prompt"] = df_master["Trait_clean"]

categories = df_master["Category"].unique().tolist()

print(f"Categories: {categories}")
print(f"Total traits: {len(df_master)}")


# ============================================================
# ENCODE ALL TRAIT PROMPTS ONCE
# ============================================================
# Text embeddings do not depend on the image, so they are computed a
# single time here instead of once per image / category / valence.

print("Encoding trait prompts...")

with torch.no_grad():
    text_tokens = tokenizer(df_master["Prompt"].tolist()).to(DEVICE)
    text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

# Row i of text_features corresponds to row i of df_master
# (df_master keeps its default RangeIndex 0..N-1)


# ============================================================
# FUNCTION: ENCODE ONE IMAGE
# ============================================================

def encode_image(image):

    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = F.normalize(
            model.encode_image(image_tensor),
            dim=-1
        )

    return image_features


# ============================================================
# FUNCTION: CALCULATE COSINE SIMILARITIES
# ============================================================

def calculate_similarities(image_features, trait_df):

    # Pick the precomputed text embeddings for these traits
    idx = torch.as_tensor(trait_df.index.values, device=DEVICE)
    feats = text_features[idx]

    # Both sides are L2-normalized, so this is cosine similarity
    similarities = F.cosine_similarity(image_features, feats, dim=-1)

    return similarities.cpu().tolist()


# ============================================================
# FUNCTION: PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, image_output_dir):

    print(f"\nProcessing image: {image_path}")

    image = Image.open(image_path).convert("RGB")

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    os.makedirs(image_output_dir, exist_ok=True)

    # Encode the image once; it is reused for every category / valence
    image_features = encode_image(image)
    image.close()

    # --------------------------------------------------------
    # Create two Excel workbooks (same layout as the CLIP pipeline):
    # 1. Positive traits
    # 2. Negative traits
    # Each workbook has one worksheet per Big Five category.
    # --------------------------------------------------------

    pos_file = os.path.join(
        image_output_dir,
        f"dfn_big5_{base_name}_pos.xlsx"
    )

    neg_file = os.path.join(
        image_output_dir,
        f"dfn_big5_{base_name}_neg.xlsx"
    )

    for valence, output_file in [
        ("Positive", pos_file),
        ("Negative", neg_file)
    ]:

        print(f"  Processing {valence} traits...")

        with pd.ExcelWriter(output_file) as writer:

            for category in categories:

                # Select traits belonging to this category
                # AND this valence
                trait_df = df_master[
                    (df_master["Category"] == category) &
                    (df_master["Valence"] == valence)
                ].copy()

                if trait_df.empty:
                    print(
                        f"    Warning: No {valence} traits "
                        f"found for {category}"
                    )
                    continue

                trait_df["Similarity"] = calculate_similarities(
                    image_features,
                    trait_df
                )

                # Sort traits from highest to lowest similarity
                trait_df = trait_df.sort_values(
                    by="Similarity",
                    ascending=False
                ).reset_index(drop=True)

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


# ============================================================
# FUNCTION: PROCESS ALL IMAGES IN ONE SUBJECT FOLDER
# ============================================================

def process_subject_folder(subject_folder):

    # e.g. "female/female_long_hair"
    folder_name = os.path.relpath(subject_folder, INPUT_ROOT)

    print("\n" + "=" * 60)
    print(f"PROCESSING FOLDER: {folder_name}")
    print("=" * 60)

    image_files = sorted([
        filename
        for filename in os.listdir(subject_folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not image_files:
        print(f"No images found in {subject_folder}")
        return

    print(f"Found {len(image_files)} images.")

    for filename in image_files:

        image_path = os.path.join(subject_folder, filename)

        match = SUBJECT_PATTERN.match(filename)

        if not match:
            print(f"  Skipping (no subject ID in name): {filename}")
            continue

        # e.g. gender = "female", subject = "female10"
        gender = match.group(1).lower()
        subject = f"{gender}{int(match.group(2))}"

        # Separate output folder per image, like the CLIP pipeline
        image_name = os.path.splitext(filename)[0]

        image_output_dir = os.path.join(
            OUTPUT_ROOT,
            gender,
            subject,
            image_name
        )

        process_image(image_path, image_output_dir)


# ============================================================
# MAIN PIPELINE: PROCESS ALL SUBJECT FOLDERS
# ============================================================

def main():

    print("\nStarting Big Five DFN Pipeline")
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

    # Any folder (at any depth) that directly contains images
    subject_folders = sorted([
        dirpath
        for dirpath, _, filenames in os.walk(INPUT_ROOT)
        if any(f.lower().endswith(IMAGE_EXTENSIONS) for f in filenames)
    ])

    print(f"\nFound {len(subject_folders)} subject folders.")

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