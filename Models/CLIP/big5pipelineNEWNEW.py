
import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

# Input folder containing physical-trait subfolders
INPUT_ROOT = r"C:\Users\yuqin\OneDrive - Texas A&M University\Research\genderBias\GenderBias-SRL-Implementation\New Pictures\female"

# Output folder
OUTPUT_ROOT = "female_data6_50"

# Excel file containing Big Five traits
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

df_master["Trait_clean"] = (
    df_master["Trait"].astype(str).str.lower().str.strip()
)

df_master["Category"] = (
    df_master["Category"].astype(str).str.strip()
)

df_master["Valence"] = (
    df_master["Valence"].astype(str).str.strip().str.capitalize()
)

df_master["Prompt"] = df_master["Trait_clean"]

categories = df_master["Category"].unique().tolist()

print(f"Categories: {categories}")
print(f"Total traits: {len(df_master)}")


# ============================================================
# FUNCTION: CALCULATE CLIP SIMILARITIES
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

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    os.makedirs(image_output_dir, exist_ok=True)

    pos_file = os.path.join(
        image_output_dir,
        f"clip_big5_{base_name}_pos.xlsx"
    )

    neg_file = os.path.join(
        image_output_dir,
        f"clip_big5_{base_name}_neg.xlsx"
    )

    image = Image.open(image_path).convert("RGB")

    try:

        for valence, output_file in [
            ("Positive", pos_file),
            ("Negative", neg_file)
        ]:

            print(f"  Processing {valence} traits...")

            with pd.ExcelWriter(
                output_file,
                engine="xlsxwriter"
            ) as writer:

                for category in categories:

                    print(f"    Category: {category}")

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

                    similarities = calculate_similarities(
                        image,
                        trait_df
                    )

                    trait_df["Similarity"] = similarities

                    trait_df = trait_df.sort_values(
                        by="Similarity",
                        ascending=False
                    ).reset_index(drop=True)

                    result_df = pd.DataFrame({
                        "Trait": trait_df["Trait"].values,
                        "Similarity": trait_df["Similarity"].values
                    })

                    result_df.to_excel(
                        writer,
                        sheet_name=category,
                        index=False
                    )

            print(f"  Saved: {output_file}")

    finally:
        image.close()

    print(f"Finished processing: {base_name}")


# ============================================================
# FUNCTION: PROCESS ONE PHYSICAL-TRAIT FOLDER
# ============================================================

def process_trait_folder(trait_folder):

    trait_folder_name = os.path.basename(trait_folder)

    print("\n" + "=" * 60)
    print(f"PROCESSING PHYSICAL TRAIT: {trait_folder_name}")
    print("=" * 60)

    # Corresponding output folder
    trait_output_dir = os.path.join(
        OUTPUT_ROOT,
        trait_folder_name
    )

    os.makedirs(trait_output_dir, exist_ok=True)

    # Find images directly inside this physical-trait folder
    image_files = sorted([
        filename
        for filename in os.listdir(trait_folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(trait_folder, filename))
    ])

    if not image_files:

        print(f"No images found in {trait_folder}")
        return

    print(f"Found {len(image_files)} images.")

    # Process every image
    for filename in image_files:

        image_path = os.path.join(
            trait_folder,
            filename
        )

        # Preserve original image filename as output folder name
        image_name = os.path.splitext(filename)[0]

        image_output_dir = os.path.join(
            trait_output_dir,
            image_name
        )

        process_image(
            image_path,
            image_output_dir
        )


# ============================================================
# MAIN PIPELINE: PROCESS ALL PHYSICAL-TRAIT FOLDERS
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

    # Find all physical-trait subfolders
    trait_folders = sorted([
        os.path.join(INPUT_ROOT, folder)
        for folder in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, folder))
    ])

    print(f"\nFound {len(trait_folders)} physical-trait folders.")

    # Process every physical-trait folder
    for trait_folder in trait_folders:

        process_trait_folder(trait_folder)

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED!")
    print(f"Results saved in: {OUTPUT_ROOT}")
    print("=" * 60)


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":
    main()
