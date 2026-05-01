import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os

# ---- Load CLIP once ----
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load Big Five trait mapping once ----
df_master = pd.read_excel("The_Big_Five_traits.xlsx")
df_master["Trait_clean"] = df_master["Trait"].str.lower().str.strip()
df_master["Prompt"] = df_master["Trait_clean"]

traits = df_master["Prompt"].tolist()

# ---- Function to process ONE image ----
def process_image(image_path):
    print(f"Processing: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Copy dataframe so original isn't modified
    df = df_master.copy()

    # Run CLIP
    inputs = processor(text=traits, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    df["Probability"] = probs.squeeze(0).tolist()

    # ---- Output folder setup ----
    output_dir = "male1" + "data" # change this to the name of the folder with results for each run
    os.makedirs(output_dir, exist_ok=True)

    # ---- Output filename ----
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_dir, f"clip_big5_{base_name}.xlsx")

    writer = pd.ExcelWriter(output_file, engine="xlsxwriter")

    # ---- Process categories ----
    for category in df["Category"].unique():
        cat_df = df[df["Category"] == category]

        pos_df = (
            cat_df[cat_df["Valence"] == "Positive"]
            .sort_values(by="Probability", ascending=False)
            .reset_index(drop=True)
        )

        neg_df = (
            cat_df[cat_df["Valence"] == "Negative"]
            .sort_values(by="Probability", ascending=False)
            .reset_index(drop=True)
        )

        max_len = max(len(pos_df), len(neg_df))

        pos_df = pos_df.reindex(range(max_len))
        neg_df = neg_df.reindex(range(max_len))

        combined = pd.DataFrame({
            "Positive Trait": pos_df["Trait"],
            "Positive Probability": pos_df["Probability"],
            "Negative Trait": neg_df["Trait"],
            "Negative Probability": neg_df["Probability"],
        })

        combined.to_excel(writer, sheet_name=category, index=False)

    writer.close()
    print(f"Saved: {output_file}\n")


# ---- Process ALL images in a folder ----
image_folder = "male1"  # change this to the name of the folder with input images

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        process_image(image_path)