import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ---- Load CLIP ----
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load image ----
image = Image.open("female1_short_hair.png").convert("RGB")

# ---- Load Big Five trait mapping ----
df = pd.read_excel("The_Big_Five_traits.xlsx")
# Expected columns: Category, Trait, Valence

# ---- Prepare prompts ----
df["Trait_clean"] = df["Trait"].str.lower().str.strip()
df["Prompt"] = df["Trait_clean"]

traits = df["Prompt"].tolist()

# ---- Run CLIP ----
inputs = processor(text=traits, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

df["Probability"] = probs.squeeze(0).tolist()

# ---- Create Excel writer ----
output_file = "clip_big5_by_category_female1_short_hair.xlsx"
writer = pd.ExcelWriter(output_file, engine="xlsxwriter")

# ---- Process each category ----
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

    # Align lengths (pad shorter one)
    max_len = max(len(pos_df), len(neg_df))

    pos_df = pos_df.reindex(range(max_len))
    neg_df = neg_df.reindex(range(max_len))

    # Combine side-by-side
    combined = pd.DataFrame({
        "Positive Trait": pos_df["Trait"],
        "Positive Probability": pos_df["Probability"],
        "Negative Trait": neg_df["Trait"],
        "Negative Probability": neg_df["Probability"],
    })

    # Write to sheet (sheet name = category)
    combined.to_excel(writer, sheet_name=category, index=False)

# ---- Save file ----
writer.close()

print(f"Saved multi-sheet Excel file: {output_file}")