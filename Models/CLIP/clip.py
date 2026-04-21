import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ---- Load CLIP ----
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load image ----
image = Image.open("female1_no_jewerly.png").convert("RGB")

# ---- Trait list ----
neg_traits = [
    "Unlovable","Stupid","Unfriendly","Lazy","Impractical","Insecure"
]

pos_traits = [
    "Lovable","Intelligent","Friendly","Hardworking","Practical","Confident"
]

# Add "a ... person" format for CLIP
neg_traits = [f"a {t.lower()} person" for t in neg_traits]
pos_traits = [f"a {t.lower()} person" for t in pos_traits]

traits = neg_traits + pos_traits

# ---- Prepare inputs ----
inputs = processor(text=traits, images=image, return_tensors="pt", padding=True)

# ---- Forward pass ----
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: (1, num_traits)
    probs = logits_per_image.softmax(dim=1)      # convert to probabilities

# ---- Split probabilities ----
probs = probs.squeeze(0).tolist()

neg_probs = probs[:len(neg_traits)]
pos_probs = probs[len(neg_traits):]

# ---- Create DataFrame ----
df = pd.DataFrame({
    "Negative Trait": neg_traits,
    "Negative Probability": neg_probs,
    "Positive Trait": pos_traits,
    "Positive Probability": pos_probs
})

# ---- Format probabilities to 12 decimals ----
df.to_excel(
    "clip_trait_similarities_female1_no_jewerly.xlsx",
    index=False,
    float_format="%.12f"
)

# ---- Save to Excel ----
df.to_excel("clip_trait_similarities_female1_no_jewerly.xlsx", index=False)
print("Results saved to clip_trait_similarities_female1_no_jewerly.xlsx")