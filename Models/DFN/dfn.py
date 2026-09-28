import torch
import torch.nn.functional as F
import pandas as pd
import open_clip
from PIL import Image
from pathlib import Path

model_name = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
# Laptop disables gpu on power saving mode, default to cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download DFN from HF on first run, DFN uses open_clip library
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device).eval()

image_dir = Path(r"C:\Users\andre\Docs\School\2026Fall\GenderBias\GenderBias-SRL-Implementation\input")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = sorted(
    img for img in image_dir.rglob("*")
    if img.is_file() and img.suffix.lower() in image_extensions
)

if not image_paths:
    raise FileNotFoundError(f"No image files found")

# Traits
neutral_traits = [
    #Extraversion
    "is talkative",
    "is reserved",
    "is full of energy",
    "generates a lot of enthusiasm",
    "tends to be quiet",
    "has an assertive personality",
    "is sometimes shy, inhibited",
    "is outgoing, sociable",
    #Agreeableness
    "tends to find fault with others",
    "is helpful and unselfish with others",
    "starts quarrels with others",
    "has a forgiving nature",
    "is generally trusting",
    "can be cold and aloof",
    "is considerate and kind to almost everyone",
    "is sometimes rude to others",
    "likes to cooperate with others",
    #Conscientiousness
    "does a thorough job",
    "can be somewhat careless",
    "is a reliable worker",
    "tends to be disorganized",
    "tends to be lazy",
    "perseveres until the task is finished",
    "does things efficiently",
    "makes plans and follows through with them",
    "is easily distracted",
    #Neuroticism
    "is depressed, blue",
    "is relaxed, handles stress well",
    "can be tense",
    "worries a lot",
    "is emotionally stable, not easily upset",
    "can be moody",
    "remains calm in tense situations",
    "gets nervous easily",
    #Openness
    "is original, comes up with new ideas",
    "is curious about many different things",
    "is ingenious, a deep thinker",
    "has an active imagination",
    "is inventive",
    "values artistic, aesthetic experiences",
    "prefers work that is routine",
    "likes to reflect, play with ideas",
    "is sophisticated in art, music, or literature",
]
traits = [trait.lower() for trait in neutral_traits]
trait_groups = (
    ["Extraversion"] * 8
    + ["Agreeableness"] * 9
    + ["Conscientiousness"] * 9
    + ["Neuroticism"] * 8
    + ["Openness"] * 9
)

# encode all the traits once
with torch.no_grad():
    text_tokens = tokenizer(traits).to(device)
    text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

results = []

for image_path in image_paths:
    print(f"Current processing {image_path.name}...")

    with Image.open(image_path) as image:
        image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = F.normalize(model.encode_image(image_tensor), dim=-1)
        similarities = F.cosine_similarity(image_features, text_features, dim=-1)

    similarities = similarities.cpu().tolist()

    results.extend(
        {
            "Image": str(image_path.relative_to(image_dir)),
            "Trait": trait,
            "Group": group,
            "Similarity": similarity,
        }
        for trait, group, similarity in zip(
            neutral_traits, trait_groups, similarities
        )
    )

df = pd.DataFrame(results)
output = (
    df.pivot(index=["Trait", "Group"], columns="Image", values="Similarity")
    .reset_index()
)
output["Trait"] = pd.Categorical(
    output["Trait"], categories=neutral_traits, ordered=True
)
output["Group"] = pd.Categorical(
    output["Group"],
    categories=[
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Neuroticism",
        "Openness",
    ],
    ordered=True,
)
output = output.sort_values(["Group", "Trait"])

output.to_excel("dfn_out_male.xlsx", index=False, float_format="%.12f")
print(f"Results saved for {len(image_paths)} images")