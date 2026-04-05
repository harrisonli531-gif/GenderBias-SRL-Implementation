import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ---- Load CLIP ----
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load image ----
image = Image.open("full-length-portrait-cute-woman-gray-wall.jpg").convert("RGB")

# ---- Trait list ----
traits = [
    "Unlovable","Cruel","Criminal","Monstrous","Fraudulent","Hateful","Predatory","Dishonest",
    "Malicious","Stupid","Sadistic","Dirty","Venomous","Barbaric","Unlikeable","Deceitful",
    "Neglectful","False","Disloyal","Heartless","Uncaring","Vindictive","Narcissistic",
    "Ignorant","Unreliable","Ungrateful","Treacherous","Disturbing","Selfish","Destructive",
    "Inconsiderate","Obnoxious","Disrespectful","Insincere","Desperate","Deceptive","Hostile",
    "Grim","Unfriendly","Unstable","Irresponsible","Cowardly","Weak-willed","Shallow",
    "Insulting","Unprincipled","Prejudiced","Narrow-minded","Perverse","Greedy","Weak",
    "Thoughtless","Artificial","Unimpressive","Aggressive","Unappreciative","Frightening",
    "Dull","Brutal","Sloppy","Scheming","Callous","Devious","Intolerant","Sly","Miserable",
    "Tasteless","Remorseless","Discouraging","Lazy","Careless","Uncooperative","Insensitive",
    "Irrational","Morbid","Paranoid","Resentful","Pompous","Foolish","Scornful","Slow",
    "Cold","Conceited","One-dimensional","Tactless","Superficial","Unhealthy",
    "Undisciplined","Uncharitable","Ridiculous","Charmless","Gloomy","Bland", 
    "Unimaginative","Sneaky","Petty","Neurotic","Crass","Stingy","Disruptive","Oppressed",
    "Pretentious","Gullible","Crude","Graceless","Disorderly","Envious","Repressed","Possessive",
    "Egocentric","Troublesome","Abrasive","Meddlesome","Impractical","Dismissive","Misguided",
    "Fickle","Naive","Unconvincing","Amoral","Uncreative","Aimless","Coarse","Unreflective",
    "Insecure","Disorganised","Shameless","Difficult","Unambitious","Short-sighted","Erratic",
    "Obsessive","Childish","Irritable","Absentminded","Unpolished","Unrealistic","Easily discouraged",
    "Confused","Disobedient","Submissive","Contradictory","Presumptuous","Rowdy","Moody",
    "Abrupt","Vulnerable","Mistaken","Rigid","Simple"
]

# Add "a ... person" format for CLIP
traits = [f"a {t.lower()} person" for t in traits]

# ---- Prepare inputs ----
inputs = processor(text=traits, images=image, return_tensors="pt", padding=True)

# ---- Forward pass ----
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: (1, num_traits)
    probs = logits_per_image.softmax(dim=1)      # convert to probabilities

# ---- Convert to DataFrame ----
probs_list = probs.squeeze(0).tolist()
df = pd.DataFrame({
    "Trait": traits,
    "Probability": probs_list
})

# Sort descending by probability
df = df.sort_values(by="Probability", ascending=False)

# ---- Format probabilities to 6 decimals ----
df["Probability"] = df["Probability"].map("{:.12f}".format)

# ---- Save to Excel ----
df.to_excel("clip_trait_similarities_full_float.xlsx", index=False)
print("Results saved to clip_trait_similarities_full_float.xlsx")