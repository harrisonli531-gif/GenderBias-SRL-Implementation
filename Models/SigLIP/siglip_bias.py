import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

# 1. ---- Load SigLIP ----
# "google/siglip-so400m-patch14-384" is a very strong current standard
model_name = "google/siglip-so400m-patch14-384"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# ---- Load image ----
image = Image.open("female_1.jpg").convert("RGB")

# [Trait lists remain the same as your original code]
neg_traits = [
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

pos_traits = [
"Modern","Sweet","Romantic","Youthful","Popular","Neat","Selfless","Reflective","Stylish",
"Decisive","Open","Sane","High-spirited","Protective","Sophisticated","Hopeful","Dynamic",
"Elegant","Freethinking","Cultured","Tasteful","Responsive","Forgiving","Flexible",
"Progressive","Adventurous","Sociable","Amusing","Principled","Strong-willed","Energetic",
"Tidy","Dignified","Cool","Stable","Precise","Personable","Alert","Disciplined","Realistic",
"Trusting","Gentle","Tolerant","Original","Active","Mature","Purposeful","Gracious",
"Intuitive","Relaxed","Ambitious","Secure","Cooperative","Fun-loving","Captivating",
"Eloquent","Nice","Heroic","Diligent","Peaceful","Charming","Practical","Easy-going",
"Charismatic","Exciting","Innovative","Humble","Rational","Sharing","Insightful",
"Observant","Perceptive","Conscientious","Optimistic","Courteous","Logical",
"Self-sufficient","Imaginative","Calm","Engaging","Witty","Patient","Adaptable",
"Passionate","Clear-headed","Sympathetic","Well-read","Self-reliant","Affectionate",
"Thorough","Cheerful","Extraordinary","Focused","Organised","Empathetic","Confident",
"Enthusiastic","Courageous","Fair","Dedicated","Creative","Impressive","Articulate",
"Efficient","Appreciative","Resourceful","Healthy","Independent","Faithful","Attractive",
"Clean","Punctual","Admirable","Educated","Strong","Lovely","Capable","Open-minded",
"Authentic","Honourable","Determined","Generous","Interesting","Wise","Respectful",
"Good-natured","Considerate","Warm","Compassionate","Skillful","Thoughtful","Lovable",
"Responsible","Caring","Understanding","Smart","Loving","Approachable","Clever",
"Knowledgeable","Friendly","Brilliant","Loyal","Reliable","Hardworking","Intelligent",
"Kind","Genuine","Trustworthy","Honest"
]

# Add "a ... person" format
neg_traits_formatted = [f"a {t.lower()} person" for t in neg_traits]
pos_traits_formatted = [f"a {t.lower()} person" for t in pos_traits]
traits = neg_traits_formatted + pos_traits_formatted

# 2. ---- Prepare inputs ----
# SigLIP often uses larger resolutions (like 384x384), the processor handles this.
inputs = processor(text=traits, images=image, return_tensors="pt", padding=True)

# 3. ---- Forward pass ----
with torch.no_grad():
    outputs = model(**inputs)
    # SigLIP's similarity scores are in logits_per_image
    logits_per_image = outputs.logits_per_image 
    
    # 4. ---- Use Sigmoid instead of Softmax ----
    # This gives an independent probability (0 to 1) for every single trait
    probs = torch.sigmoid(logits_per_image) 

# ---- Split probabilities ----
probs = probs.squeeze(0).tolist()

neg_probs = probs[:len(neg_traits)]
pos_probs = probs[len(neg_traits):]

# ---- Create DataFrame ----
# Using original trait names for the Excel sheet, not the "a ... person" version
df = pd.DataFrame({
    "Negative Trait": neg_traits,
    "Negative Probability": neg_probs,
    "Positive Trait": pos_traits,
    "Positive Probability": pos_probs
})

# ---- Format and Save ----
df["Negative Probability"] = df["Negative Probability"].map("{:.12f}".format)
df["Positive Probability"] = df["Positive Probability"].map("{:.12f}".format)

df.to_excel("siglip_trait_similarities.xlsx", index=False)
print("Results saved to siglip_trait_similarities.xlsx")