import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ---- Load CLIP ----
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load image ----
image = Image.open("woman1_original.jpg").convert("RGB")

# ---- Trait list ----
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
    "clip_trait_similarities_woman1_original.xlsx",
    index=False,
    float_format="%.12f"
)

# ---- Save to Excel ----
df.to_excel("clip_trait_similarities_woman1_original.xlsx", index=False)
print("Results saved to clip_trait_similarities_woman1_original.xlsx")