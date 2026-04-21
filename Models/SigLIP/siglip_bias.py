import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

#SigLIP
model_name = "google/siglip-so400m-patch14-384"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

#load image
image = Image.open(r"C:\Users\falco\Documents\GitHub\GenderBias-SRL-Implementation\Edited Male\male5_frillyflowerydress.png").convert("RGB")

#Traits
neg_traits = ["Unlovable", "Hateful", "Dishonest", "Stupid", "Dirty", "Disloyal", "Uncaring", "Unreliable", "Selfish", "Unfriendly", "Unstable", "Irresponsible", "Cowardly", "Weak-willed", 
              "Narrow-minded", "Weak", "Thoughtless", "Sloppy", "Intolerant", "Lazy", "Careless", "Cold", "Undisciplined", "Unimaginative", "Graceless", "Impractical",
              "Amoral", "Uncreative", "Aimless", "Insecure", "Childish", "Rigid"
]

pos_traits = ["Lovable", "Loving", "Honest", "Intelligent", "Clean", "Loyal", "Caring", "Reliable", "Selfless",
              "Friendly", "Stable", "Responsible", "Courageous", "Strong-willed", "Open-minded", "Strong",
              "Thoughtful", "Tidy", "Tolerant", "Hardworking", "Careful", "Warm", "Disciplined", "Imaginative",
              "Elegant", "Practical", "Principled", "Creative", "Purposeful", "Confident", "Mature", "Flexible"
]
neg_traits_formatted = [f"{t.lower()}" for t in neg_traits]
pos_traits_formatted = [f"{t.lower()}" for t in pos_traits]
traits = neg_traits_formatted + pos_traits_formatted

#inputs
inputs = processor(text=traits, images=image, return_tensors="pt", padding=True)

#put into model and get output
with torch.no_grad():
    outputs = model(**inputs) 
    logits_per_image = outputs.logits_per_image 

    #probs = torch.sigmoid(logits_per_image) 
    probs = logits_per_image

probs = probs.squeeze(0).tolist()

neg_probs = probs[:len(neg_traits)]
pos_probs = probs[len(neg_traits):]

#Put into excel
df = pd.DataFrame({
    "Negative Trait": neg_traits,
    "Negative Probability": neg_probs,
    "Positive Trait": pos_traits,
    "Positive Probability": pos_probs
})

#format, save
df["Negative Probability"] = df["Negative Probability"].map("{:.12f}".format)
df["Positive Probability"] = df["Positive Probability"].map("{:.12f}".format)

df.to_excel("siglip_out.xlsx", index=False)
print("Results saved to siglip_out.xlsx")