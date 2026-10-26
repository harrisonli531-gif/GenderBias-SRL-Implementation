import os
import io
import csv
import torch
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def setup_pipeline():
    """Sets up the output directory and confirms compute device."""
    output_dir = Path("./output_images")
    output_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Directories ready. System computing on device: {device.upper()}")
    return output_dir, device

def build_batch_csv_template():
    """Helper to generate a batch template file if it doesn't exist yet."""
    csv_path = Path("pipeline_inputs.csv")
    if not csv_path.exists():
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["input_image_path", "prompt", "output_filename"])
            writer.writerow([
                "test_input.jpg", 
                "Transform this into an intricate neon cyber-punk crystal artifact, highly detailed", 
                "cyberpunk_result"
            ])
        print(f"📝 Created a fresh batch template spreadsheet at: {csv_path.resolve()}")
    return csv_path

def run_batch_pipeline():
    """Systematically processes pairs of prompts and images from a CSV file."""
    try:
        output_dir, device = setup_pipeline()
        csv_path = build_batch_csv_template()
        
        # 1. Load the ML engine layers into memory ONCE
        print("🧠 Loading local BLIP Vision model into memory...")
        vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        
        print("🤖 Loading local Stable Diffusion engine into memory...")
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float32)
        
        # ⚡ OPTIMIZATION 1: Swap scheduler to DPM-Solver for faster CPU rendering
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        print(f"📊 Reading processing instructions from {csv_path}...")
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                img_path = row['input_image_path']
                user_prompt = row['prompt']
                out_name = row['output_filename']
                
                print(f"\n========================================\n🚀 Processing asset: {out_name}...")
                
                if not os.path.exists(img_path):
                    print(f"❌ Skipping: Source image file not found at '{img_path}'")
                    continue
                
                # Step A: Run local image captioning
                raw_image = Image.open(img_path).convert('RGB')
                inputs = vision_processor(raw_image, return_tensors="pt").to(device)
                out = vision_model.generate(**inputs, max_new_tokens=20)
                image_caption = vision_processor.decode(out, skip_special_tokens=True)
                
                # Step B: Merge descriptions 
                refined_prompt = f"{user_prompt}, keeping the core geometry layout matching a {image_caption}"
                print(f"✨ Custom Prompt Matrix: '{refined_prompt}'")
                
                # Step C: Render final output locally
                print(f"🎨 Rendering image output (Running on {device.upper()})...")
                
                # ⚡ OPTIMIZATION 2: Cut generation steps down to 15 (Perfect balance for CPUs)
                generator_output = pipe(
                    refined_prompt, 
                    num_inference_steps=15
                ).images[0] # Grab the first generated PIL Image object explicitly
                
                # Step D: Save output
                final_path = output_dir / f"{out_name}.png"
                generator_output.save(final_path)
                print(f"🎉 Success! Asset saved to: {final_path.resolve()}")
                
        print("\n🏁 Batch processing pipeline completed successfully!")
            
    except Exception as e:
        print(f"💥 Batch pipeline execution crashed: {e}")

if __name__ == "__main__":
    run_batch_pipeline()
