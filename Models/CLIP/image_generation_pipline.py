import os
import csv
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def setup_pipeline_directories():
    """Ensures input and output master folders exist on disk."""
    input_dir = Path("./input_images")
    output_master_dir = Path("./output_images")
    
    input_dir.mkdir(exist_ok=True)
    output_master_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Environment initialized. Compute device: {device.upper()}")
    return input_dir, output_master_dir, device

def run_nested_batch_pipeline():
    """Processes every image in a folder against every prompt in a CSV with high realism."""
    try:
        input_dir, output_master_dir, device = setup_pipeline_directories()
        csv_path = Path("prompts.csv")
        
        if not csv_path.exists():
            print(f"❌ Error: The 'prompts.csv' file was not found at {csv_path.resolve()}")
            return
            
        # 1. Gather all target image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"⚠️ No images found in '{input_dir.resolve()}'. Please drop some photos in there and re-run!")
            return
            
        # 2. Read and cache the list of prompts from the CSV
        prompts_list = []
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row_clean = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                if 'prompt_id' in row_clean and 'prompt' in row_clean:
                    prompts_list.append({
                        'id': row_clean['prompt_id'],
                        'text': row_clean['prompt']
                    })
                
        if not prompts_list:
            print("⚠️ The prompts.csv file is empty or missing required headers.")
            return

        print(f"📊 Target Matrix: {len(image_files)} image(s) × {len(prompts_list)} prompt(s) = {len(image_files) * len(prompts_list)} total variations.")

        # 3. Load the ML Engines into system memory
        print("🧠 Loading local BLIP Vision model...")
        vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        
        print("🤖 Loading photorealistic epiCRealism pipeline...")
        model_id = "emilianJR/epiCRealism"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float32)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

        # 4. Outer Loop: Process each image sequentially
        for img_path in image_files:
            img_stem = img_path.stem  
            print(f"\n📂 [STARTING IMAGE GROUP] Processing source picture: {img_path.name}")
            
            # Create a dedicated output subdirectory for this specific input picture
            image_output_folder = output_master_dir / img_stem
            image_output_folder.mkdir(exist_ok=True)
            
            # Step A: Extract visual context from the input picture once
            raw_image = Image.open(img_path).convert('RGB')
            inputs = vision_processor(raw_image, return_tensors="pt").to(device)
            out = vision_model.generate(**inputs, max_new_tokens=20)
            image_caption = vision_processor.decode(out, skip_special_tokens=True)
            
            # 5. Inner Loop: Process every prompt against the current image context
            for index, prompt_item in enumerate(prompts_list, start=1):
                p_id = prompt_item['id']
                user_prompt = prompt_item['text']
                
                print(f"  🎬 [{index}/{len(prompts_list)}] Generating realistic variant: '{p_id}'...")
                
                try:
                    # Appended standard high-fidelity rendering tokens for realistic quality weights
                    refined_prompt = (
                        f"{user_prompt}, keeping the core geometry layout matching a {image_caption}. "
                        f"photorealistic, highly detailed face, sharp focus, 8k resolution, studio lighting, masterpiece"
                    )
                    
                    # Render the frame
                    generator_output = pipe(
                        refined_prompt, 
                        num_inference_steps=15
                    ).images
                    
                    # ⚡ FIXED LINE: Safely pull the first generated PIL Image out of the returned list object
                    final_image = generator_output[0]
                    
                    # Save into the dedicated subfolder
                    final_filename = f"{img_stem}_{p_id}.png"
                    final_path = image_output_folder / final_filename
                    final_image.save(final_path)
                    
                    print(f"    💾 Saved successfully -> {final_path.resolve()}")
                    
                except Exception as inner_err:
                    print(f"    ❌ Error running prompt '{p_id}': {inner_err}")
                    print("    Skipping directly to the next variation...")
                
        print("\n🏁 All nested matrix operations completed successfully with photorealism settings!")
            
    except Exception as e:
        print(f"💥 Batch pipeline matrix execution failed: {e}")

if __name__ == "__main__":
    run_nested_batch_pipeline()
